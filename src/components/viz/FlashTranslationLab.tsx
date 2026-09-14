import { useEffect, useMemo, useRef, useState } from 'react';
import {
  VizPanel,
  Slider,
  Choice,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useTicker,
  useSize,
  makeRng,
  fmtNum,
  fmtTime,
} from './Viz';

/* ------------------------------------------------------------------ model */

export const PAGES_PER_BLOCK = 8;
export const DATA_BLOCKS = 12;
/** Host-visible capacity, in flash pages. 12 blocks x 8 pages of 16 KB = 1.5 MB. */
export const LOGICAL_PAGES = DATA_BLOCKS * PAGES_PER_BLOCK; // 96
export const PAGE_KB = 16;

/** Timing model, in microseconds. Real TLC: tPROG ~700us, tR ~60us, tBERS ~5ms;
 *  the controller interleaves across ~8 dies, so a lone host write is cheap and a
 *  host write that lands behind garbage collection is not. */
const T_WRITE_US = 25;
const T_COPY_US = 95; // one live-page relocation: read + program, amortized over the dies
const T_ERASE_US = 1500; // the erase the blocked write has to wait out
const T_DISCARD_US = 6;

export type Workload = 'seq' | 'hot' | 'uniform';

export const WORKLOAD_OPTIONS: readonly { value: Workload; label: string }[] = [
  { value: 'seq', label: 'Sequential overwrite (WAL / log)' },
  { value: 'hot', label: 'Hot / cold 80-20 (B-tree updates)' },
  { value: 'uniform', label: 'Uniform random' },
];

export type Op = { kind: 'write' | 'discard'; lba: number };

/**
 * The request stream both halves of this page replay. It is a property of the
 * *host*, not of the device: the same writes and deletes are issued whether they
 * land on NAND or on a platter.
 */
export function makeRequestStream(workload: Workload, n: number): Op[] {
  const rng = makeRng(workload === 'seq' ? 7 : workload === 'hot' ? 11 : 23);
  const ops: Op[] = [];
  const written: number[] = [];
  const seen = new Uint8Array(LOGICAL_PAGES);
  const hot = Math.max(1, Math.round(LOGICAL_PAGES * 0.2));
  let cursor = 0;
  // A pure sequential overwrite (a WAL being rewritten in place) issues no deletes;
  // the update-in-place workloads delete roughly one page in ten.
  const discardRate = workload === 'seq' ? 0 : 0.1;
  for (let i = 0; i < n; i++) {
    if (written.length > 12 && rng() < discardRate) {
      ops.push({ kind: 'discard', lba: written[Math.floor(rng() * written.length) % written.length] });
      continue;
    }
    let lba: number;
    if (workload === 'seq') {
      lba = cursor % LOGICAL_PAGES;
      cursor++;
    } else if (workload === 'hot') {
      lba = rng() < 0.8 ? Math.floor(rng() * hot) : hot + Math.floor(rng() * (LOGICAL_PAGES - hot));
    } else {
      lba = Math.floor(rng() * LOGICAL_PAGES);
    }
    lba = Math.max(0, Math.min(LOGICAL_PAGES - 1, lba));
    ops.push({ kind: 'write', lba });
    if (!seen[lba]) {
      seen[lba] = 1;
      written.push(lba);
    }
  }
  return ops;
}

type TraceRow = {
  i: number;
  kind: 'write' | 'discard';
  lba: number;
  us: number;
  copies: number;
  erased: number;
};

type FlashState = {
  nBlocks: number;
  /** 0 = free, 1 = valid (live), 2 = invalid (garbage) */
  st: Int8Array;
  lbaOf: Int32Array;
  erases: Int32Array;
  valid: Int32Array;
  invalid: Int32Array;
  openBlk: number;
  lastVictim: number;
  hostWrites: number;
  nandWrites: number;
  gcCopies: number;
  totalErase: number;
  gcRuns: number;
  stalls: number;
  livePages: number;
  trace: TraceRow[];
};

function simulateFlash(
  workload: Workload,
  spareBlocks: number,
  trim: boolean,
  steps: number,
): FlashState {
  const ppb = PAGES_PER_BLOCK;
  const nBlocks = DATA_BLOCKS + spareBlocks;
  const st = new Int8Array(nBlocks * ppb);
  const lbaOf = new Int32Array(nBlocks * ppb).fill(-1);
  const erases = new Int32Array(nBlocks);
  const valid = new Int32Array(nBlocks);
  const invalid = new Int32Array(nBlocks);
  const map = new Int32Array(LOGICAL_PAGES).fill(-1);

  let openBlk = 0;
  let openPg = 0;
  let lastVictim = -1;
  let hostWrites = 0;
  let nandWrites = 0;
  let gcCopies = 0;
  let totalErase = 0;
  let gcRuns = 0;
  let stalls = 0;
  const trace: TraceRow[] = [];

  const isFree = (b: number) => valid[b] === 0 && invalid[b] === 0;
  const freeBlocks = () => {
    let c = 0;
    for (let b = 0; b < nBlocks; b++) if (b !== openBlk && isFree(b)) c++;
    return c;
  };
  /** Erased pages the controller can still program without erasing anything. */
  const freePages = () => ppb - openPg + freeBlocks() * ppb;

  function openNewBlock(): boolean {
    for (let b = 0; b < nBlocks; b++) {
      if (b !== openBlk && isFree(b)) {
        openBlk = b;
        openPg = 0;
        return true;
      }
    }
    return false;
  }

  /** The write pointer only ever moves forward inside an erased block. */
  function alloc(): number {
    if (openPg >= ppb && !openNewBlock()) return -1;
    const p = openBlk * ppb + openPg;
    openPg++;
    return p;
  }

  function invalidate(lba: number) {
    const p = map[lba];
    if (p < 0) return;
    const b = Math.floor(p / ppb);
    st[p] = 2;
    lbaOf[p] = -1;
    valid[b]--;
    invalid[b]++;
    map[lba] = -1;
  }

  function erase(b: number) {
    for (let i = 0; i < ppb; i++) {
      st[b * ppb + i] = 0;
      lbaOf[b * ppb + i] = -1;
    }
    valid[b] = 0;
    invalid[b] = 0;
    erases[b]++;
    totalErase++;
  }

  /**
   * Greedy garbage collection. It runs when the free-page pool drops below two
   * blocks' worth, picks the block holding the most garbage, relocates whatever is
   * still live in it, and erases it. Everything expensive about an SSD is here.
   */
  function collect() {
    let copies = 0;
    let erased = 0;
    let guard = 0;
    while (freePages() < 2 * ppb && guard++ < 24) {
      let v = -1;
      let best = 0;
      for (let b = 0; b < nBlocks; b++) {
        if (b === openBlk || isFree(b)) continue;
        if (invalid[b] > best) {
          best = invalid[b];
          v = b;
        }
      }
      if (v < 0) break; // nothing to reclaim: every programmed page is still live
      lastVictim = v;
      gcRuns++;
      let failed = false;
      for (let i = 0; i < ppb; i++) {
        const p = v * ppb + i;
        if (st[p] !== 1) continue;
        const np = alloc();
        if (np < 0) {
          failed = true;
          break;
        }
        const l = lbaOf[p];
        const nb = Math.floor(np / ppb);
        st[np] = 1;
        lbaOf[np] = l;
        valid[nb]++;
        map[l] = np;
        st[p] = 2;
        lbaOf[p] = -1;
        valid[v]--;
        invalid[v]++;
        nandWrites++;
        gcCopies++;
        copies++;
      }
      if (failed) break; // never erase a block whose live pages are not all relocated
      erase(v);
      erased++;
    }
    return { copies, erased };
  }

  const ops = makeRequestStream(workload, steps);
  for (let i = 0; i < steps; i++) {
    const op = ops[i];
    if (op.kind === 'discard') {
      // Without TRIM the drive never hears about the delete: the page stays "live"
      // and garbage collection will keep relocating data nobody wants.
      if (trim) invalidate(op.lba);
      trace.push({ i, kind: 'discard', lba: op.lba, us: T_DISCARD_US, copies: 0, erased: 0 });
      continue;
    }
    invalidate(op.lba);
    const g = collect();
    const p = alloc();
    const us = T_WRITE_US + g.copies * T_COPY_US + g.erased * T_ERASE_US;
    if (p < 0) {
      stalls++;
      trace.push({ i, kind: 'write', lba: op.lba, us, copies: g.copies, erased: g.erased });
      continue;
    }
    const b = Math.floor(p / ppb);
    st[p] = 1;
    lbaOf[p] = op.lba;
    valid[b]++;
    map[op.lba] = p;
    hostWrites++;
    nandWrites++;
    trace.push({ i, kind: 'write', lba: op.lba, us, copies: g.copies, erased: g.erased });
  }

  let livePages = 0;
  for (let b = 0; b < nBlocks; b++) livePages += valid[b];

  return {
    nBlocks,
    st,
    lbaOf,
    erases,
    valid,
    invalid,
    openBlk,
    lastVictim,
    hostWrites,
    nandWrites,
    gcCopies,
    totalErase,
    gcRuns,
    stalls,
    livePages,
    trace,
  };
}

/* ---------------------------------------------------------------- drawing */

const CELL = 13;
const GAP = 2;
const COLS = 4;
const BW = COLS * (CELL + GAP) - GAP; // 58
const BH = 2 * (CELL + GAP) - GAP; // 28
const BLOCK_W = BW + 20;
const BLOCK_H = BH + 24;

const MAX_STEPS = 600;
const TRACE_N = 48;

const PAGE_FILL = ['var(--viz-neutral)', 'var(--viz-clean)', 'var(--viz-stale)'];

function waColor(wa: number) {
  if (wa <= 1.5) return 'var(--viz-good)';
  if (wa <= 2.5) return 'var(--viz-warning)';
  if (wa <= 4) return 'var(--viz-serious)';
  return 'var(--viz-critical)';
}

export default function FlashTranslationLab() {
  const [workload, setWorkload] = useState<Workload>('hot');
  const [spare, setSpare] = useState(2);
  const [trim, setTrim] = useState(true);
  const [steps, setSteps] = useState(120);
  const [running, setRunning] = useState(false);
  const [ref, width] = useSize(760);
  const tip = useTip();
  const acc = useRef(0);

  const sim = useMemo(() => simulateFlash(workload, spare, trim, steps), [workload, spare, trim, steps]);

  useTicker((dt) => {
    acc.current += dt;
    if (acc.current < 80) return;
    acc.current = 0;
    setSteps((s) => (s >= MAX_STEPS ? s : s + 1));
  }, running);

  useEffect(() => {
    if (steps >= MAX_STEPS) setRunning(false);
  }, [steps]);

  const perRow = Math.max(3, Math.min(7, Math.floor((width - 8) / BLOCK_W)));
  const rows = Math.ceil(sim.nBlocks / perRow);
  const gridW = perRow * BLOCK_W;
  const gridH = rows * BLOCK_H + 6;
  const gaugeY = gridH + 16;
  const traceY = gaugeY + 52;
  const traceH = 84;
  const svgW = Math.max(340, Math.min(width, Math.max(gridW, 420)));
  const svgH = traceY + traceH + 22;

  const wa = sim.hostWrites > 0 ? sim.nandWrites / sim.hostWrites : 1;
  const opPct = (spare / DATA_BLOCKS) * 100;
  const writes = sim.trace.filter((t) => t.kind === 'write');
  const sorted = writes.map((t) => t.us).sort((a, b) => a - b);
  const p99 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.99))] : 0;
  const worst = sorted.length ? sorted[sorted.length - 1] : 0;
  const tail = sim.trace.slice(-TRACE_N);
  const last = sim.trace[sim.trace.length - 1];
  const freeBlockCount = (() => {
    let c = 0;
    for (let b = 0; b < sim.nBlocks; b++) if (b !== sim.openBlk && sim.valid[b] === 0 && sim.invalid[b] === 0) c++;
    return c;
  })();
  const wearSpread = (() => {
    let lo = Infinity;
    let hi = 0;
    for (let b = 0; b < sim.nBlocks; b++) {
      lo = Math.min(lo, sim.erases[b]);
      hi = Math.max(hi, sim.erases[b]);
    }
    return hi - (lo === Infinity ? 0 : lo);
  })();

  // Latency trace: log scale from 10us to 30ms.
  const LO = Math.log10(10);
  const HI = Math.log10(30000);
  const ty = (us: number) => traceY + traceH - ((Math.log10(Math.max(10, us)) - LO) / (HI - LO)) * traceH;
  const barW = Math.max(2, Math.min(9, (svgW - 46) / TRACE_N - 1));

  const gaugeW = Math.min(svgW - 130, 260);
  const gx = (v: number) => 96 + ((Math.min(6, Math.max(1, v)) - 1) / 5) * gaugeW;

  return (
    <VizPanel
      title="An SSD is a log that pretends to be an array of sectors"
      subtitle="Every host write lands on a fresh flash page; the old copy becomes garbage. Issue writes and watch the FTL map move, blocks fill with invalid pages, and garbage collection pay the bill."
      controls={
        <>
          <Choice label="Workload" value={workload} onChange={setWorkload} options={WORKLOAD_OPTIONS} />
          <Slider
            label="Over-provisioning"
            min={1}
            max={8}
            value={spare}
            onChange={setSpare}
            format={(n) => `${((n / DATA_BLOCKS) * 100).toFixed(0)}% (${n} spare blocks)`}
          />
          <Check label="TRIM / discard honored" checked={trim} onChange={setTrim} />
          <Button onClick={() => setSteps((s) => Math.min(MAX_STEPS, s + 1))}>Step</Button>
          <Button onClick={() => setSteps((s) => Math.min(MAX_STEPS, s + 25))}>+25</Button>
          <Button primary onClick={() => setRunning((r) => !r)}>
            {running ? 'Pause' : 'Run'}
          </Button>
          <Button
            onClick={() => {
              setRunning(false);
              setSteps(0);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Erased / free page', color: 'var(--viz-neutral)' },
            { label: 'Live page (mapped)', color: 'var(--viz-clean)' },
            { label: 'Invalid page (garbage)', color: 'var(--viz-stale)' },
            { label: 'Open block (write pointer)', color: 'var(--viz-dirty)', shape: 'line' },
            { label: 'Last GC victim', color: 'var(--viz-warning)', shape: 'line' },
            { label: 'Write that paid for GC relocations', color: 'var(--viz-warning)', shape: 'dot' },
            { label: 'Write blocked behind a block erase (ms)', color: 'var(--viz-critical)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Host writes', value: fmtNum(sim.hostWrites), hint: 'Pages the filesystem asked for' },
            { label: 'NAND programs', value: fmtNum(sim.nandWrites), hint: 'Pages actually programmed, including GC relocations' },
            { label: 'Write amplification', value: `${wa.toFixed(2)}×`, hint: 'NAND programs / host writes' },
            { label: 'Block erases', value: fmtNum(sim.totalErase) },
            { label: 'Free blocks', value: `${freeBlockCount} / ${sim.nBlocks}` },
            { label: 'p99 write latency', value: fmtTime(p99 * 1000), hint: `worst ${fmtTime(worst * 1000)}` },
          ]}
        />
      }
      note={
        <Note>
          {last ? (
            <>
              <strong>
                op {last.i + 1}: {last.kind === 'discard' ? `DISCARD lba ${last.lba}` : `WRITE lba ${last.lba}`}
              </strong>{' '}
              {last.kind === 'discard'
                ? trim
                  ? '— the drive marks that physical page invalid immediately, so GC will never copy it again.'
                  : '— TRIM is off, so the drive still believes this page is live and will keep relocating it forever.'
                : last.copies > 0
                  ? `blocked behind GC: ${last.copies} live page${last.copies === 1 ? '' : 's'} relocated and ${last.erased} block${last.erased === 1 ? '' : 's'} erased before the host's 16 KB could land — ${fmtTime(last.us * 1000)} instead of ${fmtTime(T_WRITE_US * 1000)}.`
                  : `straight into the open block at ${fmtTime(last.us * 1000)}; no collection needed yet.`}{' '}
              Write amplification is <strong>{wa.toFixed(2)}×</strong> at {opPct.toFixed(0)}% over-provisioning
              {sim.stalls > 0 ? `, and ${sim.stalls} write(s) found no free page at all` : ''}.
            </>
          ) : (
            <>Press <strong>Run</strong>. Nothing has been written yet: every block is erased and the FTL map is empty.</>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Block</th>
                <th>Live</th>
                <th>Invalid</th>
                <th>Erased</th>
                <th>P/E cycles</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: sim.nBlocks }, (_, b) => (
                <tr key={b}>
                  <td>B{b}</td>
                  <td>{sim.valid[b]}</td>
                  <td>{sim.invalid[b]}</td>
                  <td>{PAGES_PER_BLOCK - sim.valid[b] - sim.invalid[b]}</td>
                  <td>{sim.erases[b]}</td>
                  <td>
                    {b === sim.openBlk
                      ? 'open (write pointer)'
                      : b === sim.lastVictim
                        ? 'last GC victim'
                        : sim.valid[b] === 0 && sim.invalid[b] === 0
                          ? 'free'
                          : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            Timing model: one host page program {fmtTime(T_WRITE_US * 1000)} with die interleaving, one GC
            relocation {fmtTime(T_COPY_US * 1000)} (NAND read + program), one block erase{' '}
            {fmtTime(T_ERASE_US * 1000)}. Page = {PAGE_KB} KB, block = {PAGES_PER_BLOCK} pages, host capacity ={' '}
            {LOGICAL_PAGES} pages. Wear spread (max − min P/E count) = {wearSpread}.
          </p>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label="NAND blocks, write amplification gauge and per-write latency trace">
            {/* ---- the NAND array ---- */}
            {Array.from({ length: sim.nBlocks }, (_, b) => {
              const col = b % perRow;
              const row = Math.floor(b / perRow);
              const bx = col * BLOCK_W;
              const by = row * BLOCK_H + 12;
              const free = PAGES_PER_BLOCK - sim.valid[b] - sim.invalid[b];
              const isOpen = b === sim.openBlk;
              const isVictim = b === sim.lastVictim && !isOpen;
              const stroke = isOpen
                ? 'var(--viz-dirty)'
                : isVictim
                  ? 'var(--viz-warning)'
                  : 'var(--viz-axis)';
              return (
                <g
                  key={b}
                  {...tip(
                    <>
                      <strong>Block {b}</strong>
                      <br />
                      {sim.valid[b]} live · {sim.invalid[b]} invalid · {free} erased
                      <br />
                      {sim.erases[b]} program/erase cycle{sim.erases[b] === 1 ? '' : 's'}
                      {isOpen ? ' · open block, write pointer here' : ''}
                      {isVictim ? ' · most recent GC victim' : ''}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect
                    x={bx - 4}
                    y={by - 4}
                    width={BW + 8}
                    height={BH + 8}
                    rx={5}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={isOpen || isVictim ? 2 : 1}
                  />
                  <text x={bx - 3} y={by - 8} fontSize={9} fill="var(--viz-ink-2)">
                    B{b}
                  </text>
                  <text x={bx + BW + 3} y={by - 8} fontSize={9} textAnchor="end" fill="var(--viz-ink-muted)">
                    {sim.erases[b]} P/E
                  </text>
                  {Array.from({ length: PAGES_PER_BLOCK }, (_, i) => {
                    const p = b * PAGES_PER_BLOCK + i;
                    const px = bx + (i % COLS) * (CELL + GAP);
                    const py = by + Math.floor(i / COLS) * (CELL + GAP);
                    return (
                      <rect
                        key={i}
                        x={px}
                        y={py}
                        width={CELL}
                        height={CELL}
                        rx={2}
                        fill={PAGE_FILL[sim.st[p]]}
                        stroke="var(--viz-surface)"
                        strokeWidth={1}
                      />
                    );
                  })}
                </g>
              );
            })}

            {/* ---- write-amplification gauge ---- */}
            <text x={0} y={gaugeY + 14} fontSize={11} fill="var(--viz-ink)">
              Write amp.
            </text>
            <rect x={96} y={gaugeY + 4} width={gaugeW} height={12} rx={6} fill="var(--viz-neutral)" />
            <rect
              x={96}
              y={gaugeY + 4}
              width={Math.max(3, gx(wa) - 96)}
              height={12}
              rx={6}
              fill={waColor(wa)}
            />
            {[1, 2, 3, 4, 5, 6].map((v) => (
              <g key={v}>
                <line className="viz-axis-line" x1={gx(v)} x2={gx(v)} y1={gaugeY + 18} y2={gaugeY + 22} />
                <text x={gx(v)} y={gaugeY + 33} fontSize={9} textAnchor="middle" fill="var(--viz-ink-muted)">
                  {v}×
                </text>
              </g>
            ))}
            <text
              x={96 + gaugeW + 10}
              y={gaugeY + 14}
              fontSize={12}
              fill="var(--viz-ink)"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {wa.toFixed(2)}×
            </text>

            {/* ---- per-write latency trace ---- */}
            <text x={0} y={traceY - 4} fontSize={11} fill="var(--viz-ink)">
              Latency per operation (log scale, last {TRACE_N})
            </text>
            {[10, 100, 1000, 10000].map((v) => (
              <g key={v}>
                <line className="viz-grid-line" x1={44} x2={svgW} y1={ty(v)} y2={ty(v)} />
                <text x={40} y={ty(v) + 3} fontSize={9} textAnchor="end" fill="var(--viz-ink-muted)">
                  {fmtTime(v * 1000)}
                </text>
              </g>
            ))}
            {tail.map((t, k) => {
              const x = 46 + k * ((svgW - 46) / TRACE_N);
              const y = ty(t.us);
              const fill =
                t.kind === 'discard'
                  ? 'var(--viz-stale)'
                  : t.us > 1000
                    ? 'var(--viz-critical)'
                    : t.copies > 0
                      ? 'var(--viz-warning)'
                      : 'var(--viz-clean)';
              return (
                <rect
                  key={t.i}
                  x={x}
                  y={y}
                  width={barW}
                  height={Math.max(1.5, traceY + traceH - y)}
                  rx={1}
                  fill={fill}
                  {...tip(
                    <>
                      <strong>op {t.i + 1}</strong> · {t.kind === 'discard' ? 'DISCARD' : 'WRITE'} lba {t.lba}
                      <br />
                      {fmtTime(t.us * 1000)}
                      {t.copies > 0 ? ` · ${t.copies} pages relocated, ${t.erased} erase(s)` : ''}
                    </>,
                  )}
                />
              );
            })}
            <line className="viz-axis-line" x1={44} x2={svgW} y1={traceY + traceH} y2={traceY + traceH} />
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
