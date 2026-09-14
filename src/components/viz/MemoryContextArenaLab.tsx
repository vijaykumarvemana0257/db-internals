import { useMemo, useState } from 'react';
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
  fmtBytes,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * A PostgreSQL memory context, allocating for real.
 *
 * AllocSet rounds every request up to a power-of-two size class, carves it out of blocks that
 * double from 8 kB toward 8 MB, and never returns a freed chunk to libc — it parks it on a
 * per-size-class freelist. Generation bump-allocates with no freelist and frees a block only
 * when its last chunk dies, so one survivor pins 8 kB. Slab cuts fixed-size chunks and can
 * always give a block back.
 *
 * The payoff is the same in all three: MemoryContextReset() costs one free() per *block*,
 * not per chunk, which is what makes per-tuple and per-transaction memory disposable.
 */

const KB = 1024;
const MB = 1024 * KB;

const CHUNK_HDR = 8; // PostgreSQL 16 shrank MemoryChunk's header to 8 bytes
const BLOCK_HDR = 32;
const INIT_BLOCK = 8 * KB; // ALLOCSET_DEFAULT_INITSIZE
const MAX_BLOCK = 8 * MB; // ALLOCSET_DEFAULT_MAXSIZE
const CHUNK_LIMIT = 8 * KB; // allocChunkLimit: above this a chunk gets a block of its own
const SLAB_BLOCK = 8 * KB;
const MAX_BLOCKS = 9;

const REQ_STEPS = [16, 24, 40, 72, 130, 260, 700, 3000, 9000];

type CtxKind = 'aset' | 'generation' | 'slab';

const CTX: Record<CtxKind, { label: string; where: string }> = {
  aset: {
    label: 'AllocSet (aset.c) — the default',
    where:
      'Every palloc() you have ever read about: ExecutorState, MessageContext, CacheMemoryContext, TopTransactionContext.',
  },
  generation: {
    label: 'Generation (generation.c) — bump, no freelist',
    where:
      'Allocations with a similar lifetime that are freed roughly in order — logical decoding’s ReorderBuffer is the canonical user.',
  },
  slab: {
    label: 'Slab (slab.c) — one fixed chunk size',
    where: 'Many identical structs with scattered lifetimes, again from the ReorderBuffer.',
  },
};

type Chunk = { id: number; req: number; payload: number; live: boolean };
type Block = { id: number; cap: number; used: number; dedicated: boolean; chunks: Chunk[] };

type S = {
  blocks: Block[];
  nextBlockSize: number;
  nextBlock: number;
  nextChunk: number;
  mallocs: number;
  frees: number;
  step: number;
  head: string;
  body: string;
};

const INITIAL: S = {
  blocks: [],
  nextBlockSize: INIT_BLOCK,
  nextBlock: 1,
  nextChunk: 1,
  mallocs: 0,
  frees: 0,
  step: 0,
  head: 'An empty context.',
  body:
    'A MemoryContext is a node in a tree, not a heap. palloc() carves from blocks the context owns; ' +
    'MemoryContextReset() throws all of them away at once. Allocate a few chunks and watch the blocks appear.',
};

/** Deterministic size jitter so "vary sizes" is reproducible across SSR and hydration. */
const JITTER = (() => {
  const rng = makeRng(20230711);
  return Array.from({ length: 128 }, () => 0.45 + rng() * 1.35);
})();

function align8(n: number) {
  return Math.ceil(n / 8) * 8;
}

/** AllocSetFreeIndex: round the request up to a power-of-two size class, minimum 8 bytes. */
function sizeClass(req: number) {
  let s = 8;
  while (s < req) s *= 2;
  return s;
}

function payloadFor(kind: CtxKind, req: number, slabChunk: number) {
  if (kind === 'aset') return req > CHUNK_LIMIT ? align8(req) : sizeClass(req);
  if (kind === 'slab') return slabChunk;
  return align8(req);
}

function palloc(s: S, kind: CtxKind, req: number, slabChunk: number): S {
  const blocks = s.blocks.map((b) => ({ ...b, chunks: [...b.chunks] }));
  const payload = payloadFor(kind, req, slabChunk);
  const need = payload + CHUNK_HDR;
  let mallocs = s.mallocs;
  let nextBlockSize = s.nextBlockSize;
  let nextBlock = s.nextBlock;
  let head = '';
  let body = '';

  /* 1. Reuse: AllocSet's per-size-class freelist, slab's per-block free list. */
  if (kind !== 'generation') {
    for (const b of blocks) {
      const hit = b.chunks.find((c) => !c.live && c.payload === payload);
      if (hit) {
        hit.live = true;
        hit.req = req;
        head =
          kind === 'aset'
            ? `palloc(${req}) reused a freed ${fmtBytes(payload)} chunk — no malloc().`
            : `palloc(${req}) took a free slot in block #${b.id}.`;
        body =
          kind === 'aset'
            ? 'pfree() never hands memory back to libc. It pushes the chunk onto the context’s freelist for its ' +
              'size class, where only this context can reuse it — and only for a request that rounds to the same class.'
            : 'A slab block keeps a free-chunk list of its own. Every chunk is the same size, so any free slot fits ' +
              'any request: no size classes, no rounding, no fragmentation.';
        return { ...s, blocks, head, body, step: s.step + 1 };
      }
    }
  }

  /* 2. A request past allocChunkLimit gets a block of its own (AllocSet only). */
  if (kind === 'aset' && req > CHUNK_LIMIT) {
    if (blocks.length >= MAX_BLOCKS) return s;
    blocks.push({
      id: nextBlock++,
      cap: BLOCK_HDR + need,
      used: need,
      dedicated: true,
      chunks: [{ id: s.nextChunk, req, payload, live: true }],
    });
    return {
      ...s,
      blocks,
      nextBlock,
      nextChunk: s.nextChunk + 1,
      mallocs: mallocs + 1,
      head: `palloc(${req}) exceeded allocChunkLimit — it got its own malloc().`,
      body:
        `Requests above ${fmtBytes(CHUNK_LIMIT)} bypass the size classes and the shared blocks entirely: aset.c ` +
        'mallocs a block sized exactly for this chunk and links it in. That block is freed the moment the chunk is, ' +
        'which is why one huge palloc does not permanently inflate the context.',
      step: s.step + 1,
    };
  }

  /* 3. Otherwise: the current block if it has room, else a new (doubled) block. */
  const open = [...blocks].reverse().find((b) => !b.dedicated && b.cap - BLOCK_HDR - b.used >= need);
  if (open) {
    open.used += need;
    open.chunks.push({ id: s.nextChunk, req, payload, live: true });
    head = `palloc(${req}) bumped the free pointer in block #${open.id}.`;
    body =
      kind === 'aset'
        ? `The request rounded up to the ${fmtBytes(payload)} size class and took ${fmtBytes(need)} with its 8-byte ` +
          `header. Allocation is a pointer bump and a freelist check — a few nanoseconds, no lock, no syscall.`
        : `Generation contexts do not round to size classes: the chunk is exactly ${fmtBytes(payload)} plus its ` +
          `header, bump-allocated. There is no freelist at all, which is what makes them cheap and unforgiving.`;
    return { ...s, blocks, nextChunk: s.nextChunk + 1, head, body, step: s.step + 1 };
  }

  if (blocks.length >= MAX_BLOCKS) return s;
  const cap =
    kind === 'slab'
      ? Math.max(SLAB_BLOCK, need + BLOCK_HDR)
      : Math.min(Math.max(nextBlockSize, need + BLOCK_HDR), MAX_BLOCK);
  blocks.push({
    id: nextBlock++,
    cap,
    used: need,
    dedicated: false,
    chunks: [{ id: s.nextChunk, req, payload, live: true }],
  });
  mallocs += 1;
  if (kind !== 'slab') nextBlockSize = Math.min(nextBlockSize * 2, MAX_BLOCK);
  head = `Block full — malloc(${fmtBytes(cap)}) for block #${nextBlock - 1}.`;
  body =
    kind === 'slab'
      ? `Slab blocks are a fixed ${fmtBytes(SLAB_BLOCK)} holding ${fmtNum(Math.floor((SLAB_BLOCK - BLOCK_HDR) / need))} ` +
        'identical chunks, tracked by a free list per block. No doubling, no rounding.'
      : `Block sizes double — ${fmtBytes(INIT_BLOCK)}, ${fmtBytes(2 * INIT_BLOCK)}, ${fmtBytes(4 * INIT_BLOCK)} … up to ` +
        `${fmtBytes(MAX_BLOCK)} — so a context that keeps growing makes progressively fewer malloc() calls. ` +
        'That is one malloc per thousands of pallocs.';
  return { ...s, blocks, nextBlockSize, nextBlock, nextChunk: s.nextChunk + 1, mallocs, head, body, step: s.step + 1 };
}

function pfree(s: S, kind: CtxKind): S {
  const live = s.blocks.flatMap((b) => b.chunks.filter((c) => c.live).map((c) => ({ b, c })));
  if (live.length === 0) return s;
  const pick = live[Math.floor(JITTER[s.step % JITTER.length] * live.length) % live.length];

  let blocks = s.blocks.map((b) => ({
    ...b,
    chunks: b.chunks.map((c) => (c.id === pick.c.id ? { ...c, live: false } : c)),
  }));
  let frees = s.frees;
  let head = `pfree(chunk #${pick.c.id}) — ${fmtBytes(pick.c.payload)}.`;
  let body = '';

  if (kind === 'aset') {
    const dedicated = blocks.find((b) => b.id === pick.b.id)?.dedicated;
    if (dedicated) {
      blocks = blocks.filter((b) => b.id !== pick.b.id);
      frees += 1;
      body =
        'That chunk had a block to itself, so freeing it really did call free() and the memory went back to libc. ' +
        'Chunks inside shared blocks never do.';
    } else {
      body =
        `The chunk is now on the ${fmtBytes(pick.c.payload)} freelist. RSS did not move: the block is still mapped, ` +
        'and only a palloc() from this same context, rounding to this same size class, can use the hole. This is ' +
        'why a long-lived context with a mixed allocation pattern grows and never shrinks.';
    }
  } else {
    const b = blocks.find((x) => x.id === pick.b.id)!;
    const anyLive = b.chunks.some((c) => c.live);
    if (!anyLive) {
      blocks = blocks.filter((x) => x.id !== b.id);
      frees += 1;
      body =
        kind === 'generation'
          ? 'That was the block’s last live chunk, so the whole block went back to libc. Generation contexts only ' +
            'ever free at block granularity — which is perfect when lifetimes are ordered and pathological when they are not.'
          : 'The block emptied and was returned to libc. A slab context can always do this, because every chunk in a ' +
            'block is interchangeable.';
    } else {
      body =
        kind === 'generation'
          ? `Block #${b.id} still holds ${fmtNum(b.chunks.filter((c) => c.live).length)} live chunk(s), so all ` +
            `${fmtBytes(b.cap)} of it stays mapped. One long-lived row pinning an entire block is exactly how a ` +
            'logical decoding slot with one long transaction inflates a walsender’s RSS.'
          : `The slot is back on block #${b.id}’s free list and the next palloc of this size will take it.`;
    }
  }
  return { ...s, blocks, frees, head, body, step: s.step + 1 };
}

function reset(s: S): S {
  const keeper = s.blocks.find((b) => !b.dedicated);
  const freed = s.blocks.length - (keeper ? 1 : 0);
  const blocks = keeper ? [{ ...keeper, used: 0, chunks: [] }] : [];
  return {
    ...s,
    blocks,
    nextBlockSize: INIT_BLOCK,
    mallocs: s.mallocs,
    frees: s.frees + freed,
    step: s.step + 1,
    head: `MemoryContextReset(): ${fmtNum(freed)} free() call${freed === 1 ? '' : 's'}, every chunk gone.`,
    body:
      'The cost of releasing the context is the number of blocks, not the number of chunks — nothing walks the ' +
      'allocations, nothing runs a destructor, no chunk is inspected. The first block is the keeper and stays ' +
      'mapped, so a context that is reset for every tuple (ExprContext) never calls malloc again after the first ' +
      'tuple. This is also what makes elog(ERROR) safe: the abort path resets the transaction context and every ' +
      'palloc made since the last savepoint disappears, leak-free, without any cleanup code having run.',
  };
}

function del(s: S): S {
  return {
    ...INITIAL,
    mallocs: s.mallocs,
    frees: s.frees + s.blocks.length,
    step: s.step + 1,
    head: `MemoryContextDelete(): ${fmtNum(s.blocks.length)} free() call${s.blocks.length === 1 ? '' : 's'}, keeper included.`,
    body:
      'Delete is reset plus the keeper block plus the context header, and it recurses into every child context. ' +
      'Deleting a portal deletes everything the executor allocated under it — which is why the executor can be ' +
      'written as if memory were infinite and still not leak.',
  };
}

/* ------------------------------------------------------------------ render */

const C_LIVE = 'var(--viz-1)';
const C_PAD = 'var(--viz-2)';
const C_HDR = 'var(--viz-5)';
const C_FREE = 'var(--viz-stale)';
const C_TAIL = 'var(--viz-neutral)';

type FigProps = { s: S; kind: CtxKind; svgW: number; height: number; rowH: number; maxCap: number };

function ArenaFigure({ s, kind, svgW, height, rowH, maxCap }: FigProps) {
  const tip = useTip();
  const labelW = 132;
  const availW = Math.max(200, svgW - labelW - 12);

  return (
    <svg width={svgW} height={height} role="img" aria-label="Blocks of a PostgreSQL memory context, with the chunks carved out of each one">
      {s.blocks.length === 0 ? (
        <text x={4} y={30} fontSize={13} fill="var(--viz-ink-muted)">
          No blocks. The context header exists; it owns no memory yet.
        </text>
      ) : null}
      {s.blocks.map((b, i) => {
        const y = 8 + i * rowH;
        const w = Math.max(56, (b.cap / maxCap) * availW);
        const px = (bytes: number) => (bytes / b.cap) * w;
        let cx = labelW + px(BLOCK_HDR);
        return (
          <g key={b.id}>
            <text x={0} y={y + 15} fontSize={12} fill="var(--viz-ink)">
              block #{b.id} · {fmtBytes(b.cap)}
            </text>
            <text x={0} y={y + 29} fontSize={11} fill="var(--viz-ink-muted)">
              {b.dedicated ? 'dedicated' : i === 0 ? 'keeper' : `${fmtNum(b.chunks.filter((c) => c.live).length)} live`}
            </text>
            <rect x={labelW} y={y} width={w} height={rowH - 12} rx={4} fill={C_TAIL} stroke="var(--viz-border)" />
            <rect x={labelW} y={y} width={Math.max(1.5, px(BLOCK_HDR))} height={rowH - 12} fill="var(--viz-axis)" />
            {b.chunks.map((c) => {
              const hw = px(CHUNK_HDR);
              const pw = px(c.live ? c.req : c.payload);
              const padw = px(c.live ? c.payload - c.req : 0);
              const x0 = cx;
              cx += px(c.payload + CHUNK_HDR);
              return (
                <g
                  key={c.id}
                  {...tip(
                    <>
                      <strong>
                        chunk #{c.id} — {c.live ? `palloc(${fmtNum(c.req)})` : 'freed'}
                      </strong>
                      <br />
                      {fmtBytes(c.payload)} chunk{' '}
                      {kind === 'aset' && c.payload !== align8(c.req)
                        ? `(request of ${fmtNum(c.req)} B rounded up to a power-of-two size class — ${fmtNum(
                            c.payload - c.req,
                          )} B wasted)`
                        : '(no size-class rounding in this context type)'}{' '}
                      + {CHUNK_HDR} B header.
                      <br />
                      {c.live
                        ? 'Live: reachable from whatever called palloc().'
                        : kind === 'generation'
                          ? 'Freed, but the block stays mapped until every chunk in it is freed.'
                          : 'Freed onto the freelist. Still mapped; reusable only by this context, at this size class.'}
                    </>,
                  )}
                >
                  <rect x={x0} y={y} width={Math.max(1, hw)} height={rowH - 12} fill={C_HDR} />
                  <rect
                    x={x0 + hw}
                    y={y}
                    width={Math.max(1, pw)}
                    height={rowH - 12}
                    fill={c.live ? C_LIVE : C_FREE}
                    stroke="var(--viz-surface)"
                    strokeWidth={0.5}
                  />
                  {padw > 0.6 ? <rect x={x0 + hw + pw} y={y} width={padw} height={rowH - 12} fill={C_PAD} /> : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

export default function MemoryContextArenaLab() {
  const [kind, setKind] = useState<CtxKind>('aset');
  const [reqIdx, setReqIdx] = useState(2);
  const [vary, setVary] = useState(true);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(820);

  const baseReq = REQ_STEPS[reqIdx];
  const slabChunk = align8(baseReq);

  const sizeAt = (step: number) => (vary ? Math.max(8, Math.round(baseReq * JITTER[step % JITTER.length])) : baseReq);

  const run = (n: number) =>
    setS((cur) => {
      let next = cur;
      for (let i = 0; i < n; i++) {
        const req = kind === 'slab' ? slabChunk : sizeAt(next.step);
        next = palloc(next, kind, req, slabChunk);
      }
      return next;
    });

  const switchKind = (k: CtxKind) => {
    setKind(k);
    setS(INITIAL);
  };

  const m = useMemo(() => {
    const chunks = s.blocks.flatMap((b) => b.chunks);
    const live = chunks.filter((c) => c.live);
    const requested = live.reduce((a, c) => a + c.req, 0);
    const mapped = s.blocks.reduce((a, b) => a + b.cap, 0);
    const rounding = live.reduce((a, c) => a + (c.payload - c.req), 0);
    const headers = chunks.length * CHUNK_HDR;
    const freeHeld = chunks.filter((c) => !c.live).reduce((a, c) => a + c.payload + CHUNK_HDR, 0);
    return {
      chunks,
      live,
      requested,
      mapped,
      rounding,
      headers,
      freeHeld,
      overhead: mapped > 0 ? 1 - requested / mapped : 0,
    };
  }, [s]);

  const rowH = 40;
  const svgW = Math.max(width, 620);
  const height = Math.max(64, 16 + Math.max(1, s.blocks.length) * rowH);
  const maxCap = Math.max(INIT_BLOCK, ...s.blocks.map((b) => b.cap));

  return (
    <VizPanel
      title="Inside one memory context"
      subtitle={
        <>
          palloc() carves chunks out of blocks the context owns. Free them one at a time and watch what does not
          come back; reset the context and watch everything go at once. <em>Used for:</em> {CTX[kind].where}
        </>
      }
      controls={
        <>
          <Choice
            label="Context type"
            value={kind}
            onChange={(v) => switchKind(v as CtxKind)}
            options={(Object.keys(CTX) as CtxKind[]).map((k) => ({ value: k, label: CTX[k].label }))}
          />
          <Slider
            label="palloc size"
            min={0}
            max={REQ_STEPS.length - 1}
            value={reqIdx}
            onChange={(n) => {
              setReqIdx(n);
              setS(INITIAL);
            }}
            format={(n) => `${fmtNum(REQ_STEPS[n])} B`}
          />
          <Check label="Vary sizes" checked={vary} onChange={setVary} />
          <Button onClick={() => run(1)} primary disabled={s.blocks.length >= MAX_BLOCKS}>
            palloc()
          </Button>
          <Button onClick={() => run(12)} disabled={s.blocks.length >= MAX_BLOCKS}>
            palloc() ×12
          </Button>
          <Button onClick={() => setS((c) => pfree(c, kind))} disabled={m.live.length === 0}>
            pfree() one
          </Button>
          <Button onClick={() => setS(reset)} disabled={s.blocks.length === 0}>
            MemoryContextReset()
          </Button>
          <Button onClick={() => setS(del)} disabled={s.blocks.length === 0}>
            MemoryContextDelete()
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Live payload', color: C_LIVE },
            { label: 'Size-class padding', color: C_PAD },
            { label: 'Chunk header (8 B)', color: C_HDR },
            { label: 'Freed, still mapped', color: C_FREE },
            { label: 'Unused block tail', color: C_TAIL },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Live chunks', value: fmtNum(m.live.length), hint: `${fmtNum(m.chunks.length)} chunks carved in total` },
            { label: 'Requested', value: fmtBytes(m.requested), hint: 'What the caller asked palloc() for' },
            { label: 'Mapped by malloc', value: fmtBytes(m.mapped), hint: 'What this context holds from libc — what shows up in RSS' },
            {
              label: 'Overhead',
              value: `${(m.overhead * 100).toFixed(0)}%`,
              hint: `${fmtBytes(m.rounding)} size-class padding + ${fmtBytes(m.headers)} headers + ${fmtBytes(m.freeHeld)} freed-but-held + block tails`,
            },
            { label: 'malloc() calls', value: fmtNum(s.mallocs), hint: 'One per block, not one per palloc' },
            {
              label: 'free() to release all',
              value: `${fmtNum(s.blocks.length)} vs ${fmtNum(m.live.length)}`,
              hint: 'Reset (per block) versus freeing every live chunk by hand',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Block</th>
              <th>malloc size</th>
              <th>Chunks</th>
              <th>Live</th>
              <th>Requested</th>
              <th>Padding + headers</th>
              <th>Unused tail</th>
            </tr>
          </thead>
          <tbody>
            {s.blocks.length === 0 ? (
              <tr>
                <td colSpan={7}>No blocks allocated.</td>
              </tr>
            ) : (
              s.blocks.map((b) => {
                const live = b.chunks.filter((c) => c.live);
                const req = live.reduce((a, c) => a + c.req, 0);
                const over = b.chunks.reduce((a, c) => a + CHUNK_HDR + (c.live ? c.payload - c.req : c.payload), 0);
                return (
                  <tr key={b.id}>
                    <td>#{b.id}{b.dedicated ? ' (dedicated)' : ''}</td>
                    <td>{fmtBytes(b.cap)}</td>
                    <td>{fmtNum(b.chunks.length)}</td>
                    <td>{fmtNum(live.length)}</td>
                    <td>{fmtBytes(req)}</td>
                    <td>{fmtBytes(over)}</td>
                    <td>{fmtBytes(b.cap - BLOCK_HDR - b.used)}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <ArenaFigure s={s} kind={kind} svgW={svgW} height={height} rowH={rowH} maxCap={maxCap} />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
