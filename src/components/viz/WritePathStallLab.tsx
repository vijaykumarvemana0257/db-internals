import { useMemo, useState } from 'react';
import { VizPanel, Slider, Choice, Button, Legend, Stats, Note, fmtNum, useTicker } from './Viz';

/**
 * A time-stepped model of RocksDB's write path under sustained load.
 * Units are real-sized (MB, seconds) so the triggers are the documented defaults:
 *   - memtables waiting to flush >= max_write_buffer_number            -> stop
 *   - max_write_buffer_number > 3 and waiting >= max_write_buffer_number - 1 -> slowdown
 *   - L0 files >= level0_slowdown_writes_trigger (20)                   -> slowdown
 *   - L0 files >= level0_stop_writes_trigger (36)                       -> stop
 * Slowdown admits writes at delayed_write_rate (16 MB/s). Pending-compaction-bytes limits
 * (64 GB soft / 256 GB hard) are far beyond this model's backlog and are only reported.
 */
const L0_COMPACTION_TRIGGER = 4;
const L0_SLOWDOWN = 20;
const L0_STOP = 36;
const DELAYED_WRITE_RATE = 16; // MB/s
const OPS_PER_MB = 1000; // ~1 KB records, only for the sequence-number counter
const MAX_TICKS = 240;

type Cfg = { writeRate: number; memtableMB: number; maxWbn: number; flushRate: number; compactRate: number };
type State = 'normal' | 'slowdown' | 'stop';
type Tick = {
  t: number;
  admitted: number; // MB/s actually accepted this second
  state: State;
  reason: string;
  active: number; // MB in the active memtable
  immutables: number[]; // MB in each immutable memtable, oldest first
  l0: number; // L0 files
  walMB: number; // WAL not yet retired by a flush
  seq: number;
  pendingMB: number; // L0 bytes awaiting compaction
};

export function simulate(c: Cfg): Tick[] {
  const out: Tick[] = [];
  let active = 0;
  let imm: number[] = [];
  let l0 = 0;
  let l0Partial = 0; // MB of the L0 file currently being compacted away
  let flushProgress = 0;
  let wal = 0;
  let seq = 0;
  for (let t = 1; t <= MAX_TICKS; t++) {
    const waiting = imm.length;
    let state: State = 'normal';
    let reason = 'Below every trigger.';
    if (waiting >= c.maxWbn) {
      state = 'stop';
      reason = `${waiting} memtables waiting to flush ≥ max_write_buffer_number (${c.maxWbn})`;
    } else if (l0 >= L0_STOP) {
      state = 'stop';
      reason = `${l0} L0 files ≥ level0_stop_writes_trigger (${L0_STOP})`;
    } else if (c.maxWbn > 3 && waiting >= c.maxWbn - 1) {
      state = 'slowdown';
      reason = `${waiting} memtables waiting ≥ max_write_buffer_number − 1 (${c.maxWbn - 1})`;
    } else if (l0 >= L0_SLOWDOWN) {
      state = 'slowdown';
      reason = `${l0} L0 files ≥ level0_slowdown_writes_trigger (${L0_SLOWDOWN})`;
    }
    const admitted = state === 'stop' ? 0 : state === 'slowdown' ? Math.min(c.writeRate, DELAYED_WRITE_RATE) : c.writeRate;

    // Foreground: append to the WAL, insert into the active memtable; freeze it when full.
    active += admitted;
    wal += admitted;
    seq += Math.round(admitted * OPS_PER_MB);
    while (active >= c.memtableMB) {
      imm.push(c.memtableMB);
      active -= c.memtableMB;
    }

    // Background flush: one immutable memtable at a time becomes one L0 file.
    let flushBudget = c.flushRate;
    while (imm.length && flushBudget > 0) {
      const need = imm[0] - flushProgress;
      if (flushBudget >= need) {
        flushBudget -= need;
        flushProgress = 0;
        imm = imm.slice(1);
        l0 += 1;
        wal = Math.max(0, wal - c.memtableMB); // that memtable's WAL segment can be retired
      } else {
        flushProgress += flushBudget;
        flushBudget = 0;
      }
    }

    // Background compaction drains L0 once it reaches the compaction trigger.
    if (l0 >= L0_COMPACTION_TRIGGER) {
      l0Partial += c.compactRate;
      while (l0Partial >= c.memtableMB && l0 > 0) {
        l0Partial -= c.memtableMB;
        l0 -= 1;
      }
    } else {
      l0Partial = 0;
    }

    out.push({ t, admitted, state, reason, active, immutables: [...imm], l0, walMB: wal, seq, pendingMB: l0 * c.memtableMB });
  }
  return out;
}

const STATE_COLOR: Record<State, string> = {
  normal: 'var(--viz-good)',
  slowdown: 'var(--viz-warning)',
  stop: 'var(--viz-critical)',
};
const STATE_LABEL: Record<State, string> = { normal: '● Normal', slowdown: '▲ Slowdown', stop: '■ Stopped' };

export default function WritePathStallLab() {
  const [writeRate, setWriteRate] = useState(120);
  const [memtableMB, setMemtableMB] = useState(64);
  const [maxWbn, setMaxWbn] = useState('2');
  const [flushRate, setFlushRate] = useState(100);
  const [compactRate, setCompactRate] = useState(60);
  const [t, setT] = useState(90);
  const [playing, setPlaying] = useState(false);

  const cfg: Cfg = { writeRate, memtableMB, maxWbn: Number(maxWbn), flushRate, compactRate };
  const ticks = useMemo(() => simulate(cfg), [writeRate, memtableMB, maxWbn, flushRate, compactRate]);
  const cur = ticks[Math.min(t, ticks.length) - 1];

  useTicker(
    (dt) => {
      setT((x) => {
        const n = x + Math.max(1, Math.round(dt / 50));
        if (n >= MAX_TICKS) {
          setPlaying(false);
          return MAX_TICKS;
        }
        return n;
      });
    },
    playing,
  );

  const firstStall = ticks.find((k) => k.state !== 'normal');
  const firstStop = ticks.find((k) => k.state === 'stop');
  const stoppedSeconds = ticks.slice(0, t).filter((k) => k.state === 'stop').length;
  const avgAdmitted = ticks.slice(0, t).reduce((a, k) => a + k.admitted, 0) / Math.max(1, t);

  // ---- figure geometry
  const W = 700;
  const laneX = 150;
  const barW = W - laneX - 20;
  const rateMax = Math.max(200, writeRate);

  return (
    <VizPanel
      title="The write path under sustained load"
      subtitle="Writes append to the WAL and fill the active memtable; a full memtable freezes and waits for a background flush, which turns it into an L0 file; compaction drains L0. When either queue backs up past its trigger, RocksDB throttles and then stops foreground writes."
      controls={
        <>
          <Slider label="Incoming writes" min={10} max={300} step={10} value={writeRate} onChange={setWriteRate} format={(n) => `${n} MB/s`} />
          <Slider label="write_buffer_size" min={16} max={256} step={16} value={memtableMB} onChange={setMemtableMB} format={(n) => `${n} MB`} />
          <Choice
            label="max_write_buffer_number"
            value={maxWbn}
            onChange={setMaxWbn}
            options={['2', '3', '4', '6'].map((v) => ({ value: v, label: v }))}
          />
          <Slider label="Flush throughput" min={20} max={300} step={10} value={flushRate} onChange={setFlushRate} format={(n) => `${n} MB/s`} />
          <Slider label="L0 compaction throughput" min={10} max={300} step={10} value={compactRate} onChange={setCompactRate} format={(n) => `${n} MB/s`} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Active memtable (in memory only)', color: 'var(--viz-dirty)' },
            { label: 'Immutable, waiting to flush', color: 'var(--viz-3)' },
            { label: 'L0 SSTable file (durable)', color: 'var(--viz-clean)' },
            { label: 'WAL not yet retired', color: 'var(--viz-4)' },
            { label: '● normal ▲ slowdown ■ stopped', color: 'var(--viz-ink-muted)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `Write state at t=${cur.t}s`, value: STATE_LABEL[cur.state] },
            { label: 'Admitted now', value: `${fmtNum(cur.admitted)} MB/s`, hint: `Requested ${writeRate} MB/s` },
            { label: 'Average admitted so far', value: `${fmtNum(avgAdmitted, 1)} MB/s` },
            { label: 'Sequence number', value: fmtNum(cur.seq), hint: `≈ ${OPS_PER_MB} writes per MB (1 KB records)` },
            { label: 'Seconds fully stopped', value: fmtNum(stoppedSeconds) },
          ]}
        />
      }
      note={
        <Note>
          <strong style={{ color: 'var(--viz-ink)' }}>
            t = {cur.t}s · {STATE_LABEL[cur.state]}:
          </strong>{' '}
          {cur.reason}{' '}
          {firstStall
            ? `The first stall in this run starts at t=${firstStall.t}s${firstStop ? `, and writes first stop completely at t=${firstStop.t}s` : ', and writes never stop completely'}.`
            : 'Flush and compaction keep up with this write rate: no trigger is ever crossed in this run.'}{' '}
          Pending compaction bytes here peak at {fmtNum(Math.max(...ticks.map((k) => k.pendingMB)) / 1024, 1)} GB — far below the 64 GB soft limit, so that third trigger never fires in this model.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>t (s)</th>
              <th>State</th>
              <th>Admitted MB/s</th>
              <th>Active MB</th>
              <th>Waiting to flush</th>
              <th>L0 files</th>
              <th>WAL MB</th>
            </tr>
          </thead>
          <tbody>
            {ticks.filter((k) => k.t % 10 === 0).map((k) => (
              <tr key={k.t}>
                <td>{k.t}</td>
                <td>{k.state}</td>
                <td>{fmtNum(k.admitted)}</td>
                <td>{fmtNum(k.active)}</td>
                <td>{k.immutables.length}</td>
                <td>{k.l0}</td>
                <td>{fmtNum(k.walMB)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'end', marginBottom: 8 }}>
        <Button primary onClick={() => { if (t >= MAX_TICKS) setT(1); setPlaying((p) => !p); }}>
          {playing ? 'Pause' : t >= MAX_TICKS ? 'Replay' : 'Play'}
        </Button>
        <Slider label="Time" min={1} max={MAX_TICKS} value={t} onChange={(n) => { setPlaying(false); setT(n); }} format={(n) => `${n} s`} />
      </div>

      <svg width={W} height={330} role="img" aria-label={`Write path at ${cur.t} seconds: ${cur.state}`}>
        {/* write-rate gauge */}
        <text x={0} y={22} fill="var(--viz-ink)">Write rate</text>
        <rect x={laneX} y={10} width={barW} height={16} rx={4} fill="var(--viz-plane)" stroke="var(--viz-border)" />
        <rect x={laneX} y={10} width={(barW * cur.admitted) / rateMax} height={16} rx={4} fill={STATE_COLOR[cur.state]} />
        <line x1={laneX + (barW * writeRate) / rateMax} x2={laneX + (barW * writeRate) / rateMax} y1={6} y2={30} stroke="var(--viz-ink)" strokeDasharray="3 2" />
        <text x={laneX + (barW * writeRate) / rateMax + 4} y={40} fontSize={10}>requested</text>
        <line x1={laneX + (barW * DELAYED_WRITE_RATE) / rateMax} x2={laneX + (barW * DELAYED_WRITE_RATE) / rateMax} y1={6} y2={30} stroke="var(--viz-ink-muted)" />
        <text x={laneX + (barW * DELAYED_WRITE_RATE) / rateMax + 4} y={52} fontSize={10}>delayed_write_rate 16 MB/s</text>

        {/* WAL */}
        <text x={0} y={82} fill="var(--viz-ink)">WAL (unretired)</text>
        <rect x={laneX} y={70} width={barW} height={14} rx={4} fill="var(--viz-plane)" stroke="var(--viz-border)" />
        <rect x={laneX} y={70} width={Math.min(barW, (barW * cur.walMB) / (memtableMB * (Number(maxWbn) + 1)))} height={14} rx={4} fill="var(--viz-4)" />
        <text x={laneX + barW} y={98} textAnchor="end" fontSize={10}>{fmtNum(cur.walMB)} MB — retired only when the memtable it covers is flushed</text>

        {/* memtables */}
        <text x={0} y={132} fill="var(--viz-ink)">Memtables</text>
        <rect x={laneX} y={114} width={120} height={28} rx={5} fill="var(--viz-plane)" stroke="var(--viz-border)" />
        <rect x={laneX} y={114} width={(120 * cur.active) / memtableMB} height={28} rx={5} fill="var(--viz-dirty)" />
        <text x={laneX + 60} y={132} textAnchor="middle" fill="var(--viz-ink)" fontSize={10}>active {Math.round((100 * cur.active) / memtableMB)}%</text>
        {cur.immutables.slice(0, 8).map((_, i) => (
          <g key={i}>
            <rect x={laneX + 132 + i * 50} y={114} width={44} height={28} rx={5} fill="var(--viz-3)" stroke="var(--viz-surface)" strokeWidth={2} />
            <text x={laneX + 154 + i * 50} y={132} textAnchor="middle" fill="var(--viz-ink)" fontSize={10}>imm</text>
          </g>
        ))}
        <text x={laneX} y={158} fontSize={10}>
          {cur.immutables.length} waiting to flush · stop at {maxWbn}
          {Number(maxWbn) > 3 ? ` · slowdown at ${Number(maxWbn) - 1}` : ' · no memtable slowdown when max_write_buffer_number ≤ 3'}
        </text>

        {/* L0 */}
        <text x={0} y={202} fill="var(--viz-ink)">L0 files</text>
        {Array.from({ length: 40 }, (_, i) => {
          const x = laneX + i * ((barW - 10) / 40);
          return <rect key={i} x={x} y={186} width={(barW - 10) / 40 - 2} height={24} rx={2} fill={i < cur.l0 ? 'var(--viz-clean)' : 'var(--viz-plane)'} stroke="var(--viz-border)" />;
        })}
        {[
          [L0_COMPACTION_TRIGGER, 'compaction starts (4)'],
          [L0_SLOWDOWN, 'slowdown (20)'],
          [L0_STOP, 'stop (36)'],
        ].map(([n, label], k) => {
          const x = laneX + (n as number) * ((barW - 10) / 40) - 1;
          return (
            <g key={label as string}>
              <line x1={x} x2={x} y1={180} y2={216 + k * 12} stroke="var(--viz-ink)" strokeDasharray="3 2" />
              <text x={x + 3} y={226 + k * 12} fontSize={10}>{label}</text>
            </g>
          );
        })}

        {/* timeline of states */}
        <text x={0} y={292} fill="var(--viz-ink)">State over time</text>
        {ticks.map((k) => (
          <rect key={k.t} x={laneX + ((k.t - 1) * barW) / MAX_TICKS} y={280} width={barW / MAX_TICKS + 0.5} height={16} fill={STATE_COLOR[k.state]} opacity={k.t <= t ? 1 : 0.25} />
        ))}
        <line x1={laneX + (t * barW) / MAX_TICKS} x2={laneX + (t * barW) / MAX_TICKS} y1={274} y2={302} stroke="var(--viz-ink)" strokeWidth={2} />
        <text x={laneX} y={318} fontSize={10}>0 s</text>
        <text x={laneX + barW} y={318} fontSize={10} textAnchor="end">{MAX_TICKS} s</text>
      </svg>
    </VizPanel>
  );
}
