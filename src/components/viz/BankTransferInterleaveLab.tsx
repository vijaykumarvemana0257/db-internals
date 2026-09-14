import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
} from './Viz';

/**
 * Two bank transfers, one shared pair of rows.
 *
 * T1 moves $30 from A to B; T2 moves $50 from B to A. Both are written the way almost
 * every application writes them: SELECT the balance, compute the new value in the client,
 * UPDATE it back as a literal. The learner drags the interleaving, chooses whether the
 * two statements sit inside one transaction or are two autocommit transactions, turns
 * row locking on and off, and cuts the power after any step.
 *
 * The model is small but real:
 *   - one stored value per row (updates are in place), plus a global undo log of
 *     before-images, which is what a crash or a rollback walks backwards;
 *   - a per-row version counter, so a write whose SELECT was overtaken by another
 *     transaction's write is detectable — that is a lost update;
 *   - strict two-phase locking when isolation is on: an exclusive lock taken at first
 *     touch and held to commit, with deadlock detection and a victim.
 *
 * Everything is a pure function of the five controls: same inputs, same render.
 */

const START = 100;
const INVARIANT = 2 * START;

type RowId = 'A' | 'B';
type TxnId = 1 | 2;

type Step =
  | { kind: 'read'; row: RowId }
  | { kind: 'write'; row: RowId; delta: number }
  | { kind: 'commit' };

const PROGRAM: Record<TxnId, Step[]> = {
  1: [
    { kind: 'read', row: 'A' },
    { kind: 'write', row: 'A', delta: -30 },
    { kind: 'read', row: 'B' },
    { kind: 'write', row: 'B', delta: 30 },
    { kind: 'commit' },
  ],
  2: [
    { kind: 'read', row: 'B' },
    { kind: 'write', row: 'B', delta: -50 },
    { kind: 'read', row: 'A' },
    { kind: 'write', row: 'A', delta: 50 },
    { kind: 'commit' },
  ],
};

const SUB: Record<TxnId, string> = { 1: '₁', 2: '₂' };
const TITLE: Record<TxnId, string> = {
  1: 'T₁ — transfer $30 A→B',
  2: 'T₂ — transfer $50 B→A',
};

type Mode = 'autocommit' | 'explicit';
type Iso = 'off' | 'on';

type Cfg = { mode: Mode; iso: Iso; lead: number; stride: number; crashAt: number };

type OpKind = 'read' | 'write' | 'commit' | 'wait' | 'abort' | 'noop';

type TraceRow = {
  n: number;
  txn: TxnId;
  op: OpKind;
  row?: RowId;
  chip: string;
  sql: string;
  sees: string;
  a: number;
  b: number;
  uncA: boolean;
  uncB: boolean;
  note: string;
};

type Sim = {
  trace: TraceRow[];
  a: number;
  b: number;
  crashed: boolean;
  victim: TxnId | null;
  waits: number;
  undone: TxnId[];
  durable: Record<TxnId, number>;
  anomalies: string[];
};

/** Schedule family: `lead` steps of T1 first, then the two swap every `stride` steps. */
function buildOrder(lead: number, stride: number): TxnId[] {
  const left: Record<TxnId, number> = { 1: PROGRAM[1].length, 2: PROGRAM[2].length };
  const out: TxnId[] = [];
  const take = (t: TxnId, n: number) => {
    const k = Math.min(n, left[t]);
    for (let i = 0; i < k; i++) out.push(t);
    left[t] -= k;
  };
  take(1, lead);
  let turn: TxnId = 2;
  while (left[1] + left[2] > 0) {
    if (left[turn] === 0) {
      turn = turn === 1 ? 2 : 1;
      continue;
    }
    take(turn, stride);
    turn = turn === 1 ? 2 : 1;
  }
  return out;
}

function simulate(cfg: Cfg): Sim {
  const val: Record<RowId, number> = { A: START, B: START };
  const ver: Record<RowId, number> = { A: 0, B: 0 };
  const lock: Record<RowId, TxnId | null> = { A: null, B: null };
  const blocked: Record<TxnId, RowId | null> = { 1: null, 2: null };
  const pc: Record<TxnId, number> = { 1: 0, 2: 0 };
  const state: Record<TxnId, 'run' | 'committed' | 'aborted'> = { 1: 'run', 2: 'run' };
  const reg: Record<TxnId, Partial<Record<RowId, { v: number; ver: number }>>> = { 1: {}, 2: {} };
  const readFrom: Record<TxnId, Set<TxnId>> = { 1: new Set(), 2: new Set() };
  const undo: { txn: TxnId; row: RowId; before: number; open: boolean; reverted: boolean }[] = [];
  const trace: TraceRow[] = [];
  const anomalies = new Set<string>();
  const staleWrites: number[] = [];

  /** Row locks are only held across statements inside an explicit transaction. */
  const locking = cfg.iso === 'on' && cfg.mode === 'explicit';
  let waits = 0;
  let victim: TxnId | null = null;
  let lastBlocked: TxnId | null = null;
  let crashed = false;

  const other = (t: TxnId): TxnId => (t === 1 ? 2 : 1);
  const unc = (r: RowId) => undo.some((u) => u.row === r && u.open);

  const push = (txn: TxnId, op: OpKind, chip: string, sql: string, sees: string, note: string, row?: RowId) => {
    trace.push({
      n: trace.length + 1,
      txn,
      op,
      row,
      chip,
      sql,
      sees,
      a: val.A,
      b: val.B,
      uncA: unc('A'),
      uncB: unc('B'),
      note,
    });
  };

  const release = (t: TxnId) => {
    (['A', 'B'] as RowId[]).forEach((r) => {
      if (lock[r] === t) lock[r] = null;
    });
    const o = other(t);
    if (blocked[o] && lock[blocked[o] as RowId] === null) blocked[o] = null;
  };

  const rollback = (t: TxnId, reason: 'deadlock') => {
    for (let i = undo.length - 1; i >= 0; i--) {
      if (undo[i].txn === t && undo[i].open) {
        val[undo[i].row] = undo[i].before;
        undo[i].open = false;
        undo[i].reverted = true;
      }
    }
    state[t] = 'aborted';
    blocked[t] = null;
    victim = t;
    release(t);
    push(
      t,
      'abort',
      `abort${SUB[t]}`,
      'ROLLBACK  -- deadlock victim',
      '—',
      reason === 'deadlock'
        ? `T₁ holds a lock on the row T₂ wants and vice versa — a cycle in the wait-for graph. The engine breaks it by killing a transaction: Postgres aborts the backend whose request closed the cycle after deadlock_timeout (1 s by default) and raises SQLSTATE 40P01; InnoDB picks the transaction that changed the fewest rows and returns error 1213. Its undo records are applied in reverse, so the money is exactly where it started.`
        : '',
    );
  };

  let queue = buildOrder(cfg.lead, cfg.stride);
  let guard = 0;

  while (queue.length > 0 && guard++ < 200) {
    const idx = queue.findIndex((t) => state[t] === 'run' && blocked[t] === null);
    if (idx < 0) {
      const stuck = queue.filter((t) => state[t] === 'run');
      if (stuck.length === 0) break;
      rollback(lastBlocked ?? stuck[0], 'deadlock');
      queue = queue.filter((t) => state[t] === 'run');
      if (cfg.crashAt > 0 && trace.length >= cfg.crashAt) {
        crashed = true;
        break;
      }
      continue;
    }

    const t = queue[idx];
    const step = PROGRAM[t][pc[t]];
    let done = true;

    if (step.kind === 'read') {
      const r = step.row;
      if (locking && lock[r] !== null && lock[r] !== t) {
        blocked[t] = r;
        lastBlocked = t;
        waits++;
        done = false;
        push(
          t,
          'wait',
          `wait${SUB[t]}`,
          `-- waiting on row ${r}`,
          '—',
          `T${SUB[t]} needs row ${r}; T${SUB[other(t)]} took an exclusive lock on it at first touch and strict two-phase locking holds locks until commit. This is what isolation costs: the read does not see stale data, it does not run at all.`,
          r,
        );
      } else {
        if (locking) lock[r] = t;
        const dirtySrc = undo.find((u) => u.row === r && u.open && u.txn !== t);
        const seen = val[r];
        reg[t][r] = { v: seen, ver: ver[r] };
        if (dirtySrc) readFrom[t].add(dirtySrc.txn);
        push(
          t,
          'read',
          `r${SUB[t]}(${r})=${seen}`,
          `SELECT bal FROM acct WHERE id='${r}'`,
          `$${seen}`,
          dirtySrc
            ? `Dirty read: row ${r} currently holds $${seen}, written by T${SUB[dirtySrc.txn]} and not committed. With no lock and no snapshot there is nothing in the engine that distinguishes this value from a real one — and T${SUB[dirtySrc.txn]} may still roll back.`
            : `T${SUB[t]} pulls $${seen} out of row ${r} into a client variable. The engine now has no idea that a later UPDATE will be derived from this number; that link exists only in your application.`,
          r,
        );
      }
    } else if (step.kind === 'write') {
      const r = step.row;
      if (locking && lock[r] !== null && lock[r] !== t) {
        blocked[t] = r;
        lastBlocked = t;
        waits++;
        done = false;
        push(
          t,
          'wait',
          `wait${SUB[t]}`,
          `-- waiting on row ${r}`,
          '—',
          `The UPDATE blocks: T${SUB[other(t)]} holds the exclusive lock on row ${r}. Every engine takes a row lock on write, at every isolation level — this is the one lock you cannot turn off.`,
          r,
        );
      } else {
        if (locking) lock[r] = t;
        const base = reg[t][r]?.v ?? val[r];
        const stale = reg[t][r] !== undefined && reg[t][r]!.ver !== ver[r];
        const next = base + step.delta;
        undo.push({ txn: t, row: r, before: val[r], open: cfg.mode === 'explicit', reverted: false });
        val[r] = next;
        ver[r] += 1;
        if (stale) staleWrites.push(undo.length - 1);
        push(
          t,
          'write',
          `w${SUB[t]}(${r}←${next})`,
          `UPDATE acct SET bal=${next} WHERE id='${r}'`,
          `$${next}`,
          stale
            ? `Lost update. T${SUB[t]} computed $${next} from the $${base} it read, but row ${r} has moved on since — it now held $${undo[undo.length - 1].before}. The UPDATE carries a literal, so the engine cannot tell this is stale arithmetic: it overwrites, and the other transaction's change is gone with no error anywhere.`
            : cfg.mode === 'autocommit'
              ? `The UPDATE is its own transaction: the row lock is taken, the value written, the commit record flushed and the lock dropped, all before the next statement. $${next} is durable right now, halfway through a transfer.`
              : `Written in place. The before-image $${undo[undo.length - 1].before} is in the undo log, which is the only thing that can put this row back — and nothing has been made durable yet.`,
          r,
        );
      }
    } else {
      if (cfg.mode === 'autocommit') {
        state[t] = 'committed';
        push(
          t,
          'noop',
          `—${SUB[t]}`,
          '-- nothing to commit',
          '—',
          `There is no COMMIT to issue: in autocommit each statement already committed itself. The transfer never existed as a unit, so there is no boundary for atomicity or isolation to defend.`,
        );
      } else {
        undo.forEach((u) => {
          if (u.txn === t) u.open = false;
        });
        state[t] = 'committed';
        release(t);
        push(
          t,
          'commit',
          `c${SUB[t]}`,
          'COMMIT',
          '—',
          `COMMIT: the commit record goes to the WAL and is flushed, the transaction is marked committed, its locks drop. Both legs of this transfer became durable in the same instant — that is atomicity and durability arriving together, in one fsync.`,
        );
      }
    }

    if (done) {
      queue.splice(idx, 1);
      pc[t] += 1;
    }
    if (cfg.crashAt > 0 && trace.length >= cfg.crashAt) {
      crashed = true;
      break;
    }
  }

  const undone: TxnId[] = [];
  if (crashed) {
    for (let i = undo.length - 1; i >= 0; i--) {
      if (undo[i].open) {
        val[undo[i].row] = undo[i].before;
        undo[i].open = false;
        undo[i].reverted = true;
        if (!undone.includes(undo[i].txn)) undone.push(undo[i].txn);
      }
    }
  }

  /* A lost update only counts if the clobbering write is still there at the end: a crash
     that rolled it back destroyed the evidence along with everything else. */
  if (staleWrites.some((i) => !undo[i].reverted)) anomalies.add('lost-update');

  const durable: Record<TxnId, number> = { 1: 0, 2: 0 };
  ([1, 2] as TxnId[]).forEach((t) => {
    durable[t] = new Set(undo.filter((u) => u.txn === t && !u.reverted).map((u) => u.row)).size;
    if (durable[t] === 1) anomalies.add('torn');
    for (const src of readFrom[t]) {
      if (durable[t] > 0 && undo.some((u) => u.txn === src && u.reverted)) anomalies.add('dirty-read');
    }
  });

  const sum = val.A + val.B;
  if (sum !== INVARIANT && anomalies.size === 0) anomalies.add('clobber');

  return {
    trace,
    a: val.A,
    b: val.B,
    crashed,
    victim,
    waits,
    undone,
    durable,
    anomalies: [...anomalies],
  };
}

/* -------------------------------------------------------------- presentation */

const ANOMALY: Record<string, string> = {
  'lost-update': 'Lost update',
  'dirty-read': 'Dirty read, writer rolled back',
  torn: 'Half a transfer',
  clobber: 'Blind overwrite',
};

const OP_FILL: Record<OpKind, string> = {
  read: 'var(--viz-plane)',
  write: 'var(--viz-dirty)',
  commit: 'var(--viz-clean)',
  wait: 'var(--viz-warning)',
  abort: 'var(--viz-stale)',
  noop: 'var(--viz-neutral)',
};

export default function BankTransferInterleaveLab() {
  const [mode, setMode] = useState<Mode>('explicit');
  const [iso, setIso] = useState<Iso>('off');
  const [lead, setLead] = useState(1);
  const [stride, setStride] = useState(2);
  const [crashAt, setCrashAt] = useState(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const full = useMemo(() => simulate({ mode, iso, lead, stride, crashAt: 0 }), [mode, iso, lead, stride]);
  const maxCrash = full.trace.length;
  const crashStep = Math.min(crashAt, maxCrash);
  const sim = useMemo(
    () => (crashStep > 0 ? simulate({ mode, iso, lead, stride, crashAt: crashStep }) : full),
    [mode, iso, lead, stride, crashStep, full],
  );

  const cols = sim.trace.length;
  const extra = sim.crashed ? 1 : 0;
  const labelW = 140;
  const colW = 72;
  const svgW = Math.max(width, labelW + (cols + extra) * colW + 16);
  const headY = 12;
  const laneY = (i: number) => 20 + i * 52;
  const rowY = (i: number) => 20 + 2 * 52 + 14 + i * 26;
  const height = rowY(3) + 6;
  const x = (i: number) => labelW + i * colW;

  const sum = sim.a + sim.b;
  const ok = sum === INVARIANT;
  const label = sim.anomalies.length ? sim.anomalies.map((k) => ANOMALY[k]).join(' + ') : 'none';

  const headline = (() => {
    if (sim.anomalies.includes('torn'))
      return 'Half a transfer is durable and the other half never happened.';
    if (sim.anomalies.includes('dirty-read'))
      return 'A committed transaction was built on a value that was rolled back.';
    if (sim.anomalies.includes('lost-update')) return 'One transfer silently overwrote the other.';
    if (sim.anomalies.includes('clobber')) return 'Two in-place writes to one row, no lock between them.';
    if (sim.victim) return `Deadlock: T${SUB[sim.victim]} was chosen as the victim and rolled back.`;
    if (sim.crashed) return 'Crash: everything committed came back, everything else was undone.';
    return 'Both transfers applied, and the books balance.';
  })();

  const body = (() => {
    if (sim.anomalies.includes('torn'))
      return `Autocommit made each UPDATE its own transaction, so the debit was flushed and acknowledged on its own. The power cut had nothing to undo — recovery faithfully redid a committed write that was only ever half of a transfer. ${ok ? 'The totals balance here only because a second anomaly cancels the first; neither transfer is correct.' : `The invariant is off by $${Math.abs(INVARIANT - sum)} permanently, and no error was returned to anyone.`}`;
    if (sim.anomalies.includes('dirty-read'))
      return `Atomicity worked perfectly here: the uncommitted transaction's before-images were applied in reverse and its rows went back where they were. The problem is that the other transaction had already read one of those rows, computed from it, and committed — so undoing the writer also destroyed a committed transaction's work.${ok ? ' The totals happen to balance, which is luck rather than correctness.' : ''} That is the failure only isolation can prevent, and it is why no production engine lets one transaction read another transaction's uncommitted rows.`;
    if (sim.anomalies.includes('lost-update'))
      return `Both transactions did a read-modify-write on the same row, and the second UPDATE carried a literal computed before the first one landed. Fix it in the statement — UPDATE acct SET bal = bal - 30 — or hold the row with SELECT ... FOR UPDATE inside one transaction. Raising the isolation level does nothing while the SELECT and the UPDATE are separate transactions.`;
    if (sim.victim)
      return `Nothing is corrupt: the victim's undo records ran and the survivor committed the whole of its transfer. The cost is an error your application has to catch and retry, and it is entirely caused by the two transactions touching A and B in opposite orders. Touch rows in a fixed global order and the cycle cannot form.`;
    if (sim.crashed)
      return `Recovery is a two-part rule: redo every transaction whose commit record reached the log, undo every transaction whose did not. ${sim.undone.length ? `T${sim.undone.map((t) => SUB[t]).join(', T')} had no commit record, so ${sim.undone.length === 1 ? 'its' : 'their'} before-images were applied in reverse.` : 'Every transaction had committed, so there was nothing to undo.'} The sum is ${ok ? 'still $200' : `$${sum}`}.`;
    return `Every read saw a value that was committed or its own, and each transfer's two writes became durable together. Drag "T₂ starts after" to 1 and "swap every" to 3 with locking off to break it.`;
  })();

  return (
    <VizPanel
      title="Two transfers, one pair of rows"
      subtitle="T₁ moves $30 from A to B, T₂ moves $50 from B to A, and both are written the usual way: SELECT the balance, compute in the client, UPDATE the literal back. A + B must always be $200."
      controls={
        <>
          <Segmented
            label="Transaction"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'explicit', label: 'BEGIN … COMMIT', title: 'Both statements inside one transaction' },
              { value: 'autocommit', label: 'autocommit', title: 'Each statement is its own transaction' },
            ]}
          />
          <Segmented
            label="Row locking"
            value={iso}
            onChange={setIso}
            options={[
              { value: 'off', label: 'off', title: 'Read whatever the row currently holds' },
              { value: 'on', label: 'strict 2PL', title: 'Exclusive lock at first touch, held until commit' },
            ]}
          />
          <Slider label="T₂ starts after" min={0} max={5} value={lead} onChange={setLead} format={(n) => `${n} step${n === 1 ? '' : 's'} of T₁`} />
          <Slider label="Swap every" min={1} max={5} value={stride} onChange={setStride} format={(n) => (n >= 5 ? 'never' : `${n} step${n === 1 ? '' : 's'}`)} />
          <Slider label="Power cut" min={0} max={maxCrash} value={Math.min(crashAt, maxCrash)} onChange={setCrashAt} format={(n) => (n === 0 ? 'none' : `after step ${n}`)} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'SELECT (outlined)', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'UPDATE — written, not yet durable', color: 'var(--viz-dirty)' },
            { label: 'COMMIT / durable', color: 'var(--viz-clean)' },
            { label: 'blocked on a row lock', color: 'var(--viz-warning)' },
            { label: 'rolled back', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Row A', value: `$${sim.a}`, hint: 'Balance after the schedule, and after recovery if the power was cut' },
            { label: 'Row B', value: `$${sim.b}` },
            {
              label: 'A + B',
              value: <span style={{ color: ok ? 'var(--viz-good)' : 'var(--viz-critical)' }}>{`$${sum}`}</span>,
              hint: 'The invariant no constraint in SQL can express: it is delivered by isolation and atomicity, not by the checker',
            },
            { label: 'Anomaly', value: label, hint: 'What the schedule produced, if anything' },
            { label: 'Transfers durable', value: `${([1, 2] as TxnId[]).filter((t) => sim.durable[t] === 2).length} of 2`, hint: 'Transactions whose debit and credit both survived' },
            { label: 'Lock waits', value: sim.waits + (sim.victim ? ' → deadlock' : ''), hint: 'Blocked statements; a cycle costs one transaction its work' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{headline}</strong> {body}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Txn</th>
              <th>Statement</th>
              <th>Value</th>
              <th>Row A</th>
              <th>Row B</th>
              <th>What happened</th>
            </tr>
          </thead>
          <tbody>
            {sim.trace.map((r) => (
              <tr key={r.n}>
                <td>{r.n}</td>
                <td>T{SUB[r.txn]}</td>
                <td>
                  <code>{r.sql}</code>
                </td>
                <td>{r.sees}</td>
                <td>
                  ${r.a}
                  {r.uncA ? ' (uncommitted)' : ''}
                </td>
                <td>
                  ${r.b}
                  {r.uncB ? ' (uncommitted)' : ''}
                </td>
                <td>{r.note}</td>
              </tr>
            ))}
            <tr>
              <td>{sim.trace.length + 1}</td>
              <td>—</td>
              <td>
                <code>{sim.crashed ? '-- crash recovery' : '-- end of schedule'}</code>
              </td>
              <td>—</td>
              <td>${sim.a}</td>
              <td>${sim.b}</td>
              <td>
                A + B = ${sum}
                {ok ? ' — the invariant holds.' : ` — the invariant is broken by $${Math.abs(INVARIANT - sum)}.`}
              </td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label="A schedule of two interleaved bank transfers, with the stored balances and their sum after every step"
          >
            {([1, 2] as TxnId[]).map((t, li) => (
              <g key={t}>
                <rect
                  x={labelW}
                  y={laneY(li)}
                  width={svgW - labelW - 8}
                  height={44}
                  rx={8}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={0} y={laneY(li) + 20} fill="var(--viz-ink)" fontWeight={600}>
                  {TITLE[t]}
                </text>
                <text x={0} y={laneY(li) + 35} fill="var(--viz-ink-muted)">
                  {mode === 'autocommit' ? 'autocommit: 4 transactions' : 'BEGIN … COMMIT'}
                </text>
              </g>
            ))}

            {sim.trace.map((r, i) => (
              <g key={r.n}>
                <text x={x(i) + colW / 2} y={headY} textAnchor="middle" fill="var(--viz-ink-muted)">
                  {r.n}
                </text>
                <g {...tip(<>{r.note}</>)} style={{ cursor: 'help' }}>
                  <rect
                    x={x(i) + 4}
                    y={laneY(r.txn === 1 ? 0 : 1) + 6}
                    width={colW - 8}
                    height={20}
                    rx={5}
                    fill={OP_FILL[r.op]}
                    stroke={r.op === 'read' ? 'var(--viz-ink-2)' : 'var(--viz-surface)'}
                    strokeWidth={r.op === 'read' ? 1.5 : 2}
                    strokeDasharray={r.op === 'noop' ? '3 2' : undefined}
                  />
                  <text
                    x={x(i) + colW / 2}
                    y={laneY(r.txn === 1 ? 0 : 1) + 38}
                    textAnchor="middle"
                    fill="var(--viz-ink)"
                  >
                    {r.chip}
                  </text>
                </g>
              </g>
            ))}

            {(['A', 'B'] as RowId[]).map((rid, ri) => (
              <g key={rid}>
                <text x={0} y={rowY(ri) + 12} fill="var(--viz-ink)">
                  {`row ${rid} (stored)`}
                </text>
                {sim.trace.map((r, i) => {
                  const v = rid === 'A' ? r.a : r.b;
                  const dirty = rid === 'A' ? r.uncA : r.uncB;
                  return (
                    <g key={r.n}>
                      {dirty ? (
                        <rect
                          x={x(i) + 10}
                          y={rowY(ri)}
                          width={colW - 20}
                          height={17}
                          rx={4}
                          fill="none"
                          stroke="var(--viz-dirty)"
                          strokeDasharray="3 2"
                        />
                      ) : null}
                      <text
                        x={x(i) + colW / 2}
                        y={rowY(ri) + 12}
                        textAnchor="middle"
                        fill={dirty ? 'var(--viz-dirty)' : 'var(--viz-ink)'}
                      >
                        {`$${v}`}
                      </text>
                    </g>
                  );
                })}
              </g>
            ))}

            <text x={0} y={rowY(2) + 12} fill="var(--viz-ink)" fontWeight={600}>
              A + B
            </text>
            {sim.trace.map((r, i) => {
              const s = r.a + r.b;
              return (
                <text
                  key={r.n}
                  x={x(i) + colW / 2}
                  y={rowY(2) + 12}
                  textAnchor="middle"
                  fill={s === INVARIANT ? 'var(--viz-good)' : 'var(--viz-critical)'}
                >
                  {`$${s}${s === INVARIANT ? '' : ' ✗'}`}
                </text>
              );
            })}

            {sim.crashed ? (
              <g>
                <line
                  x1={x(cols) + 2}
                  x2={x(cols) + 2}
                  y1={4}
                  y2={height - 4}
                  stroke="var(--viz-critical)"
                  strokeWidth={1.5}
                  strokeDasharray="6 4"
                />
                <text x={x(cols) + 8} y={headY} fill="var(--viz-critical)">
                  power cut
                </text>
                {([1, 2] as TxnId[]).map((t, li) => (
                  <g key={t}>
                    <rect
                      x={x(cols) + 8}
                      y={laneY(li) + 6}
                      width={colW - 12}
                      height={20}
                      rx={5}
                      fill={
                        sim.undone.includes(t)
                          ? 'var(--viz-stale)'
                          : sim.durable[t] > 0
                            ? 'var(--viz-clean)'
                            : 'var(--viz-neutral)'
                      }
                      stroke="var(--viz-surface)"
                      strokeWidth={2}
                    />
                    <text x={x(cols) + colW / 2 + 2} y={laneY(li) + 38} textAnchor="middle" fill="var(--viz-ink)">
                      {sim.undone.includes(t) ? 'undone' : sim.durable[t] > 0 ? 'redone' : 'never ran'}
                    </text>
                  </g>
                ))}
                {[0, 1, 2].map((ri) => (
                  <text
                    key={ri}
                    x={x(cols) + colW / 2 + 2}
                    y={rowY(ri) + 12}
                    textAnchor="middle"
                    fill={ri === 2 ? (ok ? 'var(--viz-good)' : 'var(--viz-critical)') : 'var(--viz-ink)'}
                    fontWeight={ri === 2 ? 600 : 400}
                  >
                    {ri === 0 ? `$${sim.a}` : ri === 1 ? `$${sim.b}` : `$${sum}${ok ? '' : ' ✗'}`}
                  </text>
                ))}
              </g>
            ) : null}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
