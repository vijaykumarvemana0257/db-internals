import { useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtTime,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * The same order-taking workload, run twice: once against an append-only CSV file and
 * once against a table in an engine that owns a log, an index and a set of constraints.
 *
 * Everything the file lane does wrong here is a real POSIX behaviour, not a strawman:
 *  - two writers that each compute their own end-of-file offset overwrite each other
 *    (O_APPEND fixes exactly this and nothing else),
 *  - a crash mid-write leaves a prefix of a record with no framing and no checksum, so
 *    no reader can tell a short line from a legitimate one,
 *  - a client that retries an un-acked write appends a second copy, because a file has
 *    no notion of a key,
 *  - a lookup with no declared uniqueness cannot stop at the first match, so it is a
 *    full scan whose cost is linear in the dataset.
 *
 * The engine lane answers each one with a specific structure: a write-ahead log with a
 * per-record CRC, a primary-key B-tree, and typed/checked columns.
 */

/* ------------------------------------------------------------------ the cost model */

const ROW_BYTES = 48; // "10047,ACME-812,3,49.90\n" plus slack for longer ids/amounts
const PAGE = 8192; // Postgres BLCKSZ
const ENTRY_BYTES = 20; // 8-byte bigint key + IndexTupleData header + line pointer
const FANOUT = Math.floor(((PAGE - 24) * 0.9) / ENTRY_BYTES); // ~367 entries per btree page
const SEQ_BW = 3.5e9; // NVMe streaming bandwidth, bytes/s
const RANDOM_NS = 20_000; // one random page read on NVMe

function btreeHeight(rows: number) {
  return Math.max(1, Math.ceil(Math.log(Math.max(rows, 2)) / Math.log(FANOUT)));
}

type Path = 'scan' | 'sorted';

function fileLookup(rows: number, path: Path) {
  if (path === 'sorted') {
    const probes = Math.max(1, Math.ceil(Math.log2(Math.max(rows, 2))));
    return { unit: 'probes', count: probes, bytes: probes * 4096, ns: probes * RANDOM_NS };
  }
  const bytes = rows * ROW_BYTES;
  return { unit: 'rows', count: rows, bytes, ns: (bytes / SEQ_BW) * 1e9 };
}

function fileAppend(rows: number, path: Path) {
  if (path === 'sorted') {
    // Inserting in key order into a flat sorted file means rewriting the tail: half the
    // file on average, read and written back.
    const bytes = (rows * ROW_BYTES) / 2;
    return { bytes, ns: ((bytes * 2) / SEQ_BW) * 1e9 };
  }
  return { bytes: ROW_BYTES, ns: 2_000 };
}

function dbLookup(rows: number) {
  const h = btreeHeight(rows);
  return { h, pages: h + 1, bytes: (h + 1) * PAGE, ns: (h + 1) * RANDOM_NS };
}

function dbAppend(rows: number) {
  const h = btreeHeight(rows);
  return { bytes: ROW_BYTES + ENTRY_BYTES, ns: h * RANDOM_NS };
}

/* --------------------------------------------------------------------- the records */

/** Deterministic order bodies — same render on the server and in the browser. */
const BODY = (() => {
  const rng = makeRng(19700101);
  return Array.from({ length: 64 }, () => {
    const qty = 1 + Math.floor(rng() * 9);
    const amt = (10 + rng() * 90).toFixed(2);
    return { qty, amt };
  });
})();

function orderText(id: number) {
  const b = BODY[id % BODY.length];
  return `${10_000 + id},ACME-${812 + id},${b.qty},${b.amt}`;
}

type Kind = 'ok' | 'torn' | 'dup' | 'clobbered' | 'bad' | 'inflight' | 'gone';

const GLYPH: Record<Kind, string> = {
  ok: '✓',
  torn: '⚠',
  dup: '⧉',
  clobbered: '✕',
  bad: '!',
  inflight: '…',
  gone: '✕',
};

// Semantic roles first (--viz-clean/-dirty/-stale, status colors for damage). A duplicate
// is a distinct kind and must not reuse --viz-2, because --viz-dirty already IS --viz-2;
// --viz-7 is the first free slot that is neither low-contrast on light nor status-coded.
const KIND_COLOR: Record<Kind, string> = {
  ok: 'var(--viz-clean)',
  torn: 'var(--viz-critical)',
  dup: 'var(--viz-7)',
  clobbered: 'var(--viz-critical)',
  bad: 'var(--viz-warning)',
  inflight: 'var(--viz-dirty)',
  gone: 'var(--viz-stale)',
};

type Line = { key: string; kind: Kind; text: string; hint: string };
type Wal = { key: string; kind: 'ok' | 'inflight' | 'gone'; lsn: string; hint: string };

type S = {
  file: Line[];
  table: Line[];
  wal: Wal[];
  next: number; // next order number
  lastOk: number; // last order id that actually committed (what a client would retry)
  lsn: number; // fake LSN byte offset
  crashed: boolean;
  fileBroken: number; // records the file lane silently lost or mangled
  dbRejects: number;
  head: string;
  body: string;
};

const INITIAL: S = {
  file: [],
  table: [],
  wal: [],
  next: 1,
  lastOk: 0,
  lsn: 0x1a2f,
  crashed: false,
  fileBroken: 0,
  dbRejects: 0,
  head: 'Two clients, one order stream, two storage strategies.',
  body:
    'Press "2 clients append" to have client A and client B each write one order. Then break things: ' +
    'pull the power mid-write, retry an un-acked request, or submit a row with a garbage quantity. ' +
    'The left lane is a CSV file opened O_WRONLY; the right lane is a table with a primary key, a ' +
    'write-ahead log and typed columns.',
};

type Cfg = { interleave: boolean; oappend: boolean; rows: number; path: Path };
type Op = 'append' | 'crash' | 'restart' | 'retry' | 'bad' | 'reset';

function lsnStr(n: number) {
  return `0/${n.toString(16).toUpperCase()}`;
}

function apply(s: S, op: Op, cfg: Cfg): S {
  if (op === 'reset') return INITIAL;
  const n = { ...s, file: [...s.file], table: [...s.table], wal: [...s.wal] };

  switch (op) {
    case 'append': {
      if (s.crashed) return s;
      const a = s.next;
      const b = s.next + 1;
      n.next = s.next + 2;
      const ta = orderText(a);
      const tb = orderText(b);

      if (cfg.interleave && !cfg.oappend) {
        // Both writers did lseek(fd, 0, SEEK_END) then write(). They agreed on the same
        // offset, so B's bytes land on top of A's and the leftover tail of A survives.
        const tail = ta.slice(Math.max(0, tb.length));
        n.file.push({
          key: `f${a}`,
          kind: 'clobbered',
          text: tb + tail,
          hint:
            'Both writers took the end-of-file offset, then wrote there. B overwrote A, and whatever of A ' +
            'was longer is still on disk as a tail. One order vanished and nothing returned an error.',
        });
        n.fileBroken += 1;
      } else {
        n.file.push({
          key: `f${a}`,
          kind: 'ok',
          text: ta,
          hint: cfg.oappend
            ? 'O_APPEND makes the seek-to-end and the write one atomic operation under the inode lock, so ' +
              'concurrent appends of small records do not interleave on a local filesystem.'
            : 'Written while the other client was idle, so the offset race never happened.',
        });
        n.file.push({ key: `f${b}`, kind: 'ok', text: tb, hint: 'Appended cleanly.' });
      }

      for (const [id, t] of [
        [a, ta],
        [b, tb],
      ] as const) {
        n.lsn += 64;
        n.wal.push({
          key: `w${id}`,
          kind: 'ok',
          lsn: lsnStr(n.lsn),
          hint: `Insert of order ${10_000 + id}, then the commit record. CRC-32C over the record body; recovery replays it.`,
        });
        n.table.push({
          key: `t${id}`,
          kind: 'ok',
          text: t,
          hint: 'A committed heap tuple, reachable through orders_pkey. Two concurrent inserts take a row lock each, never a byte range.',
        });
        n.lastOk = id;
      }

      n.head =
        cfg.interleave && !cfg.oappend
          ? 'Two concurrent appends, one surviving row in the file.'
          : 'Two orders appended to both lanes.';
      n.body =
        cfg.interleave && !cfg.oappend
          ? 'A file offset is per-descriptor state, not shared state. Two writers that each compute "the end" ' +
            'write to the same place. The engine never had this problem: each insert allocates its own tuple ' +
            'slot under a page-level lock, and the log records both.'
          : cfg.interleave
            ? 'O_APPEND closed the offset race — and closed only that. Try the crash and the retry: neither of ' +
              'them is an offset problem, and O_APPEND does nothing for either.'
            : 'Serialized writers, so both lanes look identical. Turn on "interleave the writers" and try again.';
      break;
    }

    case 'crash': {
      if (s.crashed) return s;
      const id = s.next;
      n.next = s.next + 1;
      const full = orderText(id);
      const cut = Math.max(6, Math.floor(full.length * 0.55));
      n.file.push({
        key: `f${id}`,
        kind: 'torn',
        text: full.slice(0, cut),
        hint:
          'The power went while the record was being written out. A prefix reached the media. There is no ' +
          'length prefix, no checksum and no commit marker, so nothing downstream can tell this from a ' +
          'legitimate short line.',
      });
      n.fileBroken += 1;
      n.lsn += 64;
      n.wal.push({
        key: `w${id}`,
        kind: 'inflight',
        lsn: lsnStr(n.lsn),
        hint: 'A partial WAL record: the bytes stop mid-record and the CRC does not match. Recovery treats this as the end of the log.',
      });
      n.table.push({
        key: `t${id}`,
        kind: 'inflight',
        text: full,
        hint: 'Uncommitted. Its xmin belongs to a transaction with no commit record, so no snapshot will ever see it.',
      });
      n.crashed = true;
      n.head = 'Power cut mid-write.';
      n.body =
        'Both lanes now hold a half-written record. The difference is what happens next: the file has no ' +
        'framing to detect it, while the log has a CRC per record and a commit record that this transaction ' +
        'never reached. Press restart.';
      break;
    }

    case 'restart': {
      if (!s.crashed) return s;
      n.wal = s.wal.map((w) =>
        w.kind === 'inflight'
          ? {
              ...w,
              kind: 'gone',
              hint: 'CRC mismatch at recovery. Redo stops here; this is now the end of the log and the segment tail is reused.',
            }
          : w,
      );
      n.table = s.table.filter((t) => t.kind !== 'inflight');
      n.file = s.file.map((f) =>
        f.kind === 'torn'
          ? {
              ...f,
              hint:
                'Still here after the restart, and still indistinguishable from a real row. Every consumer of ' +
                'this file now needs its own guess about what a valid line looks like.',
            }
          : f,
      );
      n.crashed = false;
      n.head = 'Restart: one lane recovered, the other did not know it was broken.';
      n.body =
        'Recovery scanned forward from the last checkpoint, replayed every record whose CRC matched, hit the ' +
        'torn record, and stopped — the table is exactly at the last commit, no partial order, nothing to ' +
        'clean up by hand. The CSV came back byte-identical, torn line included, because nothing in the file ' +
        'layer ever knew a record had a beginning and an end.';
      break;
    }

    case 'retry': {
      if (s.crashed || s.lastOk === 0) return s;
      const last = s.lastOk;
      const t = orderText(last);
      n.file.push({
        key: `f${last}r${s.file.length}`,
        kind: 'dup',
        text: t,
        hint:
          'The client timed out waiting for an ack and sent the order again. A file has no keys, so this is ' +
          'simply a second line. Downstream you will ship the order twice.',
      });
      n.fileBroken += 1;
      n.dbRejects += 1;
      n.head = 'The retry duplicated the order in the file and was rejected by the table.';
      n.body =
        'This is the ordinary failure of every network client: the request succeeded and the response was ' +
        'lost. The engine answers with ERROR: duplicate key value violates unique constraint "orders_pkey" ' +
        '(SQLSTATE 23505; MySQL reports ER_DUP_ENTRY, 1062), so the retry is idempotent for free. The file ' +
        'answers by growing by one line.';
      break;
    }

    case 'bad': {
      if (s.crashed) return s;
      const id = s.next;
      n.next = s.next + 1;
      n.file.push({
        key: `f${id}b`,
        kind: 'bad',
        text: `${10_000 + id},ACME-${812 + id},two,`,
        hint:
          'A quantity of "two" and an empty amount. write() has no opinion about your bytes; the error will ' +
          'surface in whichever consumer parses this line, months later.',
      });
      n.fileBroken += 1;
      n.dbRejects += 1;
      n.head = 'The file accepted a malformed row. The table refused it.';
      n.body =
        'qty is an integer column with CHECK (qty > 0) and amount is NOT NULL, so the insert fails at parse ' +
        'time: ERROR: invalid input syntax for type integer: "two" (SQLSTATE 22P02). The constraint is ' +
        'declared once, next to the data, and holds for every writer that will ever exist — including the ' +
        'ad-hoc script somebody writes next year.';
      break;
    }
  }
  return n;
}

/* ----------------------------------------------------------------- the component */

export default function FileVsEngineLanes() {
  const [rowsExp, setRowsExp] = useState(5); // 10^5 = 100k rows
  const [path, setPath] = useState<Path>('scan');
  const [interleave, setInterleave] = useState(true);
  const [oappend, setOappend] = useState(false);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const rows = Math.round(10 ** rowsExp);
  const cfg: Cfg = { interleave, oappend, rows, path };
  const run = (op: Op) => setS((cur) => apply(cur, op, cfg));

  const fl = fileLookup(rows, path);
  const fa = fileAppend(rows, path);
  const dl = dbLookup(rows);
  const da = dbAppend(rows);

  /* ---- layout */
  const MAX = 8;
  const svgW = Math.max(560, width);
  const gap = 14;
  const laneW = (svgW - gap) / 2;
  const titleH = 34;
  const walH = 58;
  const boxH = 26 + MAX * 24 + 8; // one lane box holding MAX record chips
  const tableY = titleH + walH + 10;
  const bodyH = walH + 10 + boxH; // file box spans the WAL strip + table box
  const costY = titleH + bodyH + 22;
  const costH = 104;
  const height = costY + costH;

  const fileShown = s.file.slice(-MAX);
  const tableShown = s.table.slice(-MAX);
  const WAL_SLOTS = 4;
  const walShown = s.wal.slice(-WAL_SLOTS);

  const lookupMax = Math.max(fl.ns, dl.ns);
  const appendMax = Math.max(fa.ns, da.ns);
  const barX = 132;
  const barW = Math.max(90, svgW - barX - 150);
  const bar = (v: number, max: number) => Math.max(3, (v / max) * barW);

  const chip = (
    x: number,
    y: number,
    w: number,
    l: { kind: Kind; text: string; hint: string; key: string },
  ) => (
    <g key={l.key} {...tip(<>{l.hint}</>)} style={{ cursor: 'help' }}>
      <rect x={x} y={y} width={w} height={20} rx={4} fill="var(--viz-plane)" stroke="var(--viz-border)" />
      <rect x={x} y={y} width={4} height={20} rx={2} fill={KIND_COLOR[l.kind]} />
      <text x={x + 12} y={y + 14} fill={l.kind === 'gone' ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}>
        {GLYPH[l.kind]} {l.text}
      </text>
    </g>
  );

  return (
    <VizPanel
      title="The same orders, into a file and into an engine"
      subtitle="Two clients append orders to a CSV on the left and to a table on the right. Break the workload — concurrent writers, a crash mid-write, a retried request, a malformed row — and grow the dataset to see what a lookup costs on each side."
      controls={
        <>
          <Slider
            label="Rows in the dataset"
            min={3}
            max={7}
            step={0.25}
            value={rowsExp}
            onChange={setRowsExp}
            format={() => fmtNum(rows)}
          />
          <Segmented
            label="File lookup path"
            value={path}
            onChange={setPath}
            options={[
              { value: 'scan', label: 'Full scan', title: 'No declared key, so a reader cannot stop at the first match' },
              { value: 'sorted', label: 'Keep it sorted', title: 'Hand-built index: binary search, at the cost of rewriting the tail on every insert' },
            ]}
          />
          <Check label="Interleave the writers" checked={interleave} onChange={setInterleave} />
          <Check label="O_APPEND" checked={oappend} onChange={setOappend} />
          <Button onClick={() => run('append')} disabled={s.crashed} primary>
            2 clients append
          </Button>
          <Button onClick={() => run('crash')} disabled={s.crashed} title="Power cut while a record is being written">
            Crash mid-write
          </Button>
          <Button onClick={() => run('restart')} disabled={!s.crashed}>
            Restart
          </Button>
          <Button onClick={() => run('retry')} disabled={s.crashed || s.lastOk === 0} title="Client timed out and re-sent the same order">
            Retry un-acked order
          </Button>
          <Button onClick={() => run('bad')} disabled={s.crashed} title="qty = 'two', amount empty">
            Submit a bad row
          </Button>
          <Button onClick={() => run('reset')}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: '✓ committed / durable record', color: 'var(--viz-clean)' },
            { label: '… in flight, uncommitted', color: 'var(--viz-dirty)' },
            { label: '⧉ duplicate (no key to stop it)', color: 'var(--viz-7)' },
            { label: '! malformed, accepted anyway', color: 'var(--viz-warning)' },
            { label: '⚠ torn / ✕ clobbered', color: 'var(--viz-critical)' },
            { label: 'dashed WAL record: CRC failed, discarded by recovery', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'File lookup',
              value: `${fmtNum(fl.count)} ${fl.unit}`,
              hint:
                path === 'scan'
                  ? 'No uniqueness is declared anywhere, so a correct reader must examine every row — it cannot stop at the first match'
                  : 'Binary search over a sorted flat file: one random read per probe',
            },
            { label: 'File lookup time', value: fmtTime(fl.ns), hint: `${fmtBytes(fl.bytes)} read` },
            {
              label: 'Index lookup',
              value: `${dl.pages} pages`,
              hint: `B-tree of height ${dl.h} (fan-out ${FANOUT} entries per 8 KB page) plus one heap page`,
            },
            { label: 'Index lookup time', value: fmtTime(dl.ns) },
            { label: 'Lookup ratio', value: `${fmtNum(Math.max(fl.ns, dl.ns) / Math.min(fl.ns, dl.ns), 1)}×`, hint: 'How much more one lookup costs on the slower lane' },
            {
              label: 'File append cost',
              value: fmtBytes(fa.bytes),
              hint: path === 'sorted' ? 'Rewriting the tail of a sorted file to make room' : 'One write() at the end of the file',
            },
            { label: 'Rows the file lane broke', value: fmtNum(s.fileBroken), hint: 'Clobbered, torn, duplicated or malformed — none of them reported an error' },
            { label: 'Writes the engine refused', value: fmtNum(s.dbRejects), hint: 'Constraint violations returned to the client at insert time' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Rows</th>
                <th>Scan: rows examined</th>
                <th>Scan: bytes read</th>
                <th>Scan time @ 3.5 GB/s</th>
                <th>B-tree height</th>
                <th>Index: pages read</th>
                <th>Index time @ 20 µs/page</th>
                <th>Ratio</th>
              </tr>
            </thead>
            <tbody>
              {[1e3, 1e4, 1e5, 1e6, 1e7, 1e8].map((r) => {
                const f = fileLookup(r, 'scan');
                const d = dbLookup(r);
                return (
                  <tr key={r}>
                    <td>{fmtNum(r)}</td>
                    <td>{fmtNum(f.count)}</td>
                    <td>{fmtBytes(f.bytes)}</td>
                    <td>{fmtTime(f.ns)}</td>
                    <td>{d.h}</td>
                    <td>{d.pages}</td>
                    <td>{fmtTime(d.ns)}</td>
                    <td>{fmtNum(f.ns / d.ns, 1)}×</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Lane</th>
                <th>Record</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {s.file.length === 0 && s.table.length === 0 ? (
                <tr>
                  <td colSpan={3}>Nothing written yet.</td>
                </tr>
              ) : (
                <>
                  {s.file.map((f) => (
                    <tr key={`ft-${f.key}`}>
                      <td>orders.csv</td>
                      <td>{f.text}</td>
                      <td>{f.kind}</td>
                    </tr>
                  ))}
                  {s.table.map((t) => (
                    <tr key={`tt-${t.key}`}>
                      <td>orders</td>
                      <td>{t.text}</td>
                      <td>{t.kind === 'ok' ? 'committed' : t.kind}</td>
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label="A CSV file lane and a database table lane taking the same order stream, with lookup and append costs"
          >
            {/* lane headings */}
            <text x={0} y={14} fill="var(--viz-ink)" fontWeight={600}>
              orders.csv — bytes on disk
            </text>
            <text x={0} y={28} fill="var(--viz-ink-muted)">
              open(O_WRONLY{oappend ? ' | O_APPEND' : ''}) · no framing · no keys · no types
            </text>
            <text x={laneW + gap} y={14} fill="var(--viz-ink)" fontWeight={600}>
              orders table — log, index, constraints
            </text>
            <text x={laneW + gap} y={28} fill="var(--viz-ink-muted)">
              WAL (CRC per record) · orders_pkey B-tree, height {dl.h} · qty INT CHECK (qty &gt; 0)
            </text>

            {/* file lane box */}
            <rect
              x={0}
              y={titleH}
              width={laneW}
              height={bodyH}
              rx={8}
              fill="var(--viz-plane)"
              stroke="var(--viz-border)"
            />
            {s.file.length === 0 ? (
              <text x={12} y={titleH + 26} fill="var(--viz-ink-muted)">
                (empty file)
              </text>
            ) : null}
            {s.file.length > MAX ? (
              <text x={laneW - 12} y={titleH + 16} textAnchor="end" fill="var(--viz-ink-muted)">
                +{s.file.length - MAX} earlier lines
              </text>
            ) : null}
            {fileShown.map((l, i) => chip(10, titleH + 24 + i * 24, laneW - 20, l))}

            {/* WAL strip */}
            <rect
              x={laneW + gap}
              y={titleH}
              width={laneW}
              height={walH}
              rx={8}
              fill="var(--viz-plane)"
              stroke="var(--viz-border)"
            />
            <text x={laneW + gap + 10} y={titleH + 16} fill="var(--viz-ink-2)">
              write-ahead log
            </text>
            {walShown.length === 0 ? (
              <text x={laneW + gap + 10} y={titleH + 40} fill="var(--viz-ink-muted)">
                (no records)
              </text>
            ) : null}
            {walShown.map((w, i) => {
              const cw = Math.max(44, (laneW - 20 - (WAL_SLOTS - 1) * 6) / WAL_SLOTS);
              const x = laneW + gap + 10 + i * (cw + 6);
              return (
                <g key={w.key} {...tip(<>{w.hint}</>)} style={{ cursor: 'help' }}>
                  <rect
                    x={x}
                    y={titleH + 24}
                    width={cw}
                    height={22}
                    rx={4}
                    fill="var(--viz-plane)"
                    stroke={w.kind === 'ok' ? 'var(--viz-clean)' : w.kind === 'inflight' ? 'var(--viz-dirty)' : 'var(--viz-stale)'}
                    strokeWidth={1.5}
                    strokeDasharray={w.kind === 'gone' ? '3 2' : undefined}
                  />
                  <text
                    x={x + cw / 2}
                    y={titleH + 39}
                    textAnchor="middle"
                    fill={w.kind === 'gone' ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}
                  >
                    {w.lsn}
                  </text>
                </g>
              );
            })}

            {/* table box */}
            <rect
              x={laneW + gap}
              y={tableY}
              width={laneW}
              height={boxH}
              rx={8}
              fill="var(--viz-plane)"
              stroke="var(--viz-border)"
            />
            {s.table.length === 0 ? (
              <text x={laneW + gap + 12} y={tableY + 26} fill="var(--viz-ink-muted)">
                (0 rows)
              </text>
            ) : null}
            {s.table.length > MAX ? (
              <text x={laneW + gap + laneW - 12} y={tableY + 16} textAnchor="end" fill="var(--viz-ink-muted)">
                +{s.table.length - MAX} earlier rows
              </text>
            ) : null}
            {tableShown.map((l, i) => chip(laneW + gap + 10, tableY + 24 + i * 24, laneW - 20, l))}

            {/* cost band */}
            <line
              x1={0}
              x2={svgW}
              y1={costY - 12}
              y2={costY - 12}
              stroke="var(--viz-grid)"
              strokeWidth={1}
            />
            {[
              {
                label: `Lookup one order (${fmtNum(rows)} rows)`,
                bars: [
                  { name: path === 'sorted' ? 'file, binary search' : 'file, full scan', v: fl.ns, c: 'var(--viz-2)', sub: `${fmtNum(fl.count)} ${fl.unit}` },
                  { name: 'index descent', v: dl.ns, c: 'var(--viz-1)', sub: `${dl.pages} pages` },
                ],
                max: lookupMax,
                y: costY,
              },
              {
                label: 'Insert one order',
                bars: [
                  { name: path === 'sorted' ? 'file, keep sorted' : 'file, append', v: fa.ns, c: 'var(--viz-2)', sub: fmtBytes(fa.bytes) },
                  { name: 'table insert', v: da.ns, c: 'var(--viz-1)', sub: fmtBytes(da.bytes) },
                ],
                max: appendMax,
                y: costY + 54,
              },
            ].map((grp) => (
              <g key={grp.label}>
                <text x={0} y={grp.y + 2} fill="var(--viz-ink)" fontWeight={600}>
                  {grp.label}
                </text>
                {grp.bars.map((b, i) => (
                  <g key={b.name}>
                    <text x={barX - 8} y={grp.y + 20 + i * 20} textAnchor="end" fill="var(--viz-ink-2)">
                      {b.name}
                    </text>
                    <rect x={barX} y={grp.y + 9 + i * 20} width={bar(b.v, grp.max)} height={14} rx={3} fill={b.c} />
                    <text
                      x={barX + bar(b.v, grp.max) + 8}
                      y={grp.y + 20 + i * 20}
                      fill="var(--viz-ink-2)"
                      style={{ fontVariantNumeric: 'tabular-nums' }}
                    >
                      {fmtTime(b.v)} · {b.sub}
                    </text>
                  </g>
                ))}
              </g>
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
