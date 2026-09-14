import { useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
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
 * Log-before-page, made literal.
 *
 * Three heap pages, one WAL strip. Every update appends a record whose LSN *is* its byte
 * offset in the log, and stamps the page it touched with that record's end LSN. Two rules
 * then govern everything:
 *
 *   WAL rule    — a dirty page may not be written to disk until the log is flushed
 *                 through that page's pageLSN.
 *   commit rule — a transaction is committed when the log is flushed through its
 *                 commit record.
 *
 * Crash, and redo replays from the checkpoint's redo point, applying a record only when
 * its LSN exceeds the on-disk pageLSN. That single comparison is what makes replay
 * idempotent, and it is the only reason it is safe to replay a log twice.
 */

/* ------------------------------------------------------------------- LSN math */

const SEG_BYTES = 0x1000000; // 16 MB — the default WAL segment size
const BASE_LSN = 0x02bca000; // an arbitrary but fixed point inside segment 2
const MAX_RECS = 15;
const PAGE_BYTES = 8192;

/** PostgreSQL prints an LSN as high/low, hex, slash-separated: 0/02BCA000. */
function fmtLsn(n: number) {
  const hi = Math.floor(n / 2 ** 32);
  const lo = n % 2 ** 32;
  return `${hi.toString(16).toUpperCase()}/${lo.toString(16).toUpperCase().padStart(8, '0')}`;
}

/** The last five hex digits — enough to read a strip of records at a glance. */
function shortLsn(n: number) {
  return `…${(n % 2 ** 32).toString(16).toUpperCase().padStart(8, '0').slice(-5)}`;
}

/** timelineID + logical-log high word + segment number, the 24-hex pg_wal filename. */
function segName(n: number) {
  const hi = Math.floor(n / 2 ** 32);
  const seg = Math.floor((n % 2 ** 32) / SEG_BYTES);
  return `00000001${hi.toString(16).toUpperCase().padStart(8, '0')}${seg
    .toString(16)
    .toUpperCase()
    .padStart(8, '0')}`;
}

/* ---------------------------------------------------------------- record sizes */

/**
 * Deterministic sizes in the right ballpark. A PostgreSQL WAL record is a 24-byte
 * XLogRecord header plus block references plus payload, MAXALIGNed to 8; a full-page
 * image adds the 8 KB block minus its free-space "hole".
 */
const SIZES = (() => {
  const rng = makeRng(0x2bca); // fixed seed: same strip on the server and in the browser
  return {
    update: Array.from({ length: 64 }, () => 8 * Math.round((64 + rng() * 40) / 8)),
    hole: Array.from({ length: 64 }, () => 8 * Math.round((rng() * 2400) / 8)),
  };
})();

const COMMIT_SIZE = 34; // what pg_waldump reports for a bare commit: no subxacts, no invalidations
const CKPT_SIZE = 112; // XLogRecord + CheckPoint struct

/* ----------------------------------------------------------------- the state */

type Kind = 'update' | 'commit' | 'ckpt';

type Rec = {
  i: number;
  lsn: number; // byte offset of the record's first byte — this IS the LSN
  size: number;
  end: number; // lsn + size; what gets stamped into pageLSN
  kind: Kind;
  fpi: boolean;
  txn: number | null;
  page: number | null;
  val: number | null; // balance after this update
  prev: number | null;
};

type PageS = {
  name: string;
  memVal: number;
  memLSN: number;
  dirty: boolean;
  diskVal: number;
  diskLSN: number;
  dirtiedSinceCkpt: boolean;
};

type ReplayRow = {
  rec: number; // index into recs
  lsn: number;
  page: number | null;
  decision: 'apply' | 'skip' | 'noop';
  pageLSNBefore: number;
  valBefore: number | null;
  valAfter: number | null;
};

type Phase = 'run' | 'crashed' | 'replay' | 'done';

type LogRow = {
  n: number;
  op: string;
  insert: number;
  flushed: number;
  unflushed: number;
  dirty: number;
  detail: string;
};

type S = {
  recs: Rec[];
  insertLSN: number;
  flushedLSN: number;
  ackLSN: number; // the client has been told "committed" through here
  redoLSN: number; // the redo point of the last checkpoint
  pages: PageS[];
  openTxn: number | null;
  nextTxn: number;
  acks: { txn: number; lsn: number }[];
  blocked: number;
  phase: Phase;
  survivors: Rec[];
  cursor: number;
  replay: ReplayRow[];
  head: string;
  body: string;
  log: LogRow[];
};

const PAGE_NAMES = ['block 0', 'block 1', 'block 2'];
const START_VALS = [100, 200, 300];

function initialPages(): PageS[] {
  return PAGE_NAMES.map((name, i) => ({
    name,
    memVal: START_VALS[i],
    memLSN: 0,
    dirty: false,
    diskVal: START_VALS[i],
    diskLSN: 0,
    dirtiedSinceCkpt: false,
  }));
}

const INITIAL: S = {
  recs: [],
  insertLSN: BASE_LSN,
  flushedLSN: BASE_LSN,
  ackLSN: BASE_LSN,
  redoLSN: BASE_LSN,
  pages: initialPages(),
  openTxn: null,
  nextTxn: 7431,
  acks: [],
  blocked: 0,
  phase: 'run',
  survivors: [],
  cursor: 0,
  replay: [],
  head: 'Clean start: three heap blocks, an empty log, nothing dirty.',
  body:
    'Press UPDATE to change a row. The change goes to the in-memory copy of the block and to a log ' +
    'record — never to the block on disk. Watch the record\'s LSN, which is literally its byte offset ' +
    'in the WAL, get stamped into the page it touched.',
  log: [],
};

type Cfg = { fpw: boolean; sync: 'on' | 'off' };
type Act = 'update' | 'commit' | 'flush' | 'write' | 'ckpt' | 'crash' | 'step' | 'all';

function unflushed(s: S) {
  return Math.max(0, s.insertLSN - s.flushedLSN);
}

function dirtyCount(s: S) {
  return s.pages.filter((p) => p.dirty).length;
}

function push(s: S, op: string, detail: string, next: S): S {
  return {
    ...next,
    log: [
      ...s.log,
      {
        n: s.log.length + 1,
        op,
        insert: next.insertLSN,
        flushed: next.flushedLSN,
        unflushed: unflushed(next),
        dirty: dirtyCount(next),
        detail,
      },
    ],
  };
}

function apply(s: S, act: Act, cfg: Cfg, target: number): S {
  if (s.phase === 'crashed' && act !== 'step' && act !== 'all') return s;
  if ((s.phase === 'replay' || s.phase === 'done') && act !== 'step' && act !== 'all') return s;

  switch (act) {
    /* ------------------------------------------------------------- an update */
    case 'update': {
      if (s.recs.length >= MAX_RECS) return s;
      const k = s.recs.length;
      const p = s.pages[target];
      const txn = s.openTxn ?? s.nextTxn;
      const fpi = cfg.fpw && !p.dirtiedSinceCkpt;
      const hole = SIZES.hole[k % SIZES.hole.length];
      const size = SIZES.update[k % SIZES.update.length] + (fpi ? PAGE_BYTES - hole : 0);
      const lsn = s.insertLSN;
      const end = lsn + size;
      const rec: Rec = {
        i: k,
        lsn,
        size,
        end,
        kind: 'update',
        fpi,
        txn,
        page: target,
        val: p.memVal + 10,
        prev: p.memVal,
      };
      const pages = s.pages.map((q, i) =>
        i === target
          ? { ...q, memVal: q.memVal + 10, memLSN: end, dirty: true, dirtiedSinceCkpt: true }
          : q,
      );
      const next: S = {
        ...s,
        recs: [...s.recs, rec],
        insertLSN: end,
        pages,
        openTxn: txn,
        head: fpi
          ? `Record ${fmtLsn(lsn)} carries a full-page image — ${fmtBytes(size)}, not ${fmtBytes(
              SIZES.update[k % SIZES.update.length],
            )}.`
          : `Record ${fmtLsn(lsn)} appended, ${size} bytes. ${PAGE_NAMES[target]}.pageLSN = ${fmtLsn(end)}.`,
        body: fpi
          ? `This is the first change to ${PAGE_NAMES[target]} since the last checkpoint, and full_page_writes ` +
            `is on, so the record carries the whole 8 KB block image (minus its ${hole}-byte free-space hole) ` +
            `instead of a description of the edit. That is the torn-page defence: if the OS writes half of ` +
            `this block during a crash, redo does not need the half that survived. The next update to the ` +
            `same block in the same checkpoint interval is small again.`
          : `The record describes the change — "on block ${target}, offset, old → new" — and it went into the ` +
            `WAL insertion buffers, not to disk. The block's in-memory copy now differs from the on-disk copy: ` +
            `it is dirty, and its pageLSN says which log record it last absorbed.`,
      };
      return push(s, 'UPDATE', `txn ${txn} → ${PAGE_NAMES[target]}${fpi ? ' + FPI' : ''}`, next);
    }

    /* ------------------------------------------------------------- a commit */
    case 'commit': {
      if (s.openTxn === null || s.recs.length >= MAX_RECS) return s;
      const lsn = s.insertLSN;
      const end = lsn + COMMIT_SIZE;
      const rec: Rec = {
        i: s.recs.length,
        lsn,
        size: COMMIT_SIZE,
        end,
        kind: 'commit',
        fpi: false,
        txn: s.openTxn,
        page: null,
        val: null,
        prev: null,
      };
      const durable = cfg.sync === 'on';
      const next: S = {
        ...s,
        recs: [...s.recs, rec],
        insertLSN: end,
        flushedLSN: durable ? end : s.flushedLSN,
        ackLSN: end,
        acks: [...s.acks, { txn: s.openTxn, lsn: end }],
        openTxn: null,
        nextTxn: s.nextTxn + 1,
        head: durable
          ? `COMMIT txn ${s.openTxn}: log flushed through ${fmtLsn(end)}, then the client was told "ok".`
          : `COMMIT txn ${s.openTxn}: the client was told "ok" at ${fmtLsn(end)} — before any fsync.`,
        body: durable
          ? `synchronous_commit = on. The backend called XLogFlush(commitLSN): write() the WAL buffers, then ` +
            `fdatasync() the segment, and only then return to the client. The commit record is the whole of ` +
            `what makes this transaction durable — not one of the dirty blocks it touched has been written, ` +
            `and none of them needs to be.`
          : `synchronous_commit = off (or innodb_flush_log_at_trx_commit = 2). The commit record is in memory; ` +
            `the WAL writer will flush it within a fraction of a second. The client already believes the ` +
            `transaction is durable. Crash now and that belief is wrong — the "acked, not durable" tile is ` +
            `counting exactly how wrong.`,
      };
      return push(s, 'COMMIT', `txn ${s.openTxn}${durable ? ' + fsync' : ' (ack first)'}`, next);
    }

    /* --------------------------------------------------------- flush the log */
    case 'flush': {
      if (unflushed(s) === 0) {
        return push(s, 'XLogFlush', 'nothing to flush', {
          ...s,
          head: 'Nothing to flush — the log is already durable through the insert point.',
          body: 'A flush with no pending bytes costs a lock acquisition and returns. Real engines check this first.',
        });
      }
      const bytes = unflushed(s);
      const next: S = {
        ...s,
        flushedLSN: s.insertLSN,
        head: `Flushed ${fmtBytes(bytes)} — durable through ${fmtLsn(s.insertLSN)}.`,
        body:
          `write() of the WAL buffers followed by fdatasync() on the segment file. One fsync, however many ` +
          `records it covers: this is why group commit works and why the cost per transaction falls as ` +
          `concurrency rises. The flushed-LSN cursor is the only thing that decides which dirty blocks are ` +
          `now allowed to be written.`,
      };
      return push(s, 'XLogFlush', fmtBytes(bytes), next);
    }

    /* ------------------------------------------ write a dirty page to disk */
    case 'write': {
      const p = s.pages[target];
      if (!p.dirty) {
        return push(s, 'write page', `${PAGE_NAMES[target]} not dirty`, {
          ...s,
          head: `${PAGE_NAMES[target]} is clean — nothing to write.`,
          body: 'The in-memory copy matches the copy on disk, so evicting it costs nothing but the frame.',
        });
      }
      if (p.memLSN > s.flushedLSN) {
        return push(s, 'write page', `BLOCKED: pageLSN ${fmtLsn(p.memLSN)} > flushed`, {
          ...s,
          blocked: s.blocked + 1,
          head: `Blocked. ${PAGE_NAMES[target]}.pageLSN = ${fmtLsn(p.memLSN)}, but the log is only durable through ${fmtLsn(
            s.flushedLSN,
          )}.`,
          body:
            `This is the write-ahead rule, and it is not advisory — PostgreSQL's FlushBuffer() calls ` +
            `XLogFlush(BufferGetLSN(buf)) before it hands the block to the OS, and InnoDB's page cleaner waits ` +
            `on the log writer the same way. Write the block first and a crash leaves a block on disk whose ` +
            `change has no log record: redo cannot replay it, undo cannot roll it back, and nothing in the ` +
            `system can tell it happened. Flush the log, then try again.`,
        });
      }
      const pages = s.pages.map((q, i) =>
        i === target ? { ...q, diskVal: q.memVal, diskLSN: q.memLSN, dirty: false } : q,
      );
      return push(s, 'write page', `${PAGE_NAMES[target]} → disk @ ${fmtLsn(p.memLSN)}`, {
        ...s,
        pages,
        head: `${PAGE_NAMES[target]} written to disk, carrying pageLSN ${fmtLsn(p.memLSN)}.`,
        body:
          `The block's own header now records the last log record it absorbed. That stamp is what makes redo ` +
          `idempotent: after a crash, recovery compares every record's LSN against this number and skips ` +
          `anything the block already contains. The block can be written a hundred times or never; replay ` +
          `reaches the same state either way.`,
      });
    }

    /* ------------------------------------------------------------ checkpoint */
    case 'ckpt': {
      if (s.recs.length >= MAX_RECS) return s;
      const redo = s.insertLSN; // the redo point is taken before the dirty pages are written
      const pages = s.pages.map((q) =>
        q.dirty ? { ...q, diskVal: q.memVal, diskLSN: q.memLSN, dirty: false, dirtiedSinceCkpt: false } : { ...q, dirtiedSinceCkpt: false },
      );
      const lsn = s.insertLSN;
      const end = lsn + CKPT_SIZE;
      const rec: Rec = {
        i: s.recs.length,
        lsn,
        size: CKPT_SIZE,
        end,
        kind: 'ckpt',
        fpi: false,
        txn: null,
        page: null,
        val: null,
        prev: null,
      };
      return push(s, 'CHECKPOINT', `redo point ${fmtLsn(redo)}`, {
        ...s,
        recs: [...s.recs, rec],
        insertLSN: end,
        flushedLSN: end,
        redoLSN: redo,
        pages,
        head: `Checkpoint: redo point ${fmtLsn(redo)}, every dirty block written, log flushed.`,
        body:
          `The redo point is recorded first, then the dirty blocks go out, then the checkpoint record is ` +
          `written and the control file is updated to point at it. Recovery will start scanning here instead ` +
          `of at the beginning of time — that, and nothing else, is what a checkpoint buys. Because every ` +
          `block is now clean, the next update to each one logs a full page image again.`,
      });
    }

    /* ----------------------------------------------------------------- crash */
    case 'crash': {
      const survivors = s.recs.filter((r) => r.end <= s.flushedLSN);
      const startIdx = survivors.findIndex((r) => r.lsn >= s.redoLSN);
      const lostAcks = s.acks.filter((a) => a.lsn > s.flushedLSN);
      const lostBytes = unflushed(s);
      return push(s, 'CRASH', `${lostBytes} B of log lost, ${lostAcks.length} acked txn lost`, {
        ...s,
        phase: 'crashed',
        survivors,
        cursor: startIdx < 0 ? survivors.length : startIdx,
        replay: [],
        openTxn: null,
        pages: s.pages.map((p) => ({ ...p, memVal: p.diskVal, memLSN: p.diskLSN, dirty: false })),
        head:
          lostAcks.length > 0
            ? `Crash. ${fmtBytes(lostBytes)} of log never reached the disk, and ${lostAcks.length} acknowledged ` +
              `commit${lostAcks.length === 1 ? '' : 's'} went with it.`
            : `Crash. The buffer pool is gone; ${fmtBytes(lostBytes)} of unflushed log is gone with it.`,
        body:
          `Everything in memory evaporated: dirty blocks, WAL buffers, transaction state. What is left on disk ` +
          `is the heap file — a mixture of blocks from different points in time — and the WAL through ` +
          `${fmtLsn(s.flushedLSN)}. Now step redo forward from the checkpoint's redo point and watch it ` +
          `reconcile the two.`,
      });
    }

    /* ---------------------------------------------------------- one redo step */
    case 'step': {
      if (s.phase !== 'crashed' && s.phase !== 'replay') return s;
      if (s.cursor >= s.survivors.length) {
        return { ...s, phase: 'done', head: 'Redo complete.', body: replaySummary(s) };
      }
      const r = s.survivors[s.cursor];
      let pages = s.pages;
      let row: ReplayRow;
      if (r.kind !== 'update' || r.page === null) {
        row = {
          rec: r.i,
          lsn: r.lsn,
          page: null,
          decision: 'noop',
          pageLSNBefore: 0,
          valBefore: null,
          valAfter: null,
        };
      } else {
        const p = s.pages[r.page];
        const applyIt = r.end > p.diskLSN;
        row = {
          rec: r.i,
          lsn: r.lsn,
          page: r.page,
          decision: applyIt ? 'apply' : 'skip',
          pageLSNBefore: p.diskLSN,
          valBefore: p.diskVal,
          valAfter: applyIt ? r.val : p.diskVal,
        };
        if (applyIt) {
          pages = s.pages.map((q, i) =>
            i === r.page ? { ...q, diskVal: r.val!, diskLSN: r.end, memVal: r.val!, memLSN: r.end } : q,
          );
        }
      }
      const next: S = {
        ...s,
        pages,
        cursor: s.cursor + 1,
        replay: [...s.replay, row],
        phase: s.cursor + 1 >= s.survivors.length ? 'done' : 'replay',
      };
      next.head =
        row.decision === 'apply'
          ? `Apply: record LSN ${fmtLsn(r.end)} > ${PAGE_NAMES[r.page!]}.pageLSN ${fmtLsn(row.pageLSNBefore)}.`
          : row.decision === 'skip'
            ? `Skip: ${PAGE_NAMES[r.page!]}.pageLSN is already ${fmtLsn(row.pageLSNBefore)}.`
            : r.kind === 'commit'
              ? `Commit record for txn ${r.txn} — noted, no page touched.`
              : 'Checkpoint record — the scan started here.';
      next.body =
        row.decision === 'apply'
          ? `The block on disk predates this record, so redo re-executes the change and stamps the block with ` +
            `the record's end LSN. Value ${row.valBefore} → ${row.valAfter}.`
          : row.decision === 'skip'
            ? `The block already absorbed this record before the crash — someone wrote it out, or a checkpoint ` +
              `did. Redo does nothing. Run recovery ten times and the answer does not move; this comparison is ` +
              `the whole of the idempotence argument.`
            : r.kind === 'commit'
              ? `Redo repeats history for committed and uncommitted transactions alike; the commit records are ` +
                `what the analysis pass uses to decide which transactions need an undo pass afterwards.`
              : `Everything before the redo point is guaranteed to be on disk already, so recovery never has to ` +
                `read it.`;
      if (next.phase === 'done') next.body = `${next.body} ${replaySummary(next)}`;
      return next;
    }

    case 'all': {
      let cur = s;
      let guard = 0;
      while ((cur.phase === 'crashed' || cur.phase === 'replay') && guard++ < 64) {
        cur = apply(cur, 'step', cfg, target);
      }
      return cur;
    }
  }
  return s;
}

function replaySummary(s: S) {
  const applied = s.replay.filter((r) => r.decision === 'apply').length;
  const skipped = s.replay.filter((r) => r.decision === 'skip').length;
  const lost = s.acks.filter((a) => a.lsn > s.flushedLSN).length;
  return (
    `${applied} record${applied === 1 ? '' : 's'} applied, ${skipped} skipped as already present. ` +
    (lost > 0
      ? `${lost} transaction${lost === 1 ? ' was' : 's were'} acknowledged to the client and is not in this ` +
        `database — that is the bill for moving the ack ahead of the fsync.`
      : `Every transaction the client was told had committed is here. The heap blocks on disk were a mess of ` +
        `different vintages and the log put them all back on the same page of history.`)
  );
}

/* -------------------------------------------------------------- the drawing */

const KIND_GLYPH: Record<Kind, string> = { update: 'U', commit: 'C', ckpt: '◆' };

export default function WalLogBeforePageLab() {
  const [fpw, setFpw] = useState(true);
  const [sync, setSync] = useState<'on' | 'off'>('on');
  const [target, setTarget] = useState(0);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const cfg: Cfg = { fpw, sync };
  const run = (a: Act) => setS((cur) => apply(cur, a, cfg, target));
  const recovering = s.phase === 'crashed' || s.phase === 'replay' || s.phase === 'done';
  const live = s.phase === 'run';

  /* geometry */
  const stripX = 108;
  const gap = 6;
  const chipW = (r: Rec) => Math.max(52, Math.round(34 + 9 * Math.log2(r.size / 32)));
  const xs: number[] = [];
  let cx = stripX;
  for (const r of s.recs) {
    xs.push(cx);
    cx += chipW(r) + gap;
  }
  const stripEnd = cx + 10;

  const stripY = 44;
  const chipH = 30;
  const pagesY = 120;
  const rowH = 62;
  const boxW = 150;
  const height = pagesY + s.pages.length * rowH + 16;
  const svgW = Math.max(width, stripEnd + 90, stripX + 2 * boxW + 260);

  /** Where a cursor at this LSN falls along the strip. */
  const lsnX = (lsn: number) => {
    if (s.recs.length === 0) return stripX;
    for (let i = 0; i < s.recs.length; i++) {
      if (lsn <= s.recs[i].lsn) return xs[i] - gap / 2;
      if (lsn < s.recs[i].end) return xs[i] + chipW(s.recs[i]) * ((lsn - s.recs[i].lsn) / s.recs[i].size);
    }
    return xs[s.recs.length - 1] + chipW(s.recs[s.recs.length - 1]) + gap / 2;
  };

  const cursorRec = recovering && s.cursor < s.survivors.length ? s.survivors[s.cursor] : null;
  const lostAcks = s.acks.filter((a) => a.lsn > s.flushedLSN).length;

  const chipFill = (r: Rec) => {
    if (recovering && r.end > s.flushedLSN) return 'var(--viz-stale)';
    const done = s.replay.find((x) => x.rec === r.i);
    if (done) return done.decision === 'apply' ? 'var(--viz-good)' : 'var(--viz-stale)';
    return r.end <= s.flushedLSN ? 'var(--viz-clean)' : 'var(--viz-dirty)';
  };

  return (
    <VizPanel
      title="Log before page: LSNs, pageLSN, and a replay you can step"
      subtitle="Update rows, flush the log, try to write a dirty block too early, then crash and step redo. Chip width is proportional to the log(record size), so a full-page image is visibly the expensive one."
      controls={
        <>
          <Choice
            label="Target block"
            value={String(target)}
            onChange={(v) => setTarget(Number(v))}
            options={PAGE_NAMES.map((n, i) => ({ value: String(i), label: n }))}
          />
          <Check label="full_page_writes" checked={fpw} onChange={setFpw} />
          <Segmented
            label="Commit durability"
            value={sync}
            onChange={setSync}
            options={[
              { value: 'on', label: 'flush, then ack', title: 'synchronous_commit = on / innodb_flush_log_at_trx_commit = 1' },
              { value: 'off', label: 'ack, then flush', title: 'synchronous_commit = off / innodb_flush_log_at_trx_commit = 2' },
            ]}
          />
          <Button onClick={() => run('update')} disabled={!live || s.recs.length >= MAX_RECS} primary>
            UPDATE
          </Button>
          <Button onClick={() => run('commit')} disabled={!live || s.openTxn === null}>
            COMMIT
          </Button>
          <Button onClick={() => run('flush')} disabled={!live} title="XLogFlush: write() the WAL buffers, then fdatasync()">
            Flush log
          </Button>
          <Button onClick={() => run('write')} disabled={!live} title="What eviction or the checkpointer does to one dirty block">
            Write block
          </Button>
          <Button onClick={() => run('ckpt')} disabled={!live || s.recs.length >= MAX_RECS}>
            Checkpoint
          </Button>
          <Button onClick={() => run('crash')} disabled={!live}>
            Crash
          </Button>
          <Button onClick={() => run('step')} disabled={!(s.phase === 'crashed' || s.phase === 'replay')} primary={recovering}>
            Redo step
          </Button>
          <Button onClick={() => run('all')} disabled={!(s.phase === 'crashed' || s.phase === 'replay')}>
            Redo all
          </Button>
          <Button onClick={() => setS(INITIAL)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'record durable (≤ flushed LSN)', color: 'var(--viz-clean)' },
            { label: 'record in WAL buffers only', color: 'var(--viz-dirty)' },
            { label: 'applied by redo', color: 'var(--viz-good)' },
            { label: 'skipped / lost in the crash', color: 'var(--viz-stale)' },
            { label: 'U update · C commit · ◆ checkpoint · +FPI full-page image', color: 'var(--viz-ink-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Insert LSN', value: fmtLsn(s.insertLSN), hint: 'Where the next record will be written — a byte offset in pg_wal' },
            { label: 'Flushed LSN', value: fmtLsn(s.flushedLSN), hint: 'The log is durable through here; nothing above it survives a crash' },
            { label: 'Unflushed log', value: fmtBytes(unflushed(s)), hint: 'Bytes in the WAL buffers with no fsync behind them' },
            { label: 'Dirty blocks', value: `${dirtyCount(s)} / ${s.pages.length}`, hint: 'Changed in memory, not on disk — and pinned there by the WAL rule' },
            {
              label: 'Acked, not durable',
              value: fmtNum(lostAcks),
              hint: 'Transactions the client believes are committed whose commit record is not on disk',
            },
            { label: 'Writes blocked', value: fmtNum(s.blocked), hint: 'Page writes refused because the log had not caught up' },
            { label: 'Redo point', value: fmtLsn(s.redoLSN), hint: 'Recovery starts scanning here — set by the last checkpoint' },
            {
              label: 'Redo applied / skipped',
              value: `${s.replay.filter((r) => r.decision === 'apply').length} / ${s.replay.filter((r) => r.decision === 'skip').length}`,
              hint: 'Skipped means the block on disk already had a pageLSN at or beyond the record',
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
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>LSN</th>
                <th>Byte offset</th>
                <th>Bytes</th>
                <th>Kind</th>
                <th>Txn</th>
                <th>Block</th>
                <th>Change</th>
                <th>End LSN → pageLSN</th>
                <th>Durable at crash</th>
              </tr>
            </thead>
            <tbody>
              {s.recs.length === 0 ? (
                <tr>
                  <td colSpan={9}>No log records yet.</td>
                </tr>
              ) : (
                s.recs.map((r) => (
                  <tr key={r.lsn}>
                    <td>{fmtLsn(r.lsn)}</td>
                    <td>{fmtNum(r.lsn)}</td>
                    <td>{fmtNum(r.size)}</td>
                    <td>{r.kind === 'update' ? (r.fpi ? 'update + FPI' : 'update') : r.kind}</td>
                    <td>{r.txn ?? '—'}</td>
                    <td>{r.page === null ? '—' : PAGE_NAMES[r.page]}</td>
                    <td>{r.val === null ? '—' : `${r.prev} → ${r.val}`}</td>
                    <td>{fmtLsn(r.end)}</td>
                    <td>{r.end <= s.flushedLSN ? 'yes' : 'no'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Block</th>
                <th>In memory</th>
                <th>pageLSN (memory)</th>
                <th>Dirty</th>
                <th>On disk</th>
                <th>pageLSN (disk)</th>
                <th>Writable now?</th>
              </tr>
            </thead>
            <tbody>
              {s.pages.map((p) => (
                <tr key={p.name}>
                  <td>{p.name}</td>
                  <td>{p.memVal}</td>
                  <td>{p.memLSN === 0 ? '0/00000000' : fmtLsn(p.memLSN)}</td>
                  <td>{p.dirty ? 'yes' : 'no'}</td>
                  <td>{p.diskVal}</td>
                  <td>{p.diskLSN === 0 ? '0/00000000' : fmtLsn(p.diskLSN)}</td>
                  <td>{!p.dirty ? '—' : p.memLSN <= s.flushedLSN ? 'yes' : 'no — log behind pageLSN'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {s.replay.length > 0 ? (
            <table className="viz-table">
              <thead>
                <tr>
                  <th>Redo step</th>
                  <th>Record LSN</th>
                  <th>Block</th>
                  <th>pageLSN on disk</th>
                  <th>Decision</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {s.replay.map((r, i) => (
                  <tr key={`${r.rec}-${i}`}>
                    <td>{i + 1}</td>
                    <td>{fmtLsn(r.lsn)}</td>
                    <td>{r.page === null ? '—' : PAGE_NAMES[r.page]}</td>
                    <td>{r.page === null ? '—' : r.pageLSNBefore === 0 ? '0/00000000' : fmtLsn(r.pageLSNBefore)}</td>
                    <td>{r.decision === 'noop' ? 'no page' : r.decision}</td>
                    <td>{r.valBefore === null ? '—' : `${r.valBefore} → ${r.valAfter}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label="A write-ahead log strip with LSN cursors above three heap blocks and their in-memory and on-disk copies"
          >
            {/* ---------------------------------------------------- the WAL strip */}
            <text x={0} y={stripY - 14} fill="var(--viz-ink)" fontWeight={600}>
              WAL
            </text>
            <text x={0} y={stripY + 2} fill="var(--viz-ink-muted)">
              {segName(s.insertLSN)}
            </text>
            <text x={0} y={stripY + 18} fill="var(--viz-ink-muted)">
              16 MB segment
            </text>
            <line
              x1={stripX - 6}
              x2={Math.max(stripEnd, stripX + 40)}
              y1={stripY + chipH + 4}
              y2={stripY + chipH + 4}
              className="viz-axis-line"
            />

            {s.recs.map((r, i) => {
              const w = chipW(r);
              const isCursor = cursorRec !== null && cursorRec.i === r.i;
              return (
                <g
                  key={r.lsn}
                  {...tip(
                    <>
                      <strong>
                        {fmtLsn(r.lsn)} · {r.kind === 'update' ? (r.fpi ? 'update + full-page image' : 'update') : r.kind}
                      </strong>
                      <br />
                      {fmtNum(r.size)} bytes, so the next record starts at {fmtLsn(r.end)}.
                      {r.page !== null ? ` Stamps ${PAGE_NAMES[r.page]}.pageLSN = ${fmtLsn(r.end)} (${r.prev} → ${r.val}).` : ''}
                      {r.txn !== null && r.kind === 'commit' ? ` Transaction ${r.txn} is durable once the log is flushed past ${fmtLsn(r.end)}.` : ''}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect
                    x={xs[i]}
                    y={stripY - chipH + 6}
                    width={w}
                    height={chipH}
                    rx={5}
                    fill={chipFill(r)}
                    stroke={isCursor ? 'var(--viz-ink)' : 'var(--viz-surface)'}
                    strokeWidth={isCursor ? 2.5 : 1.5}
                    opacity={recovering && r.end > s.flushedLSN ? 0.45 : 1}
                  />
                  <text
                    x={xs[i] + w / 2}
                    y={stripY - chipH + 26}
                    textAnchor="middle"
                    fill="var(--viz-surface)"
                    fontWeight={700}
                  >
                    {KIND_GLYPH[r.kind]}
                    {r.fpi ? '+FPI' : ''}
                  </text>
                  <text x={xs[i] + w / 2} y={stripY + 20} textAnchor="middle" fill="var(--viz-ink-muted)">
                    {shortLsn(r.lsn)}
                  </text>
                  {r.page !== null ? (
                    <text x={xs[i] + w / 2} y={stripY + 34} textAnchor="middle" fill="var(--viz-ink-2)">
                      b{r.page}
                    </text>
                  ) : null}
                </g>
              );
            })}

            {/* ------------------------------------------------------- the cursors */}
            {(() => {
              const marks: { lsn: number; label: string; color: string; dash?: string }[] = [
                { lsn: s.redoLSN, label: 'redo point', color: 'var(--viz-7)', dash: '4 3' },
                { lsn: s.flushedLSN, label: 'flushed LSN (fsync)', color: 'var(--viz-clean)' },
                { lsn: s.insertLSN, label: 'insert LSN', color: 'var(--viz-dirty)', dash: '4 3' },
              ];
              if (s.ackLSN > BASE_LSN) {
                marks.push({ lsn: s.ackLSN, label: 'client ack', color: 'var(--viz-critical)', dash: '2 3' });
              }
              return marks.map((m, i) => {
                const x = lsnX(m.lsn);
                const yTop = stripY - chipH - 16 - (i % 2) * 14;
                return (
                  <g key={m.label}>
                    <line
                      x1={x}
                      x2={x}
                      y1={yTop + 4}
                      y2={stripY + 38}
                      stroke={m.color}
                      strokeWidth={1.5}
                      strokeDasharray={m.dash}
                    />
                    <text x={x + 4} y={yTop} fill={m.color}>
                      {m.label}
                    </text>
                  </g>
                );
              });
            })()}

            {/* -------------------------------------------------------- the blocks */}
            {s.pages.map((p, i) => {
              const y = pagesY + i * rowH;
              const memX = stripX;
              const diskX = stripX + boxW + 150;
              const writable = !p.dirty || p.memLSN <= s.flushedLSN;
              return (
                <g key={p.name}>
                  <text x={0} y={y + 20} fill="var(--viz-ink)" fontWeight={600}>
                    {p.name}
                  </text>
                  <text x={0} y={y + 36} fill="var(--viz-ink-muted)">
                    rel 16384
                  </text>

                  <g
                    {...tip(
                      <>
                        <strong>Buffer pool copy of {p.name}</strong>
                        <br />
                        {p.dirty
                          ? `Dirty: the on-disk block still says ${p.diskVal}. pageLSN ${fmtLsn(p.memLSN)} — ` +
                            `${writable ? 'the log has caught up, so this block may be written.' : 'the log has not been flushed this far, so this block is pinned in memory.'}`
                          : 'Clean: identical to the block on disk.'}
                      </>,
                    )}
                    style={{ cursor: 'help' }}
                  >
                    <rect
                      x={memX}
                      y={y}
                      width={boxW}
                      height={44}
                      rx={7}
                      fill={p.dirty ? 'var(--viz-dirty)' : 'var(--viz-plane)'}
                      stroke={p.dirty ? 'var(--viz-surface)' : 'var(--viz-border)'}
                      strokeWidth={1.5}
                      opacity={p.dirty ? 0.92 : 1}
                    />
                    <text x={memX + 10} y={y + 18} fill={p.dirty ? 'var(--viz-surface)' : 'var(--viz-ink)'} fontWeight={600}>
                      memory: bal {p.memVal} {p.dirty ? '· dirty' : ''}
                    </text>
                    <text x={memX + 10} y={y + 34} fill={p.dirty ? 'var(--viz-surface)' : 'var(--viz-ink-2)'}>
                      pageLSN {p.memLSN === 0 ? '0/00000000' : fmtLsn(p.memLSN)}
                    </text>
                  </g>

                  <line
                    x1={memX + boxW + 8}
                    x2={diskX - 8}
                    y1={y + 22}
                    y2={y + 22}
                    stroke={p.dirty ? (writable ? 'var(--viz-good)' : 'var(--viz-critical)') : 'var(--viz-axis)'}
                    strokeWidth={1.5}
                    strokeDasharray={p.dirty && !writable ? '3 3' : undefined}
                  />
                  {p.dirty ? (
                    <text
                      x={(memX + boxW + diskX) / 2}
                      y={y + 14}
                      textAnchor="middle"
                      fill={writable ? 'var(--viz-good)' : 'var(--viz-critical)'}
                    >
                      {writable ? 'may write' : 'BLOCKED'}
                    </text>
                  ) : null}

                  <g
                    {...tip(
                      <>
                        <strong>On-disk block {p.name}</strong>
                        <br />
                        Its header carries pageLSN {p.diskLSN === 0 ? '0/00000000' : fmtLsn(p.diskLSN)}. Redo applies a
                        record to this block only if the record&apos;s LSN is greater than that.
                      </>,
                    )}
                    style={{ cursor: 'help' }}
                  >
                    <rect
                      x={diskX}
                      y={y}
                      width={boxW}
                      height={44}
                      rx={7}
                      fill="var(--viz-clean)"
                      stroke="var(--viz-surface)"
                      strokeWidth={1.5}
                      opacity={0.92}
                    />
                    <text x={diskX + 10} y={y + 18} fill="var(--viz-surface)" fontWeight={600}>
                      disk: bal {p.diskVal}
                    </text>
                    <text x={diskX + 10} y={y + 34} fill="var(--viz-surface)">
                      pageLSN {p.diskLSN === 0 ? '0/00000000' : fmtLsn(p.diskLSN)}
                    </text>
                  </g>
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
