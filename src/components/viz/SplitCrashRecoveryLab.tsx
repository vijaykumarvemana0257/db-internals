import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Check,
  Button,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
} from './Viz';

/**
 * One leaf split, logged three different ways, with a draggable crash point.
 *
 * The tree: root (blk 1) with downlinks to leaf blk 2 (keys 10,20,30,40, high key 60)
 * and leaf blk 5 (keys 60,70,80). Inserting 25 overflows blk 2, so it splits: blk 2
 * keeps 10,20,25 with high key 30 and a right link to the brand-new blk 4, which takes
 * 30,40 and inherits the old high key 60 and the old right link to blk 5. Three pages
 * change, plus the parent, and every one of those changes is somebody else's data.
 *
 * The learner chooses how the engine logs that — PostgreSQL nbtree's one multi-page
 * split record with a deferred downlink, InnoDB's mini-transaction group, or a naive
 * page-at-a-time log — then crashes at any point and restarts. The state that comes back
 * is computed, not scripted: which effects are durable decides which searches still work.
 */

/* ------------------------------------------------------------------- the tree */

const BLK_L = 2;
const BLK_R = 4;
const BLK_SIB = 5;

const PRE_LEFT = [10, 20, 30, 40];
const NEW_KEY = 25;
const AFTER_LEFT = [10, 20, 25];
const MOVED = [30, 40];
const SIB_KEYS = [60, 70, 80];
const SEP = 30; // the new separator: blk 2's new high key, and blk 4's downlink key
const OLD_HIGH = 60; // blk 2's high key before the split; blk 4 inherits it

/* --------------------------------------------------------- the durable effects */

type Eff = {
  left: boolean; // blk 2 rewritten: 10,20,25 + high key 30 + right link to blk 4
  right: boolean; // blk 4 initialized: 30,40 + high key 60 + right link to blk 5
  sib: boolean; // blk 5's btpo_prev repointed at blk 4
  down: boolean; // the (30 -> blk 4) downlink is in the root
  flag: boolean; // blk 2 carries BTP_INCOMPLETE_SPLIT until the downlink lands
  commit: boolean; // the inserting transaction's commit record is durable
};

const NONE: Eff = { left: false, right: false, sib: false, down: false, flag: false, commit: false };

type StepKind = 'buffer' | 'wal' | 'flush' | 'commit' | 'checkpoint';

type StepDef = {
  code: string;
  label: string;
  detail: string;
  kind: StepKind;
  eff: Eff;
};

type EngineId = 'nbtree' | 'innodb' | 'naive';

const SPLIT: Eff = { ...NONE, left: true, right: true, sib: true, flag: true };
const SPLIT_DOWN: Eff = { ...SPLIT, down: true, flag: false };

const ENGINES: Record<
  EngineId,
  { label: string; blurb: string; steps: StepDef[] }
> = {
  nbtree: {
    label: 'PostgreSQL nbtree',
    blurb:
      'One multi-page split record, then a separate record for the parent downlink. The gap between them is legal, flagged and repairable.',
    steps: [
      {
        code: '_bt_split',
        label: '_bt_split() forms both halves',
        kind: 'buffer',
        detail:
          'A new block is taken from the free space map, the right half is built in a scratch page, blk 2 is rewritten in shared buffers, and all three buffers are marked dirty inside one critical section. Nothing has been logged yet, and nothing may be written to the data files yet either.',
        eff: NONE,
      },
      {
        code: 'SPLIT_R',
        label: 'XLogInsert(XLOG_BTREE_SPLIT_R)',
        kind: 'wal',
        detail:
          'ONE record with three registered blocks: block 0 is blk 2 (rebuilt at redo from the pre-split image plus the logged split point), block 1 is blk 4 (logged in full, REGBUF_WILL_INIT — there is no prior image to rebuild from), block 2 is blk 5, whose btpo_prev has to move. The record is in the WAL buffer; a crash here still loses it.',
        eff: NONE,
      },
      {
        code: 'flush',
        label: 'XLogFlush through the split LSN',
        kind: 'flush',
        detail:
          'The split is now durable, all three pages together — redo cannot apply half of one record. The parent still has no downlink to blk 4, and blk 2 is on disk carrying BTP_INCOMPLETE_SPLIT to say so.',
        eff: SPLIT,
      },
      {
        code: 'INSERT_UPPER',
        label: 'XLogInsert(XLOG_BTREE_INSERT_UPPER)',
        kind: 'wal',
        detail:
          '_bt_insert_parent() walks back up the BTStack it recorded on the way down and inserts the (30 -> blk 4) downlink into the root. The same record registers blk 2 as a second block so that redo clears BTP_INCOMPLETE_SPLIT there too. Still only in the WAL buffer.',
        eff: SPLIT,
      },
      {
        code: 'flush',
        label: 'XLogFlush through the downlink LSN',
        kind: 'flush',
        detail: 'The tree is now a textbook B+tree again: two hops to every key, flag cleared.',
        eff: SPLIT_DOWN,
      },
      {
        code: 'COMMIT',
        label: 'commit record written and flushed',
        kind: 'commit',
        detail:
          'Only now does the client hear "COMMIT". Everything before this point was structure that other transactions were already using.',
        eff: { ...SPLIT_DOWN, commit: true },
      },
      {
        code: 'ckpt',
        label: 'checkpoint writes the dirty pages',
        kind: 'checkpoint',
        detail:
          'The three pages reach their data files, and the redo pointer may advance past the split record. Note the ordering rule: no page could be written before its own LSN was flushed.',
        eff: { ...SPLIT_DOWN, commit: true },
      },
    ],
  },
  innodb: {
    label: 'InnoDB mini-transaction',
    blurb:
      'The split and the parent downlink are one mini-transaction. Its redo records reach the log as an indivisible group, and there is no undo for any of them.',
    steps: [
      {
        code: 'mtr_start',
        label: 'mtr_start(); btr_page_split_and_insert()',
        kind: 'buffer',
        detail:
          'The split, the sibling fix-up and the parent downlink insertion all run inside one mini-transaction. Every redo record it generates goes into the mtr’s own private buffer, not the global log buffer, so none of it is visible to recovery yet.',
        eff: NONE,
      },
      {
        code: 'mtr_commit',
        label: 'mtr_commit(): group copied to the log buffer',
        kind: 'wal',
        detail:
          'The whole group is appended to the global log buffer at once, terminated by MLOG_MULTI_REC_END. Recovery applies a multi-record group only when it has found that terminator, so a group truncated by a crash is discarded whole.',
        eff: NONE,
      },
      {
        code: 'flush',
        label: 'log flushed through the group’s end LSN',
        kind: 'flush',
        detail:
          'All four page changes become durable at the same instant. There is no intermediate state to flag, because the parent downlink was never a separate action.',
        eff: SPLIT_DOWN,
      },
      {
        code: 'COMMIT',
        label: 'trx commit record flushed',
        kind: 'commit',
        detail:
          'With innodb_flush_log_at_trx_commit = 1 this is the fsync the client waits on. The split was durable one step earlier and belonged to the index, not to this transaction.',
        eff: { ...SPLIT_DOWN, commit: true },
      },
      {
        code: 'ckpt',
        label: 'page cleaner flushes the pages',
        kind: 'checkpoint',
        detail:
          'Dirty pages leave the buffer pool for the tablespace; the checkpoint LSN advances past the mini-transaction.',
        eff: { ...SPLIT_DOWN, commit: true },
      },
    ],
  },
  naive: {
    label: 'One record per page (broken)',
    blurb:
      'What you get if each modified page is logged and flushed on its own. Every gap between records is a distinct on-disk corruption.',
    steps: [
      {
        code: 'log blk 2',
        label: 'rewrite blk 2, flush',
        kind: 'flush',
        detail:
          'blk 2 now holds 10,20,25 with high key 30 and a right link to blk 4 — a block that has never been initialized. The keys 30 and 40 are in the heap and in no index page at all.',
        eff: { ...NONE, left: true },
      },
      {
        code: 'log blk 4',
        label: 'initialize blk 4, flush',
        kind: 'flush',
        detail:
          'The right half exists and the forward chain is whole again. blk 5 still believes its left neighbour is blk 2, so a backward scan hops straight over blk 4.',
        eff: { ...NONE, left: true, right: true },
      },
      {
        code: 'log blk 5',
        label: 'repoint blk 5’s btpo_prev, flush',
        kind: 'flush',
        detail:
          'Both sibling chains are consistent. The root still has no downlink to blk 4 — and nothing on any page records that fact, so nothing will ever repair it.',
        eff: { ...NONE, left: true, right: true, sib: true },
      },
      {
        code: 'log root',
        label: 'insert the downlink into the root, flush',
        kind: 'flush',
        detail: 'The structure is finally correct, by luck rather than by design.',
        eff: SPLIT_DOWN,
      },
      {
        code: 'COMMIT',
        label: 'commit record flushed',
        kind: 'commit',
        detail: 'The client hears COMMIT.',
        eff: { ...SPLIT_DOWN, commit: true },
      },
    ],
  },
};

const ENGINE_OPTS = [
  { value: 'nbtree' as const, label: 'nbtree', title: 'PostgreSQL: one split record, deferred downlink' },
  { value: 'innodb' as const, label: 'mini-txn', title: 'InnoDB: the whole SMO in one mini-transaction' },
  { value: 'naive' as const, label: 'per page', title: 'One WAL record per modified page — the broken baseline' },
];

/* ------------------------------------------------------------ recovered state */

type Leaf = {
  blk: number;
  exists: boolean;
  keys: number[];
  high: number | null;
  next: number | null;
  prev: number | null;
  flag: boolean;
};

type Tree = {
  root: { sep: number | null; blk: number }[];
  leaves: Leaf[]; // always [blk2, blk4, blk5]
  committed: boolean;
  newKeyPresent: boolean;
};

function build(eff: Eff, extraDown: boolean): Tree {
  const down = eff.down || extraDown;
  const root: { sep: number | null; blk: number }[] = [{ sep: null, blk: BLK_L }];
  if (down) root.push({ sep: SEP, blk: BLK_R });
  root.push({ sep: OLD_HIGH, blk: BLK_SIB });

  const left: Leaf = {
    blk: BLK_L,
    exists: true,
    keys: eff.left ? AFTER_LEFT : PRE_LEFT,
    high: eff.left ? SEP : OLD_HIGH,
    next: eff.left ? BLK_R : BLK_SIB,
    prev: null,
    flag: eff.flag && !down,
  };
  const right: Leaf = {
    blk: BLK_R,
    exists: eff.right,
    keys: MOVED,
    high: OLD_HIGH,
    next: BLK_SIB,
    prev: BLK_L,
    flag: false,
  };
  const sib: Leaf = {
    blk: BLK_SIB,
    exists: true,
    keys: SIB_KEYS,
    high: null,
    next: null,
    prev: eff.sib ? BLK_R : BLK_L,
    flag: false,
  };
  return { root, leaves: [left, right, sib], committed: eff.commit, newKeyPresent: eff.left };
}

type Verdict = 'none' | 'dangling' | 'orphan' | 'backlink' | 'incomplete' | 'silent' | 'clean';

const VERDICT_TEXT: Record<Verdict, { short: string; tone: 'good' | 'warn' | 'bad'; long: string }> = {
  none: {
    short: 'pre-split, intact',
    tone: 'good',
    long:
      'Redo found nothing to apply: the split never became durable, so the tree is exactly the tree that existed before the insert. The transaction had not committed, so nobody was promised otherwise.',
  },
  dangling: {
    short: 'right link into a void',
    tone: 'bad',
    long:
      'blk 2 was rewritten and blk 4 was not. The keys 30 and 40 now live in no index page at all, and blk 2’s right link points at a block that was never initialized. This is corruption: the heap has rows the index cannot find.',
  },
  orphan: {
    short: 'orphaned page',
    tone: 'warn',
    long:
      'blk 4 exists with a copy of 30 and 40, but nothing points at it and blk 2 still holds those keys. Searches are correct; the page is leaked until something reclaims it.',
  },
  backlink: {
    short: 'stale backward link',
    tone: 'warn',
    long:
      'Forward search and forward scans are correct, but blk 5’s btpo_prev still names blk 2, so a backward scan jumps over blk 4 and silently skips 30 and 40. This is why the sibling’s pointer has to be in the same atomic unit as the split.',
  },
  incomplete: {
    short: 'incomplete split (flagged)',
    tone: 'warn',
    long:
      'The split is durable and the parent has no downlink to blk 4 — a legal, documented state. blk 2 carries BTP_INCOMPLETE_SPLIT, so every descent knows the tree is mid-modification, and the next inserter through blk 2 is obliged to finish the job.',
  },
  silent: {
    short: 'missing downlink, unrecorded',
    tone: 'bad',
    long:
      'The same shape as an incomplete split, with nothing on any page saying so. Searches work by right link, forever, one extra page read at a time, and no descent will ever repair it because no descent can tell there is anything to repair.',
  },
  clean: {
    short: 'consistent',
    tone: 'good',
    long: 'Every page change is durable. Two hops from the root to any key; both sibling chains agree.',
  },
};

function verdictOf(eff: Eff, extraDown: boolean): Verdict {
  if (!eff.left && !eff.right) return 'none';
  if (eff.left && !eff.right) return 'dangling';
  if (!eff.left && eff.right) return 'orphan';
  if (!eff.sib) return 'backlink';
  if (!(eff.down || extraDown)) return eff.flag ? 'incomplete' : 'silent';
  return 'clean';
}

/* ------------------------------------------------------------------- descent */

type Hop = { blk: number; why: string };
type Descent = {
  hops: Hop[];
  result: 'found' | 'missing' | 'deadend';
  repaired: boolean;
  key: number;
  mode: 'read' | 'insert';
  text: string;
};

function descend(
  tree: Tree,
  key: number,
  followRight: boolean,
  mode: 'read' | 'insert',
  repairOn: boolean,
): Descent {
  const hops: Hop[] = [];
  let repaired = false;

  // Root: take the last downlink whose separator is <= the search key.
  let chosen = tree.root[0];
  for (const d of tree.root) if (d.sep !== null && key >= d.sep) chosen = d;
  hops.push({ blk: 1, why: `separator scan picks the downlink for blk ${chosen.blk}` });

  let blk: number | null = chosen.blk;
  for (let guard = 0; guard < 4 && blk !== null; guard++) {
    const page = tree.leaves.find((l) => l.blk === blk);
    if (!page || !page.exists) {
      hops.push({ blk: blk, why: 'page was never initialized — dead end' });
      return {
        hops,
        result: 'deadend',
        repaired,
        key,
        mode,
        text: `The descent followed a pointer to blk ${blk} and found a block that no WAL record ever initialized. PostgreSQL reports this as "index ... contains unexpected zero page at block ${blk}"; the row is in the heap and a sequential scan finds it, so the index and the table now disagree.`,
      };
    }
    hops.push({ blk: page.blk, why: `binary search within blk ${page.blk}` });

    if (mode === 'insert' && page.flag && repairOn) {
      repaired = true;
    }

    if (page.high !== null && key >= page.high) {
      if (!followRight || page.next === null) {
        return {
          hops,
          result: 'missing',
          repaired,
          key,
          mode,
          text: `Key ${key} is greater than blk ${page.blk}’s high key (${page.high}), so it is not on this page and the descent has nowhere else to go. Without the rule "if the key exceeds the high key, follow the right link", the half-finished tree is indistinguishable from a lost row.`,
        };
      }
      blk = page.next;
      continue;
    }

    const found = page.keys.includes(key);
    return {
      hops,
      result: found ? 'found' : 'missing',
      repaired,
      key,
      mode,
      text: found
        ? `Found key ${key} on blk ${page.blk} after ${hops.length} page reads${
            hops.length > 2 ? ' — one more than a healthy tree, because the last hop was a right link rather than a downlink' : ''
          }.${
            repaired
              ? ' On the way down this descent saw BTP_INCOMPLETE_SPLIT on blk 2 and called _bt_finish_split(): it re-found the parent, inserted the (30 -> blk 4) downlink and cleared the flag. The repair is charged to whoever happens to pass through.'
              : ''
          }`
        : `Key ${key} is not in this index.`,
    };
  }
  return { hops, result: 'missing', repaired, key, mode, text: 'The right-link chain did not terminate.' };
}

/* ------------------------------------------------------------------- drawing */

const STEP_FILL: Record<StepKind, string> = {
  buffer: 'var(--viz-dirty)',
  wal: 'var(--viz-dirty)',
  flush: 'var(--viz-clean)',
  commit: 'var(--viz-clean)',
  checkpoint: 'var(--viz-clean)',
};

function Arrow({
  x1,
  y1,
  x2,
  y2,
  color,
  dashed,
  width = 1.4,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  dashed?: boolean;
  width?: number;
}) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.max(1, Math.hypot(dx, dy));
  const ux = dx / len;
  const uy = dy / len;
  const hx = x2 - ux * 7;
  const hy = y2 - uy * 7;
  const px = -uy * 3.5;
  const py = ux * 3.5;
  return (
    <g>
      <line
        x1={x1}
        y1={y1}
        x2={hx}
        y2={hy}
        stroke={color}
        strokeWidth={width}
        strokeDasharray={dashed ? '4 3' : undefined}
      />
      <polygon points={`${x2},${y2} ${hx + px},${hy + py} ${hx - px},${hy - py}`} fill={color} />
    </g>
  );
}

/* ----------------------------------------------------------------- component */

export default function SplitCrashRecoveryLab() {
  const [engine, setEngine] = useState<EngineId>('nbtree');
  const [crashAt, setCrashAt] = useState(3);
  const [followRight, setFollowRight] = useState(true);
  const [repairOn, setRepairOn] = useState(true);
  const [searchKey, setSearchKey] = useState('40');
  const [mode, setMode] = useState<'read' | 'insert'>('read');
  const [run, setRun] = useState<Descent | null>(null);
  const [extraDown, setExtraDown] = useState(false);
  const [rolledBack, setRolledBack] = useState(false);
  const [ref, width] = useSize(780);
  const tip = useTip();

  const eng = ENGINES[engine];
  const n = eng.steps.length;
  const at = Math.min(crashAt, n);
  const eff = at === 0 ? NONE : eng.steps[at - 1].eff;
  const tree = build(eff, extraDown);
  const v = verdictOf(eff, extraDown);
  const crashed = at < n;

  const reset = (fn: () => void) => {
    fn();
    setRun(null);
    setExtraDown(false);
    setRolledBack(false);
  };

  const doDescend = () => {
    const d = descend(tree, Number(searchKey), followRight, mode, repairOn);
    setRun(d);
    if (d.repaired) setExtraDown(true);
  };

  /* geometry */
  const svgW = Math.max(width, 720);
  const stripY = 10;
  const stripH = 34;
  const chipGap = 8;
  const chipW = (svgW - 16 - chipGap * (n - 1)) / n;
  const rootY = 96;
  const rootH = 42;
  const rootW = Math.min(300, svgW - 120);
  const rootX = svgW / 2 - rootW / 2;
  const leafY = 208;
  const leafH = 70;
  const leafW = Math.min(196, (svgW - 60) / 3);
  const slotX = (i: number) => 16 + i * ((svgW - 32 - leafW) / 2);
  const height = 392;

  const pathBlks = run ? run.hops.map((h) => h.blk) : [];
  const keyVisible = tree.newKeyPresent && tree.committed && !rolledBack;

  const leafBox = (l: Leaf, i: number) => {
    const x = slotX(i);
    const missing = !l.exists;
    const inPath = pathBlks.includes(l.blk);
    const hopNo = pathBlks.indexOf(l.blk) + 1;
    return (
      <g
        key={l.blk}
        {...tip(
          missing ? (
            <>
              <strong>blk {l.blk} — uninitialized</strong>
              <br />
              No WAL record ever created this page, yet blk 2 points at it.
            </>
          ) : (
            <>
              <strong>blk {l.blk}</strong>
              <br />
              keys: {l.keys.join(', ')}
              <br />
              high key: {l.high === null ? '— (rightmost page)' : l.high}
              <br />
              btpo_next: {l.next ?? '— (P_NONE)'} · btpo_prev: {l.prev ?? '—'}
              <br />
              {l.flag ? 'btpo_flags: BTP_INCOMPLETE_SPLIT' : 'btpo_flags: —'}
            </>
          ),
        )}
        style={{ cursor: 'help' }}
      >
        <rect
          x={x}
          y={leafY}
          width={leafW}
          height={leafH}
          rx={8}
          fill={missing ? 'var(--viz-plane)' : 'var(--viz-surface)'}
          stroke={
            missing
              ? 'var(--viz-critical)'
              : l.flag
                ? 'var(--viz-warning)'
                : inPath
                  ? 'var(--viz-7)'
                  : 'var(--viz-border)'
          }
          strokeWidth={missing || l.flag || inPath ? 2 : 1}
          strokeDasharray={missing ? '5 4' : undefined}
        />
        <text x={x + 10} y={leafY + 17} fill="var(--viz-ink)" fontWeight={600}>
          blk {l.blk}
        </text>
        {missing ? (
          <text x={x + 10} y={leafY + 38} fill="var(--viz-critical)">
            never initialized
          </text>
        ) : (
          <>
            <text x={x + 10} y={leafY + 38} fontSize={13}>
              {l.keys.map((k, ki) => {
                const dead = k === NEW_KEY && tree.newKeyPresent && !keyVisible;
                return (
                  <tspan key={k} dx={ki ? 10 : 0} fill={dead ? 'var(--viz-stale)' : 'var(--viz-ink)'}>
                    {dead ? `${k}†` : k}
                  </tspan>
                );
              })}
            </text>
            <text x={x + 10} y={leafY + 56}>
              high key {l.high === null ? '∞' : l.high}
            </text>
            <text x={x + leafW - 10} y={leafY + 17} textAnchor="end">
              {l.flag ? 'INCOMPLETE_SPLIT' : ''}
            </text>
          </>
        )}
        {inPath ? (
          <>
            <circle cx={x + leafW - 14} cy={leafY + leafH - 14} r={9} fill="var(--viz-7)" />
            <text
              x={x + leafW - 14}
              y={leafY + leafH - 10}
              textAnchor="middle"
              fill="var(--viz-surface)"
              fontWeight={700}
            >
              {hopNo}
            </text>
          </>
        ) : null}
      </g>
    );
  };

  const rootCellW = rootW / tree.root.length;

  const stat = (t: 'good' | 'warn' | 'bad') =>
    t === 'good' ? 'var(--viz-good)' : t === 'warn' ? 'var(--viz-warning)' : 'var(--viz-critical)';

  return (
    <VizPanel
      title="One split, three ways to log it, and a crash anywhere you like"
      subtitle={`Inserting 25 overflows leaf blk 2. Drag the crash point through the log sequence, restart, and descend: the tree that comes back is whatever set of page changes happened to be durable. ${eng.label}: ${eng.blurb}`}
      controls={
        <>
          <Segmented
            label="Logging model"
            value={engine}
            onChange={(e) => reset(() => { setEngine(e); setCrashAt(Math.min(crashAt, ENGINES[e].steps.length)); })}
            options={ENGINE_OPTS}
          />
          <Slider
            label="Crash after"
            min={0}
            max={n}
            value={at}
            onChange={(x) => reset(() => setCrashAt(x))}
            format={(x) => (x === n ? 'no crash' : x === 0 ? 'nothing' : `step ${x}`)}
          />
          <Choice
            label="Descend for key"
            value={searchKey}
            onChange={(k) => { setSearchKey(k); setRun(null); }}
            options={[
              { value: '25', label: '25 — the newly inserted key' },
              { value: '30', label: '30 — first key of the right half' },
              { value: '40', label: '40 — last key of the right half' },
              { value: '70', label: '70 — untouched neighbour page' },
            ]}
          />
          <Segmented
            label="Descent"
            value={mode}
            onChange={(m) => { setMode(m); setRun(null); }}
            options={[
              { value: 'read' as const, label: 'read', title: 'A plain search: never repairs anything' },
              { value: 'insert' as const, label: 'insert', title: '_bt_moveright(forupdate=true): obliged to finish an incomplete split' },
            ]}
          />
          <Button onClick={doDescend} primary>
            Descend
          </Button>
          <Check label="Reader follows the right link" checked={followRight} onChange={(b) => { setFollowRight(b); setRun(null); }} />
          <Check label="_bt_finish_split repair" checked={repairOn} onChange={(b) => { setRepairOn(b); setRun(null); }} />
          <Button
            onClick={() => setRolledBack(true)}
            disabled={!tree.newKeyPresent || rolledBack}
            title="ROLLBACK the transaction that inserted key 25"
          >
            Roll back the inserter
          </Button>
          <Button onClick={() => reset(() => {})}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'durable — survived the restart', color: 'var(--viz-clean)' },
            { label: 'in memory only — lost by the crash', color: 'var(--viz-dirty)' },
            { label: 'BTP_INCOMPLETE_SPLIT', color: 'var(--viz-warning)' },
            { label: 'corrupt / uninitialized', color: 'var(--viz-critical)' },
            { label: 'descent path (numbered)', color: 'var(--viz-7)' },
            { label: '† present but dead', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Durable log steps', value: `${at} / ${n}`, hint: 'Everything after the crash point was in DRAM and is gone' },
            {
              label: 'Index after restart',
              value: <span style={{ color: stat(VERDICT_TEXT[v].tone) }}>{VERDICT_TEXT[v].short}</span>,
            },
            {
              label: `Key ${NEW_KEY}`,
              value: !tree.newKeyPresent
                ? 'never inserted'
                : rolledBack
                  ? 'dead entry'
                  : tree.committed
                    ? 'visible'
                    : 'dead entry',
              hint: 'An uncommitted or rolled-back index tuple is still on the page; only VACUUM or purge reclaims it',
            },
            {
              label: 'Descent result',
              value: run
                ? run.result === 'found'
                  ? 'found'
                  : run.result === 'deadend'
                    ? <span style={{ color: 'var(--viz-critical)' }}>dead end</span>
                    : <span style={{ color: 'var(--viz-critical)' }}>not found</span>
                : '—',
            },
            { label: 'Page reads', value: run ? run.hops.length : '—', hint: 'Root plus every leaf touched, right-link hops included' },
            { label: 'Split undone by rollback', value: 'never', hint: 'A structure modification is not the aborting transaction’s data to undo' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {crashed ? `Crash after step ${at} of ${n}, then restart: ` : 'No crash: '}
            {VERDICT_TEXT[v].short}.
          </strong>{' '}
          {at > 0 && crashed ? `${eng.steps[at - 1].detail} ` : at === 0 && crashed ? 'The crash landed before the engine had logged anything. ' : ''}
          {VERDICT_TEXT[v].long}
          {run ? ` — ${run.text}` : ''}
          {rolledBack
            ? engine === 'innodb'
              ? ' The rollback applied the insert’s undo record and removed the index entry; the split, the sibling links and the downlink are all still there, because no undo record was ever written for them.'
              : ' The rollback wrote no index undo at all: the tuple for key 25 simply stops being visible and waits for VACUUM. The split stays.'
            : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Crash after</th>
              <th>Log step</th>
              <th>Durable page changes</th>
              <th>Index state</th>
              <th>Search for 40</th>
              <th>Backward scan</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: n + 1 }, (_, i) => {
              const e = i === 0 ? NONE : eng.steps[i - 1].eff;
              const t = build(e, false);
              const vv = verdictOf(e, false);
              const d = descend(t, 40, true, 'read', false);
              const parts = [
                e.left ? 'blk 2' : null,
                e.right ? 'blk 4' : null,
                e.sib ? 'blk 5' : null,
                e.down ? 'root' : null,
              ].filter(Boolean);
              return (
                <tr key={i}>
                  <td>{i === 0 ? 'nothing' : `step ${i}`}</td>
                  <td>{i === 0 ? '—' : eng.steps[i - 1].code}</td>
                  <td>{parts.length ? parts.join(', ') : 'none'}</td>
                  <td>{VERDICT_TEXT[vv].short}</td>
                  <td>
                    {d.result === 'found' ? `found, ${d.hops.length} reads` : d.result === 'deadend' ? 'dead end' : 'not found'}
                  </td>
                  <td>{vv === 'backlink' ? 'skips blk 4' : vv === 'dangling' ? 'misses 30, 40' : 'correct'}</td>
                </tr>
              );
            })}
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
            aria-label="A B+tree leaf split logged as a sequence of WAL steps, with a crash point and the recovered tree below it"
          >
            {/* the log strip */}
            {eng.steps.map((s, i) => {
              const x = 8 + i * (chipW + chipGap);
              const lost = i >= at;
              return (
                <g
                  key={i}
                  {...tip(
                    <>
                      <strong>
                        {i + 1}. {s.label}
                      </strong>
                      <br />
                      {s.detail}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect
                    x={x}
                    y={stripY}
                    width={chipW}
                    height={stripH}
                    rx={6}
                    fill={lost ? 'var(--viz-neutral)' : STEP_FILL[s.kind]}
                    stroke={lost ? 'var(--viz-stale)' : 'var(--viz-surface)'}
                    strokeWidth={1.5}
                    strokeDasharray={lost ? '4 3' : undefined}
                    opacity={lost ? 0.7 : 1}
                  />
                  <text
                    x={x + chipW / 2}
                    y={stripY + 21}
                    textAnchor="middle"
                    fill={lost ? 'var(--viz-ink-muted)' : 'var(--viz-surface)'}
                    fontWeight={600}
                  >
                    {i + 1}. {s.code}
                  </text>
                  <text x={x + chipW / 2} y={stripY + stripH + 14} textAnchor="middle">
                    {s.kind === 'flush' || s.kind === 'commit' ? 'flushed' : s.kind === 'wal' ? 'log buffer' : s.kind}
                  </text>
                </g>
              );
            })}
            {crashed ? (
              <g>
                <line
                  x1={8 + at * (chipW + chipGap) - chipGap / 2}
                  x2={8 + at * (chipW + chipGap) - chipGap / 2}
                  y1={stripY - 6}
                  y2={stripY + stripH + 20}
                  stroke="var(--viz-critical)"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                />
                <text
                  x={8 + at * (chipW + chipGap) - chipGap / 2 + 4}
                  y={stripY - 10}
                  fill="var(--viz-critical)"
                  fontWeight={600}
                >
                  crash
                </text>
              </g>
            ) : null}

            {/* the root */}
            <rect
              x={rootX}
              y={rootY}
              width={rootW}
              height={rootH}
              rx={8}
              fill="var(--viz-surface)"
              stroke={pathBlks.includes(1) ? 'var(--viz-7)' : 'var(--viz-border)'}
              strokeWidth={pathBlks.includes(1) ? 2 : 1}
            />
            <text x={rootX} y={rootY - 8} fill="var(--viz-ink)" fontWeight={600}>
              blk 1 — root
            </text>
            {tree.root.map((d, i) => (
              <g key={d.blk}>
                {i > 0 ? (
                  <line
                    x1={rootX + i * rootCellW}
                    x2={rootX + i * rootCellW}
                    y1={rootY}
                    y2={rootY + rootH}
                    stroke="var(--viz-grid)"
                  />
                ) : null}
                <text x={rootX + i * rootCellW + rootCellW / 2} y={rootY + 26} textAnchor="middle" fill="var(--viz-ink)">
                  {d.sep === null ? '−∞' : `≥ ${d.sep}`} → blk {d.blk}
                </text>
              </g>
            ))}
            {tree.root.map((d, i) => {
              const li = d.blk === BLK_L ? 0 : d.blk === BLK_R ? 1 : 2;
              return (
                <Arrow
                  key={`dl${d.blk}`}
                  x1={rootX + i * rootCellW + rootCellW / 2}
                  y1={rootY + rootH}
                  x2={slotX(li) + leafW / 2}
                  y2={leafY - 3}
                  color="var(--viz-ink-2)"
                />
              );
            })}
            {!tree.root.some((d) => d.blk === BLK_R) && tree.leaves[1].exists ? (
              <text x={slotX(1) + leafW / 2} y={leafY - 12} textAnchor="middle" fill="var(--viz-warning)">
                no downlink
              </text>
            ) : null}

            {/* the leaves */}
            {tree.leaves.map((l, i) => leafBox(l, i))}

            {/* sibling chains */}
            {tree.leaves.map((l, i) => {
              if (l.next === null) return null;
              const ti = l.next === BLK_R ? 1 : 2;
              return (
                <g key={`nx${l.blk}`}>
                  <Arrow
                    x1={slotX(i) + leafW}
                    y1={leafY + 26}
                    x2={slotX(ti) - 2}
                    y2={leafY + 26}
                    color={!tree.leaves[ti].exists ? 'var(--viz-critical)' : 'var(--viz-ink-2)'}
                    dashed={!tree.leaves[ti].exists}
                  />
                  <text x={(slotX(i) + leafW + slotX(ti)) / 2} y={leafY + 20} textAnchor="middle">
                    btpo_next
                  </text>
                </g>
              );
            })}
            {tree.leaves.map((l, i) => {
              if (l.prev === null) return null;
              const ti = l.prev === BLK_L ? 0 : 1;
              const stale = l.blk === BLK_SIB && l.prev === BLK_L && tree.leaves[1].exists;
              const y = leafY + leafH + (stale ? 42 : 14);
              return (
                <g key={`pv${l.blk}`}>
                  <Arrow
                    x1={slotX(i) - 2}
                    y1={y}
                    x2={slotX(ti) + leafW}
                    y2={y}
                    color={stale ? 'var(--viz-critical)' : 'var(--viz-ink-2)'}
                    dashed={stale}
                  />
                  <text x={(slotX(i) + slotX(ti) + leafW) / 2} y={y + 14} textAnchor="middle" fill={stale ? 'var(--viz-critical)' : undefined}>
                    btpo_prev{stale ? ' — stale, skips blk 4' : ''}
                  </text>
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
