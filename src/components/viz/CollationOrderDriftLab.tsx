import { useMemo, useState } from 'react';
import {
  VizPanel,
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
import {
  COMPARATORS,
  COMPARATOR_ORDER,
  FIXTURE_PROVENANCE,
  KEYS,
  compare,
  keyOf,
  memcmpBytes,
  weightCells,
  type ComparatorId,
} from './collationOrderFixture';

/**
 * One unique btree index on a display_name column, built by one comparator and
 * searched by another.
 *
 * Everything here is real mechanism: the leaves are laid out in the order the
 * build comparator produced, the separators are copy-ups of each child's first
 * key, and the probe runs an actual binary search at every level using the
 * *runtime* comparator. Nothing about the tree is edited when the comparator
 * changes — that is the whole point. The bytes on disk are still sorted the old
 * way, and a perfectly correct binary search walks into the wrong subtree.
 *
 * Orderings come from src/components/viz/collationOrderFixture.ts, which records
 * where each column came from.
 */

const COMPARATOR_OPTIONS = COMPARATORS.map((c) => ({ value: c.id, label: c.label }));
const INDEX_NAME = 'users_display_name_key';

/* ------------------------------------------------------------------- tree */

type TNode = {
  level: number; // 0 = leaf
  keys: string[]; // leaf: the entries. internal: the separators
  children: number[]; // indexes into the level below
  low: string; // the smallest key in this subtree — what a parent copies up
};

type Tree = { levels: TNode[][]; order: string[]; rejected: string[] };

/** Bulk-load: sort by the build comparator, chunk into leaves, copy up first keys. */
function buildTree(builtWith: ComparatorId, cap: number, desc: boolean): Tree {
  const sorted = [...KEYS.map((k) => k.s)].sort((a, b) => {
    const c = compare(builtWith, a, b, desc);
    return c !== 0 ? c : memcmpBytes(keyOf(a).hex, keyOf(b).hex);
  });

  // A UNIQUE index cannot hold two keys the comparator calls equal.
  const order: string[] = [];
  const rejected: string[] = [];
  for (const s of sorted) {
    if (order.length > 0 && compare(builtWith, order[order.length - 1], s, desc) === 0) rejected.push(s);
    else order.push(s);
  }

  const leaves: TNode[] = [];
  for (let i = 0; i < order.length; i += cap) {
    const keys = order.slice(i, i + cap);
    leaves.push({ level: 0, keys, children: [], low: keys[0] });
  }
  const levels: TNode[][] = [leaves];
  while (levels[levels.length - 1].length > 1) {
    const below = levels[levels.length - 1];
    const up: TNode[] = [];
    for (let i = 0; i < below.length; i += cap) {
      const kids = below.slice(i, i + cap);
      up.push({
        level: levels.length,
        // A separator is the low key of the subtree it guards, copied up.
        keys: kids.slice(1).map((k) => k.low),
        children: kids.map((_, j) => i + j),
        low: kids[0].low,
      });
    }
    levels.push(up);
  }
  return { levels, order, rejected };
}

/* --------------------------------------------------------------- descent */

type Step = { level: number; node: number; probes: { key: string; res: number }[]; chose: number };

type Descent = {
  steps: Step[];
  leaf: number; // physical leaf the search landed in
  pos: number; // insertion point inside that leaf
  hit: string | null; // the entry the comparator called equal, if any
  comparisons: number;
};

/** Binary search for the first entry >= probe. Exactly what nbtsearch.c does per page. */
function bsearch(entries: string[], probe: string, cmpFn: (a: string, b: string) => number) {
  let lo = 0;
  let hi = entries.length;
  const probes: { key: string; res: number }[] = [];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    const res = cmpFn(probe, entries[mid]);
    probes.push({ key: entries[mid], res });
    if (res > 0) lo = mid + 1;
    else hi = mid;
  }
  return { idx: lo, probes };
}

function descend(tree: Tree, leaves: string[][], probe: string, runtime: ComparatorId, desc: boolean): Descent {
  const cmpFn = (a: string, b: string) => compare(runtime, a, b, desc);
  const steps: Step[] = [];
  let comparisons = 0;
  let level = tree.levels.length - 1;
  let node = 0;

  while (level > 0) {
    const n = tree.levels[level][node];
    // An internal node sends the probe right of separator i when probe >= separator i.
    let lo = 0;
    let hi = n.keys.length;
    const probes: { key: string; res: number }[] = [];
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      const res = cmpFn(probe, n.keys[mid]);
      probes.push({ key: n.keys[mid], res });
      if (res >= 0) lo = mid + 1;
      else hi = mid;
    }
    comparisons += probes.length;
    steps.push({ level, node, probes, chose: n.children[lo] });
    node = n.children[lo];
    level -= 1;
  }

  const entries = leaves[node] ?? [];
  const { idx, probes } = bsearch(entries, probe, cmpFn);
  comparisons += probes.length;
  steps.push({ level: 0, node, probes, chose: idx });
  const hit = idx < entries.length && cmpFn(probe, entries[idx]) === 0 ? entries[idx] : null;
  return { steps, leaf: node, pos: idx, hit, comparisons };
}

/* -------------------------------------------------------------- geometry */

const LEAF_W = 108;
const LEAF_GAP = 12;
const ROW = 15;
const LEAF_PAD = 20;
const LEVEL_GAP = 46;
const CELL = 30;

/* ------------------------------------------------------------- component */

type Admitted = { key: string; leaf: number; pos: number };

export default function CollationOrderDriftLab() {
  const [builtWith, setBuiltWith] = useState<ComparatorId>('glibc227');
  const [runtime, setRuntime] = useState<ComparatorId>('glibc227');
  const [cap, setCap] = useState(4);
  const [desc, setDesc] = useState(false);
  const [probe, setProbe] = useState('de Vries');
  const [admitted, setAdmitted] = useState<Admitted[]>([]);
  const [head, setHead] = useState('The index is sorted and searched by the same comparator.');
  const [body, setBody] = useState(
    'Pick a key and watch the binary search descend. Then change "server compares with" — or press ' +
      'the upgrade button — and run the same probe against the same bytes.',
  );
  const [ref, width] = useSize(860);
  const tip = useTip();

  const tree = useMemo(() => buildTree(builtWith, cap, desc), [builtWith, cap, desc]);

  /** The physical leaves: what the bulk load wrote, plus anything a lookup miss let in. */
  const leaves = useMemo(() => {
    const out = tree.levels[0].map((n) => [...n.keys]);
    for (const a of admitted) {
      if (out[a.leaf]) out[a.leaf].splice(Math.min(a.pos, out[a.leaf].length), 0, a.key);
    }
    return out;
  }, [tree, admitted]);

  const d = useMemo(() => descend(tree, leaves, probe, runtime, desc), [tree, leaves, probe, runtime, desc]);

  const present = leaves.some((l) => l.includes(probe));
  const realLeaf = leaves.findIndex((l) => l.includes(probe));
  const exactHere = (leaves[d.leaf] ?? []).includes(probe);
  const miss = present && !exactHere && d.hit === null;

  /** amcheck's item-order invariant, evaluated against the comparator in force now. */
  const flat = useMemo(() => leaves.flat(), [leaves]);
  const misordered = useMemo(() => {
    const bad: string[] = [];
    for (let i = 1; i < flat.length; i++) {
      if (compare(runtime, flat[i - 1], flat[i], desc) > 0) bad.push(flat[i]);
    }
    return bad;
  }, [flat, runtime, desc]);

  const unreachable = useMemo(() => {
    const out: string[] = [];
    for (const l of leaves) {
      for (const k of l) {
        const r = descend(tree, leaves, k, runtime, desc);
        if (!(leaves[r.leaf] ?? []).includes(k) && r.hit === null) out.push(k);
      }
    }
    return out;
  }, [tree, leaves, runtime, desc]);

  const dupes = admitted.length;
  const rc = COMPARATORS.find((c) => c.id === runtime)!;
  const bc = COMPARATORS.find((c) => c.id === builtWith)!;

  const resetProbes = () => setAdmitted([]);

  const say = (h: string, b: string) => {
    setHead(h);
    setBody(b);
  };

  const setBuild = (v: ComparatorId) => {
    setBuiltWith(v);
    setRuntime(v);
    resetProbes();
    const c = COMPARATORS.find((x) => x.id === v)!;
    const refused = buildTree(v, cap, desc).rejected;
    say(
      `Rebuilt: the leaves are now in ${c.short} order.`,
      `${c.gist}${
        refused.length > 0
          ? ` This collation also refused ${refused.length} of the 24 keys as duplicates: ${refused.join(', ')}.`
          : ''
      }`,
    );
  };

  const setRun = (v: ComparatorId) => {
    setRuntime(v);
    const c = COMPARATORS.find((x) => x.id === v)!;
    say(
      v === builtWith
        ? 'Index and server agree again.'
        : `The server now compares with ${c.short}; the bytes on disk are still in ${bc.short} order.`,
      v === builtWith
        ? 'Every descent is sound: the separator that routed the key is the same one the comparator would choose now.'
        : 'Nothing was rewritten. No error was raised. Run a probe and watch a correct binary search take a wrong turn.',
    );
  };

  const upgrade = () => {
    setBuiltWith('glibc227');
    setRuntime('glibc228');
    resetProbes();
    say(
      'apt upgrade: glibc 2.27 → 2.28, and every text index in the cluster is now mis-sorted.',
      'The 2.28 locale data gives space and punctuation real primary weights. Postgres notices only if you ' +
        'ask: pg_collation.collversion still records the version the index was built under, so you get ' +
        '"WARNING: database has a collation version mismatch" — and queries that keep answering, wrongly.',
    );
  };

  const runInsert = () => {
    if (d.hit !== null) {
      say(
        `INSERT rejected: duplicate key value violates unique constraint "${INDEX_NAME}"`,
        `The descent landed on "${d.hit}", which ${rc.short} calls equal to "${probe}"${
          d.hit === probe ? '' : ' even though the bytes differ'
        }. This is the index doing its job.`,
      );
      return;
    }
    setAdmitted((cur) => [...cur, { key: probe, leaf: d.leaf, pos: d.pos }]);
    say(
      present
        ? `INSERT accepted — and "${probe}" is now in the index twice.`
        : `INSERT accepted into leaf ${d.leaf + 1}.`,
      present
        ? `The unique check only looks where the descent lands. It landed in leaf ${d.leaf + 1}; the existing row ` +
            `sits in leaf ${realLeaf + 1}, which ${rc.short} never visits. No constraint was violated as far as the ` +
            `engine can see, and the table now holds two rows a UNIQUE index promised could not both exist.`
        : 'The key was genuinely absent, so this is an ordinary insert at the position the binary search found.',
    );
  };

  const runAmcheck = () => {
    if (misordered.length === 0) {
      say(
        'bt_index_check() returned void.',
        'Every adjacent pair in the leaf level is still in order under the comparator in force, and the ' +
          'downlinks agree with the keys. The structure was never damaged — only its meaning.',
      );
      return;
    }
    say(
      `ERROR: item order invariant violated for index "${INDEX_NAME}"`,
      `amcheck walked the leaf level and found ${misordered.length} adjacent pair${
        misordered.length === 1 ? '' : 's'
      } where the left item is greater than the right one under the collation the server uses now — starting at ` +
        `"${misordered[0]}". This is the one check that catches collation drift before your users do, and it needs ` +
        'no downtime: bt_index_check() takes an AccessShareLock.',
    );
  };

  const runReindex = () => {
    if (dupes > 0) {
      const k = admitted[0].key;
      say(
        `ERROR: could not create unique index "${INDEX_NAME}"`,
        `DETAIL: Key (display_name)=(${k}) is duplicated. The rebuild sorts every row afresh, which is exactly ` +
          'when the duplicates the broken index admitted finally become visible. REINDEX is the fix for the ' +
          'ordering; nothing but a manual cleanup fixes the rows that got in while it was broken.',
      );
      return;
    }
    setBuiltWith(runtime);
    resetProbes();
    say(
      `REINDEX INDEX CONCURRENTLY ${INDEX_NAME};`,
      `The leaves were rewritten in ${rc.short} order and pg_collation.collversion was stamped with the version ` +
        'that did it. Every probe is sound again — until the next base image moves.',
    );
  };

  /* ------------------------------------------------------------ layout */

  const nLeaves = leaves.length;
  const treeW = nLeaves * LEAF_W + (nLeaves - 1) * LEAF_GAP;
  const maxLeafRows = Math.max(...leaves.map((l) => l.length), 1);
  const leafH = LEAF_PAD + maxLeafRows * ROW + 6;
  const nLevels = tree.levels.length;
  const internalH = 30;

  const cells = rc.kind === 'bytes' ? keyOf(probe).hex.length : [...probe].length;
  const stripW = 120 + cells * CELL;
  const stripRows = rc.kind === 'bytes' ? 2 : runtime === 'glibc227' ? 5 : runtime === 'icu_ai' ? 2 : 4;
  const stripH = 18 + stripRows * 16;

  const svgW = Math.max(width, treeW + 24, stripW + 24);
  const topOfTree = stripH + 24;
  const height = topOfTree + (nLevels - 1) * (internalH + LEVEL_GAP) + leafH + 26;

  const leafX = (i: number) => 12 + i * (LEAF_W + LEAF_GAP);
  const leafCx = (i: number) => leafX(i) + LEAF_W / 2;
  const levelY = (lvl: number) => topOfTree + (nLevels - 1 - lvl) * (internalH + LEVEL_GAP);

  const centers: number[][] = [];
  for (let lvl = 0; lvl < nLevels; lvl++) {
    if (lvl === 0) centers.push(leaves.map((_, i) => leafCx(i)));
    else centers.push(tree.levels[lvl].map((n) => n.children.reduce((a, c) => a + centers[lvl - 1][c], 0) / n.children.length));
  }

  const onPath = new Map<string, Step>();
  for (const s of d.steps) onPath.set(`${s.level}:${s.node}`, s);

  const cellsOut = rc.kind === 'bytes' ? null : weightCells(probe, runtime === 'glibc227' ? 'old' : 'new');
  const hex = keyOf(probe).hex;

  return (
    <VizPanel
      title="One index, two comparators"
      subtitle={
        'A UNIQUE btree on users(display_name), bulk-loaded in one comparator’s order and then searched with ' +
        'another. The bytes never move; only the comparator does.'
      }
      controls={
        <>
          <Choice label="Index built with" value={builtWith} onChange={setBuild} options={COMPARATOR_OPTIONS} />
          <Choice label="Server compares with" value={runtime} onChange={setRun} options={COMPARATOR_OPTIONS} />
          <Choice
            label="Probe key"
            value={probe}
            onChange={(v) => setProbe(v)}
            options={KEYS.map((k) => ({ value: k.s, label: k.s }))}
          />
          <Slider
            label="Entries per page"
            min={3}
            max={6}
            value={cap}
            onChange={(n) => {
              setCap(n);
              resetProbes();
            }}
          />
          <Check
            label="DESC (complement the bytes)"
            checked={desc}
            onChange={(b) => {
              setDesc(b);
              resetProbes();
              say(
                b ? 'Descending index.' : 'Ascending index.',
                b
                  ? 'An engine with memcomparable keys gets DESC for free: complement every byte (0x41 → 0xbe) and ' +
                      'keep using memcmp. CockroachDB terminates a descending string with 0xff 0xfe instead of 0x00 0x01. ' +
                      'PostgreSQL instead stores a DESC flag in pg_index.indoption and reverses the comparator.'
                  : 'Back to ascending order.',
              );
            }}
          />
          <Button onClick={upgrade} primary title="Build under glibc 2.27, then serve with glibc 2.28">
            Upgrade glibc 2.27 → 2.28
          </Button>
          <Button onClick={runInsert} title={`INSERT INTO users VALUES ('${probe}')`}>
            INSERT the probe key
          </Button>
          <Button onClick={runAmcheck} title="SELECT bt_index_check('users_display_name_key')">
            Run amcheck
          </Button>
          <Button onClick={runReindex}>REINDEX</Button>
          <Button
            onClick={() => {
              setBuiltWith('glibc227');
              setRuntime('glibc227');
              setCap(4);
              setDesc(false);
              setProbe('de Vries');
              resetProbes();
              say('Reset.', 'Index and server agree, ascending, four entries per page.');
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'descent path (binary search under the runtime comparator)', color: 'var(--viz-1)' },
            { label: '✓ probe found here', color: 'var(--viz-good)' },
            { label: '✗ probe missed — search stopped here', color: 'var(--viz-critical)' },
            { label: '● the key is actually stored here', color: 'var(--viz-7)', shape: 'dot' },
            { label: '‼ out of order under the runtime comparator', color: 'var(--viz-serious)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Keys in the index', value: flat.length, hint: '24 rows, minus anything UNIQUE refused' },
            { label: 'Height', value: nLevels, hint: 'Levels from root to leaf, at this page capacity' },
            { label: 'Comparisons', value: d.comparisons, hint: 'Binary-search calls to the comparator for this probe' },
            {
              label: 'Probe result',
              value: exactHere || d.hit !== null ? 'found' : present ? 'MISS' : 'absent',
              hint: present ? 'The row exists in the index' : 'Not in the index at all',
            },
            {
              label: 'Unreachable keys',
              value: `${unreachable.length} / ${flat.length}`,
              hint: 'Rows an index scan can no longer find, though a seq scan would',
            },
            {
              label: 'Out-of-order pairs',
              value: misordered.length,
              hint: 'What bt_index_check() reports as an item order invariant violation',
            },
            { label: 'Duplicates admitted', value: dupes, hint: 'Rows a UNIQUE index accepted that it should not have' },
            {
              label: 'UNIQUE rejections at build',
              value: tree.rejected.length,
              hint: tree.rejected.length ? `Refused: ${tree.rejected.join(', ')}` : 'None: every key is distinct to this comparator',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{head}</strong> {body}
        </Note>
      }
      table={
        <>
          <p>
            Rank of each key under each comparator. Equal ranks mean the comparator calls those keys equal.{' '}
            {FIXTURE_PROVENANCE}
          </p>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Key</th>
                <th>UTF-8 bytes</th>
                {COMPARATORS.map((c) => (
                  <th key={c.id}>{c.short}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {KEYS.map((k) => (
                <tr key={k.s}>
                  <td>{k.s}</td>
                  <td>{k.hex.map((b) => b.toString(16).padStart(2, '0')).join(' ')}</td>
                  {COMPARATOR_ORDER.map((id, i) => (
                    <td key={id}>{k.r[i]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Level</th>
                <th>Compared against</th>
                <th>Result</th>
                <th>Went</th>
              </tr>
            </thead>
            <tbody>
              {d.steps.flatMap((s, si) =>
                s.probes.length === 0
                  ? [
                      <tr key={`${si}-empty`}>
                        <td>{si + 1}</td>
                        <td>{s.level === 0 ? 'leaf' : `internal L${s.level}`}</td>
                        <td colSpan={2}>nothing to compare</td>
                        <td>{s.level === 0 ? `slot ${s.chose}` : `child ${s.chose + 1}`}</td>
                      </tr>,
                    ]
                  : s.probes.map((p, pi) => (
                      <tr key={`${si}-${pi}`}>
                        <td>{si + 1}</td>
                        <td>{s.level === 0 ? 'leaf' : `internal L${s.level}`}</td>
                        <td>{p.key}</td>
                        <td>
                          {probe} {p.res < 0 ? '<' : p.res > 0 ? '>' : '='} {p.key}
                        </td>
                        <td>
                          {pi === s.probes.length - 1
                            ? s.level === 0
                              ? `slot ${s.chose}`
                              : `child ${s.chose + 1}`
                            : p.res >= 0
                              ? 'right'
                              : 'left'}
                        </td>
                      </tr>
                    )),
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
            aria-label="A btree built by one comparator and searched by another, showing the descent path and where the probe key really lives"
          >
            {/* --- what the runtime comparator actually compares --- */}
            <text x={0} y={12} fill="var(--viz-ink)" fontWeight={600}>
              {rc.kind === 'bytes'
                ? `memcmp sees "${probe}" as bytes${desc ? ', complemented for DESC' : ''}`
                : `${rc.short} sees "${probe}" as weight levels`}
            </text>
            {rc.kind === 'bytes'
              ? hex.map((b, i) => {
                  const v = desc ? 0xff - b : b;
                  return (
                    <g
                      key={i}
                      {...tip(
                        <>
                          <strong>{`0x${v.toString(16).padStart(2, '0')}`}</strong>
                          <br />
                          {desc
                            ? `0xff − 0x${b.toString(16).padStart(2, '0')}: complementing every byte turns an ascending memcmp into a descending scan.`
                            : 'UTF-8 is self-ordering: byte order is codepoint order, so no table is consulted at all.'}
                        </>,
                      )}
                    >
                      <rect
                        x={120 + i * CELL}
                        y={20}
                        width={CELL - 4}
                        height={16}
                        rx={3}
                        fill="var(--viz-plane)"
                        stroke="var(--viz-border)"
                      />
                      <text x={120 + i * CELL + (CELL - 4) / 2} y={32} textAnchor="middle" fill="var(--viz-ink)">
                        {v.toString(16).padStart(2, '0')}
                      </text>
                    </g>
                  );
                })
              : null}
            {rc.kind === 'bytes' ? (
              <>
                <text x={0} y={32} fill="var(--viz-ink-2)">
                  stored bytes
                </text>
                <text x={120} y={50} fill="var(--viz-ink-muted)">
                  {desc
                    ? 'terminator ff fe — descending. Any 0xff inside the key would be escaped as ff 00.'
                    : 'terminator 00 01 — the key ends here. Any 0x00 inside the key would be escaped as 00 ff.'}
                </text>
              </>
            ) : (
              (['l1', 'l2', 'l3', 'l4'] as const).slice(0, stripRows - 1).map((lvl, row) => (
                <g key={lvl}>
                  <text x={0} y={32 + row * 16} fill="var(--viz-ink-2)">
                    {lvl === 'l1'
                      ? 'L1 base letter'
                      : lvl === 'l2'
                        ? 'L2 accent'
                        : lvl === 'l3'
                          ? 'L3 case'
                          : 'L4 codepoint'}
                  </text>
                  {(cellsOut ?? []).map((c, i) => (
                    <text
                      key={i}
                      x={120 + i * CELL + (CELL - 4) / 2}
                      y={32 + row * 16}
                      textAnchor="middle"
                      fill={c.ignored ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}
                    >
                      {c[lvl]}
                    </text>
                  ))}
                </g>
              ))
            )}
            {rc.kind === 'weights' ? (
              <text x={120} y={32 + (stripRows - 1) * 16} fill="var(--viz-ink-muted)">
                {runtime === 'glibc227'
                  ? 'punctuation is IGNORE until L4; an unassigned character never appears at all'
                  : runtime === 'icu_ai'
                    ? 'comparison stops after L1: accents and case never get a vote, so Adams = adams and Backer = Bäcker'
                    : 'punctuation and symbols carry primary weights, below every digit and letter'}
              </text>
            ) : null}

            {/* --- the tree --- */}
            {tree.levels.map((nodes, lvl) =>
              lvl === 0 ? null : (
                <g key={`edges${lvl}`}>
                  {nodes.map((n, ni) =>
                    n.children.map((c) => {
                      const step = onPath.get(`${lvl}:${ni}`);
                      const taken = step?.chose === c;
                      return (
                        <line
                          key={`${ni}-${c}`}
                          x1={centers[lvl][ni]}
                          y1={levelY(lvl) + internalH}
                          x2={centers[lvl - 1][c]}
                          y2={levelY(lvl - 1)}
                          stroke={taken ? 'var(--viz-1)' : 'var(--viz-axis)'}
                          strokeWidth={taken ? 2.5 : 1}
                        />
                      );
                    }),
                  )}
                </g>
              ),
            )}

            {tree.levels.map((nodes, lvl) =>
              lvl === 0 ? null : (
                <g key={`nodes${lvl}`}>
                  {nodes.map((n, ni) => {
                    const step = onPath.get(`${lvl}:${ni}`);
                    const w = Math.max(72, n.keys.length * 76 + 16);
                    const x = centers[lvl][ni] - w / 2;
                    return (
                      <g
                        key={ni}
                        {...tip(
                          <>
                            <strong>{lvl === nLevels - 1 ? 'root' : `internal page, level ${lvl}`}</strong>
                            <br />
                            {n.keys.length} separator{n.keys.length === 1 ? '' : 's'}, {n.children.length} downlinks.
                            Separators are copy-ups of each child&rsquo;s first key, frozen in {bc.short} order.
                          </>,
                        )}
                      >
                        <rect
                          x={x}
                          y={levelY(lvl)}
                          width={w}
                          height={internalH}
                          rx={6}
                          fill="var(--viz-plane)"
                          stroke={step ? 'var(--viz-1)' : 'var(--viz-border)'}
                          strokeWidth={step ? 2 : 1}
                        />
                        {n.keys.map((k, ki) => (
                          <text
                            key={`${k}-${ki}`}
                            x={x + 8 + ki * 76}
                            y={levelY(lvl) + 19}
                            fill={step?.probes.some((p) => p.key === k) ? 'var(--viz-1)' : 'var(--viz-ink-2)'}
                            fontWeight={step?.probes.some((p) => p.key === k) ? 600 : 400}
                          >
                            {k}
                          </text>
                        ))}
                        {n.keys.length === 0 ? (
                          <text x={x + 8} y={levelY(lvl) + 19} fill="var(--viz-ink-muted)">
                            (no separators)
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                </g>
              ),
            )}

            {leaves.map((entries, i) => {
              const landed = d.leaf === i;
              const real = realLeaf === i;
              const y = levelY(0);
              return (
                <g key={i}>
                  <rect
                    x={leafX(i)}
                    y={y}
                    width={LEAF_W}
                    height={leafH}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={
                      landed ? (exactHere || d.hit !== null ? 'var(--viz-good)' : 'var(--viz-critical)') : 'var(--viz-border)'
                    }
                    strokeWidth={landed ? 2.5 : 1}
                  />
                  <text x={leafX(i) + 6} y={y + 13} fill="var(--viz-ink-muted)">
                    leaf {i + 1}
                    {real && !landed ? ' ●' : ''}
                  </text>
                  {entries.map((k, ki) => {
                    const bad = misordered.includes(k);
                    const isProbe = k === probe;
                    const hitHere = landed && d.hit === k;
                    const dup = admitted.some((a) => a.key === k) && isProbe;
                    return (
                      <g
                        key={`${k}-${ki}`}
                        {...tip(
                          <>
                            <strong>{k}</strong>
                            <br />
                            {`${keyOf(k).hex.map((b) => b.toString(16).padStart(2, '0')).join(' ')}`}
                            <br />
                            {bad
                              ? 'Sorts before the entry to its left under the comparator in force now — amcheck calls this an item order invariant violation.'
                              : hitHere
                                ? 'The binary search stopped here and the comparator called it equal to the probe.'
                                : 'Written here by the bulk load, in the build comparator’s order.'}
                          </>,
                        )}
                      >
                        <text
                          x={leafX(i) + 6}
                          y={y + LEAF_PAD + 10 + ki * ROW}
                          fill={
                            hitHere
                              ? 'var(--viz-good)'
                              : dup
                                ? 'var(--viz-critical)'
                                : isProbe
                                  ? 'var(--viz-7)'
                                  : bad
                                    ? 'var(--viz-serious)'
                                    : 'var(--viz-ink)'
                          }
                          fontWeight={isProbe || hitHere ? 600 : 400}
                        >
                          {bad ? '‼ ' : ''}
                          {k}
                          {hitHere ? ' ✓' : ''}
                          {isProbe && !hitHere ? ' ●' : ''}
                        </text>
                      </g>
                    );
                  })}
                  {landed && !exactHere && d.hit === null ? (
                    <text
                      x={leafX(i) + 6}
                      y={y + LEAF_PAD + 10 + entries.length * ROW}
                      fill="var(--viz-critical)"
                      fontWeight={600}
                    >
                      ✗ not here
                    </text>
                  ) : null}
                </g>
              );
            })}

            {miss ? (
              <line
                x1={leafCx(d.leaf)}
                y1={levelY(0) + leafH + 8}
                x2={leafCx(realLeaf)}
                y2={levelY(0) + leafH + 8}
                stroke="var(--viz-7)"
                strokeWidth={1.5}
                strokeDasharray="5 4"
              />
            ) : null}
            {miss ? (
              <text x={(leafCx(d.leaf) + leafCx(realLeaf)) / 2} y={levelY(0) + leafH + 22} textAnchor="middle" fill="var(--viz-7)">
                the row is {Math.abs(realLeaf - d.leaf)} leaf
                {Math.abs(realLeaf - d.leaf) === 1 ? '' : 's'} away, and this descent never goes there
              </text>
            ) : null}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
