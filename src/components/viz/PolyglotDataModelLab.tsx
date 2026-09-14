import { useEffect, useRef, useState } from 'react';
import {
  VizPanel,
  Choice,
  Check,
  Slider,
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
  useTicker,
} from './Viz';

/**
 * One blog dataset — users, posts, comments, tags, follows — stored five ways at once,
 * then asked the same question.
 *
 * The point is not that one model wins. It is that each model has exactly one cheap
 * access path (primary key, document id, key, partition key, adjacency pointer), and
 * every other question is paid for somewhere: an extra round trip, a scatter across
 * partitions, a scan, or a copy of the fact written at insert time that you now have
 * to keep correct. The "denormalize on write" switch moves the cost between the read
 * path and the write path without ever deleting it.
 */

type ModelId = 'rel' | 'doc' | 'kv' | 'wc' | 'graph';
type Kind = 'index' | 'read' | 'traverse' | 'scan' | 'write';
type QueryId = 'post' | 'tag' | 'fof' | 'rename';

const KIND: Record<Kind, { color: string; letter: string; label: string }> = {
  index: { color: 'var(--viz-1)', letter: 'I', label: 'I — index / key lookup' },
  read: { color: 'var(--viz-2)', letter: 'R', label: 'R — unit fetched by key' },
  traverse: { color: 'var(--viz-3)', letter: 'T', label: 'T — pointer traversal (no index)' },
  scan: { color: 'var(--viz-4)', letter: 'S', label: 'S — scan / scatter' },
  write: { color: 'var(--viz-5)', letter: 'W', label: 'W — write' },
};

const MODELS: { id: ModelId; label: string; engine: string; unit: string }[] = [
  { id: 'rel', label: 'Relational', engine: 'PostgreSQL, InnoDB', unit: 'row in a table' },
  { id: 'doc', label: 'Document', engine: 'MongoDB, jsonb', unit: 'nested document' },
  { id: 'kv', label: 'Key–value', engine: 'Redis, RocksDB', unit: 'opaque value' },
  { id: 'wc', label: 'Wide-column', engine: 'Cassandra, Bigtable', unit: 'partition of cells' },
  { id: 'graph', label: 'Property graph', engine: 'Neo4j', unit: 'node + relationship' },
];

/* ------------------------------------------------------------------ dataset */

/** Modelled payload sizes, in bytes. Real engines round these up to whole pages/blocks. */
const SZ = { post: 1520, user: 96, tag: 28, ptag: 16, follow: 16, proj: 224, cell: 40 };

/** Comment bodies vary; deterministic so SSR and hydration agree. */
const COMMENT_BYTES = (() => {
  const rng = makeRng(19700101);
  return Array.from({ length: 24 }, () => 140 + Math.round(rng() * 130));
})();
const cSum = (n: number) => COMMENT_BYTES.slice(0, n).reduce((a, b) => a + b, 0);

type Unit = { id: string; label: string; sub: string; subAlt?: string; ghost?: 'dn' | 'nodn' };

const UNITS: Record<ModelId, Unit[]> = {
  rel: [
    { id: 'posts', label: 'posts', sub: 'heap + pkey' },
    { id: 'comments', label: 'comments', sub: 'idx(post_id)' },
    { id: 'users', label: 'users', sub: 'heap + pkey' },
    { id: 'tags', label: 'tags', sub: 'uniq(name)' },
    { id: 'post_tags', label: 'post_tags', sub: 'idx(tag_id)' },
    { id: 'follows', label: 'follows', sub: 'idx(follower)' },
  ],
  doc: [
    { id: 'posts/p1', label: 'posts/p1', sub: 'post + comments', subAlt: 'post only' },
    { id: 'posts/p2', label: 'posts/p2', sub: 'post' },
    { id: 'posts/p3', label: 'posts/p3', sub: 'post' },
    { id: 'comments', label: 'comments', sub: 'folded into post', subAlt: 'own collection', ghost: 'dn' },
    { id: 'users/u1', label: 'users/u1', sub: 'profile' },
    { id: 'users/u2', label: 'users/u2', sub: 'profile' },
    { id: 'users/u3', label: 'users/u3', sub: 'profile' },
    { id: 'users/u4', label: 'users/u4', sub: 'profile' },
  ],
  kv: [
    { id: 'post:p1', label: 'post:p1', sub: 'blob: post+cmts', subAlt: 'post JSON' },
    { id: 'post:p2', label: 'post:p2', sub: 'post JSON' },
    { id: 'post:p3', label: 'post:p3', sub: 'post JSON' },
    { id: 'comments:p1', label: 'comments:p1', sub: 'in the blob', subAlt: 'LIST', ghost: 'dn' },
    { id: 'tag:storage', label: 'tag:storage', sub: 'SET {p1,p2}', subAlt: 'not maintained', ghost: 'nodn' },
    { id: 'user:u2', label: 'user:u2', sub: 'profile' },
    { id: 'follows:u1', label: 'follows:u1', sub: 'SET' },
    { id: 'follows:u2', label: 'follows:u2', sub: 'SET' },
    { id: 'follows:u4', label: 'follows:u4', sub: 'SET' },
  ],
  wc: [
    { id: 'by_id/p1', label: 'posts_by_id/p1', sub: 'static + n rows' },
    { id: 'by_id/p2', label: 'posts_by_id/p2', sub: 'static cols' },
    { id: 'by_id/p3', label: 'posts_by_id/p3', sub: 'static cols' },
    { id: 'by_tag/st', label: 'posts_by_tag/storage', sub: 'clustered by date', subAlt: 'no such table', ghost: 'nodn' },
    { id: 'users/u2', label: 'users/u2', sub: 'profile row' },
    { id: 'follows/u1', label: 'follows/u1', sub: 'clustered' },
    { id: 'follows/u2', label: 'follows/u2', sub: 'clustered' },
    { id: 'follows/u4', label: 'follows/u4', sub: 'clustered' },
  ],
  graph: [],
};

/** The graph lane is drawn as nodes and relationships, not chips. */
const GNODES: { id: string; label: string; x: number; y: number }[] = [
  { id: 'u1', label: 'u1', x: 54, y: 34 },
  { id: 'u2', label: 'u2', x: 174, y: 34 },
  { id: 'u3', label: 'u3', x: 294, y: 34 },
  { id: 'u4', label: 'u4', x: 414, y: 34 },
  { id: 'p1', label: 'p1', x: 54, y: 100 },
  { id: 'c1', label: 'c1', x: 174, y: 100 },
  { id: 'c2', label: 'c2', x: 264, y: 100 },
  { id: 'c3', label: 'c3', x: 354, y: 100 },
  { id: 'tg', label: 'tg', x: 478, y: 100 },
  { id: 'p2', label: 'p2', x: 598, y: 100 },
];

type GEdge = { id: string; a: string; b: string; arc: 'up' | 'down' | 'none' };
const GEDGES: GEdge[] = [
  { id: 'f12', a: 'u1', b: 'u2', arc: 'up' },
  { id: 'f14', a: 'u1', b: 'u4', arc: 'up' },
  { id: 'f23', a: 'u2', b: 'u3', arc: 'up' },
  { id: 'f24', a: 'u2', b: 'u4', arc: 'up' },
  { id: 'f43', a: 'u4', b: 'u3', arc: 'up' },
  { id: 'a1', a: 'u1', b: 'p1', arc: 'none' },
  { id: 'h1', a: 'p1', b: 'c1', arc: 'down' },
  { id: 'h2', a: 'p1', b: 'c2', arc: 'down' },
  { id: 'h3', a: 'p1', b: 'c3', arc: 'down' },
  { id: 'b1', a: 'c1', b: 'u2', arc: 'none' },
  { id: 'b2', a: 'c2', b: 'u3', arc: 'none' },
  { id: 'b3', a: 'c3', b: 'u4', arc: 'none' },
  { id: 't1', a: 'p1', b: 'tg', arc: 'down' },
  { id: 't2', a: 'p2', b: 'tg', arc: 'none' },
];

/* -------------------------------------------------------------------- plan */

type Step = {
  model: ModelId;
  units: string[];
  edges?: string[];
  kind: Kind;
  rt: number;
  idx: number;
  items: number;
  bytes: number;
  op: string;
};

const QUERIES: { value: QueryId; label: string }[] = [
  { value: 'post', label: 'Read a post with its comments and their authors' },
  { value: 'tag', label: 'All posts tagged "storage"' },
  { value: 'fof', label: 'Friends-of-friends of u1 who commented on p1' },
  { value: 'rename', label: 'Write: u2 changes their display name' },
];

function plan(q: QueryId, dn: boolean, n: number): Step[] {
  const S: Step[] = [];
  const c = cSum(n);
  const u2c = Math.max(1, Math.round(n / 3)); // comments on p1 written by u2
  const add = (
    model: ModelId,
    units: string[],
    kind: Kind,
    rt: number,
    idx: number,
    items: number,
    bytes: number,
    op: string,
    edges?: string[],
  ) => S.push({ model, units, kind, rt, idx, items, bytes, op, edges });

  if (q === 'post') {
    add('rel', ['posts'], 'index', 1, 1, 1, SZ.post, 'one statement: posts_pkey descent → 1 heap tuple');
    add('rel', ['comments'], 'index', 0, 1, n, c, `idx(post_id) range scan → ${n} comment tuples`);
    add('rel', ['users'], 'index', 0, 4, 4, 4 * SZ.user, 'nested-loop join → author + 3 distinct commenters');

    if (dn) {
      add('doc', ['posts/p1'], 'read', 1, 1, 1, SZ.post + c, 'findOne({_id:"p1"}) — comments and author names embedded: one seek, zero joins');
    } else {
      add('doc', ['posts/p1'], 'read', 1, 1, 1, SZ.post, 'findOne({_id:"p1"})');
      add('doc', ['comments'], 'index', 1, 1, n, c, 'second query: find({post_id:"p1"}) — or a server-side $lookup');
      add('doc', ['users/u2', 'users/u3', 'users/u4'], 'index', 1, 3, 3, 3 * SZ.user, 'third query: find({_id:{$in:[…]}}) just for the names');
    }

    if (dn) {
      add('kv', ['post:p1'], 'read', 1, 1, 1, SZ.post + c, 'GET post:p1 — one prebuilt render blob, rewritten on every comment');
    } else {
      add('kv', ['post:p1'], 'read', 1, 1, 1, SZ.post, 'GET post:p1');
      add('kv', ['comments:p1'], 'read', 1, 1, n, c, `LRANGE comments:p1 0 -1 → ${n} ids, then a second fetch`);
      add('kv', ['user:u2'], 'read', 1, 3, 3, 3 * SZ.user, 'MGET user:u2 user:u3 user:u4 — the join is your for-loop');
    }

    add('wc', ['by_id/p1'], 'read', 1, 1, 1 + n, SZ.post + c, `one partition: static post columns + ${n} comment rows, already sorted by the clustering key`);
    if (!dn) add('wc', ['users/u2'], 'read', 3, 3, 3, 3 * SZ.user, 'no join exists: 3 more single-partition reads for the names');

    add('graph', ['p1'], 'index', 1, 1, 1, SZ.post, 'the only index lookup in the whole query: Post(slug) → node record');
    add('graph', ['c1', 'c2', 'c3'], 'traverse', 0, 0, n, c, `walk the HAS_COMMENT relationship chain → ${n} comment nodes`, ['h1', 'h2', 'h3']);
    add('graph', ['u2', 'u3', 'u4'], 'traverse', 0, 0, 3, 3 * SZ.user, 'each comment record stores the id of its author node — one pointer hop each', ['b1', 'b2', 'b3']);
  }

  if (q === 'tag') {
    add('rel', ['tags'], 'index', 1, 1, 1, SZ.tag, 'uniq(name) → tag id');
    add('rel', ['post_tags'], 'index', 0, 1, 2, 2 * SZ.ptag, 'idx(tag_id) → 2 junction rows');
    add('rel', ['posts'], 'index', 0, 2, 2, 2 * SZ.post, 'pkey lookups → 2 posts. The junction table is what makes many-to-many cheap both ways');

    add(
      'doc',
      ['posts/p1', 'posts/p2'],
      'index',
      1,
      1,
      2,
      dn ? 2 * SZ.post + c : 2 * SZ.post,
      dn
        ? 'multikey index on tags[] → 2 documents, and p1 drags all its embedded comments along'
        : 'multikey index on tags[] → 2 lean documents',
    );

    if (dn) {
      add('kv', ['tag:storage'], 'read', 1, 1, 1, 64, 'SMEMBERS tag:storage — a secondary index you build and maintain by hand');
      add('kv', ['post:p1', 'post:p2'], 'read', 1, 2, 2, 2 * SZ.post + c, 'MGET the two blobs');
    } else {
      add('kv', ['post:p1', 'post:p2', 'post:p3'], 'scan', 2, 0, 3, 3 * SZ.post, 'no secondary access path: SCAN the keyspace, deserialize, filter in the client');
    }

    if (dn) {
      add('wc', ['by_tag/st'], 'read', 1, 1, 2, 2 * SZ.proj, 'one partition of posts_by_tag, clustered by (created_at, post_id) — a whole table that exists only for this query');
    } else {
      add('wc', ['by_id/p1', 'by_id/p2', 'by_id/p3'], 'scan', 3, 0, 3, 3 * SZ.post + c, 'ALLOW FILTERING: the coordinator scatters to every token range and filters after reading');
    }

    add('graph', ['tg'], 'index', 1, 1, 1, SZ.tag, 'index on Tag(name) → one node');
    add('graph', ['p1', 'p2'], 'traverse', 0, 0, 2, 2 * SZ.post, 'walk the incoming TAGGED_AS chain → the tagged posts', ['t1', 't2']);
  }

  if (q === 'fof') {
    add('rel', ['follows'], 'index', 1, 1, 2, 2 * SZ.follow, 'one statement: idx(follower_id) → u1 follows u2, u4');
    add('rel', ['follows'], 'index', 0, 2, 3, 3 * SZ.follow, 'self-join, second hop → u3, u4 (u4 twice; DISTINCT drops it)');
    add('rel', ['comments'], 'index', 0, 1, n, c, 'join to comments on p1 and keep only the ones who commented');
    add('rel', ['users'], 'index', 0, 2, 2, 2 * SZ.user, 'join to users for the display names');

    add('doc', ['users/u1'], 'index', 1, 1, 1, SZ.user, '$graphLookup starts from u1');
    add('doc', ['users/u2', 'users/u4'], 'index', 0, 2, 2, 2 * SZ.user, 'hop 1: an index lookup per edge — a document store has no pointers between documents');
    add('doc', ['users/u3'], 'index', 0, 3, 1, SZ.user, 'hop 2: 3 more index lookups');
    add('doc', [dn ? 'posts/p1' : 'comments'], 'index', 1, 1, n, c, 'second query: who actually commented on p1');

    add('kv', ['follows:u1'], 'read', 1, 1, 2, 2 * SZ.follow, 'SMEMBERS follows:u1');
    add('kv', ['follows:u2', 'follows:u4'], 'read', 1, 2, 3, 3 * SZ.follow, 'pipeline the next hop — you are writing the traversal, one round trip per level');
    add('kv', [dn ? 'post:p1' : 'comments:p1'], 'read', 1, 1, n, c, 'fetch the comment list and intersect the two sets in your process');

    add('wc', ['follows/u1'], 'read', 1, 1, 2, 2 * SZ.follow, 'one partition: SELECT … FROM follows WHERE follower = u1');
    add('wc', ['follows/u2'], 'read', 1, 1, 2, 2 * SZ.follow, 'hop 2, partition 1 — a different token, probably a different coordinator');
    add('wc', ['follows/u4'], 'read', 1, 1, 1, SZ.follow, 'hop 2, partition 2 — the scatter is the join');
    add('wc', ['by_id/p1'], 'read', 1, 1, n, c, 'read p1 to see who commented; the filter runs in your code');

    add('graph', ['u1'], 'index', 1, 1, 1, SZ.user, 'index lookup on User(handle) → the start node, and that is the last index touched');
    add('graph', ['u2', 'u4'], 'traverse', 0, 0, 2, 2 * SZ.user, 'hop 1: follow the FOLLOWS chain out of u1', ['f12', 'f14']);
    add('graph', ['u3'], 'traverse', 0, 0, 3, 3 * SZ.user, 'hop 2: three more pointer dereferences, cost independent of graph size', ['f23', 'f24', 'f43']);
    add('graph', ['c2', 'c3', 'p1'], 'traverse', 0, 0, 3, 2 * COMMENT_BYTES[1], 'intersect with the comment edges on p1 — u3 and u4 qualify', ['b2', 'b3', 'h2', 'h3']);
  }

  if (q === 'rename') {
    add('rel', ['users'], 'write', 1, 1, 1, SZ.user, 'UPDATE users SET display_name=… WHERE id=2 — one row, because the fact is stored once');

    if (dn) {
      add('doc', ['users/u2'], 'write', 1, 1, 1, SZ.user, 'updateOne on the profile document');
      add('doc', ['posts/p2'], 'write', 1, 1, 1, SZ.cell, 'updateMany: author_name was copied into every post u2 wrote');
      add('doc', ['posts/p1'], 'write', 1, 1, u2c, SZ.cell * u2c, `updateMany with arrayFilters: ${u2c} embedded comment subdocuments carry the stale name — atomic per document, not across them`);
    } else {
      add('doc', ['users/u2'], 'write', 1, 1, 1, SZ.user, 'updateOne — comments reference the author by _id, so one write is enough (and every read pays a join)');
    }

    if (dn) {
      add('kv', ['user:u2'], 'write', 1, 1, 1, SZ.user, 'SET user:u2');
      add('kv', ['post:p1'], 'write', 1, 1, 1, SZ.post + c, 'the whole render blob is rewritten — a key–value store updates values, never fields');
      add('kv', ['post:p2'], 'write', 1, 1, 1, SZ.post, 'nothing in the store tells you which blobs embed the name; you maintain that list yourself');
    } else {
      add('kv', ['user:u2'], 'write', 1, 1, 1, SZ.user, 'SET user:u2 — one write, and every read now costs an extra MGET');
    }

    if (dn) {
      add('wc', ['users/u2'], 'write', 1, 1, 1, SZ.user, 'INSERT into users (an upsert — Cassandra has no UPDATE/INSERT distinction)');
      add('wc', ['by_id/p1'], 'write', 1, 1, u2c, SZ.cell * u2c, `${u2c} comment rows carry author_name as a cell, each with its own write timestamp`);
      add('wc', ['by_tag/st'], 'write', 1, 1, 1, SZ.cell, 'and the query table too. A logged batch buys atomicity across partitions, never isolation');
    } else {
      add('wc', ['users/u2'], 'write', 1, 1, 1, SZ.user, 'one row — but then every read path needs an extra partition lookup per name');
    }

    add('graph', ['u2'], 'write', 1, 1, 1, SZ.user, 'SET u2.display_name — every relationship still points at the same node id');
  }

  return S;
}

function workSite(q: QueryId, m: ModelId, dn: boolean): string {
  if (q === 'rename') {
    if (m === 'rel' || m === 'graph') return 'one copy of the fact';
    return dn ? 'fan-out: every copy written at insert time' : 'one copy — the read path pays instead';
  }
  if (m === 'rel') return 'in the engine (nested loop / hash join)';
  if (m === 'graph') return 'pointer traversal, no join operator';
  if (m === 'doc') return dn ? 'at write time (embedded)' : 'application code or $lookup';
  if (m === 'kv') return dn ? 'at write time (blob / index key)' : 'application code';
  return dn ? 'at write time (a table per query)' : 'application code / ALLOW FILTERING';
}

/* --------------------------------------------------------------- geometry */

const LBL = 172;
const CW = 100;
const CH = 36;
const CG = 8;
const LANE_H = 74;
const GRAPH_H = 196;
const LANE_GAP = 10;

function trunc(s: string, max: number) {
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}

export default function PolyglotDataModelLab() {
  const [query, setQuery] = useState<QueryId>('post');
  const [dn, setDn] = useState(true);
  const [n, setN] = useState(6);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ref, width] = useSize(920);
  const tip = useTip();
  const acc = useRef(0);

  const steps = plan(query, dn, n);
  const byModel: Record<ModelId, Step[]> = {
    rel: steps.filter((s) => s.model === 'rel'),
    doc: steps.filter((s) => s.model === 'doc'),
    kv: steps.filter((s) => s.model === 'kv'),
    wc: steps.filter((s) => s.model === 'wc'),
    graph: steps.filter((s) => s.model === 'graph'),
  };
  const maxLen = Math.max(...MODELS.map((m) => byModel[m.id].length));

  useEffect(() => {
    acc.current = 0;
    setStep(0);
    setPlaying(true);
  }, [query, dn]);

  useTicker((dt) => {
    acc.current += dt;
    if (acc.current < 900) return;
    acc.current = 0;
    if (step >= maxLen) setPlaying(false);
    else setStep(step + 1);
  }, playing);

  /** Per-model totals for everything revealed so far. */
  const totals = MODELS.map((m) => {
    const seen = byModel[m.id].slice(0, step);
    return {
      id: m.id,
      label: m.label,
      rt: seen.reduce((a, s) => a + s.rt, 0),
      idx: seen.reduce((a, s) => a + s.idx, 0),
      items: seen.reduce((a, s) => a + s.items, 0),
      bytes: seen.reduce((a, s) => a + s.bytes, 0),
      last: seen.length ? seen[seen.length - 1] : null,
    };
  });
  const full = MODELS.map((m) => {
    const all = byModel[m.id];
    return {
      id: m.id,
      label: m.label,
      rt: all.reduce((a, s) => a + s.rt, 0),
      idx: all.reduce((a, s) => a + s.idx, 0),
      items: all.reduce((a, s) => a + s.items, 0),
      bytes: all.reduce((a, s) => a + s.bytes, 0),
    };
  });
  const best = full.reduce((a, b) => (b.rt < a.rt ? b : a));
  const worst = full.reduce((a, b) => (b.rt > a.rt ? b : a));
  const fattest = full.reduce((a, b) => (b.bytes > a.bytes ? b : a));
  const graphIdx = full.find((f) => f.id === 'graph')!.idx;
  const mostIdx = full.reduce((a, b) => (b.idx > a.idx ? b : a));

  /** Which units are lit, by which step, in which colour. */
  const mark: Record<string, { kind: Kind; order: number }> = {};
  const edgeMark: Record<string, { kind: Kind; order: number }> = {};
  for (const m of MODELS) {
    byModel[m.id].slice(0, step).forEach((s, i) => {
      for (const u of s.units) mark[`${m.id}:${u}`] = { kind: s.kind, order: i + 1 };
      for (const e of s.edges ?? []) edgeMark[e] = { kind: s.kind, order: i + 1 };
    });
  }

  const maxChips = Math.max(...MODELS.filter((m) => m.id !== 'graph').map((m) => UNITS[m.id].length));
  const svgW = Math.max(width, LBL + maxChips * (CW + CG) + 16);
  const laneTop: Record<ModelId, number> = { rel: 0, doc: 0, kv: 0, wc: 0, graph: 0 };
  let y = 6;
  for (const m of MODELS) {
    laneTop[m.id] = y;
    y += (m.id === 'graph' ? GRAPH_H : LANE_H) + LANE_GAP;
  }
  const svgH = y;

  const gx = (id: string) => GNODES.find((g) => g.id === id)!.x + LBL;
  const gy = (id: string) => GNODES.find((g) => g.id === id)!.y + laneTop.graph + 26;

  const notes: Record<QueryId, string> = {
    post: dn
      ? 'Embedding buys one-seek locality: the document, the blob and the partition each answer this query with a single read, and the graph reaches the same rows with one index lookup plus pointer hops. Relational pays two extra index descents and joins them in the engine.'
      : 'With nothing denormalized, every model except relational and graph has to issue the join itself — two or three round trips, each one a chance for the client to see a half-updated picture.',
    tag: dn
      ? 'A second access path is never free: the document store gets one via a multikey index the engine maintains, Redis and Cassandra get one only because you wrote a second copy of the data at insert time.'
      : 'Without a maintained secondary path, "all posts tagged storage" degrades into a keyspace SCAN in Redis and an ALLOW FILTERING scatter in Cassandra — both O(dataset), both fine on 3 posts and fatal on 3 million.',
    fof: 'Two hops is where the models separate. Relational expresses it as a self-join; the document store does an index lookup per edge; Cassandra turns each hop into another partition on another node. The graph does one index lookup and then dereferences pointers — index-free adjacency, cost proportional to the neighbourhood, not the database.',
    rename: dn
      ? 'This is the bill for the fast reads above. The name was copied into documents, blobs, comment cells and query tables, so one logical change is N physical writes with no transaction spanning them — and every missed copy is a permanent inconsistency nothing will detect.'
      : 'Normalized, the rename is one write everywhere, including Cassandra and Redis. The cost moved back to the read path, which is exactly the trade the switch controls.',
  };

  return (
    <VizPanel
      title="One dataset, five data models, one query"
      subtitle="Users, posts, comments, tags and follows, stored five ways at once. Pick a query and watch each model reach the answer — index descents, extra round trips, scatters, pointer hops and write fan-out."
      controls={
        <>
          <Choice label="Query" value={query} onChange={(v) => setQuery(v)} options={QUERIES} />
          <Check label="Denormalize on write (embed / blobs / query tables / index keys)" checked={dn} onChange={setDn} />
          <Slider label="Comments on p1" min={2} max={24} value={n} onChange={setN} format={(v) => `${v}`} />
          <Button onClick={() => setPlaying(!playing)} primary disabled={step >= maxLen && !playing}>
            {playing ? 'Pause' : 'Run'}
          </Button>
          <Button onClick={() => { setPlaying(false); setStep(Math.min(step + 1, maxLen)); }} disabled={step >= maxLen}>
            Step
          </Button>
          <Button onClick={() => { setPlaying(false); setStep(0); acc.current = 0; }}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            ...(Object.keys(KIND) as Kind[]).map((k) => ({ label: KIND[k].label, color: KIND[k].color })),
            { label: 'dashed — structure absent in this configuration', color: 'var(--viz-stale)', shape: 'line' as const },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Step', value: `${Math.min(step, maxLen)} / ${maxLen}` },
            { label: 'Fewest round trips', value: `${best.label} · ${best.rt}`, hint: 'Client↔server requests needed to answer the whole query' },
            { label: 'Most round trips', value: `${worst.label} · ${worst.rt}` },
            { label: 'Most bytes touched', value: `${fattest.label} · ${fmtBytes(fattest.bytes)}`, hint: 'Payload bytes of every unit read or written; real engines round this up to whole pages and blocks' },
            {
              label: query === 'rename' ? 'Writes, graph vs worst' : 'Index lookups, graph vs worst',
              value: query === 'rename'
                ? `1 vs ${fmtNum(full.reduce((a, b) => (b.items > a.items ? b : a)).items)}`
                : `${graphIdx} vs ${mostIdx.idx} (${mostIdx.label})`,
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{QUERIES.find((x) => x.value === query)!.label}.</strong> {notes[query]}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Storage unit</th>
                <th>Round trips</th>
                <th>Index lookups</th>
                <th>{query === 'rename' ? 'Units written' : 'Units read'}</th>
                <th>Bytes touched</th>
                <th>Where the work lands</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => {
                const f = full.find((x) => x.id === m.id)!;
                return (
                  <tr key={m.id}>
                    <td>{m.label}</td>
                    <td>{m.unit}</td>
                    <td>{f.rt}</td>
                    <td>{f.idx}</td>
                    <td>{fmtNum(f.items)}</td>
                    <td>{fmtBytes(f.bytes)}</td>
                    <td>{workSite(query, m.id, dn)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>#</th>
                <th>Operation</th>
                <th>Units</th>
                <th>Bytes</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.flatMap((m) =>
                byModel[m.id].map((s, i) => (
                  <tr key={`${m.id}-${i}`}>
                    <td>{m.label}</td>
                    <td>{i + 1}</td>
                    <td>{s.op}</td>
                    <td>{s.units.join(', ')}</td>
                    <td>{fmtBytes(s.bytes)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          <svg width={svgW} height={svgH} role="img" aria-label="The same query executed against five data models">
            {MODELS.map((m) => {
              const top = laneTop[m.id];
              const h = m.id === 'graph' ? GRAPH_H : LANE_H;
              const t = totals.find((x) => x.id === m.id)!;
              return (
                <g key={m.id}>
                  <rect x={0} y={top} width={svgW} height={h} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                  <text x={10} y={top + 17} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
                    {m.label}
                  </text>
                  <text x={10} y={top + 32} fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
                    {m.engine}
                  </text>
                  <text x={10} y={top + 50} fill="var(--viz-ink-2)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {t.rt} rt · {t.idx} idx
                  </text>
                  <text x={10} y={top + 64} fill="var(--viz-ink-2)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {fmtNum(t.items)} units · {fmtBytes(t.bytes)}
                  </text>
                  {m.id === 'graph' ? (
                    <text x={10} y={top + 82} fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
                      u=user p=post
                    </text>
                  ) : null}
                  {m.id === 'graph' ? (
                    <text x={10} y={top + 95} fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
                      c=comment tg=tag
                    </text>
                  ) : null}

                  {/* chip lanes */}
                  {m.id !== 'graph' &&
                    UNITS[m.id].map((u, i) => {
                      const x = LBL + i * (CW + CG);
                      const cy = top + 8;
                      const ghost = (u.ghost === 'dn' && dn) || (u.ghost === 'nodn' && !dn);
                      const hit = mark[`${m.id}:${u.id}`];
                      const stroke = hit ? KIND[hit.kind].color : ghost ? 'var(--viz-stale)' : 'var(--viz-border)';
                      const sub = u.subAlt && !dn ? u.subAlt : u.sub;
                      return (
                        <g
                          key={u.id}
                          {...tip(
                            <>
                              <strong>{u.label}</strong> — {sub}
                              {ghost ? '. This structure does not exist in the current configuration.' : ''}
                              {hit ? ` Touched at step ${hit.order}: ${KIND[hit.kind].label.slice(4)}.` : ''}
                            </>,
                          )}
                        >
                          <rect
                            x={x}
                            y={cy}
                            width={CW}
                            height={CH}
                            rx={6}
                            fill={ghost ? 'none' : 'var(--viz-neutral)'}
                            stroke={stroke}
                            strokeWidth={hit ? 2 : 1}
                            strokeDasharray={ghost ? '4 3' : undefined}
                          />
                          <text x={x + 7} y={cy + 15} fill={ghost ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}>
                            {trunc(u.label, 16)}
                          </text>
                          <text x={x + 7} y={cy + 28} fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
                            {trunc(sub, 17)}
                          </text>
                          {hit ? (
                            <>
                              <rect x={x + CW - 24} y={cy - 7} width={22} height={14} rx={4} fill="var(--viz-surface)" stroke={KIND[hit.kind].color} />
                              <text x={x + CW - 13} y={cy + 3} textAnchor="middle" fill="var(--viz-ink)" style={{ fontSize: 10, fontWeight: 600 }}>
                                {KIND[hit.kind].letter}
                                {hit.order}
                              </text>
                            </>
                          ) : null}
                        </g>
                      );
                    })}

                  {/* graph lane */}
                  {m.id === 'graph' ? (
                    <>
                      {GEDGES.map((e) => {
                        const hit = edgeMark[e.id];
                        const x1 = gx(e.a);
                        const y1 = gy(e.a);
                        const x2 = gx(e.b);
                        const y2 = gy(e.b);
                        const bulge = Math.min(e.arc === 'up' ? 76 : 52, Math.abs(x2 - x1) / 1.6);
                        const d =
                          e.arc === 'none'
                            ? `M ${x1} ${y1} L ${x2} ${y2}`
                            : e.arc === 'up'
                              ? `M ${x1} ${y1 - 16} Q ${(x1 + x2) / 2} ${y1 - 16 - bulge} ${x2} ${y2 - 16}`
                              : `M ${x1} ${y1 + 16} Q ${(x1 + x2) / 2} ${y1 + 16 + bulge} ${x2} ${y2 + 16}`;
                        return (
                          <path
                            key={e.id}
                            d={d}
                            fill="none"
                            stroke={hit ? KIND[hit.kind].color : 'var(--viz-axis)'}
                            strokeWidth={hit ? 2.5 : 1}
                            strokeDasharray={hit ? undefined : '3 3'}
                          />
                        );
                      })}
                      {GNODES.map((g) => {
                        const hit = mark[`graph:${g.id}`];
                        return (
                          <g
                            key={g.id}
                            {...tip(
                              <>
                                <strong>{g.label}</strong> — a node record with its first relationship id.
                                {hit ? ` Reached at step ${hit.order} by ${hit.kind === 'index' ? 'an index lookup' : hit.kind === 'write' ? 'a property write' : 'following a pointer'}.` : ''}
                              </>,
                            )}
                          >
                            <circle
                              cx={gx(g.id)}
                              cy={gy(g.id)}
                              r={15}
                              fill="var(--viz-neutral)"
                              stroke={hit ? KIND[hit.kind].color : 'var(--viz-border)'}
                              strokeWidth={hit ? 2.5 : 1}
                            />
                            <text x={gx(g.id)} y={gy(g.id) + 4} textAnchor="middle" fill="var(--viz-ink)">
                              {g.label}
                            </text>
                            {hit ? (
                              <text x={gx(g.id) + 17} y={gy(g.id) - 10} fill="var(--viz-ink)" style={{ fontSize: 10, fontWeight: 600 }}>
                                {KIND[hit.kind].letter}
                                {hit.order}
                              </text>
                            ) : null}
                          </g>
                        );
                      })}
                    </>
                  ) : null}

                  {/* the narration line for this lane */}
                  <text x={LBL} y={top + (m.id === 'graph' ? GRAPH_H - 10 : LANE_H - 10)} fill="var(--viz-ink-2)">
                    {t.last ? `${Math.min(step, byModel[m.id].length)}. ${t.last.op}` : byModel[m.id].length ? 'waiting — press Run' : '—'}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
