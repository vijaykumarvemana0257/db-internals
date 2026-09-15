import { useDeferredValue, useMemo, useState, type ReactElement } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Legend, Stats, Note, makeRng, fmtBytes, fmtNum } from './Viz';

/**
 * Choosing an index for semi-structured predicates, modelled on PostgreSQL 18 (sources: jsonb_gin.c,
 * contrib/pg_trgm, tsginidx.c, ginarrayproc.c, ginfast.c, ginpostinglist.c).
 *
 * What is real:
 * - jsonb_ops extracts one entry per key and per scalar value; string array elements are stored as KEY
 *   entries ("pretend string array elements are keys"), so `?` also matches them and always rechecks.
 *   `@>` ANDs every key and value of the query and always rechecks.
 * - jsonb_path_ops extracts one hash per scalar value, mixing in every key on the path to it; arrays add
 *   nothing to the path. No entry records that a key exists, so `?` cannot use it.
 * - pg_trgm lowercases, splits words on non-alphanumerics and pads each word with two spaces before and
 *   one after; a LIKE pattern pads only where no wildcard is adjacent, and a part under three characters
 *   yields no trigram, which forces a full index scan. GIN always rechecks trigram matches.
 * - gist_trgm_ops keeps each value's exact trigram array at the leaf and a 12-byte signature
 *   (bit = trigram % 95) on inner pages.
 * - array_ops: && and @> need no recheck. tsvector_ops: & needs none; <-> rechecks (no positions).
 * - Index expressions must match the query's expression; PostgreSQL does not substitute a generated
 *   column for its expression.
 * - GIN postings encode heap TIDs (block << 11 | offset) as varbyte deltas after a 6-byte first TID.
 * - The fastupdate pending list stores one index tuple per entry; an insert that pushes it past
 *   gin_pending_list_limit (pages x 8,160 bytes > limit) merges it in the foreground.
 *
 * Model assumptions (labelled in the UI):
 * - 10,000 synthetic rows, inserted in order into 8 KB heap pages; row size = text length + 28 bytes
 *   (jsonb's binary format is approximated by the JSON text).
 * - Index pages packed to 90%; GIN lookups read one entry-tree descent per query entry plus every page of a
 *   posting tree (an upper bound: GIN can skip inside long posting trees).
 * - Hashes are FNV-1a stand-ins, not PostgreSQL's hash_any, so entry counts are exact but hash values are not.
 * - Lexemes are lowercase words minus a subset of English stopwords, without Snowball stemming, so lexeme
 *   text differs from PostgreSQL's ('update', not 'updat'); no two body words share a stem, so counts hold.
 *   Phrase adjacency counts stopword positions, as to_tsvector does.
 * - The trigram index on body extracts from every 20th row and scales the size up.
 * - GiST leaves are filled in insertion order (real picksplit groups similar signatures).
 */

/* ------------------------------------------------------------------ constants */

export const N_ROWS = 10_000;
const HEAP_USABLE = 8192 - 24;
const INDEX_USABLE = 8192 - 40;
const PACK = 0.9;
const GIN_MAX_ITEM = 2712;
export const GIN_PAGE_FREESIZE = 8192 - 24 - 8;
const maxalign = (n: number) => Math.ceil(n / 8) * 8;

/* ---------------------------------------------------------------------- data */

export type JsonVal = null | boolean | number | string | JsonVal[] | { [k: string]: JsonVal };
export type JsonObj = { [k: string]: JsonVal };
export type Row = { data?: JsonObj; tags?: string[]; title?: string; body?: string; words?: string[]; bytes: number; page: number; tid: number };
export type ScenarioId = 'products' | 'articles';

const COLORS = ['black', 'white', 'navy', 'grey', 'blue', 'red', 'green', 'olive', 'tan', 'brown', 'beige', 'cream', 'pink', 'burgundy', 'teal', 'mustard', 'charcoal', 'khaki', 'rust', 'sage', 'lilac', 'ivory', 'denim', 'coral'];
const ATTR_KEYS: { key: string; values: string[] }[] = [
  { key: 'size', values: ['XS', 'S', 'M', 'L', 'XL', 'XXL'] },
  { key: 'trim', values: COLORS },
  { key: 'material', values: ['cotton', 'linen', 'wool', 'silk', 'denim', 'polyester', 'cashmere', 'viscose'] },
  { key: 'fit', values: ['slim', 'regular', 'relaxed', 'oversized'] },
  { key: 'lining', values: COLORS },
  { key: 'pattern', values: ['solid', 'striped', 'checked', 'floral', 'plain'] },
  { key: 'care', values: ['machine wash', 'hand wash', 'dry clean'] },
  { key: 'origin', values: ['PT', 'IT', 'VN', 'BD', 'TR', 'IN', 'CN', 'US', 'MX', 'RO'] },
  { key: 'season', values: ['SS25', 'AW25', 'SS26', 'AW26'] },
  { key: 'collar', values: ['button-down', 'spread', 'mandarin', 'crew', 'v-neck'] },
  { key: 'stitching', values: COLORS },
  { key: 'closure', values: ['button', 'zip', 'pullover', 'snap'] },
];
const PRODUCT_TAGS = ['new', 'sale', 'bestseller', 'organic', 'recycled', 'gift', 'limited', 'basics', 'petite', 'tall', 'vegan', 'exclusive', 'online-only', 'last-chance', 'members', 'bundle'];
const ADJ = ['classic', 'slim', 'relaxed', 'cropped', 'heavyweight', 'lightweight', 'washed', 'brushed', 'everyday', 'essential', 'premium', 'vintage', 'tailored', 'soft', 'boxy', 'ribbed', 'quilted', 'waffle'];
const NOUNS = ['shirt', 'tee', 'polo', 'chino', 'blazer', 'jacket', 'parka', 'hoodie', 'sweater', 'cardigan', 'jeans', 'shorts', 'skirt', 'dress', 'coat', 'vest', 'henley', 'jogger', 'overshirt', 'trouser', 'cord pant', 'windbreaker'];
const MATERIAL_WORDS = ['cotton', 'linen', 'wool', 'merino', 'denim', 'jersey', 'fleece', 'twill', 'poplin', 'corduroy', 'flannel', 'seersucker'];

const STOPWORDS = new Set(['the', 'a', 'an', 'of', 'to', 'and', 'in', 'is', 'it', 'for', 'on', 'with', 'when', 'that', 'this', 'by', 'as', 'be', 'are', 'from', 'at', 'or', 'not', 'we', 'you', 'can', 'but', 'if', 'then', 'so']);
const STOP_LIST = [...STOPWORDS];
const RARE_VOCAB = ['xid', 'wraparound', 'multixact', 'fsm', 'relfilenode', 'oid', 'catalog', 'syscache', 'relcache', 'lsn', 'timeline', 'archiver', 'walsender', 'walreceiver', 'bgwriter', 'checksum', 'sequence', 'domain', 'collation', 'icu', 'libpq', 'pgoutput', 'decoder', 'lwlock', 'spinlock', 'semaphore', 'hugepage', 'numa', 'io_uring', 'direct', 'fsync', 'crc', 'rewind', 'basebackup', 'manifest', 'pgaudit', 'rls', 'policy'];
const BODY_VOCAB = ['index', 'vacuum', 'heap', 'lock', 'wal', 'replica', 'checkpoint', 'planner', 'cluster', 'shard', 'latch', 'buffer', 'commit', 'snapshot', 'toast', 'btree', 'gin', 'gist', 'brin', 'hash', 'join', 'sort', 'scan', 'trigger', 'backup', 'parser', 'cursor', 'schema', 'column', 'row', 'bloat', 'fillfactor', 'pgbouncer', 'pooler', 'failover', 'standby', 'upgrade', 'partition', 'tablespace', 'extension', 'grant', 'role', 'jsonb', 'array', 'tsvector', 'trigram', 'posting', 'entry', 'pending', 'merge', 'insert', 'delete', 'update', 'select', 'explain', 'analyze', 'latency', 'throughput', 'memory', 'disk', 'cpu', 'kernel', 'page', 'block', 'tuple', 'visibility', 'map', 'horizon', 'slot', 'publication', 'subscriber', 'conflict', 'deadlock', 'timeout', 'retry', 'queue', 'worker', 'parallel', 'hot', 'chain', 'prune', 'freeze'];
const TITLE_VOCAB = ['postgres', 'tuning', 'guide', 'notes', 'deep', 'dive', 'lessons', 'from', 'production', 'debugging', 'slow', 'queries', 'scaling', 'writes', 'reads', 'migrating', 'to', 'understanding', 'internals', 'field', 'report', 'benchmarks', 'mistakes', 'we', 'made', 'with', 'indexes', 'replication', 'locks', 'storage', 'jsonb', 'search'];
const ARTICLE_TAGS = ['postgres', 'performance', 'indexing', 'operations', 'replication', 'backup', 'security', 'cloud', 'kubernetes', 'python', 'go', 'java', 'rust', 'analytics', 'migrations', 'monitoring', 'vacuum', 'jsonb', 'search', 'sharding'];

function weightedPick(rng: () => number, items: string[], rare: Record<string, number>) {
  let r = rng();
  for (const [item, p] of Object.entries(rare)) {
    if (r < p) return item;
    r -= p;
  }
  const rest = 1 - Object.values(rare).reduce((s, p) => s + p, 0);
  const common = items.filter((x) => !(x in rare));
  return common[Math.min(common.length - 1, Math.floor((r / rest) * common.length))];
}

/** A Zipf-ish pick: low indexes are much more frequent. */
function zipfPick<T>(rng: () => number, items: T[], s = 1.1) {
  const u = rng();
  const i = Math.floor(Math.pow(u, 1 + s) * items.length);
  return items[Math.min(items.length - 1, i)];
}

export const SKU_TARGET = 'KT-14242';
const skuFor = (i: number) => `${String.fromCharCode(65 + ((i * 7) % 26))}${String.fromCharCode(65 + ((i * 11) % 26))}-${10000 + i}`;

function placeRows(rows: Omit<Row, 'page' | 'tid'>[]): Row[] {
  let page = 0;
  let used = 0;
  let offset = 0;
  return rows.map((r) => {
    const sz = maxalign(24 + r.bytes) + 4;
    if (used + sz > HEAP_USABLE) {
      page++;
      used = 0;
      offset = 0;
    }
    used += sz;
    offset++;
    return { ...r, page, tid: page * 2048 + offset };
  });
}

export function makeProducts(attrKeys: number, tagsPer: number, n = N_ROWS): Row[] {
  const rng = makeRng(20260914);
  const out: Omit<Row, 'page' | 'tid'>[] = [];
  for (let i = 0; i < n; i++) {
    const attrs: JsonObj = { color: weightedPick(rng, COLORS, { coral: 0.004 }) };
    for (let k = 0; k < attrKeys; k++) {
      const a = ATTR_KEYS[k];
      attrs[a.key] = a.values === COLORS ? COLORS[Math.floor(rng() * COLORS.length)] : a.values[Math.floor(rng() * a.values.length)];
    }
    const tags = new Set<string>();
    for (let t = 0; t < tagsPer; t++) tags.add(weightedPick(rng, PRODUCT_TAGS, { clearance: 0.0015, discount: 0.0008 }));
    const material = MATERIAL_WORDS[Math.floor(rng() * MATERIAL_WORDS.length)];
    const noun = rng() < 0.004 ? 'oxford' : NOUNS[Math.floor(rng() * NOUNS.length)];
    const title = `${ADJ[Math.floor(rng() * ADJ.length)]} ${material} ${noun}`.replace(/^./, (c) => c.toUpperCase());
    const data: JsonObj = {
      sku: i === 4242 ? SKU_TARGET : skuFor(i),
      title,
      price: Math.round((9 + rng() * 190) * 100) / 100,
      attrs,
      tags: [...tags],
    };
    if (rng() < 0.005) data.discount = [10, 15, 20, 25][Math.floor(rng() * 4)];
    out.push({ data, bytes: JSON.stringify(data).length });
  }
  return placeRows(out);
}

export function makeArticles(tagsPer: number, bodyWords: number, n = N_ROWS): Row[] {
  const rng = makeRng(777);
  const out: Omit<Row, 'page' | 'tid'>[] = [];
  for (let i = 0; i < n; i++) {
    const tags = new Set<string>();
    for (let t = 0; t < tagsPer; t++) tags.add(weightedPick(rng, ARTICLE_TAGS, { sqlite: 0.0015, duckdb: 0.001, clickhouse: 0.001 }));
    const tw: string[] = [];
    const titleLen = 4 + Math.floor(rng() * 4);
    for (let w = 0; w < titleLen; w++) tw.push(rng() < 0.5 ? TITLE_VOCAB[Math.floor(rng() * TITLE_VOCAB.length)] : BODY_VOCAB[Math.floor(rng() * BODY_VOCAB.length)]);
    if (rng() < 0.004) tw.splice(1 + Math.floor(rng() * (tw.length - 1)), 0, 'autovacuum');
    const title = tw.join(' ').replace(/^./, (c) => c.toUpperCase());
    const tokens: string[] = [];
    for (let w = 0; w < bodyWords; w++) {
      const u = rng();
      if (u < 0.35) tokens.push(STOP_LIST[Math.floor(rng() * STOP_LIST.length)]);
      else if (u < 0.38) tokens.push(RARE_VOCAB[Math.floor(rng() * RARE_VOCAB.length)]);
      else tokens.push(zipfPick(rng, BODY_VOCAB, 1.6));
    }
    if (rng() < 0.005) tokens.splice(Math.floor(rng() * tokens.length), 0, 'xid', 'wraparound');
    const body = tokens.join(' ');
    const tagBytes = [...tags].reduce((s, t) => s + t.length + 4, 24);
    out.push({ tags: [...tags], title, body, words: tokens.filter((t) => !STOPWORDS.has(t)), bytes: title.length + body.length + tagBytes + 8 });
  }
  return placeRows(out);
}

/* --------------------------------------------------------------- extraction */

function fnv(s: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
const rotl1 = (h: number) => ((h << 1) | (h >>> 31)) >>> 0;

const scalarText = (v: JsonVal) => (v === null ? '' : typeof v === 'boolean' ? (v ? 't' : 'f') : String(v));
const valueFlag = (v: JsonVal) => (v === null ? 'null' : typeof v === 'boolean' ? 'bool' : typeof v === 'number' ? 'num' : 'str');

/** jsonb_ops entries, as "key:x" / "str:x" / "num:x"... (flag + text, like make_scalar_key). */
export function jsonbOpsEntries(v: JsonVal, out = new Set<string>()): Set<string> {
  if (Array.isArray(v)) {
    for (const e of v) {
      if (e !== null && typeof e === 'object') jsonbOpsEntries(e, out);
      else out.add(typeof e === 'string' ? `key:${e}` : `${valueFlag(e)}:${scalarText(e)}`);
    }
  } else if (v !== null && typeof v === 'object') {
    for (const [k, e] of Object.entries(v)) {
      out.add(`key:${k}`);
      if (e !== null && typeof e === 'object') jsonbOpsEntries(e, out);
      else out.add(`${valueFlag(e)}:${scalarText(e)}`);
    }
  }
  return out;
}

/** jsonb_path_ops entries: one hash per scalar, mixing every key on its path (arrays add nothing). */
export function jsonbPathEntries(v: JsonVal, hash = 0, path = '', out = new Map<number, string>()): Map<number, string> {
  if (Array.isArray(v)) {
    for (const e of v) {
      if (e !== null && typeof e === 'object') jsonbPathEntries(e, hash, `${path}[]`, out);
      else {
        const h = (rotl1(hash) ^ fnv(`${valueFlag(e)}${scalarText(e)}`)) >>> 0;
        out.set(h, `${path}[]=${JSON.stringify(e)}`);
      }
    }
  } else if (v !== null && typeof v === 'object') {
    for (const [k, e] of Object.entries(v)) {
      const hk = (rotl1(hash) ^ fnv(`key${k}`)) >>> 0;
      const p = path ? `${path}.${k}` : k;
      if (e !== null && typeof e === 'object') jsonbPathEntries(e, hk, p, out);
      else out.set((rotl1(hk) ^ fnv(`${valueFlag(e)}${scalarText(e)}`)) >>> 0, `${p}=${JSON.stringify(e)}`);
    }
  }
  return out;
}

const isWordChar = (c: string) => /[a-z0-9]/i.test(c);

/** pg_trgm generate_trgm: lowercase, words of alphanumerics, "  word " padding, unique trigrams. */
export function trigrams(s: string): Set<string> {
  const out = new Set<string>();
  for (const w of s.toLowerCase().split(/[^a-z0-9]+/)) {
    if (!w) continue;
    const p = `  ${w} `;
    for (let i = 0; i + 3 <= p.length; i++) out.add(p.slice(i, i + 3));
  }
  return out;
}

/** pg_trgm generate_wildcard_trgm: trigrams every string matching a LIKE pattern must contain. */
export function wildcardTrigrams(pattern: string): Set<string> {
  const out = new Set<string>();
  const s = pattern;
  let i = 0;
  while (i < s.length) {
    let leadingWild = false;
    while (i < s.length && !isWordChar(s[i])) {
      leadingWild = s[i] === '%' || s[i] === '_' ? true : false;
      i++;
    }
    if (i >= s.length) break;
    let w = '';
    while (i < s.length && isWordChar(s[i])) w += s[i++];
    const trailingWild = i < s.length && (s[i] === '%' || s[i] === '_');
    const p = `${leadingWild ? '' : '  '}${w.toLowerCase()}${trailingWild ? '' : ' '}`;
    for (let j = 0; j + 3 <= p.length; j++) out.add(p.slice(j, j + 3));
  }
  return out;
}

export function lexemes(body: string): Set<string> {
  const out = new Set<string>();
  for (const w of body.toLowerCase().split(/[^a-z0-9_]+/)) if (w && !STOPWORDS.has(w)) out.add(w);
  return out;
}

/** gist_trgm_ops signature bit for one trigram (CPTRGM into an int, then % (siglen*8-1)). */
export const trgmBit = (t: string, siglen: number) => (t.charCodeAt(0) + (t.charCodeAt(1) << 8) + (t.charCodeAt(2) << 16)) % (siglen * 8 - 1);

/* ---------------------------------------------------------------- predicates */

export type PredKind =
  | { k: 'contains'; query: JsonObj }
  | { k: 'exists'; key: string }
  | { k: 'tags-exists'; key: string }
  | { k: 'text-eq'; expr: 'sku' | 'color'; value: string }
  | { k: 'col-eq'; value: string }
  | { k: 'ilike'; col: 'title'; pattern: string }
  | { k: 'arr-contains'; values: string[] }
  | { k: 'arr-any'; value: string }
  | { k: 'arr-overlap'; values: string[] }
  | { k: 'ts'; form: 'expr' | 'expr1' | 'col'; op: 'and' | 'phrase'; words: [string, string] };

export type Pred = { id: string; sql: string; kind: PredKind };
export type Slot = { label: string; options: Pred[] };

export function jsonContains(doc: JsonVal, q: JsonVal): boolean {
  if (q === null || typeof q !== 'object') return doc === q;
  if (Array.isArray(q)) {
    if (!Array.isArray(doc)) return false;
    return q.every((qe) => doc.some((de) => jsonContains(de, qe)));
  }
  if (doc === null || typeof doc !== 'object' || Array.isArray(doc)) return false;
  return Object.entries(q).every(([k, qv]) => k in doc && jsonContains(doc[k], qv));
}

function likeRegex(pattern: string) {
  const esc = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/%/g, '.*').replace(/_/g, '.');
  return new RegExp(`^${esc}$`, 'is');
}

export function truth(p: PredKind, r: Row): boolean {
  const d = r.data;
  switch (p.k) {
    case 'contains':
      return !!d && jsonContains(d, p.query);
    case 'exists':
      return !!d && p.key in d;
    case 'tags-exists':
      return !!d && Array.isArray(d.tags) && d.tags.includes(p.key);
    case 'text-eq':
      if (!d) return false;
      if (p.expr === 'sku') return d.sku === p.value;
      return !!d.attrs && typeof d.attrs === 'object' && !Array.isArray(d.attrs) && (d.attrs as JsonObj).color === p.value;
    case 'col-eq':
      return !!d && d.sku === p.value;
    case 'ilike': {
      const t = d ? String(d.title ?? '') : r.title ?? '';
      return likeRegex(p.pattern).test(t);
    }
    case 'arr-contains':
      return !!r.tags && p.values.every((v) => r.tags!.includes(v));
    case 'arr-any':
      return !!r.tags && r.tags.includes(p.value);
    case 'arr-overlap':
      return !!r.tags && p.values.some((v) => r.tags!.includes(v));
    case 'ts': {
      const words = r.words ?? [];
      if (p.op === 'and') return words.includes(p.words[0]) && words.includes(p.words[1]);
      // <-> needs consecutive positions, and a dropped stopword still takes a position
      const toks = (r.body ?? '').split(' ');
      for (let i = 0; i + 1 < toks.length; i++) if (toks[i] === p.words[0] && toks[i + 1] === p.words[1]) return true;
      return false;
    }
  }
}

export const SLOTS: Record<ScenarioId, Slot[]> = {
  products: [
    {
      label: 'Q1 nested attribute',
      options: [
        { id: 'attr-contains', sql: `data @> '{"attrs": {"color": "coral"}}'`, kind: { k: 'contains', query: { attrs: { color: 'coral' } } } },
        { id: 'attr-eq', sql: `data->'attrs'->>'color' = 'coral'`, kind: { k: 'text-eq', expr: 'color', value: 'coral' } },
      ],
    },
    {
      label: 'Q2 tag or flag',
      options: [
        { id: 'tag-contains', sql: `data @> '{"tags": ["clearance"]}'`, kind: { k: 'contains', query: { tags: ['clearance'] } } },
        { id: 'tag-exists', sql: `data->'tags' ? 'clearance'`, kind: { k: 'tags-exists', key: 'clearance' } },
        { id: 'discount-exists', sql: `data ? 'discount'`, kind: { k: 'exists', key: 'discount' } },
      ],
    },
    {
      label: 'Q3 SKU lookup',
      options: [
        { id: 'sku-expr', sql: `data->>'sku' = '${SKU_TARGET}'`, kind: { k: 'text-eq', expr: 'sku', value: SKU_TARGET } },
        { id: 'sku-col', sql: `sku = '${SKU_TARGET}'`, kind: { k: 'col-eq', value: SKU_TARGET } },
        { id: 'sku-contains', sql: `data @> '{"sku": "${SKU_TARGET}"}'`, kind: { k: 'contains', query: { sku: SKU_TARGET } } },
      ],
    },
    {
      label: 'Q4 title search',
      options: [
        { id: 'title-oxford', sql: `data->>'title' ILIKE '%oxford%'`, kind: { k: 'ilike', col: 'title', pattern: '%oxford%' } },
        { id: 'title-ox', sql: `data->>'title' ILIKE '%ox%'`, kind: { k: 'ilike', col: 'title', pattern: '%ox%' } },
      ],
    },
  ],
  articles: [
    {
      label: 'Q1 one tag',
      options: [
        { id: 'arr-contains', sql: `tags @> ARRAY['sqlite']`, kind: { k: 'arr-contains', values: ['sqlite'] } },
        { id: 'arr-any', sql: `'sqlite' = ANY (tags)`, kind: { k: 'arr-any', value: 'sqlite' } },
      ],
    },
    {
      label: 'Q2 any of two tags',
      options: [{ id: 'arr-overlap', sql: `tags && ARRAY['duckdb', 'clickhouse']`, kind: { k: 'arr-overlap', values: ['duckdb', 'clickhouse'] } }],
    },
    {
      label: 'Q3 full text',
      options: [
        { id: 'ts-expr', sql: `to_tsvector('english', body) @@ to_tsquery('english', 'xid & wraparound')`, kind: { k: 'ts', form: 'expr', op: 'and', words: ['xid', 'wraparound'] } },
        { id: 'ts-phrase', sql: `to_tsvector('english', body) @@ to_tsquery('english', 'xid <-> wraparound')`, kind: { k: 'ts', form: 'expr', op: 'phrase', words: ['xid', 'wraparound'] } },
        { id: 'ts-expr1', sql: `to_tsvector(body) @@ to_tsquery('xid & wraparound')`, kind: { k: 'ts', form: 'expr1', op: 'and', words: ['xid', 'wraparound'] } },
        { id: 'ts-col', sql: `body_tsv @@ to_tsquery('english', 'xid & wraparound')`, kind: { k: 'ts', form: 'col', op: 'and', words: ['xid', 'wraparound'] } },
      ],
    },
    {
      label: 'Q4 title search',
      options: [
        { id: 'title-autovac', sql: `title ILIKE '%autovacuum%'`, kind: { k: 'ilike', col: 'title', pattern: '%autovacuum%' } },
        { id: 'title-db', sql: `title ILIKE '%db%'`, kind: { k: 'ilike', col: 'title', pattern: '%db%' } },
      ],
    },
  ],
};

/* ---------------------------------------------------------------- candidates */

export type CandKind = 'gin-jsonb' | 'gin-jsonb-tags' | 'gin-path' | 'btree-sku' | 'gencol-sku' | 'btree-color' | 'gin-trgm' | 'gist-trgm' | 'gin-array' | 'btree-array' | 'gin-tsv' | 'gencol-tsv' | 'gin-trgm-body';
export type Cand = { id: CandKind; family: 'GIN' | 'B-tree' | 'GiST'; sql: string; short: string };

export const CANDIDATES: Record<ScenarioId, Cand[]> = {
  products: [
    { id: 'gin-jsonb', family: 'GIN', short: 'GIN jsonb_ops', sql: 'CREATE INDEX ON products USING gin (data)' },
    { id: 'gin-path', family: 'GIN', short: 'GIN jsonb_path_ops', sql: 'CREATE INDEX ON products USING gin (data jsonb_path_ops)' },
    { id: 'gin-jsonb-tags', family: 'GIN', short: "GIN on data->'tags'", sql: "CREATE INDEX ON products USING gin ((data->'tags'))" },
    { id: 'btree-sku', family: 'B-tree', short: "B-tree on data->>'sku'", sql: "CREATE INDEX ON products ((data->>'sku'))" },
    { id: 'gencol-sku', family: 'B-tree', short: 'generated column sku', sql: "ALTER TABLE products ADD sku text GENERATED ALWAYS AS (data->>'sku') STORED; CREATE INDEX ON products (sku)" },
    { id: 'btree-color', family: 'B-tree', short: "B-tree on attrs->>'color'", sql: "CREATE INDEX ON products ((data->'attrs'->>'color'))" },
    { id: 'gin-trgm', family: 'GIN', short: 'GIN pg_trgm on title', sql: "CREATE INDEX ON products USING gin ((data->>'title') gin_trgm_ops)" },
    { id: 'gist-trgm', family: 'GiST', short: 'GiST pg_trgm on title', sql: "CREATE INDEX ON products USING gist ((data->>'title') gist_trgm_ops)" },
  ],
  articles: [
    { id: 'gin-array', family: 'GIN', short: 'GIN array_ops on tags', sql: 'CREATE INDEX ON articles USING gin (tags)' },
    { id: 'btree-array', family: 'B-tree', short: 'B-tree on tags', sql: 'CREATE INDEX ON articles (tags)' },
    { id: 'gin-tsv', family: 'GIN', short: 'GIN on to_tsvector', sql: "CREATE INDEX ON articles USING gin (to_tsvector('english', body))" },
    { id: 'gencol-tsv', family: 'GIN', short: 'generated tsvector column', sql: "ALTER TABLE articles ADD body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED; CREATE INDEX ON articles USING gin (body_tsv)" },
    { id: 'gin-trgm', family: 'GIN', short: 'GIN pg_trgm on title', sql: 'CREATE INDEX ON articles USING gin (title gin_trgm_ops)' },
    { id: 'gist-trgm', family: 'GiST', short: 'GiST pg_trgm on title', sql: 'CREATE INDEX ON articles USING gist (title gist_trgm_ops)' },
    { id: 'gin-trgm-body', family: 'GIN', short: 'GIN pg_trgm on body', sql: 'CREATE INDEX ON articles USING gin (body gin_trgm_ops)' },
  ],
};

export type Serve = { status: 'serves' | 'no' | 'full'; reason: string; entries: string[]; mode: 'and' | 'or'; recheck: boolean };

const no = (reason: string): Serve => ({ status: 'no', reason, entries: [], mode: 'and', recheck: false });

/** Can this index answer this predicate as written, and with which index entries? */
export function serve(c: CandKind, p: PredKind): Serve {
  switch (c) {
    case 'gin-jsonb':
      if (p.k === 'contains') {
        const e = [...jsonbOpsEntries(p.query)];
        return { status: 'serves', entries: e, mode: 'and', recheck: true, reason: `@> is a jsonb_ops operator: the query's keys and values become ${e.length} entries, ANDed, and every candidate is rechecked because the entries do not record where in the document they sat.` };
      }
      if (p.k === 'exists') return { status: 'serves', entries: [`key:${p.key}`], mode: 'and', recheck: true, reason: `? looks up the single key entry "${p.key}". Recheck is always on: the same entry is written for nested keys and for string array elements.` };
      if (p.k === 'tags-exists') return no(`The ? is applied to data->'tags', not to the indexed column data. Rewrite it as data @> '{"tags": ["${p.key}"]}' or index the expression.`);
      if (p.k === 'text-eq') return no(`->> returns text, and text = is not a jsonb_ops operator. Rewrite it as containment, or build a B-tree on the expression.`);
      if (p.k === 'col-eq') return no('The query compares the column sku, which this index does not contain.');
      return no('Not an operator on data.');
    case 'gin-path':
      if (p.k === 'contains') {
        const e = [...jsonbPathEntries(p.query).values()].map((x) => `hash(${x})`);
        return { status: 'serves', entries: e, mode: 'and', recheck: true, reason: `Each scalar in the query becomes one hash of its value and the keys above it: ${e.length} entr${e.length === 1 ? 'y' : 'ies'}. Always rechecked (hash collisions).` };
      }
      if (p.k === 'exists') return no('jsonb_path_ops stores only hashes of paths to values. No entry says "this key exists", so ? is not in the operator class.');
      if (p.k === 'tags-exists') return no(`The ? is applied to data->'tags', and jsonb_path_ops does not support ? at all.`);
      if (p.k === 'text-eq') return no('->> = compares text; the operator class only has @>, @? and @@.');
      if (p.k === 'col-eq') return no('The query compares the column sku, which this index does not contain.');
      return no('Not an operator on data.');
    case 'gin-jsonb-tags':
      if (p.k === 'tags-exists') return { status: 'serves', entries: [`key:${p.key}`], mode: 'and', recheck: true, reason: `The query applies ? to exactly the indexed expression data->'tags'; one key entry, rechecked.` };
      if (p.k === 'contains') return no(`The index is on data->'tags', but the query applies @> to data. Operators must be applied to the indexed expression itself.`);
      return no(`Only operators applied to data->'tags' can use this index.`);
    case 'btree-sku':
      if (p.k === 'text-eq' && p.expr === 'sku') return { status: 'serves', entries: [`'${p.value}'`], mode: 'and', recheck: false, reason: `The query repeats the indexed expression data->>'sku' with =, so it is one B-tree descent.` };
      if (p.k === 'col-eq') return no(`The query names the stored generated column sku. PostgreSQL does not match that column reference to an index on data->>'sku'.`);
      if (p.k === 'contains') return no('@> is not a B-tree operator.');
      return no(`The index holds only data->>'sku'.`);
    case 'gencol-sku':
      if (p.k === 'col-eq') return { status: 'serves', entries: [`'${p.value}'`], mode: 'and', recheck: false, reason: 'The query names the stored generated column, which has a plain B-tree.' };
      if (p.k === 'text-eq' && p.expr === 'sku') return no(`The query spells out data->>'sku'. PostgreSQL does not substitute the generated column for its expression (MySQL's optimizer does), so rewrite the query to use sku.`);
      if (p.k === 'contains') return no('@> is not a B-tree operator.');
      return no('The index holds only the sku column.');
    case 'btree-color':
      if (p.k === 'text-eq' && p.expr === 'color') return { status: 'serves', entries: [`'${p.value}'`], mode: 'and', recheck: false, reason: `The query repeats data->'attrs'->>'color' exactly: one B-tree range of equal keys.` };
      if (p.k === 'contains') return no('@> is not a B-tree operator, even when it asks for the same value.');
      return no(`The index holds only data->'attrs'->>'color'.`);
    case 'gin-trgm':
    case 'gist-trgm':
      if (p.k === 'ilike') {
        const t = [...wildcardTrigrams(p.pattern)];
        if (t.length === 0) return { status: 'full', entries: [], mode: 'and', recheck: true, reason: `'${p.pattern}' has no trigram: every word between wildcards is shorter than three characters, so the scan must read the whole index.` };
        return { status: 'serves', entries: t.map((x) => `"${x}"`), mode: 'and', recheck: true, reason: `The pattern yields ${t.length} trigrams every match must contain${c === 'gist-trgm' ? '; inner pages are tested against signature bits, leaves against each title’s exact trigram set' : ', ANDed'}. Always rechecked: trigrams do not record order.` };
      }
      if (p.k === 'ts') return no('pg_trgm serves LIKE, ILIKE, regex and similarity, not @@.');
      return no('Not a predicate on the title.');
    case 'gin-trgm-body':
      if (p.k === 'ts') return no('A trigram index on body serves body LIKE and ILIKE, not @@: it holds trigrams, not lexemes.');
      return no('None of the queries searches body with LIKE.');
    case 'gin-array':
      if (p.k === 'arr-contains') return { status: 'serves', entries: p.values.map((v) => `'${v}'`), mode: 'and', recheck: false, reason: '@> between arrays is an array_ops operator; each element is one entry. No recheck.' };
      if (p.k === 'arr-overlap') return { status: 'serves', entries: p.values.map((v) => `'${v}'`), mode: 'or', recheck: false, reason: '&& is an array_ops operator: the posting lists are unioned. No recheck.' };
      if (p.k === 'arr-any') return no(`= ANY applies text = to each element. array_ops only knows &&, @>, <@ and = between arrays: write tags @> ARRAY['${p.value}'].`);
      return no('Not a predicate on tags.');
    case 'btree-array':
      if (p.k === 'arr-contains' || p.k === 'arr-overlap' || p.k === 'arr-any') return no('A B-tree orders whole arrays (=, <, >). It cannot look inside one.');
      return no('Not a predicate on tags.');
    case 'gin-tsv':
      if (p.k === 'ts') {
        if (p.form === 'expr1') return no(`to_tsvector(body) without a configuration name is a different expression from to_tsvector('english', body), so the index does not match.`);
        if (p.form === 'col') return no(`The query names body_tsv; the index is on the expression to_tsvector('english', body).`);
        return { status: 'serves', entries: p.words.map((w) => `'${w}'`), mode: 'and', recheck: p.op === 'phrase', reason: p.op === 'phrase' ? 'Two lexeme entries, ANDed. GIN stores no positions, so <-> is rechecked against the row.' : 'Two lexeme entries, ANDed. Plain & needs no recheck.' };
      }
      return no('Not a text search predicate.');
    case 'gencol-tsv':
      if (p.k === 'ts') {
        if (p.form !== 'col') return no(`The query computes to_tsvector itself; this index is on the stored column body_tsv. Rewrite the query to use body_tsv.`);
        return { status: 'serves', entries: p.words.map((w) => `'${w}'`), mode: 'and', recheck: p.op === 'phrase', reason: 'The query names body_tsv: two lexeme entries, ANDed, no recheck for &.' };
      }
      return no('Not a text search predicate.');
  }
}

/* ------------------------------------------------------------- index builds */

const varbyteLen = (x: number) => (x < 128 ? 1 : x < 16384 ? 2 : x < 2097152 ? 3 : x < 268435456 ? 4 : 5);

/** Bytes of a GIN posting list for sorted TIDs: 6-byte first item + 2-byte length, then varbyte deltas. */
function postingBytes(tids: number[]) {
  let b = 8;
  for (let i = 1; i < tids.length; i++) b += varbyteLen(tids[i] - tids[i - 1]);
  return b;
}

const textKey = (s: string) => 1 + s.length;

export type IndexStats = {
  kind: 'gin' | 'btree' | 'gist';
  entriesPerRow: number;
  pendingBytesPerRow: number;
  bytes: number;
  pages: number;
  height: number;
  extraHeapBytesPerRow: number;
  postings?: Map<string, number[]>;
  postingPages?: Map<string, number>;
  leafTuplesPerPage?: number;
  gist?: { siglen: number; leafPages: { rows: number[]; sig: Uint8Array }[]; parents: { children: number[]; sig: Uint8Array }[]; trgm: Set<string>[] };
  sampleEntries: string[];
};

function treeHeight(leafPages: number, fanout: number) {
  let h = 1;
  let n = leafPages;
  let internal = 0;
  while (n > 1) {
    n = Math.ceil(n / fanout);
    internal += n;
    h++;
  }
  return { h, internal };
}

function ginStats(rows: Row[], extract: (r: Row) => string[], keyBytes: (e: string) => number, sampleEvery = 1): IndexStats {
  const postings = new Map<string, number[]>();
  let entries = 0;
  let pendBytes = 0;
  let sampled = 0;
  for (let i = 0; i < rows.length; i += sampleEvery) {
    const es = extract(rows[i]);
    sampled++;
    entries += es.length;
    for (const e of es) {
      pendBytes += maxalign(8 + keyBytes(e)) + 4;
      let p = postings.get(e);
      if (!p) postings.set(e, (p = []));
      p.push(rows[i].tid);
    }
  }
  const scale = rows.length / sampled;
  let leafBytes = 0;
  let postingTreePages = 0;
  const postingPages = new Map<string, number>();
  let keyTotal = 0;
  for (const [e, tids] of postings) {
    const kb = keyBytes(e);
    keyTotal += kb;
    const pb = postingBytes(tids);
    const inline = maxalign(8 + kb + pb);
    if (inline <= GIN_MAX_ITEM) {
      leafBytes += inline + 4;
    } else {
      leafBytes += maxalign(8 + kb + 4) + 4;
      const pages = Math.ceil((pb + 8 * Math.ceil(pb / 256)) / (INDEX_USABLE * PACK));
      postingTreePages += pages;
      postingPages.set(e, pages);
    }
  }
  const leafPages = Math.max(1, Math.ceil((leafBytes * scale) / (INDEX_USABLE * PACK)));
  const avgKey = postings.size ? keyTotal / postings.size : 4;
  const fanout = Math.max(2, Math.floor((INDEX_USABLE * PACK) / (maxalign(8 + avgKey) + 4)));
  const { h, internal } = treeHeight(leafPages, fanout);
  const pages = 1 + leafPages + internal + Math.round(postingTreePages * scale);
  const first = extract(rows[0]);
  return { kind: 'gin', entriesPerRow: entries / sampled, pendingBytesPerRow: pendBytes / sampled, bytes: pages * 8192, pages, height: h, extraHeapBytesPerRow: 0, postings: sampleEvery === 1 ? postings : undefined, postingPages, sampleEntries: first };
}

function btreeStats(rows: Row[], keyBytes: (r: Row) => number, sample: string, extraHeap = 0): IndexStats {
  let bytes = 0;
  for (const r of rows) bytes += maxalign(8 + keyBytes(r)) + 4;
  const avg = bytes / rows.length;
  const perPage = Math.floor((INDEX_USABLE * PACK) / avg);
  const leafPages = Math.max(1, Math.ceil(rows.length / perPage));
  const fanout = Math.max(2, Math.floor((INDEX_USABLE * 0.7) / avg));
  const { h, internal } = treeHeight(leafPages, fanout);
  const pages = 1 + leafPages + internal;
  return { kind: 'btree', entriesPerRow: 1, pendingBytesPerRow: 0, bytes: pages * 8192, pages, height: h, extraHeapBytesPerRow: extraHeap, leafTuplesPerPage: perPage, sampleEntries: [sample] };
}

function gistTrgmStats(rows: Row[], text: (r: Row) => string, siglen: number): IndexStats {
  const SIGBITS = siglen * 8 - 1;
  const trgm = rows.map((r) => trigrams(text(r)));
  const leafPages: { rows: number[]; sig: Uint8Array }[] = [];
  let cur: { rows: number[]; sig: Uint8Array } = { rows: [], sig: new Uint8Array(SIGBITS) };
  let used = 0;
  rows.forEach((_, i) => {
    const sz = maxalign(8 + 5 + 3 * trgm[i].size) + 4;
    if (used + sz > INDEX_USABLE * PACK && cur.rows.length) {
      leafPages.push(cur);
      cur = { rows: [], sig: new Uint8Array(SIGBITS) };
      used = 0;
    }
    used += sz;
    cur.rows.push(i);
    for (const t of trgm[i]) cur.sig[trgmBit(t, siglen)] = 1;
  });
  leafPages.push(cur);
  const innerTuple = maxalign(8 + 5 + siglen) + 4;
  const fanout = Math.floor((INDEX_USABLE * PACK) / innerTuple);
  const parents: { children: number[]; sig: Uint8Array }[] = [];
  for (let i = 0; i < leafPages.length; i += fanout) {
    const sig = new Uint8Array(SIGBITS);
    const children: number[] = [];
    for (let j = i; j < Math.min(leafPages.length, i + fanout); j++) {
      children.push(j);
      leafPages[j].sig.forEach((b, k) => {
        if (b) sig[k] = 1;
      });
    }
    parents.push({ children, sig });
  }
  const internalPages = parents.length > 1 ? parents.length + 1 : 1;
  const pages = leafPages.length + internalPages;
  const first = [...trgm[0]].map((t) => `"${t}"`);
  return { kind: 'gist', entriesPerRow: 1, pendingBytesPerRow: 0, bytes: pages * 8192, pages, height: parents.length > 1 ? 3 : 2, extraHeapBytesPerRow: 0, gist: { siglen, leafPages, parents, trgm }, sampleEntries: [`{${first.join(', ')}}`] };
}

const quoted = (s: IndexStats) => ({ ...s, sampleEntries: s.sampleEntries.map((e) => `'${e}'`) });

export function buildIndex(c: CandKind, rows: Row[], siglen = 12, prior?: IndexStats): IndexStats {
  switch (c) {
    case 'gin-jsonb':
      return ginStats(rows, (r) => [...jsonbOpsEntries(r.data ?? {})], (e) => {
        const len = e.length - e.indexOf(':') - 1;
        return 2 + (len > 125 ? 8 : len);
      });
    case 'gin-path': {
      const s = ginStats(rows, (r) => [...jsonbPathEntries(r.data ?? {}).keys()].map(String), () => 4);
      s.sampleEntries = [...jsonbPathEntries(rows[0].data ?? {}).values()].map((x) => `hash(${x})`);
      return s;
    }
    case 'gin-jsonb-tags':
      return ginStats(rows, (r) => [...jsonbOpsEntries((r.data?.tags as JsonVal) ?? [])], (e) => 2 + e.length - e.indexOf(':') - 1);
    case 'btree-sku':
      return btreeStats(rows, (r) => textKey(String(r.data?.sku ?? '')), `'${rows[0].data?.sku}'`);
    case 'gencol-sku':
      return btreeStats(rows, (r) => textKey(String(r.data?.sku ?? '')), `'${rows[0].data?.sku}'`, textKey(String(rows[0].data?.sku ?? '')));
    case 'btree-color':
      return btreeStats(rows, (r) => textKey(String((r.data?.attrs as JsonObj | undefined)?.color ?? '')), `'${(rows[0].data?.attrs as JsonObj).color}'`);
    case 'gin-trgm': {
      const s = ginStats(rows, (r) => [...trigrams(r.data ? String(r.data.title ?? '') : r.title ?? '')], () => 4);
      s.sampleEntries = s.sampleEntries.map((t) => `"${t}"`);
      return s;
    }
    case 'gist-trgm':
      return gistTrgmStats(rows, (r) => (r.data ? String(r.data.title ?? '') : r.title ?? ''), siglen);
    case 'gin-trgm-body': {
      const s = ginStats(rows, (r) => [...trigrams(r.body ?? '')], () => 4, 20);
      s.sampleEntries = s.sampleEntries.map((t) => `"${t}"`);
      return s;
    }
    case 'gin-array': {
      const s = ginStats(rows, (r) => [...new Set(r.tags ?? [])], (e) => textKey(e));
      s.sampleEntries = s.sampleEntries.map((e) => `'${e}'`);
      return s;
    }
    case 'btree-array': {
      const s = btreeStats(rows, (r) => 4 + 20 + (r.tags ?? []).reduce((a, t) => a + Math.ceil((4 + t.length) / 4) * 4, 0), `{${(rows[0].tags ?? []).join(',')}}`);
      return s;
    }
    case 'gin-tsv':
      return quoted(ginStats(rows, (r) => [...new Set(r.words ?? [])], (e) => textKey(e)));
    case 'gencol-tsv': {
      const s = { ...(prior ?? ginStats(rows, (r) => [...new Set(r.words ?? [])], (e) => textKey(e))) };
      let extra = 0;
      for (const r of rows) {
        const counts = new Map<string, number>();
        for (const w of r.words ?? []) counts.set(w, (counts.get(w) ?? 0) + 1);
        // tsvector (ts_type.h): varlena header + int32 count, a 4-byte WordEntry per lexeme, the lexeme bytes,
        // a uint16 position count and a uint16 per position (alignment padding ignored)
        let b = 8;
        for (const [l, n] of counts) b += 4 + l.length + 2 + 2 * n;
        extra += b;
      }
      s.extraHeapBytesPerRow = extra / rows.length;
      return quoted(s);
    }
  }
}

/* ------------------------------------------------------------------- lookup */

export type Lookup = {
  cand: CandKind | 'seq';
  serve: Serve;
  postingRows: { entry: string; rows: number }[];
  candidates: number;
  matches: number;
  indexPages: number;
  heapPages: number;
  pages: number;
  seqPages: number;
  served: boolean;
};

function intersect(a: number[], b: number[]) {
  const out: number[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      out.push(a[i]);
      i++;
      j++;
    } else if (a[i] < b[j]) i++;
    else j++;
  }
  return out;
}
function union(a: number[], b: number[]) {
  return [...new Set([...a, ...b])].sort((x, y) => x - y);
}

export type Corpus = { rows: Row[]; byTid: Map<number, number>; heapPages: number; indexes: Map<CandKind, IndexStats> };

export function makeCorpus(scenario: ScenarioId, a: number, b: number, siglen = 12): Corpus {
  const rows = scenario === 'products' ? makeProducts(a, b) : makeArticles(a, b);
  const byTid = new Map<number, number>();
  rows.forEach((r, i) => byTid.set(r.tid, i));
  const indexes = new Map<CandKind, IndexStats>();
  for (const c of CANDIDATES[scenario]) {
    // the stored tsvector column indexes exactly the lexemes the expression index does
    const prior = c.id === 'gencol-tsv' ? indexes.get('gin-tsv') : undefined;
    indexes.set(c.id, buildIndex(c.id, rows, siglen, prior ? { ...prior, sampleEntries: prior.sampleEntries.map((e) => e.slice(1, -1)) } : undefined));
  }
  return { rows, byTid, heapPages: rows[rows.length - 1].page + 1, indexes };
}

const truthCache = new WeakMap<Corpus, Map<string, number[]>>();
function truthFor(corpus: Corpus, p: PredKind) {
  let m = truthCache.get(corpus);
  if (!m) truthCache.set(corpus, (m = new Map()));
  const key = JSON.stringify(p);
  let t = m.get(key);
  if (!t) {
    t = [];
    corpus.rows.forEach((r, i) => {
      if (truth(p, r)) t!.push(i);
    });
    m.set(key, t);
  }
  return t;
}

const distinctPages = (rows: Row[], idx: number[]) => new Set(idx.map((i) => rows[i].page)).size;

export function lookup(corpus: Corpus, c: CandKind | 'seq', p: PredKind, budget: number): Lookup {
  const { rows, heapPages } = corpus;
  const truthRows = truthFor(corpus, p);
  const seq: Lookup = { cand: 'seq', serve: no('Sequential scan: every heap page is read and every row tested.'), postingRows: [], candidates: rows.length, matches: truthRows.length, indexPages: 0, heapPages, pages: heapPages, seqPages: heapPages, served: false };
  if (c === 'seq') return seq;
  const s = serve(c, p);
  const ix = corpus.indexes.get(c)!;
  if (s.status === 'no') return { ...seq, cand: c, serve: s };
  if (s.status === 'full') {
    return { cand: c, serve: s, postingRows: [], candidates: rows.length, matches: truthRows.length, indexPages: ix.pages, heapPages, pages: ix.pages + heapPages, seqPages: heapPages, served: false };
  }
  let cand: number[] = [];
  let indexPages = 0;
  const postingRows: { entry: string; rows: number }[] = [];
  if (ix.kind === 'btree') {
    cand = truthRows;
    indexPages = ix.height + Math.max(0, Math.ceil(truthRows.length / (ix.leafTuplesPerPage ?? 1)) - 1);
    postingRows.push({ entry: s.entries[0], rows: truthRows.length });
  } else if (ix.kind === 'gist' && ix.gist) {
    const q = [...wildcardTrigrams((p as { pattern: string }).pattern)];
    const g = ix.gist;
    const bits = q.map((t) => trgmBit(t, g.siglen));
    indexPages = g.parents.length > 1 ? 1 : 0;
    for (const par of g.parents) {
      indexPages++;
      if (!bits.every((bt) => par.sig[bt])) continue;
      for (const li of par.children) {
        const leaf = g.leafPages[li];
        if (!bits.every((bt) => leaf.sig[bt])) continue;
        indexPages++;
        for (const ri of leaf.rows) if (q.every((t) => g.trgm[ri].has(t))) cand.push(ri);
      }
    }
    postingRows.push({ entry: `${q.length} query bits`, rows: cand.length });
  } else {
    const keyOf = (e: string) => {
      if (c === 'gin-path') {
        const m = [...jsonbPathEntries((p as { query: JsonObj }).query).entries()].find(([, v]) => `hash(${v})` === e);
        return m ? String(m[0]) : e;
      }
      if (c === 'gin-trgm') return e.slice(1, -1);
      if (c === 'gin-array' || c === 'gin-tsv' || c === 'gencol-tsv') return e.slice(1, -1);
      return e;
    };
    let acc: number[] | null = null;
    for (const e of s.entries) {
      const k = keyOf(e);
      const tids = ix.postings?.get(k) ?? [];
      const idxs = tids.map((t) => corpus.byTid.get(t)!);
      postingRows.push({ entry: e, rows: idxs.length });
      indexPages += ix.height + (ix.postingPages?.get(k) ?? 0);
      acc = acc === null ? idxs : s.mode === 'and' ? intersect(acc, idxs) : union(acc, idxs);
    }
    cand = acc ?? [];
  }
  const matches = s.recheck ? cand.filter((i) => truth(p, rows[i])).length : cand.length;
  const hp = distinctPages(rows, cand);
  const pages = indexPages + hp;
  return { cand: c, serve: s, postingRows, candidates: cand.length, matches, indexPages, heapPages: hp, pages, seqPages: heapPages, served: pages <= budget * heapPages };
}

/* ------------------------------------------------------------------ scoring */

export function evaluate(corpus: Corpus, scenario: ScenarioId, preds: PredKind[], budget: number) {
  const cands = CANDIDATES[scenario];
  const grid = cands.map((c) => preds.map((p) => lookup(corpus, c.id, p, budget)));
  const seq = preds.map((p) => lookup(corpus, 'seq', p, budget));
  const cost = (c: CandKind) => corpus.indexes.get(c)!.entriesPerRow;
  const bytes = (c: CandKind) => corpus.indexes.get(c)!.bytes + corpus.indexes.get(c)!.extraHeapBytesPerRow * corpus.rows.length;
  let best: { set: number; entries: number; bytes: number } | null = null;
  for (let mask = 1; mask < 1 << cands.length; mask++) {
    const covered = preds.every((_, q) => cands.some((_, ci) => mask & (1 << ci) && grid[ci][q].served));
    if (!covered) continue;
    let e = 0;
    let by = 0;
    cands.forEach((c, ci) => {
      if (mask & (1 << ci)) {
        e += cost(c.id);
        by += bytes(c.id);
      }
    });
    if (!best || e < best.entries - 1e-9 || (Math.abs(e - best.entries) < 1e-9 && by < best.bytes)) best = { set: mask, entries: e, bytes: by };
  }
  return { grid, seq, best, cost, bytes };
}

/** Scoreboard for a chosen index set. */
export function score(corpus: Corpus, scenario: ScenarioId, ev: ReturnType<typeof evaluate>, built: Set<CandKind>) {
  const cands = CANDIDATES[scenario];
  const perPred = ev.seq.map((sq, q) => {
    let bestL: Lookup = sq;
    cands.forEach((c, ci) => {
      const l = ev.grid[ci][q];
      if (built.has(c.id) && l.served && l.pages < bestL.pages) bestL = l;
    });
    return bestL;
  });
  let entries = 0;
  let bytes = 0;
  for (const c of built) {
    entries += ev.cost(c);
    bytes += ev.bytes(c);
  }
  const covered = perPred.filter((l) => l.cand !== 'seq').length;
  const pages = perPred.reduce((s, l) => s + l.pages, 0);
  return { perPred, entries, bytes, covered, pages };
}

/* -------------------------------------------------------- pending list model */

export type StreamPoint = { merged: number; appended: number; pending: number; foreground: boolean; vacuum: boolean };

/**
 * One GIN index under a stream of single-row inserts. With fastupdate each row's entries are appended to the
 * pending list; when pages x GIN_PAGE_FREESIZE exceed the limit, that insert merges the whole list. A VACUUM or
 * autoanalyze every `vacuumEvery` inserts drains it in the background. Without fastupdate every entry goes
 * straight into the main structure.
 */
export function pendingStream(entriesPerRow: number, bytesPerRow: number, fastupdate: boolean, limitKb: number, vacuumEvery: number, inserts: number): StreamPoint[] {
  const out: StreamPoint[] = [];
  let pendingEntries = 0;
  let pendingBytes = 0;
  const threshold = limitKb * 1024;
  for (let i = 1; i <= inserts; i++) {
    if (!fastupdate) {
      out.push({ merged: entriesPerRow, appended: 0, pending: 0, foreground: false, vacuum: false });
      continue;
    }
    pendingEntries += entriesPerRow;
    pendingBytes += bytesPerRow;
    const pages = Math.ceil(pendingBytes / GIN_PAGE_FREESIZE);
    let merged = 0;
    let foreground = false;
    if (pages * GIN_PAGE_FREESIZE > threshold) {
      merged = pendingEntries;
      foreground = true;
      pendingEntries = 0;
      pendingBytes = 0;
    }
    const vacuum = vacuumEvery > 0 && i % vacuumEvery === 0;
    if (vacuum) {
      pendingEntries = 0;
      pendingBytes = 0;
    }
    out.push({ merged, appended: entriesPerRow, pending: pendingEntries, foreground, vacuum });
  }
  return out;
}

/* ======================================================================= UI */

type Panel = 'set' | 'stream';

const STATUS_COLOR = { served: 'var(--viz-good)', over: 'var(--viz-warning)', full: 'var(--viz-serious)', no: 'var(--viz-grid)' } as const;
type CellState = keyof typeof STATUS_COLOR;

function cellState(l: Lookup): CellState {
  if (l.serve.status === 'no') return 'no';
  if (l.serve.status === 'full') return 'full';
  return l.served ? 'served' : 'over';
}

const SHAPES: Record<ScenarioId, { a: { label: string; min: number; max: number; def: number }; b: { label: string; min: number; max: number; step: number; def: number } }> = {
  products: { a: { label: 'Attribute keys per product', min: 1, max: 12, def: 6 }, b: { label: 'Tags per product', min: 0, max: 8, step: 1, def: 3 } },
  articles: { a: { label: 'Tags per article', min: 1, max: 10, def: 3 }, b: { label: 'Body length (words)', min: 20, max: 200, step: 10, def: 80 } },
};

const TABLE_DDL: Record<ScenarioId, string> = {
  products: 'products (id bigint, data jsonb)',
  articles: 'articles (id bigint, tags text[], title text, body text)',
};

function chipLabel(e: string) {
  const m = /^(key|str|num|bool|null):(.*)$/.exec(e);
  if (!m) return e;
  return m[1] === 'key' || m[1] === 'str' ? `${m[1]} "${m[2]}"` : `${m[1]} ${m[2]}`;
}

function Chips({ items, x, y, width, highlight }: { items: string[]; x: number; y: number; width: number; highlight: Set<string> }) {
  let cx = x;
  let cy = y;
  const els: ReactElement[] = [];
  const shown = items.slice(0, 40);
  shown.forEach((it, i) => {
    const label = chipLabel(it);
    const w = Math.min(width, label.length * 6.1 + 12);
    if (cx + w > x + width) {
      cx = x;
      cy += 22;
    }
    const hi = highlight.has(it);
    els.push(
      <g key={i}>
        <rect x={cx} y={cy} width={w} height={17} rx={4} fill="var(--viz-surface)" stroke={hi ? 'var(--viz-ink)' : 'var(--viz-border)'} strokeWidth={hi ? 1.6 : 1} strokeDasharray={hi ? '3 2' : undefined} />
        <text x={cx + 6} y={cy + 12.5} fontSize={10.5} fill="var(--viz-ink)">
          {label.length > 60 ? `${label.slice(0, 58)}…` : label}
        </text>
      </g>,
    );
    cx += w + 5;
  });
  if (items.length > shown.length) {
    els.push(
      <text key="more" x={cx} y={cy + 12.5} fontSize={10.5} fill="var(--viz-ink-2)">
        +{items.length - shown.length} more
      </text>,
    );
  }
  return { els, bottom: cy + 17 };
}

function Trace({ corpus, cand, pred, lk, budget }: { corpus: Corpus; cand: Cand | null; pred: Pred; lk: Lookup; budget: number }) {
  const W = 700;
  const LEFT = 150;
  const barW = W - LEFT - 150;
  const n = corpus.rows.length;
  const ix = cand ? corpus.indexes.get(cand.id)! : null;
  const highlight = new Set(lk.serve.entries);
  const sample = ix ? ix.sampleEntries : [];
  const rowWrites = ix ? (ix.kind === 'gin' ? `Row 1 writes ${sample.length} entr${sample.length === 1 ? 'y' : 'ies'} into this index` : ix.kind === 'gist' ? 'Row 1 writes one leaf tuple: its whole trigram set' : 'Row 1 writes one index tuple') : 'No index: rows are only in the heap';
  const chips = Chips({ items: sample, x: 10, y: 24, width: W - 20, highlight });
  let y = chips.bottom + 26;
  const els: ReactElement[] = [];
  const bar = (v: number, max: number) => Math.max(v > 0 ? 2 : 0, (v / Math.max(1, max)) * barW);

  if (lk.serve.status === 'serves' && lk.postingRows.length) {
    els.push(
      <text key="ph" x={10} y={y} fontSize={11.5} fill="var(--viz-ink)">
        {ix?.kind === 'btree' ? 'Descend the B-tree to the key' : ix?.kind === 'gist' ? 'Walk the tree, skipping pages whose signature lacks a query bit' : `Look up ${lk.postingRows.length} query entr${lk.postingRows.length === 1 ? 'y' : 'ies'} and read ${lk.postingRows.length === 1 ? 'its posting list' : `their posting lists (${lk.serve.mode === 'and' ? 'intersect' : 'union'})`}`}
      </text>,
    );
    y += 10;
    lk.postingRows.slice(0, 10).forEach((p, i) => {
      els.push(
        <g key={`p${i}`}>
          <text x={10} y={y + 13} fontSize={10.5} fill="var(--viz-ink-2)">
            {(ix?.kind === 'gist' ? p.entry : chipLabel(p.entry)).slice(0, 24)}
          </text>
          <rect x={LEFT} y={y + 3} width={barW} height={12} rx={3} fill="var(--viz-plane)" stroke="var(--viz-border)" />
          <rect x={LEFT} y={y + 3} width={bar(p.rows, n)} height={12} rx={3} fill="var(--viz-1)" />
          <text x={LEFT + barW + 8} y={y + 13} fontSize={10.5} fill="var(--viz-ink-2)">
            {fmtNum(p.rows)} row{p.rows === 1 ? '' : 's'}
          </text>
        </g>,
      );
      y += 18;
    });
    if (lk.postingRows.length > 10) {
      els.push(
        <text key="pm" x={10} y={y + 12} fontSize={10.5} fill="var(--viz-ink-2)">
          … and {lk.postingRows.length - 10} more entries
        </text>,
      );
      y += 18;
    }
    y += 8;
  } else if (lk.serve.status === 'full') {
    els.push(
      <text key="full" x={10} y={y} fontSize={11.5} fill="var(--viz-ink)">
        No entry to look up: all {fmtNum(ix?.pages ?? 0)} index pages are read, and every row is a candidate
      </text>,
    );
    y += 14;
  } else {
    els.push(
      <text key="noix" x={10} y={y} fontSize={11.5} fill="var(--viz-ink)">
        {cand ? 'The index cannot answer this predicate, so the plan is a sequential scan' : 'Sequential scan: every heap page, every row tested'}
      </text>,
    );
    y += 14;
  }

  const usesIndex = lk.serve.status !== 'no';
  const removed = lk.candidates - lk.matches;
  els.push(
    <g key="cand">
      <text x={10} y={y + 13} fontSize={10.5} fill="var(--viz-ink-2)">
        {usesIndex ? 'Candidates' : 'Rows tested'}
      </text>
      <rect x={LEFT} y={y + 3} width={barW} height={12} rx={3} fill="var(--viz-plane)" stroke="var(--viz-border)" />
      <rect x={LEFT} y={y + 3} width={bar(lk.matches, n)} height={12} rx={3} fill="var(--viz-3)" />
      <rect x={LEFT + bar(lk.matches, n)} y={y + 3} width={bar(removed, n)} height={12} rx={3} fill="var(--viz-stale)" />
      <text x={LEFT + barW + 8} y={y + 13} fontSize={10.5} fill="var(--viz-ink-2)">
        {fmtNum(lk.candidates)} → {fmtNum(lk.matches)} match
      </text>
    </g>,
  );
  y += 26;
  const seq = lk.seqPages;
  const maxPages = Math.max(seq, lk.pages);
  const pbar = (v: number) => Math.max(v > 0 ? 2 : 0, (v / maxPages) * barW);
  els.push(
    <g key="pages">
      <text x={10} y={y + 13} fontSize={10.5} fill="var(--viz-ink-2)">
        Pages read
      </text>
      <rect x={LEFT} y={y + 3} width={pbar(lk.indexPages)} height={12} rx={3} fill="var(--viz-7)" />
      <rect x={LEFT + pbar(lk.indexPages)} y={y + 3} width={pbar(usesIndex ? lk.heapPages : seq)} height={12} rx={3} fill="var(--viz-5)" />
      <text x={LEFT + barW + 8} y={y + 13} fontSize={10.5} fill="var(--viz-ink-2)">
        {usesIndex ? `${fmtNum(lk.indexPages)} index + ${fmtNum(lk.heapPages)} heap` : `${fmtNum(seq)} heap`}
      </text>
      <text x={10} y={y + 35} fontSize={10.5} fill="var(--viz-ink-2)">
        Sequential scan
      </text>
      <rect x={LEFT} y={y + 25} width={pbar(seq)} height={12} rx={3} fill="var(--viz-plane)" stroke="var(--viz-5)" />
      <text x={LEFT + barW + 8} y={y + 35} fontSize={10.5} fill="var(--viz-ink-2)">
        {fmtNum(seq)} heap
      </text>
      <line x1={LEFT + pbar(seq * budget)} x2={LEFT + pbar(seq * budget)} y1={y - 2} y2={y + 42} stroke="var(--viz-ink)" strokeDasharray="3 3" />
      <text x={LEFT + pbar(seq * budget) + 4} y={y + 50} fontSize={10} fill="var(--viz-ink-2)">
        budget {Math.round(budget * 100)}%
      </text>
    </g>,
  );
  y += 58;
  const title = `${cand ? cand.short : 'No index'} answering ${pred.sql}`;
  return (
    <svg viewBox={`0 0 ${W} ${y}`} width={W} height={y} style={{ width: '100%', minWidth: 580, maxWidth: 'none' }} role="img" aria-label={`${title}: ${fmtNum(lk.candidates)} candidates, ${fmtNum(lk.matches)} matches, ${fmtNum(lk.pages)} pages read`}>
      <text x={10} y={14} fontSize={11.5} fill="var(--viz-ink)">
        {rowWrites}
        {cand?.id === 'gencol-sku' || cand?.id === 'gencol-tsv' ? ` (plus ${fmtNum(ix?.extraHeapBytesPerRow ?? 0)} bytes in the heap for the stored column)` : ''}
      </text>
      {chips.els}
      {els}
    </svg>
  );
}

function StreamChart({ points, fastupdate }: { points: StreamPoint[]; fastupdate: boolean }) {
  const W = 700;
  const H = 230;
  const L = 56;
  const R = 12;
  const T = 14;
  const B = 34;
  const cols = 280;
  const per = Math.ceil(points.length / cols);
  const buckets = Array.from({ length: Math.ceil(points.length / per) }, (_, i) => {
    const slice = points.slice(i * per, (i + 1) * per);
    return {
      merged: Math.max(...slice.map((p) => p.merged)),
      appended: Math.max(...slice.map((p) => p.appended)),
      pending: Math.max(...slice.map((p) => p.pending)),
      foreground: slice.some((p) => p.foreground),
      vacuum: slice.some((p) => p.vacuum),
    };
  });
  const maxV = Math.max(10, ...buckets.map((b) => Math.max(b.merged, b.pending, b.appended)));
  const top = Math.pow(10, Math.ceil(Math.log10(maxV)));
  const yOf = (v: number) => T + (H - T - B) * (1 - Math.log10(Math.max(1, v)) / Math.log10(top));
  const bw = (W - L - R) / buckets.length;
  const ticks: number[] = [];
  for (let v = 1; v <= top; v *= 10) ticks.push(v);
  const path = buckets.map((b, i) => `${i === 0 ? 'M' : 'L'}${(L + i * bw + bw / 2).toFixed(1)},${yOf(b.pending).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ width: '100%', minWidth: 520, maxWidth: 'none' }} role="img" aria-label={fastupdate ? 'Per-insert index entries with fastupdate: small appends with periodic foreground merges' : 'Per-insert index entries with fastupdate off: every insert writes its entries directly'}>
      {ticks.map((v) => (
        <g key={v}>
          <line className="viz-grid-line" x1={L} x2={W - R} y1={yOf(v)} y2={yOf(v)} />
          <text x={L - 6} y={yOf(v) + 4} textAnchor="end" fontSize={10}>
            {v >= 1_000_000 ? `${fmtNum(v / 1_000_000)}M` : v >= 1000 ? `${fmtNum(v / 1000)}k` : v}
          </text>
        </g>
      ))}
      {buckets.map((b, i) => {
        const x = L + i * bw;
        return (
          <g key={i}>
            {fastupdate ? (
              <rect x={x} y={yOf(b.appended)} width={Math.max(1, bw - 0.5)} height={H - B - yOf(b.appended)} fill="var(--viz-3)" />
            ) : (
              <rect x={x} y={yOf(b.merged)} width={Math.max(1, bw - 0.5)} height={H - B - yOf(b.merged)} fill="var(--viz-1)" />
            )}
            {fastupdate && b.merged > 0 ? <rect x={x - 0.5} y={yOf(b.merged)} width={Math.max(2, bw)} height={H - B - yOf(b.merged)} fill="var(--viz-2)" /> : null}
            {b.vacuum ? <path d={`M${x + bw / 2},${H - B + 3} l-4,7 h8 z`} fill="var(--viz-ink-muted)" /> : null}
          </g>
        );
      })}
      {fastupdate ? <path d={path} fill="none" stroke="var(--viz-7)" strokeWidth={1.8} /> : null}
      <line className="viz-axis-line" x1={L} x2={W - R} y1={H - B} y2={H - B} />
      <text x={L} y={H - 8} fontSize={10}>
        insert 1
      </text>
      <text x={W - R} y={H - 8} textAnchor="end" fontSize={10}>
        insert {fmtNum(points.length)}
      </text>
      <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" fontSize={10}>
        single-row INSERTs over time →
      </text>
      <text x={12} y={T + 2} fontSize={10} transform={`rotate(-90 12 ${T + 2})`} textAnchor="end">
        index entries (log)
      </text>
    </svg>
  );
}

export default function JsonbIndexChooserLab() {
  const [panel, setPanel] = useState<Panel>('set');
  const [scenario, setScenario] = useState<ScenarioId>('products');
  const [shapeA, setShapeA] = useState<Record<ScenarioId, number>>({ products: 6, articles: 3 });
  const [shapeB, setShapeB] = useState<Record<ScenarioId, number>>({ products: 3, articles: 80 });
  const [budgetPct, setBudgetPct] = useState(25);
  const [forms, setForms] = useState<Record<ScenarioId, string[]>>({
    products: SLOTS.products.map((s) => s.options[0].id),
    articles: SLOTS.articles.map((s) => s.options[0].id),
  });
  const [built, setBuilt] = useState<Record<ScenarioId, CandKind[]>>({ products: ['gin-jsonb'], articles: ['gin-array'] });
  const [inspect, setInspect] = useState<Record<ScenarioId, { c: CandKind | 'seq'; q: number }>>({ products: { c: 'gin-jsonb', q: 0 }, articles: { c: 'gin-array', q: 0 } });
  const [fastupdate, setFastupdate] = useState(true);
  const [limitKb, setLimitKb] = useState('4096');
  const [vacuumEvery, setVacuumEvery] = useState(0);
  const [streamIdx, setStreamIdx] = useState<Record<ScenarioId, CandKind>>({ products: 'gin-jsonb', articles: 'gin-tsv' });

  const key = `${scenario}|${shapeA[scenario]}|${shapeB[scenario]}`;
  const deferredKey = useDeferredValue(key);
  const [ds, dA, dB] = deferredKey.split('|') as [ScenarioId, string, string];
  const corpus = useMemo(() => makeCorpus(ds, Number(dA), Number(dB)), [ds, dA, dB]);
  const stale = deferredKey !== key;

  const slots = SLOTS[ds];
  const cands = CANDIDATES[ds];
  const preds = slots.map((s, i) => s.options.find((o) => o.id === forms[ds][i]) ?? s.options[0]);
  const budget = budgetPct / 100;
  const predKey = preds.map((p) => p.id).join(',');
  const ev = useMemo(() => evaluate(corpus, ds, preds.map((p) => p.kind), budget), [corpus, ds, predKey, budget]);
  const builtSet = new Set(built[ds]);
  const sc = score(corpus, ds, ev, builtSet);
  const optimal = !!ev.best && sc.covered === slots.length && sc.entries <= ev.best.entries + 1e-6;
  const bestNames = ev.best ? cands.filter((_, i) => ev.best!.set & (1 << i)).map((c) => c.short) : [];

  const insp = inspect[ds];
  const inspCand = insp.c === 'seq' ? null : cands.find((c) => c.id === insp.c) ?? null;
  const inspLookup = insp.c === 'seq' ? ev.seq[insp.q] : ev.grid[cands.findIndex((c) => c.id === insp.c)][insp.q];

  const toggle = (c: CandKind) => setBuilt((b) => ({ ...b, [ds]: b[ds].includes(c) ? b[ds].filter((x) => x !== c) : [...b[ds], c] }));

  // stream panel
  const ginCands = cands.filter((c) => c.family === 'GIN');
  const sIdx = ginCands.find((c) => c.id === streamIdx[ds]) ?? ginCands[0];
  const sStats = corpus.indexes.get(sIdx.id)!;
  const INSERTS = 60_000;
  const points = useMemo(() => pendingStream(sStats.entriesPerRow, sStats.pendingBytesPerRow, fastupdate, Number(limitKb), vacuumEvery, INSERTS), [sStats, fastupdate, limitKb, vacuumEvery]);
  const fg = points.filter((p) => p.foreground);
  const firstFg = points.findIndex((p) => p.foreground) + 1;
  const worstMerge = Math.max(0, ...fg.map((p) => p.merged));
  const maxPending = Math.max(0, ...points.map((p) => p.pending));

  const unserved = preds.map((p, q) => ({ p, q })).filter(({ q }) => sc.perPred[q].cand === 'seq');

  const shape = SHAPES[scenario];
  return (
    <VizPanel
      title={panel === 'set' ? 'Pick the index set for four semi-structured predicates' : 'What a GIN index costs each INSERT'}
      subtitle={
        panel === 'set'
          ? `${fmtNum(N_ROWS)} modelled rows of ${TABLE_DDL[ds]}. Build candidates, click any cell to trace how that index answers that predicate, and serve all four within the page budget at the fewest index entries written per insert.`
          : 'With fastupdate on, each row’s entries are appended to the pending list; the insert that pushes the list past gin_pending_list_limit merges all of it into the index. With it off, every insert writes every entry into the index directly.'
      }
      controls={
        <>
          <Segmented label="Panel" value={panel} onChange={setPanel} options={[{ value: 'set', label: 'Index set' }, { value: 'stream', label: 'Insert stream' }]} />
          <Choice label="Table" value={scenario} onChange={setScenario} options={[{ value: 'products', label: 'products: one jsonb column' }, { value: 'articles', label: 'articles: text[] and full text' }]} />
          <Slider label={shape.a.label} min={shape.a.min} max={shape.a.max} value={shapeA[scenario]} onChange={(v) => setShapeA((s) => ({ ...s, [scenario]: v }))} />
          <Slider label={shape.b.label} min={shape.b.min} max={shape.b.max} step={shape.b.step} value={shapeB[scenario]} onChange={(v) => setShapeB((s) => ({ ...s, [scenario]: v }))} />
          {panel === 'set' ? <Slider label="Page budget per lookup" min={5} max={100} step={5} value={budgetPct} onChange={setBudgetPct} format={(v) => `${v}% of a seq scan`} /> : null}
        </>
      }
      legend={
        panel === 'set' ? (
          <Legend
            items={[
              { label: 'Served within the page budget', color: STATUS_COLOR.served },
              { label: 'Usable, but reads more than the budget', color: STATUS_COLOR.over },
              { label: 'Full index scan (nothing to look up)', color: STATUS_COLOR.full },
              { label: 'Cannot use this index', color: STATUS_COLOR.no },
              { label: 'Posting list length', color: 'var(--viz-1)' },
              { label: 'Rows that match', color: 'var(--viz-3)' },
              { label: 'Candidates removed by recheck', color: 'var(--viz-stale)' },
              { label: 'Index pages read', color: 'var(--viz-7)' },
              { label: 'Heap pages read', color: 'var(--viz-5)' },
            ]}
          />
        ) : (
          <Legend
            items={
              fastupdate
                ? [
                    { label: 'Entries appended to the pending list', color: 'var(--viz-3)' },
                    { label: 'Entries merged by one INSERT (foreground cleanup)', color: 'var(--viz-2)' },
                    { label: 'Pending entries every search must also scan', color: 'var(--viz-7)', shape: 'line' },
                    { label: 'VACUUM or autoanalyze drains the list', color: 'var(--viz-ink-muted)' },
                  ]
                : [
                    { label: 'Entries written into the index by each INSERT', color: 'var(--viz-1)' },
                    { label: 'No pending list: nothing for searches to scan', color: 'var(--viz-ink-muted)' },
                  ]
            }
          />
        )
      }
      stats={
        panel === 'set' ? (
          <Stats
            items={[
              { label: 'Predicates served', value: `${sc.covered} of ${slots.length}` },
              { label: 'Entries written per insert', value: fmtNum(sc.entries, 1), hint: 'Sum over the built indexes: GIN writes one entry per distinct key it extracts, B-tree and GiST one tuple per row.' },
              { label: 'Index size', value: fmtBytes(sc.bytes), hint: 'Model estimate from exact entry counts and varbyte-encoded postings; includes the heap bytes of stored generated columns.' },
              { label: 'Pages read, four lookups', value: `${fmtNum(sc.pages)} / ${fmtNum(ev.seq.reduce((s, l) => s + l.pages, 0))}`, hint: 'Best built index per predicate, or a sequential scan; the second number is four sequential scans.' },
              { label: 'Cheapest set that serves all four', value: ev.best ? `${fmtNum(ev.best.entries, 1)} entries` : 'none', hint: ev.best ? bestNames.join(' + ') : 'At least one predicate cannot be served within the budget by any candidate.' },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Entries per row', value: fmtNum(sStats.entriesPerRow, 1) },
              { label: 'Pending-list bytes per row', value: fmtBytes(sStats.pendingBytesPerRow), hint: 'One index tuple per entry: MAXALIGN(8-byte header + key) + a 4-byte line pointer.' },
              { label: 'Rows until a foreground merge', value: !fastupdate ? '—' : firstFg > 0 ? fmtNum(firstFg) : 'never', hint: 'Cleanup starts when pending pages × 8,160 bytes exceed gin_pending_list_limit.' },
              { label: 'Worst INSERT merges', value: fastupdate && worstMerge > 0 ? `${fmtNum(worstMerge)} entries` : fastupdate ? 'none' : `${fmtNum(sStats.entriesPerRow, 1)} each`, hint: 'The foreground merge runs inside that one INSERT; another backend that finds a cleanup already running just returns.' },
              { label: 'Most pending entries a search scans', value: fastupdate ? fmtNum(maxPending) : '0' },
            ]}
          />
        )
      }
      note={
        <Note>
          {panel === 'set' ? (
            optimal ? (
              <>
                <strong>All {slots.length} served at the minimum write cost: {fmtNum(sc.entries, 1)} index entries per insert.</strong> {fmtBytes(sc.bytes)} of indexes. Change a predicate’s form or the page budget and the cheapest set changes with it.
              </>
            ) : sc.covered === slots.length ? (
              <>
                <strong>All {slots.length} served, at {fmtNum(sc.entries, 1)} entries per insert.</strong> A cheaper set exists at {fmtNum(ev.best!.entries, 1)}: find the index whose entries serve the least, and drop it.
              </>
            ) : (
              <>
                <strong>
                  {unserved.map((u) => `Q${u.q + 1}`).join(', ')} {unserved.length === 1 ? 'is' : 'are'} not served by what you built.
                </strong>{' '}
                {(() => {
                  const u = unserved[0];
                  const able = cands.filter((_, ci) => ev.grid[ci][u.q].served);
                  if (able.length) return `Q${u.q + 1} can be served by: ${able.map((c) => c.short).join(', ')}.`;
                  const usable = cands.map((c, ci) => ({ c, l: ev.grid[ci][u.q] })).find((x) => x.l.serve.status !== 'no');
                  return usable
                    ? `No candidate serves Q${u.q + 1} within ${budgetPct}% of a sequential scan: ${usable.c.short} reads ${fmtNum(usable.l.pages)} of ${fmtNum(usable.l.seqPages)} pages. ${usable.l.serve.status === 'full' ? usable.l.serve.reason : 'Raise the budget or change the query.'}`
                    : `No candidate can use an index for Q${u.q + 1} as written. Try another form of the query.`;
                })()}
              </>
            )
          ) : !fastupdate ? (
            <>
              <strong>fastupdate is off: every INSERT writes its {fmtNum(sStats.entriesPerRow, 1)} entries straight into {sIdx.short}.</strong> No spikes and no pending list for searches to scan, but every row pays for every entry, one index insertion at a time.
            </>
          ) : firstFg > 0 ? (
            <>
              <strong>
                Every {fmtNum(firstFg)} rows, one INSERT merges about {fmtNum(worstMerge)} pending entries into {sIdx.short} before it returns.
              </strong>{' '}
              The other inserts only append. The limit sets how many entries each merge carries; the entries per row set how often it comes. {vacuumEvery > 0 ? `A VACUUM every ${fmtNum(vacuumEvery)} inserts is too late to prevent it.` : 'Add a VACUUM or autoanalyze more often than that and the merges move to the background.'}
            </>
          ) : vacuumEvery === 0 ? (
            <>
              <strong>In {fmtNum(INSERTS)} inserts the pending list never grows past gin_pending_list_limit, so no INSERT merges in the foreground.</strong> With no VACUUM it only grows: searches scan up to {fmtNum(maxPending)} pending entries.
            </>
          ) : (
            <>
              <strong>Background VACUUM every {fmtNum(vacuumEvery)} inserts drains the pending list before it reaches the limit.</strong> No INSERT merges in the foreground; searches scan at most {fmtNum(maxPending)} pending entries.
            </>
          )}
          {stale ? ' (recomputing…)' : ''}
        </Note>
      }
      table={
        panel === 'set' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Index</th>
                {preds.map((p, q) => (
                  <th key={q}>Q{q + 1} pages (candidates → matches)</th>
                ))}
                <th>Entries / insert</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {cands.map((c, ci) => (
                <tr key={c.id}>
                  <td>{c.short}</td>
                  {ev.grid[ci].map((l, q) => (
                    <td key={q}>{l.serve.status === 'no' ? 'cannot use' : `${fmtNum(l.pages)} (${fmtNum(l.candidates)} → ${fmtNum(l.matches)})${l.serve.status === 'full' ? ' full scan' : l.served ? ' served' : ' over budget'}`}</td>
                  ))}
                  <td>{fmtNum(corpus.indexes.get(c.id)!.entriesPerRow, 1)}</td>
                  <td>{fmtBytes(corpus.indexes.get(c.id)!.bytes + corpus.indexes.get(c.id)!.extraHeapBytesPerRow * corpus.rows.length)}</td>
                </tr>
              ))}
              <tr>
                <td>No index (sequential scan)</td>
                {ev.seq.map((l, q) => (
                  <td key={q}>{fmtNum(l.pages)} ({fmtNum(l.matches)} matches)</td>
                ))}
                <td>0</td>
                <td>0</td>
              </tr>
            </tbody>
          </table>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>GIN index</th>
                <th>Entries per row</th>
                <th>Pending bytes per row</th>
                <th>Rows per foreground merge at 1 MB / 4 MB / 16 MB</th>
              </tr>
            </thead>
            <tbody>
              {ginCands.map((c) => {
                const s = corpus.indexes.get(c.id)!;
                const rowsAt = (kb: number) => Math.floor((Math.floor((kb * 1024) / GIN_PAGE_FREESIZE) * GIN_PAGE_FREESIZE) / s.pendingBytesPerRow) + 1;
                return (
                  <tr key={c.id}>
                    <td>{c.short}</td>
                    <td>{fmtNum(s.entriesPerRow, 1)}</td>
                    <td>{fmtNum(s.pendingBytesPerRow)}</td>
                    <td>
                      {fmtNum(rowsAt(1024))} / {fmtNum(rowsAt(4096))} / {fmtNum(rowsAt(16384))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )
      }
    >
      {panel === 'set' ? (
        <>
          <div className="viz-controls">
            {slots.map((s, i) => (
              <Choice
                key={s.label}
                label={s.label}
                value={forms[ds][i]}
                onChange={(v) => {
                  setForms((f) => ({ ...f, [ds]: f[ds].map((x, j) => (j === i ? v : x)) }));
                  setInspect((ins) => ({ ...ins, [ds]: { ...ins[ds], q: i } }));
                }}
                options={s.options.map((o) => ({ value: o.id, label: o.sql }))}
              />
            ))}
          </div>
          <table className="viz-table" style={{ minWidth: 540, marginTop: 4, fontSize: '0.72rem' }}>
            <thead>
              <tr>
                <th scope="col" aria-label="Build" style={{ padding: '0.25rem 0.2rem' }} />
                <th scope="col">Candidate (tick to build)</th>
                {preds.map((p, q) => (
                  <th scope="col" key={q} title={p.sql} style={{ padding: '0.25rem 0.2rem' }}>
                    Q{q + 1} pages
                  </th>
                ))}
                <th scope="col" style={{ padding: '0.25rem 0.3rem' }}>Entries per insert</th>
                <th scope="col" style={{ padding: '0.25rem 0.3rem' }}>Size</th>
              </tr>
            </thead>
            <tbody>
              {cands.map((c, ci) => {
                const s = corpus.indexes.get(c.id)!;
                return (
                  <tr key={c.id}>
                    <td style={{ padding: '0.25rem 0.2rem' }}>
                      <input type="checkbox" aria-label={`Build ${c.short}`} checked={builtSet.has(c.id)} onChange={() => toggle(c.id)} />
                    </td>
                    <td title={c.sql} style={{ color: 'var(--viz-ink)', fontWeight: builtSet.has(c.id) ? 600 : 400 }}>
                      {c.short}
                    </td>
                    {ev.grid[ci].map((l, q) => {
                      const st = cellState(l);
                      const sel = insp.c === c.id && insp.q === q;
                      return (
                        <td key={q} style={{ padding: '2px 0.15rem' }}>
                          <button
                            type="button"
                            aria-pressed={sel}
                            aria-label={`${c.short}, Q${q + 1}: ${st === 'no' ? 'cannot use' : st === 'full' ? `full index scan, ${l.pages} pages` : `${st === 'served' ? 'served' : 'over budget'}, ${l.pages} pages`}`}
                            title={l.serve.reason}
                            onClick={() => setInspect((ins) => ({ ...ins, [ds]: { c: c.id, q } }))}
                            style={{ width: '100%', minWidth: 54, whiteSpace: 'nowrap', textAlign: 'left', font: 'inherit', fontSize: '0.7rem', padding: '2px 4px', borderRadius: 5, cursor: 'pointer', background: 'var(--viz-surface)', color: st === 'no' ? 'var(--viz-ink-muted)' : 'var(--viz-ink)', border: sel ? '2px solid var(--viz-ink)' : '1px solid var(--viz-border)' }}
                          >
                            <span aria-hidden="true" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, marginRight: 4, background: STATUS_COLOR[st] }} />
                            {st === 'no' ? '✗' : st === 'full' ? `full ${fmtNum(l.pages)}` : `${st === 'served' ? '✓' : '~'} ${fmtNum(l.pages)}`}
                          </button>
                        </td>
                      );
                    })}
                    <td style={{ padding: '0.25rem 0.3rem' }}>{fmtNum(s.entriesPerRow, 1)}</td>
                    <td style={{ padding: '0.25rem 0.3rem', whiteSpace: 'nowrap' }}>{fmtBytes(s.bytes + s.extraHeapBytesPerRow * corpus.rows.length)}</td>
                  </tr>
                );
              })}
              <tr>
                <td />
                <td style={{ color: 'var(--viz-ink)' }}>No index (Seq Scan)</td>
                {ev.seq.map((l, q) => {
                  const sel = insp.c === 'seq' && insp.q === q;
                  return (
                    <td key={q} style={{ padding: 2 }}>
                      <button
                        type="button"
                        aria-pressed={sel}
                        aria-label={`No index, Q${q + 1}: sequential scan, ${l.pages} pages`}
                        title="Sequential scan"
                        onClick={() => setInspect((ins) => ({ ...ins, [ds]: { c: 'seq', q } }))}
                        style={{ width: '100%', minWidth: 54, whiteSpace: 'nowrap', textAlign: 'left', font: 'inherit', fontSize: '0.7rem', padding: '2px 4px', borderRadius: 5, cursor: 'pointer', background: 'var(--viz-surface)', color: 'var(--viz-ink)', border: sel ? '2px solid var(--viz-ink)' : '1px solid var(--viz-border)' }}
                      >
                        {fmtNum(l.pages)}
                      </button>
                    </td>
                  );
                })}
                <td>0</td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
          <div style={{ margin: '0.75rem 0 0.25rem', fontSize: '0.8rem', color: 'var(--viz-ink)' }}>
            <code style={{ fontSize: '0.75rem' }}>{inspCand ? inspCand.sql : 'no index'}</code>
            <br />
            <code style={{ fontSize: '0.75rem' }}>WHERE {preds[insp.q].sql}</code>
            <div style={{ color: 'var(--viz-ink-2)', marginTop: 4 }}>{inspLookup.serve.reason}</div>
          </div>
          <Trace corpus={corpus} cand={inspCand} pred={preds[insp.q]} lk={inspLookup} budget={budget} />
        </>
      ) : (
        <>
          <div className="viz-controls">
            <Choice label="GIN index" value={sIdx.id} onChange={(v) => setStreamIdx((s) => ({ ...s, [ds]: v }))} options={ginCands.map((c) => ({ value: c.id, label: c.short }))} />
            <Check label="fastupdate" checked={fastupdate} onChange={setFastupdate} />
            <Choice label="gin_pending_list_limit" value={limitKb} onChange={setLimitKb} options={[{ value: '1024', label: '1MB' }, { value: '4096', label: '4MB (default)' }, { value: '16384', label: '16MB' }]} />
            <Slider label="VACUUM or autoanalyze every" min={0} max={30000} step={1000} value={vacuumEvery} onChange={setVacuumEvery} format={(v) => (v === 0 ? 'never' : `${fmtNum(v)} inserts`)} />
          </div>
          <StreamChart points={points} fastupdate={fastupdate} />
        </>
      )}
    </VizPanel>
  );
}
