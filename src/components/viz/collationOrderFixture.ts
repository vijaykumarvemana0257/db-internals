/**
 * A committed ordering fixture: 24 keys, six comparators, one rank table.
 *
 * A browser has exactly one collation library — whatever ICU its engine was built
 * with — and no way to ask for "glibc 2.27". So the orderings are generated offline
 * and shipped as data. Provenance, per column, is recorded honestly below and is
 * printed in the visual's "Show the numbers" table:
 *
 *  memcmp    Bytewise comparison of the UTF-8 encoding. Computed, not modelled:
 *            Buffer.compare() on the exact bytes in `hex`. For UTF-8, byte order
 *            IS codepoint order, so this is also the C / POSIX / C.UTF-8 order.
 *  glibc227  MODELLED from the pre-2.28 en_US.UTF-8 weight tables, in which SPACE
 *            and the punctuation marks carry `IGNORE;IGNORE;IGNORE;<code>` — they
 *            do not exist until the fourth pass — and characters the table never
 *            assigned (the octopus) are ignorable at every level, so strcoll() can
 *            return 0 for two different strings.
 *  glibc228  MODELLED from the ISO 14651:2016 tables glibc adopted in 2.28, in
 *            which space, punctuation and symbols carry real primary weights.
 *  icu       CAPTURED: Intl.Collator('en').compare under Node v22.13.0,
 *            ICU 76.1 / CLDR 46 / Unicode 16.0.
 *  icu_sv    CAPTURED the same way with Intl.Collator('sv') — Swedish, where ä and
 *            ö are letters after z rather than accented a and o.
 *  icu_ai    CAPTURED with Intl.Collator('en', { sensitivity: 'base' }) — primary
 *            strength only, the ICU equivalent of MySQL's utf8mb4_0900_ai_ci.
 *            Equal ranks here mean the collation calls the two strings equal.
 *
 * The two glibc columns are a model of the documented rule applied to these 24
 * keys, not a capture from a running libc. Everything the model decides for this
 * fixture is a difference you can reproduce with `sort` on the two glibc versions:
 * a hyphen or a space that used to be invisible until the last pass now compares
 * first, which moves "Al-Rashid", "de Vries", "O'Brien" and "van Dijk".
 *
 * The upgrades page in Part 10 reuses this table for its own collation-trap drill.
 */

export type ComparatorId = 'memcmp' | 'glibc227' | 'glibc228' | 'icu' | 'icu_sv' | 'icu_ai';

export type Comparator = {
  id: ComparatorId;
  label: string;
  short: string;
  /** Bytewise comparators need no library at all; weight comparators do. */
  kind: 'bytes' | 'weights';
  /** A deterministic comparator breaks a weight tie with memcmp, so ties never merge rows. */
  deterministic: boolean;
  provenance: string;
  gist: string;
};

export const COMPARATORS: Comparator[] = [
  {
    id: 'memcmp',
    label: 'C / C.UTF-8 — memcmp on the UTF-8 bytes',
    short: 'memcmp',
    kind: 'bytes',
    deterministic: true,
    provenance: 'computed from the stored bytes',
    gist:
      'No table, no library, no version. Uppercase before lowercase because 0x41 < 0x61, and every ' +
      'non-ASCII key after every ASCII one because a UTF-8 lead byte is >= 0xC2.',
  },
  {
    id: 'glibc227',
    label: 'glibc 2.27 en_US.UTF-8 (pre-2.28 tables)',
    short: 'glibc 2.27',
    kind: 'weights',
    deterministic: true,
    provenance: 'modelled from the pre-2.28 weight rules',
    gist:
      'Space and punctuation are IGNORE at the first three levels and only reappear at the fourth, ' +
      'so "Al-Rashid" collates as "alrashid". Unassigned characters are ignorable everywhere, so ' +
      'strcoll() can return 0 for distinct strings.',
  },
  {
    id: 'glibc228',
    label: 'glibc 2.28+ en_US.UTF-8 (ISO 14651:2016 tables)',
    short: 'glibc 2.28',
    kind: 'weights',
    deterministic: true,
    provenance: 'modelled from the 2.28 weight rules',
    gist:
      'Space, punctuation and symbols now carry primary weights below the digits and letters. Every ' +
      'key holding one of them moves — with no change to the data and no error anywhere.',
  },
  {
    id: 'icu',
    label: 'ICU en (CLDR root order)',
    short: 'ICU en',
    kind: 'weights',
    deterministic: true,
    provenance: 'captured: ICU 76.1 / CLDR 46',
    gist:
      'CLDR root sets alternate = non-ignorable, so punctuation compares at the primary level. On this ' +
      'fixture ICU agrees with post-2.28 glibc key for key — which is the point: the 2.28 change moved ' +
      'glibc towards the table ICU was already using.',
  },
  {
    id: 'icu_sv',
    label: 'ICU sv (Swedish)',
    short: 'ICU sv',
    kind: 'weights',
    deterministic: true,
    provenance: 'captured: ICU 76.1 / CLDR 46',
    gist:
      'Same provider, same version, different locale: in Swedish ä and ö are letters in their own right ' +
      'after z, so Backer < Bakker < Bäcker and Munro < Müller.',
  },
  {
    id: 'icu_ai',
    label: 'ICU en, primary strength (≈ utf8mb4_0900_ai_ci)',
    short: 'ai_ci',
    kind: 'weights',
    deterministic: false,
    provenance: 'captured: ICU 76.1 / CLDR 46, sensitivity = base',
    gist:
      'Comparison stops after the primary level: Adams = adams, Backer = Bäcker, Muller = Müller. ' +
      'Equality, GROUP BY and UNIQUE all inherit that, so a unique index refuses the second of each pair.',
  },
];

export const COMPARATOR_ORDER: ComparatorId[] = ['memcmp', 'glibc227', 'glibc228', 'icu', 'icu_sv', 'icu_ai'];

export type FixtureKey = {
  s: string;
  /** The exact UTF-8 bytes. */
  hex: number[];
  /** Rank under each comparator, in COMPARATOR_ORDER. Equal ranks mean the comparator says equal. */
  r: number[];
};

/** A display_name column: case pairs, accents, a hyphen, an apostrophe, two spaces, one emoji. */
export const KEYS: FixtureKey[] = [
  { s: 'Adams', hex: [0x41, 0x64, 0x61, 0x6d, 0x73], r: [0, 1, 2, 2, 2, 1] },
  { s: 'adams', hex: [0x61, 0x64, 0x61, 0x6d, 0x73], r: [19, 0, 1, 1, 1, 1] },
  { s: 'Al-Rashid', hex: [0x41, 0x6c, 0x2d, 0x52, 0x61, 0x73, 0x68, 0x69, 0x64], r: [1, 3, 3, 3, 3, 3] },
  { s: 'Alavi', hex: [0x41, 0x6c, 0x61, 0x76, 0x69], r: [2, 2, 4, 4, 4, 4] },
  { s: 'Alston', hex: [0x41, 0x6c, 0x73, 0x74, 0x6f, 0x6e], r: [3, 4, 5, 5, 5, 5] },
  { s: 'Backer', hex: [0x42, 0x61, 0x63, 0x6b, 0x65, 0x72], r: [4, 5, 6, 6, 6, 6] },
  { s: 'Bäcker', hex: [0x42, 0xc3, 0xa4, 0x63, 0x6b, 0x65, 0x72], r: [6, 6, 7, 7, 8, 6] },
  { s: 'Bakker', hex: [0x42, 0x61, 0x6b, 0x6b, 0x65, 0x72], r: [5, 7, 8, 8, 7, 8] },
  { s: 'de Vries', hex: [0x64, 0x65, 0x20, 0x56, 0x72, 0x69, 0x65, 0x73], r: [20, 10, 9, 9, 9, 9] },
  { s: 'Dean', hex: [0x44, 0x65, 0x61, 0x6e], r: [7, 8, 10, 10, 10, 10] },
  { s: 'Devlin', hex: [0x44, 0x65, 0x76, 0x6c, 0x69, 0x6e], r: [8, 9, 11, 11, 11, 11] },
  { s: 'Ferraro', hex: [0x46, 0x65, 0x72, 0x72, 0x61, 0x72, 0x6f], r: [9, 11, 12, 12, 12, 12] },
  { s: 'Ferré', hex: [0x46, 0x65, 0x72, 0x72, 0xc3, 0xa9], r: [11, 12, 13, 13, 13, 13] },
  { s: 'Ferrer', hex: [0x46, 0x65, 0x72, 0x72, 0x65, 0x72], r: [10, 13, 14, 14, 14, 14] },
  { s: 'kraken', hex: [0x6b, 0x72, 0x61, 0x6b, 0x65, 0x6e], r: [21, 14, 15, 15, 15, 15] },
  {
    s: '🐙kraken',
    hex: [0xf0, 0x9f, 0x90, 0x99, 0x6b, 0x72, 0x61, 0x6b, 0x65, 0x6e],
    r: [23, 14, 0, 0, 0, 0],
  },
  { s: 'Muller', hex: [0x4d, 0x75, 0x6c, 0x6c, 0x65, 0x72], r: [12, 16, 16, 16, 16, 16] },
  { s: 'Müller', hex: [0x4d, 0xc3, 0xbc, 0x6c, 0x6c, 0x65, 0x72], r: [14, 17, 17, 17, 18, 16] },
  { s: 'Munro', hex: [0x4d, 0x75, 0x6e, 0x72, 0x6f], r: [13, 18, 18, 18, 17, 18] },
  { s: "O'Brien", hex: [0x4f, 0x27, 0x42, 0x72, 0x69, 0x65, 0x6e], r: [15, 20, 19, 19, 19, 19] },
  { s: 'Oakley', hex: [0x4f, 0x61, 0x6b, 0x6c, 0x65, 0x79], r: [16, 19, 20, 20, 20, 20] },
  { s: 'Ochoa', hex: [0x4f, 0x63, 0x68, 0x6f, 0x61], r: [17, 21, 21, 21, 21, 21] },
  { s: 'van Dijk', hex: [0x76, 0x61, 0x6e, 0x20, 0x44, 0x69, 0x6a, 0x6b], r: [22, 23, 22, 22, 22, 22] },
  {
    s: 'Vandenberg',
    hex: [0x56, 0x61, 0x6e, 0x64, 0x65, 0x6e, 0x62, 0x65, 0x72, 0x67],
    r: [18, 22, 23, 23, 23, 23],
  },
];

const BY_KEY = new Map(KEYS.map((k) => [k.s, k]));
const COL = new Map(COMPARATOR_ORDER.map((id, i) => [id, i]));

export function keyOf(s: string): FixtureKey {
  const k = BY_KEY.get(s);
  if (!k) throw new Error(`key not in fixture: ${s}`);
  return k;
}

/** memcmp on the stored UTF-8 bytes. Also the tiebreak a deterministic collation applies. */
export function memcmpBytes(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
}

/** What the collation itself says: <0, 0 or >0. A 0 here is strcoll()'s "these are equal". */
export function rawCompare(id: ComparatorId, a: string, b: string): number {
  const i = COL.get(id)!;
  const ra = keyOf(a).r[i];
  const rb = keyOf(b).r[i];
  return ra === rb ? 0 : ra < rb ? -1 : 1;
}

/**
 * The comparator the engine actually installs in the index.
 * Deterministic collations follow PostgreSQL's varstr_cmp() and break a tie with
 * memcmp so the order stays total; a nondeterministic one lets the tie stand.
 */
export function compare(id: ComparatorId, a: string, b: string, desc = false): number {
  const c = rawCompare(id, a, b);
  const det = COMPARATORS.find((x) => x.id === id)!.deterministic;
  const v = c !== 0 ? c : det ? memcmpBytes(keyOf(a).hex, keyOf(b).hex) : 0;
  return desc ? -v : v;
}

/* --------------------------------------------------- weight-key decomposition */

/** Base letter + secondary (accent) + tertiary (case) for the characters in this fixture. */
const DECOMP: Record<string, [string, string]> = {
  ä: ['a', '¨'],
  ü: ['u', '¨'],
  é: ['e', '´'],
};

export type WeightCell = { ch: string; l1: string; l2: string; l3: string; l4: string; ignored: boolean };

/**
 * How a weight-based comparator sees one key, level by level. `era` picks the
 * pre-2.28 rule (punctuation and unassigned characters vanish from levels 1-3).
 */
export function weightCells(s: string, era: 'old' | 'new'): WeightCell[] {
  const out: WeightCell[] = [];
  for (const ch of s) {
    const lower = ch.toLowerCase();
    const d = DECOMP[lower];
    const base = d ? d[0] : lower;
    const mark = d ? d[1] : '·';
    const isLetter = base >= 'a' && base <= 'z';
    const cased = ch !== lower ? 'ᴀ' : '·';
    if (isLetter) {
      out.push({ ch, l1: base, l2: mark, l3: cased, l4: ch, ignored: false });
      continue;
    }
    const sym = ch === ' ' ? '␠' : ch;
    const punct = ch === ' ' || ch === '-' || ch === "'";
    if (era === 'old') {
      // pre-2.28: punctuation is IGNORE at levels 1-3 and reappears only at level 4;
      // a character the table never assigned is ignorable at every level, which is
      // how two distinct strings end up collating equal.
      const assigned = punct;
      out.push({ ch, l1: '—', l2: '—', l3: '—', l4: assigned ? sym : '—', ignored: true });
      continue;
    }
    out.push({ ch, l1: punct ? sym : 'sym', l2: '·', l3: '·', l4: sym, ignored: false });
  }
  return out;
}

export const FIXTURE_PROVENANCE =
  'memcmp computed from the stored bytes; glibc columns modelled from the documented pre- and ' +
  'post-2.28 weight rules; ICU columns captured from Node v22.13.0 with ICU 76.1 / CLDR 46 / Unicode 16.0.';
