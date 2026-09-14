import { useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Check,
  Button,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
  makeRng,
  fmtNum,
} from './Viz';

/**
 * Nine queries that return a plausible wrong answer, run over one tiny dataset the
 * learner mutates.
 *
 * Every trap is evaluated by hand rather than by a parser, because the interesting state
 * is per-candidate: the three-valued verdict a predicate produced for each row, which
 * join row is the second copy of an order, which tuple the sort put on both pages. The
 * dataset is five customers, five orders and up to eleven line items; dropping a NULL
 * into it, or attaching the one-to-many child, is what moves the numbers.
 */

/* ====================================================================== config */

type Eng = 'pg' | 'mysql' | 'mysql_off' | 'sqlite';
type Anti = 'notin' | 'notexists' | 'leftjoin';
type TrapId = 'neq' | 'anti' | 'agg' | 'fanout' | 'group' | 'page' | 'between' | 'intdiv' | 'cast';

type Cfg = {
  eng: Eng;
  anti: Anti;
  nullCust: boolean;
  nullRegion: boolean;
  nullAmount: boolean;
  lines: number;
  distinct: boolean;
  seed: number;
};

const ENGINES: { value: Eng; label: string }[] = [
  { value: 'pg', label: 'PostgreSQL' },
  { value: 'mysql', label: 'MySQL 8 (default sql_mode)' },
  { value: 'mysql_off', label: 'MySQL 8, ONLY_FULL_GROUP_BY off' },
  { value: 'sqlite', label: 'SQLite' },
];

const isMysql = (e: Eng) => e === 'mysql' || e === 'mysql_off';

/* ======================================================================== data */

type Cust = { id: number; name: string; code: string; region: string | null };
type Ord = { id: number; cust: number | null; cents: number | null; at: string; ref: string; lines: number };

/** Line items per order, before the fan-out slider truncates them. */
const FULL_LINES: Record<number, number> = { 900: 3, 901: 2, 902: 3, 903: 1, 904: 2 };

function customers(cfg: Cfg): Cust[] {
  return [
    { id: 1, name: 'Acme', code: 'ACME', region: 'west' },
    { id: 2, name: 'Bolt', code: 'BOLT', region: 'west' },
    { id: 3, name: 'Cog', code: 'COG', region: cfg.nullRegion ? null : 'east' },
    { id: 4, name: 'Dyn', code: 'DYN', region: 'east' },
    { id: 5, name: 'Ely', code: 'ELY', region: 'west' },
  ];
}

function orders(cfg: Cfg): Ord[] {
  const raw: Ord[] = [
    { id: 900, cust: 1, cents: 25000, at: '2026-03-01 00:00', ref: '0900', lines: 0 },
    { id: 901, cust: 1, cents: 25000, at: '2026-03-17 09:30', ref: '0901', lines: 0 },
    { id: 902, cust: 2, cents: 25000, at: '2026-03-31 14:05', ref: '0902', lines: 0 },
    { id: 903, cust: 4, cents: cfg.nullAmount ? null : 6000, at: '2026-03-09 11:00', ref: '0903', lines: 0 },
    { id: 904, cust: cfg.nullCust ? null : 4, cents: 50000, at: '2026-04-02 08:00', ref: '0904', lines: 0 },
  ];
  return raw.map((o) => ({ ...o, lines: Math.min(FULL_LINES[o.id], cfg.lines) }));
}

function money(cents: number | null): string {
  if (cents === null) return 'NULL';
  return `$${(cents / 100).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function shuffle<T>(arr: readonly T[], seed: number): T[] {
  const rng = makeRng(seed);
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
  return a;
}

/* ===================================================================== results */

/** The verdict a single candidate got. Every one of these carries a text tag too. */
type Vk = 'TRUE' | 'FALSE' | 'UNKNOWN' | 'ERROR' | 'KEEP' | 'DUP' | 'SKIP' | 'DROP';

const VK_STYLE: Record<Vk, { fill: string; text: string }> = {
  TRUE: { fill: 'var(--viz-good)', text: 'TRUE' },
  FALSE: { fill: 'var(--viz-ink-muted)', text: 'FALSE' },
  UNKNOWN: { fill: 'var(--viz-warning)', text: 'UNKNOWN' },
  ERROR: { fill: 'var(--viz-critical)', text: 'ERROR' },
  KEEP: { fill: 'var(--viz-7)', text: 'counted' },
  DUP: { fill: 'var(--viz-serious)', text: 'counted again' },
  SKIP: { fill: 'var(--viz-ink-muted)', text: 'skipped' },
  DROP: { fill: 'var(--viz-stale)', text: 'dropped' },
};

type ERow = { key: string; label: string; expr: string; vk: Vk; tag?: string; note: string };
type Metric = { label: string; written: number; meant: number; fmt: (n: number) => string };

type Out = {
  sql: string;
  fix: string;
  rows: ERow[];
  rowsHead: string;
  cols: string[];
  result: string[][];
  truthRows: number;
  metrics: Metric[];
  head: string;
  body: string;
  error?: string;
};

const cnt = (n: number) => fmtNum(n);
const pct = (n: number) => n.toFixed(4);

/* ------------------------------------------------- 1. the inequality that hides */

function trapNeq(cfg: Cfg): Out {
  const cs = customers(cfg);
  const rows: ERow[] = cs.map((c) => {
    const lit = c.region === null ? 'NULL' : `'${c.region}'`;
    const vk: Vk = c.region === null ? 'UNKNOWN' : c.region !== 'west' ? 'TRUE' : 'FALSE';
    return {
      key: `c${c.id}`,
      label: `c#${c.id} ${c.name}`,
      expr: `${lit} <> 'west'`,
      vk,
      note:
        vk === 'UNKNOWN'
          ? 'NULL is not a value, so no comparison with it can be TRUE or FALSE. WHERE keeps only TRUE, so this row disappears from both region <> \'west\' and region = \'west\' — the two "opposite" queries do not partition the table.'
          : vk === 'TRUE'
            ? 'A real value that is not west. Kept.'
            : 'A real value equal to west. Rejected, correctly.',
    };
  });
  const kept = cs.filter((c) => c.region !== null && c.region !== 'west');
  const truth = cs.filter((c) => c.region === null || c.region !== 'west').length;
  const unknowns = rows.filter((r) => r.vk === 'UNKNOWN').length;
  return {
    sql: "SELECT id, name, region\nFROM customers\nWHERE region <> 'west';",
    fix: "WHERE region IS DISTINCT FROM 'west'      -- PostgreSQL / SQLite\nWHERE NOT (region <=> 'west')            -- MySQL",
    rows,
    rowsHead: 'WHERE evaluated per row, in three-valued logic',
    cols: ['id', 'name', 'region'],
    result: kept.map((c) => [String(c.id), c.name, c.region ?? 'NULL']),
    truthRows: truth,
    metrics: [{ label: 'customers reported as “not west”', written: kept.length, meant: truth, fmt: cnt }],
    head: unknowns
      ? 'One customer is in neither answer.'
      : 'With no NULL in region, the predicate partitions the table cleanly.',
    body: unknowns
      ? 'region <> \'west\' returned ' +
        `${kept.length} row${kept.length === 1 ? '' : 's'} and region = 'west' returns ` +
        `${cs.filter((c) => c.region === 'west').length}; the table has ${cs.length}. The NULL row evaluates to UNKNOWN in both, ` +
        'and WHERE discards anything that is not TRUE. This is the most common NULL bug in reporting code, because the ' +
        'two queries look exhaustive and the missing row is never an error. IS DISTINCT FROM (or MySQL’s NULL-safe <=>) ' +
        'compares NULL as a value and brings it back.'
      : 'Turn on “customers.region has a NULL” and watch a row vanish from both this query and its supposed complement.',
  };
}

/* ---------------------------------------------------------- 2. the anti-join trio */

function trapAnti(cfg: Cfg): Out {
  const cs = customers(cfg);
  const os = orders(cfg);
  const keys = os.map((o) => o.cust);
  const hasNull = keys.some((k) => k === null);
  const list = keys.map((k) => (k === null ? 'NULL' : String(k))).join(', ');

  const rows: ERow[] = cs.map((c) => {
    const match = keys.some((k) => k === c.id);
    let vk: Vk;
    let expr: string;
    let note: string;
    if (cfg.anti === 'notin') {
      vk = match ? 'FALSE' : hasNull ? 'UNKNOWN' : 'TRUE';
      expr = `${c.id} NOT IN (${list})`;
      note = match
        ? `NOT IN expands to NOT (${c.id} = 1 OR …). One disjunct is TRUE, so the OR is TRUE and NOT TRUE is FALSE. Correctly rejected.`
        : hasNull
          ? `Every equality is FALSE except ${c.id} = NULL, which is UNKNOWN. FALSE OR UNKNOWN is UNKNOWN, and NOT UNKNOWN is UNKNOWN — so this row is dropped even though it has no orders.`
          : 'Every equality is FALSE, the OR is FALSE, NOT FALSE is TRUE. Kept.';
    } else if (cfg.anti === 'notexists') {
      vk = match ? 'FALSE' : 'TRUE';
      expr = `NOT EXISTS (… WHERE o.customer_id = ${c.id})`;
      note = match
        ? 'The correlated subquery found a row, so EXISTS is TRUE and NOT EXISTS is FALSE.'
        : 'The subquery found nothing. EXISTS is FALSE — never UNKNOWN, because EXISTS only asks whether a row was produced, and o.customer_id = ' +
          `${c.id} is simply not TRUE for the NULL row.`;
    } else {
      vk = match ? 'FALSE' : 'TRUE';
      expr = `LEFT JOIN … → o.id IS ${match ? 'NOT NULL' : 'NULL'}`;
      note = match
        ? 'The outer join found a match, so o.id is a real id and the IS NULL filter rejects the row.'
        : 'No match, so the outer join manufactured an all-NULL right side and IS NULL is TRUE. IS NULL is a two-valued test, which is why this form is immune.';
    }
    return { key: `c${c.id}`, label: `c#${c.id} ${c.name}`, expr, vk, note };
  });

  const kept = rows.filter((r) => r.vk === 'TRUE').map((r) => cs.find((c) => `c${c.id}` === r.key)!);
  const truth = cs.filter((c) => !keys.some((k) => k === c.id)).length;
  const broken = cfg.anti === 'notin' && hasNull;
  return {
    sql:
      cfg.anti === 'notin'
        ? 'SELECT id, name\nFROM customers c\nWHERE c.id NOT IN (SELECT customer_id FROM orders);'
        : cfg.anti === 'notexists'
          ? 'SELECT id, name\nFROM customers c\nWHERE NOT EXISTS (\n  SELECT 1 FROM orders o WHERE o.customer_id = c.id);'
          : 'SELECT c.id, c.name\nFROM customers c\nLEFT JOIN orders o ON o.customer_id = c.id\nWHERE o.id IS NULL;',
    fix: 'WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)\n-- or NOT IN (SELECT customer_id FROM orders WHERE customer_id IS NOT NULL)',
    rows,
    rowsHead: 'The anti-join predicate, per candidate customer',
    cols: ['id', 'name'],
    result: kept.map((c) => [String(c.id), c.name]),
    truthRows: truth,
    metrics: [{ label: 'customers with no orders', written: kept.length, meant: truth, fmt: cnt }],
    head: broken
      ? 'NOT IN returned nothing at all.'
      : cfg.anti === 'notin'
        ? 'With no NULL in the subquery, NOT IN is correct.'
        : `${cfg.anti === 'notexists' ? 'NOT EXISTS' : 'The LEFT JOIN … IS NULL form'} is unaffected by the NULL.`,
    body: broken
      ? 'A single NULL anywhere in the subquery result makes NOT IN return the empty set, for every outer row that had no ' +
        'match. The failure is total and silent: no error, no warning, just a report that says nobody is idle. Switch the ' +
        'form to NOT EXISTS and the same NULL changes nothing, because EXISTS answers “did a row come back”, a question ' +
        'with only two answers.'
      : 'Turn on “orders.customer_id has a NULL”, then switch between the three forms. Only NOT IN collapses. ' +
        'Postgres also plans them differently: NOT IN forbids a hashed anti-join precisely because of this rule, so it ' +
        'is often materialised as a hashed subplan, while NOT EXISTS becomes a Hash Anti Join.',
  };
}

/* ------------------------------------------------------ 3. aggregates and NULLs */

function trapAgg(cfg: Cfg): Out {
  const os = orders(cfg);
  const rows: ERow[] = os.map((o) => ({
    key: `o${o.id}`,
    label: `o#${o.id}`,
    expr: `amount = ${money(o.cents)}, customer_id = ${o.cust === null ? 'NULL' : o.cust}`,
    vk: o.cents === null ? 'SKIP' : 'KEEP',
    note:
      o.cents === null
        ? 'SUM, AVG, MIN, MAX and COUNT(amount) all ignore this row entirely. COUNT(*) still counts it, because COUNT(*) counts rows, not values — it never even looks at a column.'
        : 'A real amount: contributes to every aggregate.',
  }));
  const vals = os.map((o) => o.cents).filter((c): c is number => c !== null);
  const sum = vals.reduce((a, b) => a + b, 0);
  const avg = vals.length ? sum / vals.length : NaN;
  const perRow = sum / os.length;
  const cntStar = os.length;
  const cntCust = os.filter((o) => o.cust !== null).length;
  const cntDistinct = new Set(os.filter((o) => o.cust !== null).map((o) => o.cust)).size;
  return {
    sql:
      'SELECT COUNT(*)                    AS rows,\n' +
      '       COUNT(customer_id)          AS with_customer,\n' +
      '       COUNT(DISTINCT customer_id) AS customers,\n' +
      '       SUM(amount)                 AS revenue,\n' +
      '       AVG(amount)                 AS avg_order\n' +
      'FROM orders;',
    fix: 'AVG(COALESCE(amount, 0))     -- if an unpriced order really is a $0 order\nSUM(amount) / COUNT(*)       -- revenue spread over every order, priced or not\nCOALESCE(SUM(amount), 0)     -- SUM over zero rows is NULL, not 0',
    rows,
    rowsHead: 'What each order contributes to each aggregate',
    cols: ['rows', 'with_customer', 'customers', 'revenue', 'avg_order'],
    result: [[String(cntStar), String(cntCust), String(cntDistinct), money(sum), Number.isNaN(avg) ? 'NULL' : money(Math.round(avg))]],
    truthRows: 1,
    metrics: [
      { label: 'average order value', written: Number.isNaN(avg) ? 0 : Math.round(avg), meant: Math.round(perRow), fmt: (n) => money(n) },
      { label: 'orders counted', written: cntCust, meant: cntStar, fmt: cnt },
    ],
    head:
      vals.length === os.length && cntCust === os.length
        ? 'With no NULLs, every count agrees.'
        : 'Five aggregates over the same five rows, and three different denominators.',
    body:
      `COUNT(*) = ${cntStar} counts rows. COUNT(customer_id) = ${cntCust} counts non-NULL values in that column. ` +
      `AVG(amount) divides by ${vals.length}, not by ${cntStar}, so an order with no price quietly raises the average ` +
      'instead of lowering it. None of this is a rounding difference — it is a different question being answered. ' +
      'And the empty case is worse: SUM over zero rows returns NULL, not 0, so a COALESCE belongs on the outside of the ' +
      'SUM, not on the inside where it would change the meaning.',
  };
}

/* ---------------------------------------------------------- 4. join fan-out */

function trapFanout(cfg: Cfg): Out {
  const cs = customers(cfg);
  const os = orders(cfg);
  const rows: ERow[] = [];
  type Agg = { name: string; joinRows: number; sumAll: number; amounts: Set<number>; orderIds: Set<number> };
  const agg = new Map<number, Agg>();

  for (const c of cs) {
    const mine = os.filter((o) => o.cust === c.id);
    for (const o of mine) {
      if (o.lines === 0) {
        rows.push({
          key: `d${o.id}`,
          label: `c#${c.id} ${c.name} · o#${o.id}`,
          expr: `${money(o.cents)} · no line items`,
          vk: 'DROP',
          note: 'The inner join to order_lines has nothing to match, so the order vanishes from the result. Fan-out inflates some numbers and deletes others in the same query.',
        });
        continue;
      }
      for (let i = 0; i < o.lines; i++) {
        const a = agg.get(c.id) ?? { name: c.name, joinRows: 0, sumAll: 0, amounts: new Set<number>(), orderIds: new Set<number>() };
        a.joinRows += 1;
        if (o.cents !== null) {
          a.sumAll += o.cents;
          a.amounts.add(o.cents);
        }
        a.orderIds.add(o.id);
        agg.set(c.id, a);
        rows.push({
          key: `l${o.id}-${i}`,
          label: `c#${c.id} ${c.name} · o#${o.id}`,
          expr: `line ${i + 1}/${o.lines} → SUM += ${money(o.cents)}`,
          vk: i === 0 ? 'KEEP' : 'DUP',
          note:
            i === 0
              ? 'The first join row for this order: the amount is added once, correctly.'
              : `The join produced a second copy of order #${o.id} because it has ${o.lines} line items. orders.amount is a ` +
                'property of the order, not of the line, so every extra line adds the whole order total again.',
        });
      }
    }
  }

  for (const o of os) {
    if (o.cust === null) {
      rows.push({
        key: `n${o.id}`,
        label: `o#${o.id}`,
        expr: `${money(o.cents)} · customer_id IS NULL`,
        vk: 'DROP',
        note: 'o.customer_id = c.id is UNKNOWN for every customer, so an inner join finds no partner and the order is not in anybody\u2019s revenue.',
      });
    }
  }

  const groups = [...agg.entries()].sort((a, b) => a[0] - b[0]);
  const truthFor = (cid: number) =>
    os.filter((o) => o.cust === cid).reduce((s, o) => s + (o.cents ?? 0), 0);
  const withOrders = cs.filter((c) => os.some((o) => o.cust === c.id));

  const result = groups.map(([, a]) => [
    a.name,
    money(cfg.distinct ? [...a.amounts].reduce((s, v) => s + v, 0) : a.sumAll),
    String(cfg.distinct ? a.orderIds.size : a.joinRows),
  ]);
  const metrics: Metric[] = withOrders.map((c) => {
    const a = agg.get(c.id);
    const written = a ? (cfg.distinct ? [...a.amounts].reduce((s, v) => s + v, 0) : a.sumAll) : 0;
    return { label: `revenue, ${c.name}`, written, meant: truthFor(c.id), fmt: money };
  });

  const dups = rows.filter((r) => r.vk === 'DUP').length;
  const dropped = rows.filter((r) => r.vk === 'DROP').length;
  const wrong = metrics.some((m) => m.written !== m.meant);
  return {
    sql:
      'SELECT c.name,\n' +
      `       SUM(${cfg.distinct ? 'DISTINCT ' : ''}o.amount) AS revenue,\n` +
      `       COUNT(${cfg.distinct ? 'DISTINCT o.id' : '*'})   AS orders\n` +
      'FROM customers c\n' +
      'JOIN orders o      ON o.customer_id = c.id\n' +
      'JOIN order_lines l ON l.order_id   = o.id\n' +
      'GROUP BY c.name;',
    fix:
      'SELECT c.name, o.revenue\n' +
      'FROM customers c\n' +
      'JOIN LATERAL (SELECT SUM(amount) AS revenue FROM orders\n' +
      '              WHERE customer_id = c.id) o ON true;\n' +
      '-- aggregate each one-to-many side before you join it to another',
    rows,
    rowsHead: 'Every row the join actually produced, and what it added to SUM',
    cols: ['name', 'revenue', 'orders'],
    result,
    truthRows: withOrders.length,
    metrics,
    head: cfg.distinct
      ? wrong
        ? 'DISTINCT made the row count right and the money wrong.'
        : 'DISTINCT happens to agree here — because no two orders share an amount.'
      : dups > 0
        ? `${dups} join rows are duplicates, and every one of them added a full order total again.`
        : dropped > 0
          ? 'No duplicates — but orders with no line items have disappeared.'
          : 'One line item per order: fan-out is ×1 and the totals are right.',
    body: cfg.distinct
      ? 'SUM(DISTINCT o.amount) deduplicates by *value*, not by row identity. Acme placed two separate $250.00 orders; ' +
        'DISTINCT cannot tell them apart, so it reports $250.00. COUNT(DISTINCT o.id) is genuinely correct because ids are ' +
        'unique — which is exactly what makes the pattern so convincing: the row count looks fixed while the money is now ' +
        'wrong in the other direction.'
      : 'The join key is the order, but SUM is over a column that belongs to the order, so the multiplier is the number of ' +
        'child rows — and it is different per order, so the error is not even a constant factor you could divide out. ' +
        'Drag the line-items slider and watch each customer’s revenue scale by its own fan-out. At 0 line items the inner ' +
        'join silently deletes orders instead.',
  };
}

/* --------------------------------------------- 5. GROUP BY and bare columns */

function trapGroup(cfg: Cfg): Out {
  const cs = customers(cfg);
  const keys: (string | null)[] = [];
  for (const c of cs) if (!keys.some((k) => k === c.region)) keys.push(c.region);

  const rng = makeRng(cfg.seed * 7919 + 13);
  const picks = new Map<string, number>();
  for (const k of keys) {
    const members = cs.filter((c) => c.region === k);
    picks.set(String(k), members[Math.floor(rng() * members.length)].id);
  }

  const err =
    cfg.eng === 'pg'
      ? 'ERROR:  column "c.name" must appear in the GROUP BY clause or be used in an aggregate function'
      : cfg.eng === 'mysql'
        ? "ERROR 1055 (42000): Expression #2 of SELECT list is not in GROUP BY clause and contains nonaggregated column 'shop.c.name' which is not functionally dependent on columns in GROUP BY clause; this is incompatible with sql_mode=only_full_group_by"
        : undefined;

  const rows: ERow[] = cs.map((c) => {
    const chosen = picks.get(String(c.region)) === c.id;
    return {
      key: `c${c.id}`,
      label: `c#${c.id} ${c.name}`,
      expr: `group key = ${c.region === null ? 'NULL' : `'${c.region}'`}`,
      vk: (err ? 'ERROR' : chosen ? 'KEEP' : 'SKIP') as Vk,
      note: err
        ? 'No row is emitted at all: the statement is rejected before execution, because c.name is neither grouped nor aggregated and region is not a key that determines it.'
        : chosen
          ? 'This row’s name is the one the engine happened to emit for the group. Nothing in the query asked for it, and nothing guarantees it next time.'
          : 'This row is inside the group, and its name is discarded without a word. If the groups were built in a different order, a different name would win.',
    };
  });

  const result = err
    ? []
    : keys.map((k) => {
        const members = cs.filter((c) => c.region === k);
        const pick = members.find((c) => c.id === picks.get(String(k)))!;
        return [k === null ? 'NULL' : k, pick.name, String(members.length)];
      });

  return {
    sql: 'SELECT c.region, c.name, COUNT(*) AS customers\nFROM customers c\nGROUP BY c.region;',
    fix:
      'SELECT c.region, MIN(c.name) AS a_name, COUNT(*) FROM customers c GROUP BY c.region;\n' +
      '-- or group by a key: GROUP BY c.id lets both engines emit c.name by functional dependency',
    rows,
    rowsHead: 'Which row inside each group supplied the bare c.name',
    cols: ['region', 'name', 'customers'],
    result,
    truthRows: keys.length,
    metrics: [
      { label: 'rows returned', written: result.length, meant: keys.length, fmt: cnt },
      { label: 'names silently discarded', written: err ? 0 : cs.length - keys.length, meant: 0, fmt: cnt },
    ],
    error: err,
    head: err
      ? 'This query does not run here.'
      : 'It ran, and picked a name out of each group for you.',
    body: err
      ? 'c.name is neither in the GROUP BY nor inside an aggregate, and the engine cannot prove it is functionally ' +
        'dependent on the group key — region is not a key. Both PostgreSQL and MySQL 8 with the default sql_mode reject ' +
        'it. Group by c.id instead and both accept the bare c.name, because SQL:1999’s functional-dependency rule makes ' +
        'a primary key determine the whole row. Switch the engine to see what the permissive dialects do with the same text.'
      : `${cs.length} customers collapsed into ${keys.length} groups and ${cs.length - keys.length} names were thrown ` +
        'away. The value you get back is from an unspecified row — SQLite documents it as arbitrary, MySQL with ' +
        'ONLY_FULL_GROUP_BY off says the server is free to choose any value from each group — so a plan change, an index ' +
        'change or a new row can change the answer with no code change at all. ' +
        (cfg.nullRegion
          ? 'Note also the NULL group: GROUP BY puts all NULLs together even though NULL = NULL is UNKNOWN, because grouping compares with "not distinct", not with =.'
          : 'Turn on the NULL region to see GROUP BY collect all NULLs into one group, even though NULL = NULL is UNKNOWN.'),
  };
}

/* --------------------------------------------- 6. ORDER BY without a tiebreaker */

function trapPage(cfg: Cfg): Out {
  const os = orders(cfg);
  const ids = os.map((o) => o.id);
  const physA = shuffle(ids, cfg.seed);
  const physB = shuffle(ids, cfg.seed + 101);
  // PostgreSQL sorts NULL as larger than every value (so DESC puts it first); MySQL and
  // SQLite sort it as smaller.
  const nullRank = cfg.eng === 'pg' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  const amt = (id: number) => os.find((o) => o.id === id)!.cents ?? nullRank;
  const sortDesc = (phys: number[]) => phys.slice().sort((a, b) => amt(b) - amt(a));
  const page1 = sortDesc(physA).slice(0, 2);
  const page2 = sortDesc(physB).slice(2, 4);
  const seen = [...page1, ...page2];
  const dupIds = seen.filter((v, i) => seen.indexOf(v) !== i);
  const expected = sortDesc(physA).slice(0, 4);
  const missing = expected.filter((id) => !seen.includes(id));

  const rows: ERow[] = os
    .slice()
    .sort((a, b) => (b.cents ?? -1) - (a.cents ?? -1))
    .map((o) => {
      const on1 = page1.includes(o.id);
      const on2 = page2.includes(o.id);
      const vk: Vk = on1 && on2 ? 'DUP' : on1 || on2 ? 'KEEP' : 'DROP';
      const tied = os.filter((x) => x.cents === o.cents).length > 1;
      return {
        key: `o${o.id}`,
        label: `o#${o.id}`,
        expr: `amount ${money(o.cents)}${tied ? ' — tied' : ''} · ${on1 && on2 ? 'page 1 AND page 2' : on1 ? 'page 1' : on2 ? 'page 2' : 'neither page'}`,
        vk,
        note: tied
          ? 'This row is inside a tie group. ORDER BY amount DESC constrains nothing about the order within the group, so the physical scan order decides — and the physical order changed between the two page requests.'
          : 'Its sort key is unique, so its position is fully determined and it lands on the same page every time.',
      };
    });

  const label = (id: number) => `#${id}`;
  return {
    sql:
      'SELECT id, amount FROM orders ORDER BY amount DESC LIMIT 2 OFFSET 0;  -- page 1\n' +
      'SELECT id, amount FROM orders ORDER BY amount DESC LIMIT 2 OFFSET 2;  -- page 2',
    fix:
      'ORDER BY amount DESC, id DESC              -- a unique tiebreaker makes the order total\n' +
      'WHERE (amount, id) < (:last_amount, :last_id)  -- keyset pagination: no OFFSET at all',
    rows,
    rowsHead: 'Where each order landed across the two page requests',
    cols: ['page', 'id', 'amount'],
    result: [
      ...page1.map((id) => ['1', label(id), money(amt(id))]),
      ...page2.map((id) => ['2', label(id), money(amt(id))]),
    ],
    truthRows: 4,
    metrics: [
      { label: 'distinct orders across both pages', written: new Set(seen).size, meant: 4, fmt: cnt },
      { label: 'rows shown twice', written: dupIds.length, meant: 0, fmt: cnt },
    ],
    head: dupIds.length
      ? `Order ${dupIds.map(label).join(', ')} appears on both pages${missing.length ? ` and ${missing.map(label).join(', ')} appears on neither` : ''}.`
      : 'At this physical order the two pages happen to agree.',
    body:
      'Three orders share the amount $250.00, and the tie group straddles the page boundary. ORDER BY defines a total ' +
      'order only if the key is unique; otherwise the engine may emit tied rows in any order, and it will — a Postgres ' +
      'UPDATE writes a new tuple version at the end of the heap, synchronize_seqscans starts a scan wherever another ' +
      'scan already is, and a parallel plan interleaves workers non-deterministically. Press “re-plan” to change the ' +
      'physical order between the two requests, which is exactly what a concurrent write does to a paginated API.' +
      (cfg.nullAmount
        ? ' The NULL amount moves too: PostgreSQL sorts NULL as larger than every value, so DESC puts it first, while MySQL and SQLite put it last.'
        : ''),
  };
}

/* ---------------------------------------------------- 7. BETWEEN on timestamps */

function trapBetween(cfg: Cfg): Out {
  const os = orders(cfg);
  const lo = '2026-03-01 00:00';
  const hi = '2026-03-31 00:00';
  const rows: ERow[] = os.map((o) => {
    const inRange = o.at >= lo && o.at <= hi;
    const inMarch = o.at.startsWith('2026-03');
    return {
      key: `o${o.id}`,
      label: `o#${o.id}`,
      expr: `'${o.at}' BETWEEN '${lo}' AND '${hi}'`,
      vk: inRange ? 'TRUE' : 'FALSE',
      note: inRange
        ? 'Inside the closed interval on both ends.'
        : inMarch
          ? "The date literal '2026-03-31' is cast to the timestamp 2026-03-31 00:00:00, so every order placed during that day is greater than the upper bound. A whole day of revenue is missing and the query looks right."
          : 'Genuinely outside March.',
    };
  });
  const kept = os.filter((o) => o.at >= lo && o.at <= hi);
  const truth = os.filter((o) => o.at.startsWith('2026-03'));
  const sum = kept.reduce((s, o) => s + (o.cents ?? 0), 0);
  const truthSum = truth.reduce((s, o) => s + (o.cents ?? 0), 0);
  return {
    sql:
      "SELECT id, placed_at, amount\nFROM orders\nWHERE placed_at BETWEEN DATE '2026-03-01' AND DATE '2026-03-31';",
    fix: "WHERE placed_at >= DATE '2026-03-01'\n  AND placed_at <  DATE '2026-04-01'   -- half-open, and still index-friendly",
    rows,
    rowsHead: 'BETWEEN expanded to its two comparisons',
    cols: ['id', 'placed_at', 'amount'],
    result: kept.map((o) => [String(o.id), o.at, money(o.cents)]),
    truthRows: truth.length,
    metrics: [
      { label: 'March orders found', written: kept.length, meant: truth.length, fmt: cnt },
      { label: 'March revenue', written: sum, meant: truthSum, fmt: money },
    ],
    head: kept.length === truth.length ? 'No rows are lost with this data.' : 'The last day of the month is missing.',
    body:
      'BETWEEN is defined as x >= lo AND x <= hi — inclusive at both ends — and that is exactly the problem when x is a ' +
      'timestamp and hi is a date. The bound becomes midnight, so BETWEEN covers 30 days and 0 seconds of March. The ' +
      'half-open form >= start AND < next_start is correct for every resolution, for time zones, and for a column whose ' +
      'type someone later widens from date to timestamptz.',
  };
}

/* ------------------------------------------------------- 8. integer division */

function trapIntDiv(cfg: Cfg): Out {
  const os = orders(cfg);
  const big = os.filter((o) => (o.cents ?? 0) > 10000).length;
  const total = os.length;
  const exact = big / total;
  const truncates = cfg.eng === 'pg' || cfg.eng === 'sqlite';
  const written = truncates ? Math.trunc(exact) : exact;
  const rows: ERow[] = os.map((o) => ({
    key: `o${o.id}`,
    label: `o#${o.id}`,
    expr: `CASE WHEN ${o.cents === null ? 'NULL' : o.cents} > 10000 THEN 1 ELSE 0 END → ${(o.cents ?? 0) > 10000 ? 1 : 0}`,
    vk: (o.cents ?? 0) > 10000 ? 'KEEP' : 'SKIP',
    note:
      o.cents === null
        ? 'NULL > 10000 is UNKNOWN, which is not TRUE, so CASE takes the ELSE branch and contributes 0. A NULL price is counted as a small order.'
        : (o.cents ?? 0) > 10000
          ? 'Contributes 1 to the numerator.'
          : 'Contributes 0 to the numerator, and 1 to COUNT(*) all the same.',
  }));
  return {
    sql:
      'SELECT SUM(CASE WHEN amount_cents > 10000 THEN 1 ELSE 0 END)\n       / COUNT(*) AS big_order_rate\nFROM orders;',
    fix:
      'SUM(CASE WHEN amount_cents > 10000 THEN 1 ELSE 0 END)::numeric / COUNT(*)\n' +
      'AVG(CASE WHEN amount_cents > 10000 THEN 1 ELSE 0 END)   -- avg(integer) is numeric in PostgreSQL',
    rows,
    rowsHead: 'The numerator, one row at a time',
    cols: ['big_order_rate'],
    result: [[truncates ? '0' : pct(exact)]],
    truthRows: 1,
    metrics: [{ label: 'big-order rate', written, meant: exact, fmt: (n) => n.toFixed(4) }],
    head: truncates
      ? `${big} / ${total} came back as 0.`
      : `${big} / ${total} came back as ${pct(exact)} — MySQL’s / is not integer division.`,
    body: truncates
      ? 'Both operands are integers, so / is integer division and truncates toward zero. Every rate, every percentage and ' +
        'every per-capita metric computed this way is 0 until the numerator exceeds the denominator, which never happens ' +
        'for a ratio. The dashboard shows a flat zero line and nobody gets an error. ' +
        (cfg.eng === 'pg'
          ? 'Postgres also truncates rather than rounds: 9 / 5 is 1, and (-9) / 5 is -1.'
          : 'SQLite applies the same rule whenever both operands have integer values.')
      : 'MySQL’s / on integer or decimal operands produces a DECIMAL — the scale is the dividend’s scale plus ' +
        'div_precision_increment, 4 by default — ' +
        'which is why the identical SQL that returns 0 on PostgreSQL and SQLite returns a real fraction here. ' +
        'MySQL’s integer division operator is the separate DIV. Portable code casts explicitly rather than relying on ' +
        'either behaviour.',
  };
}

/* -------------------------------------------- 9. implicit casts and collation */

function trapCast(cfg: Cfg): Out {
  const cs = customers(cfg);
  const os = orders(cfg);
  const ci = isMysql(cfg.eng);
  const rows: ERow[] = [
    ...cs.map((c) => {
      const hit = ci ? c.code.toLowerCase() === 'acme' : c.code === 'acme';
      return {
        key: `c${c.id}`,
        label: `c#${c.id} ${c.code}`,
        expr: `'${c.code}' = 'acme'`,
        vk: (hit ? 'TRUE' : 'FALSE') as Vk,
        note: ci
          ? 'MySQL 8’s default collation utf8mb4_0900_ai_ci is accent- and case-insensitive, so the comparison folds case before comparing.'
          : 'A deterministic collation compares the strings as given: ACME and acme are different strings, and no index or function will change that.',
      };
    }),
    ...os.map((o) => {
      const vk: Vk = cfg.eng === 'pg' ? 'ERROR' : ci ? (Number(o.ref) === 904 ? 'TRUE' : 'FALSE') : 'FALSE';
      return {
        key: `o${o.id}`,
        label: `o#${o.id} ref='${o.ref}'`,
        expr: cfg.eng === 'pg' ? `'${o.ref}'::varchar = 904::integer` : ci ? `CAST('${o.ref}' AS DOUBLE) = 904` : `'${o.ref}' = '904'`,
        vk,
        note:
          cfg.eng === 'pg'
            ? 'PostgreSQL has no varchar = integer operator and refuses to invent one. The error is loud, at parse time, and it is the behaviour you want.'
            : ci
              ? 'MySQL converts BOTH sides to DOUBLE when a string column meets a number, so the leading zero disappears and ’0904’ matches. The conversion is applied to the column, so an index on ref cannot be used and the query becomes a full scan.'
              : 'SQLite applies TEXT affinity to the literal because the column has TEXT affinity, so 904 becomes ’904’ and never matches ’0904’.',
      };
    }),
  ];
  const codeHits = rows.slice(0, cs.length).filter((r) => r.vk === 'TRUE').length;
  const refHits = rows.slice(cs.length).filter((r) => r.vk === 'TRUE').length;
  const err =
    cfg.eng === 'pg'
      ? 'ERROR:  operator does not exist: character varying = integer\nHINT:  No operator matches the given name and argument types. You might need to add explicit type casts.'
      : undefined;
  return {
    sql: "SELECT * FROM customers WHERE code = 'acme';\nSELECT * FROM orders    WHERE ref  = 904;",
    fix: "WHERE lower(code) = 'acme'        -- and index lower(code), or use a nondeterministic ICU collation\nWHERE ref = '0904'                -- compare a string column to a string, so the index stays usable",
    rows,
    rowsHead: 'Both predicates, evaluated under this engine’s coercion and collation rules',
    cols: ['predicate', 'rows matched'],
    result: [
      ["code = 'acme'", String(codeHits)],
      ['ref = 904', err ? 'error' : String(refHits)],
    ],
    truthRows: 2,
    metrics: [
      { label: "rows matched by code = 'acme'", written: codeHits, meant: 1, fmt: cnt },
      { label: 'rows matched by ref = 904', written: err ? 0 : refHits, meant: 1, fmt: cnt },
    ],
    error: err,
    head:
      cfg.eng === 'pg'
        ? 'One query returns nothing; the other refuses to run.'
        : ci
          ? 'Both queries match — for two different reasons you did not write down.'
          : 'Both queries return nothing, silently.',
    body:
      'The same two statements give three different answers across three engines, and none of them is a bug. Case ' +
      'sensitivity is a property of the column’s collation, not of the engine, so the same schema migrated from MySQL to ' +
      'PostgreSQL changes which rows a login query finds. And an implicit cast is not merely a semantic hazard: because ' +
      'MySQL converts the column side to DOUBLE, the B-tree on ref is unusable and the plan degrades to a full scan the ' +
      'day the table gets large. Collation mismatches do the same to joins — join a utf8mb4_general_ci column to a ' +
      'utf8mb4_0900_ai_ci one and MySQL either errors with “Illegal mix of collations” or converts one side and drops the index.',
  };
}

/* ======================================================================= traps */

const TRAPS: { value: TrapId; label: string; run: (c: Cfg) => Out }[] = [
  { value: 'neq', label: '1 — region <> \'west\' loses a row', run: trapNeq },
  { value: 'anti', label: '2 — NOT IN vs NOT EXISTS vs LEFT JOIN', run: trapAnti },
  { value: 'agg', label: '3 — COUNT(*), COUNT(col) and AVG disagree', run: trapAgg },
  { value: 'fanout', label: '4 — join fan-out doubles the money', run: trapFanout },
  { value: 'group', label: '5 — a bare column in GROUP BY', run: trapGroup },
  { value: 'page', label: '6 — ORDER BY with no unique tiebreaker', run: trapPage },
  { value: 'between', label: '7 — BETWEEN drops the last day', run: trapBetween },
  { value: 'intdiv', label: '8 — integer division returns 0', run: trapIntDiv },
  { value: 'cast', label: '9 — implicit casts and collation', run: trapCast },
];

/* =================================================================== component */

const MAX_ROWS = 14;

export default function NullFanoutTrapSandbox() {
  const [trap, setTrap] = useState<TrapId>('anti');
  const [eng, setEng] = useState<Eng>('pg');
  const [anti, setAnti] = useState<Anti>('notin');
  const [nullCust, setNullCust] = useState(true);
  const [nullRegion, setNullRegion] = useState(true);
  const [nullAmount, setNullAmount] = useState(false);
  const [lines, setLines] = useState(2);
  const [distinct, setDistinct] = useState(false);
  const [seed, setSeed] = useState(1);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const cfg: Cfg = { eng, anti, nullCust, nullRegion, nullAmount, lines, distinct, seed };
  const out = TRAPS.find((t) => t.value === trap)!.run(cfg);

  const shown = out.rows.slice(0, MAX_ROWS);
  const hidden = out.rows.length - shown.length;

  /* layout */
  const svgW = Math.max(width, 880);
  const rowH = 24;
  const top = 34;
  const height = top + Math.max(shown.length, 5) * rowH + 24;
  const leftW = Math.round(svgW * 0.58);
  const labelW = 128;
  const chipW = 104;
  const barX = leftW + 18;
  const barW = svgW - barX - 96;
  const metrics = out.metrics.slice(0, 4);
  const barBandH = Math.min(46, (height - top - 20) / Math.max(metrics.length, 1));

  const unknowns = out.rows.filter((r) => r.vk === 'UNKNOWN').length;
  const dups = out.rows.filter((r) => r.vk === 'DUP').length;
  const m0 = metrics[0];

  return (
    <VizPanel
      title="Nine queries that return a plausible wrong answer"
      subtitle="Five customers, five orders, up to eleven line items. Drop a NULL into the data, attach the one-to-many child, or swap the anti-join form, and watch the result set and the aggregates move apart from the answer you meant."
      controls={
        <>
          <Choice label="Trap" value={trap} options={TRAPS.map((t) => ({ value: t.value, label: t.label }))} onChange={setTrap} />
          <Choice label="Engine" value={eng} options={ENGINES} onChange={setEng} />
          <Segmented
            label="Anti-join form"
            value={anti}
            onChange={setAnti}
            options={[
              { value: 'notin', label: 'NOT IN', title: 'NOT (x = ANY (…)) — three-valued' },
              { value: 'notexists', label: 'NOT EXISTS', title: 'Correlated subquery — two-valued' },
              { value: 'leftjoin', label: 'LEFT JOIN … IS NULL', title: 'Outer join plus a two-valued IS NULL test' },
            ]}
          />
          <Check label="orders.customer_id has a NULL" checked={nullCust} onChange={setNullCust} />
          <Check label="customers.region has a NULL" checked={nullRegion} onChange={setNullRegion} />
          <Check label="orders.amount has a NULL" checked={nullAmount} onChange={setNullAmount} />
          <Slider label="Line items per order" min={0} max={3} value={lines} onChange={setLines} format={(n) => (n === 0 ? 'none' : `up to ${n}`)} />
          <Check label="DISTINCT" checked={distinct} onChange={setDistinct} />
          <Button onClick={() => setSeed((s) => s + 1)} title="Change the physical row order between the two page requests">
            Re-plan
          </Button>
          <Button onClick={() => { setNullCust(true); setNullRegion(true); setNullAmount(false); setLines(2); setDistinct(false); setSeed(1); setAnti('notin'); }}>
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'TRUE — the row is kept', color: VK_STYLE.TRUE.fill },
            { label: 'FALSE — rejected on purpose', color: VK_STYLE.FALSE.fill },
            { label: 'UNKNOWN — neither, so WHERE drops it', color: VK_STYLE.UNKNOWN.fill },
            { label: 'ERROR — the engine refuses the statement', color: VK_STYLE.ERROR.fill },
            { label: 'counted once', color: VK_STYLE.KEEP.fill },
            { label: 'counted again (fan-out / both pages)', color: VK_STYLE.DUP.fill },
            { label: 'skipped or dropped', color: VK_STYLE.DROP.fill },
            { label: 'as written', color: 'var(--viz-2)', shape: 'square' },
            { label: 'what you meant', color: 'var(--viz-1)', shape: 'square' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Rows returned', value: out.error ? 'error' : cnt(out.result.length) },
            { label: 'The correct form returns', value: cnt(out.truthRows), hint: 'What the query in “the fix” below produces over the same data' },
            { label: 'UNKNOWN verdicts', value: cnt(unknowns), hint: 'Rows the predicate could neither accept nor reject, which WHERE therefore discards' },
            { label: 'Double-counted rows', value: cnt(dups), hint: 'Join rows that added an amount a second time, or rows served on both pages' },
            m0
              ? { label: `${m0.label}, as written`, value: m0.fmt(m0.written) }
              : { label: 'metric', value: '—' },
            m0 ? { label: `${m0.label}, as meant`, value: m0.fmt(m0.meant) } : { label: 'metric', value: '—' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{out.head}</strong> {out.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>candidate</th>
                <th>expression</th>
                <th>verdict</th>
              </tr>
            </thead>
            <tbody>
              {out.rows.length === 0 ? (
                <tr>
                  <td colSpan={3}>no candidate rows</td>
                </tr>
              ) : (
                out.rows.map((r) => (
                  <tr key={r.key}>
                    <td>{r.label}</td>
                    <td>{r.expr}</td>
                    <td>{VK_STYLE[r.vk].text}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>measure</th>
                <th>as written</th>
                <th>what you meant</th>
                <th>error</th>
              </tr>
            </thead>
            <tbody>
              {out.metrics.map((m) => (
                <tr key={m.label}>
                  <td>{m.label}</td>
                  <td>{m.fmt(m.written)}</td>
                  <td>{m.fmt(m.meant)}</td>
                  <td>{m.meant === 0 ? (m.written === 0 ? '0%' : '—') : `${(((m.written - m.meant) / m.meant) * 100).toFixed(1)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>id</th>
                <th>customer_id</th>
                <th>amount</th>
                <th>placed_at</th>
                <th>ref</th>
                <th>line items</th>
              </tr>
            </thead>
            <tbody>
              {orders(cfg).map((o) => (
                <tr key={o.id}>
                  <td>{o.id}</td>
                  <td>{o.cust === null ? 'NULL' : o.cust}</td>
                  <td>{money(o.cents)}</td>
                  <td>{o.at}</td>
                  <td>{o.ref}</td>
                  <td>{o.lines}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <pre
          style={{
            margin: '0 0 .5rem',
            padding: '.5rem .6rem',
            fontSize: '.8125rem',
            lineHeight: 1.5,
            color: 'var(--viz-ink)',
            background: 'var(--viz-plane)',
            border: `1px solid ${out.error ? 'var(--viz-critical)' : 'var(--viz-border)'}`,
            borderRadius: '6px',
            overflowX: 'auto',
          }}
        >
          {out.sql}
        </pre>

        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Per-row evaluation of the selected query, and the aggregate it produced next to the aggregate that was meant">
            <text x={0} y={14} fill="var(--viz-ink-2)">
              {out.rowsHead}
            </text>
            <text x={barX} y={14} fill="var(--viz-ink-2)">
              as written vs what you meant
            </text>

            {shown.map((r, i) => {
              const y = top + i * rowH;
              const st = VK_STYLE[r.vk];
              return (
                <g key={r.key} {...tip(<><strong>{r.label}</strong><br />{r.expr}<br />{r.note}</>)} style={{ cursor: 'help' }}>
                  <rect x={0} y={y} width={leftW} height={rowH - 4} rx={5} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                  <text x={8} y={y + 14} fill="var(--viz-ink)" fontWeight={600}>
                    {r.label}
                  </text>
                  <text x={labelW} y={y + 14} fill="var(--viz-ink-2)">
                    {r.expr.length > 46 ? `${r.expr.slice(0, 45)}…` : r.expr}
                  </text>
                  <rect
                    x={leftW - chipW - 6}
                    y={y + 2}
                    width={chipW}
                    height={rowH - 8}
                    rx={4}
                    fill={st.fill}
                    fillOpacity={0.18}
                    stroke={st.fill}
                    strokeWidth={1.5}
                    strokeDasharray={r.vk === 'FALSE' || r.vk === 'SKIP' || r.vk === 'DROP' ? '3 2' : undefined}
                  />
                  <text x={leftW - chipW / 2 - 6} y={y + 14} textAnchor="middle" fill="var(--viz-ink)" fontWeight={600}>
                    {st.text}
                  </text>
                </g>
              );
            })}
            {hidden > 0 ? (
              <text x={8} y={top + shown.length * rowH + 14} fill="var(--viz-ink-muted)">
                … {hidden} more candidate row{hidden === 1 ? '' : 's'}, all of them in “Show the numbers”
              </text>
            ) : null}

            {metrics.map((m, i) => {
              const y = top + i * barBandH;
              const max = Math.max(m.written, m.meant, 1);
              const w1 = Math.max(2, (m.written / max) * barW);
              const w2 = Math.max(2, (m.meant / max) * barW);
              const h = Math.min(11, barBandH / 4);
              return (
                <g key={m.label} {...tip(<><strong>{m.label}</strong><br />as written: {m.fmt(m.written)}<br />what you meant: {m.fmt(m.meant)}</>)}>
                  <text x={barX} y={y + 10} fill="var(--viz-ink-2)">
                    {m.label}
                  </text>
                  <rect x={barX} y={y + 15} width={w1} height={h} rx={2} fill="var(--viz-2)" />
                  <text x={barX + w1 + 6} y={y + 15 + h - 1} fill="var(--viz-ink)">
                    {m.fmt(m.written)}
                  </text>
                  <rect x={barX} y={y + 17 + h} width={w2} height={h} rx={2} fill="var(--viz-1)" />
                  <text x={barX + w2 + 6} y={y + 17 + 2 * h - 1} fill="var(--viz-ink)">
                    {m.fmt(m.meant)}
                  </text>
                </g>
              );
            })}
          </svg>
        </TooltipHost>

        <div style={{ marginTop: '.5rem' }}>
          <p style={{ margin: '0 0 .15rem', color: 'var(--viz-ink-2)', fontSize: '.78rem' }}>
            {out.error ? 'The engine refused the statement' : `Result set — ${out.result.length} row${out.result.length === 1 ? '' : 's'}`}
          </p>
          {out.error ? (
            <pre style={{ margin: 0, padding: '.5rem .6rem', fontSize: '.78rem', color: 'var(--viz-critical)', background: 'var(--viz-plane)', border: '1px solid var(--viz-border)', borderRadius: '6px', overflowX: 'auto' }}>
              {out.error}
            </pre>
          ) : (
            <table className="viz-table">
              <thead>
                <tr>
                  {out.cols.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {out.result.length === 0 ? (
                  <tr>
                    <td colSpan={out.cols.length} style={{ color: 'var(--viz-ink-muted)' }}>
                      no rows
                    </td>
                  </tr>
                ) : (
                  out.result.map((r, i) => (
                    <tr key={i}>
                      {r.map((v, j) => (
                        <td key={j} style={v === 'NULL' ? { color: 'var(--viz-ink-muted)', fontStyle: 'italic' } : undefined}>
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
          <p style={{ margin: '.4rem 0 0', color: 'var(--viz-ink-2)', fontSize: '.78rem' }}>The form that answers the question you asked:</p>
          <pre style={{ margin: '.15rem 0 0', padding: '.5rem .6rem', fontSize: '.78rem', lineHeight: 1.5, color: 'var(--viz-ink)', background: 'var(--viz-plane)', border: '1px solid var(--viz-border)', borderRadius: '6px', overflowX: 'auto' }}>
            {out.fix}
          </pre>
        </div>
      </div>
    </VizPanel>
  );
}
