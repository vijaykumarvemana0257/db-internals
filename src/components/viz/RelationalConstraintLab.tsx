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
  makeRng,
  useSize,
} from './Viz';

/**
 * A constraint enforcer over customers / orders / order_items.
 *
 * The learner edits the DDL (foreign-key actions, a NOT NULL, the NULLS DISTINCT mode),
 * fires a real statement, and watches every gate evaluate in PostgreSQL's actual order:
 * NOT NULL and CHECK first (ExecConstraints, before the tuple is in the heap), then the
 * unique indexes, then referential integrity as an after-row trigger. The first FALSE
 * aborts the statement and nothing after it runs.
 *
 * The NULL story is the point of the lab: the same NULL makes a CHECK pass, a WHERE row
 * disappear, a UNIQUE index shrug, and a foreign key skip its lookup entirely.
 */

/* ------------------------------------------------------------------- schema */

type Cust = { id: number; email: string | null; region: string };
type Ord = { id: number; customer_id: number | null; total: number | null; status: string };
type Item = { order_id: number; line: number; qty: number | null };

type FkAct = 'noaction' | 'restrict' | 'cascade' | 'setnull';

const FK_LABEL: Record<FkAct, string> = {
  noaction: 'NO ACTION',
  restrict: 'RESTRICT',
  cascade: 'CASCADE',
  setnull: 'SET NULL',
};

type Cfg = {
  fkOrd: FkAct; // orders.customer_id -> customers.id
  fkItem: FkAct; // order_items.order_id -> orders.id
  notNull: boolean; // orders.customer_id NOT NULL
  nullsDistinct: boolean; // UNIQUE (email) NULLS [NOT] DISTINCT
};

/* --------------------------------------------------------------- seed rows */

/** Deterministic seed amounts — the same table on every render and every SSR pass. */
const SEED_TOTALS = (() => {
  const rng = makeRng(19700101);
  return Array.from({ length: 3 }, () => Math.round((20 + rng() * 180) / 5) * 5);
})();

const BASE_CUST: Cust[] = [
  { id: 1, email: 'kay@vine.io', region: 'WA' },
  { id: 2, email: 'lin@harbor.dev', region: 'OR' },
  { id: 3, email: null, region: 'CA' },
];

const BASE_ORD: Ord[] = [
  { id: 5001, customer_id: 1, total: SEED_TOTALS[0], status: 'new' },
  { id: 5002, customer_id: 1, total: null, status: 'new' },
  { id: 5003, customer_id: 2, total: SEED_TOTALS[2], status: 'new' },
];

const BASE_ITEM: Item[] = [
  { order_id: 5001, line: 1, qty: 2 },
  { order_id: 5001, line: 2, qty: 1 },
  { order_id: 5003, line: 1, qty: 4 },
];

/* ------------------------------------------------------------------- state */

type Verdict = 'pass' | 'fail' | 'unknown' | 'skip';
type Gate = { name: string; expr: string; verdict: Verdict; detail: string };
type Mark = 'base' | 'added' | 'rejected' | 'deleted' | 'changed' | 'unmatched';
type Ghost = { tbl: 'c' | 'o' | 'i'; cells: (string | number | null)[] } | null;
type LogRow = { n: number; stmt: string; result: string; gate: string };

type S = {
  cust: Cust[];
  ord: Ord[];
  item: Item[];
  mark: Record<string, Mark>;
  ghost: Ghost;
  gates: Gate[];
  result: string;
  ok: boolean;
  head: string;
  body: string;
  nextCust: number;
  nextOrd: number;
  log: LogRow[];
};

const INITIAL: S = {
  cust: BASE_CUST,
  ord: BASE_ORD,
  item: BASE_ITEM,
  mark: {},
  ghost: null,
  gates: [],
  result: '—',
  ok: true,
  head: 'Nothing executed yet.',
  body:
    'Pick a statement and press Execute. Every gate runs in the order PostgreSQL runs it: ' +
    'NOT NULL and CHECK before the tuple reaches the heap, the unique indexes next, ' +
    'referential integrity last as an after-row trigger. The first FALSE aborts the statement.',
  nextCust: 4,
  nextOrd: 5004,
  log: [],
};

/* -------------------------------------------------------------- statements */

type StmtId =
  | 'ins_cust_new'
  | 'ins_cust_dup'
  | 'ins_cust_null'
  | 'ins_ord_ok'
  | 'ins_ord_orphan'
  | 'ins_ord_neg'
  | 'ins_ord_nulltotal'
  | 'ins_ord_nullfk'
  | 'ins_item_dup'
  | 'upd_paid'
  | 'del_cust';

const STMTS: { value: StmtId; label: string; sql: string }[] = [
  { value: 'ins_cust_new', label: 'INSERT customer — fresh email', sql: "INSERT INTO customers (email, region) VALUES ('ada@vine.io', 'WA')" },
  { value: 'ins_cust_dup', label: 'INSERT customer — email already used', sql: "INSERT INTO customers (email, region) VALUES ('kay@vine.io', 'OR')" },
  { value: 'ins_cust_null', label: 'INSERT customer — email NULL', sql: "INSERT INTO customers (email, region) VALUES (NULL, 'CA')" },
  { value: 'ins_ord_ok', label: 'INSERT order — valid', sql: 'INSERT INTO orders (customer_id, total) VALUES (1, 120.00)' },
  { value: 'ins_ord_orphan', label: 'INSERT order — customer_id 99', sql: 'INSERT INTO orders (customer_id, total) VALUES (99, 120.00)' },
  { value: 'ins_ord_neg', label: 'INSERT order — total -5.00', sql: 'INSERT INTO orders (customer_id, total) VALUES (1, -5.00)' },
  { value: 'ins_ord_nulltotal', label: 'INSERT order — total NULL', sql: 'INSERT INTO orders (customer_id, total) VALUES (1, NULL)' },
  { value: 'ins_ord_nullfk', label: 'INSERT order — customer_id NULL', sql: 'INSERT INTO orders (customer_id, total) VALUES (NULL, 60.00)' },
  { value: 'ins_item_dup', label: 'INSERT item — (5001, 1)', sql: 'INSERT INTO order_items (order_id, line, qty) VALUES (5001, 1, 2)' },
  { value: 'upd_paid', label: "UPDATE orders WHERE total >= 0", sql: "UPDATE orders SET status = 'paid' WHERE total >= 0" },
  { value: 'del_cust', label: 'DELETE customer 1', sql: 'DELETE FROM customers WHERE id = 1' },
];

/* ------------------------------------------------------------ the enforcer */

/** Collects gate verdicts; once one FAILs, everything after it is "not reached". */
function runner() {
  const gates: Gate[] = [];
  let err: string | null = null;
  return {
    gates,
    add(name: string, expr: string, verdict: Verdict, detail: string, error?: string) {
      if (err) {
        gates.push({ name, expr, verdict: 'skip', detail: 'Not reached — the statement already aborted and rolled back.' });
        return;
      }
      gates.push({ name, expr, verdict, detail });
      if (verdict === 'fail') err = error ?? 'ERROR';
    },
    get err() {
      return err;
    },
  };
}

const NN_OK = 'Every column with a NOT NULL declaration has a value. NOT NULL is checked first, before the tuple is ever placed in the heap.';

function money(n: number | null) {
  return n === null ? null : n.toFixed(2);
}

function insertCustomer(s: S, cfg: Cfg, email: string | null): S {
  const id = s.nextCust;
  const r = runner();

  r.add('NOT NULL', 'id, region', 'pass', `id = ${id} and region are present. email has no NOT NULL, so a NULL there is legal.`);
  r.add('CHECK', 'none on customers', 'skip', 'customers declares no CHECK constraint, so there is no row predicate to evaluate.');
  r.add('PRIMARY KEY', 'customers_pkey (id)', 'pass', `No live row has id = ${id}, so the unique index accepts the entry. A primary key is a candidate key plus an implicit NOT NULL on every one of its columns.`);

  if (email === null) {
    if (cfg.nullsDistinct) {
      r.add(
        'UNIQUE',
        'customers_email_key (email) NULLS DISTINCT',
        'unknown',
        'email IS NULL. Uniqueness is defined by equality, and NULL = NULL is UNKNOWN — not TRUE — so under the SQL default every NULL counts as its own value. You may store any number of NULL emails.',
      );
    } else if (s.cust.some((c) => c.email === null)) {
      r.add(
        'UNIQUE',
        'customers_email_key (email) NULLS NOT DISTINCT',
        'fail',
        'PostgreSQL 15 added NULLS NOT DISTINCT, which makes the index treat two NULLs as equal. customer 3 already holds a NULL email, so this one collides.',
        'ERROR 23505: duplicate key value violates unique constraint "customers_email_key"',
      );
    } else {
      r.add('UNIQUE', 'customers_email_key (email) NULLS NOT DISTINCT', 'pass', 'No other row holds a NULL email yet, so even under NULLS NOT DISTINCT this is the first one.');
    }
  } else if (s.cust.some((c) => c.email === email)) {
    r.add(
      'UNIQUE',
      'customers_email_key (email)',
      'fail',
      `'${email}' is already in the index. The uniqueness test happens inside the B-tree insert, under a page lock — which is why no amount of SELECT-then-INSERT in application code can replace it.`,
      'ERROR 23505: duplicate key value violates unique constraint "customers_email_key"',
    );
  } else {
    r.add('UNIQUE', 'customers_email_key (email)', 'pass', `'${email}' is not in the index. email is an alternate key — a candidate you did not elect as primary — and it is only a true key once it is also NOT NULL; here it is nullable, which is why NULLs get in.`);
  }

  r.add('FOREIGN KEY', 'none outgoing', 'skip', 'customers references no other relation, so no referential action fires.');

  const cells = [id, email, email === null ? 'CA' : email === 'kay@vine.io' ? 'OR' : 'WA'];
  if (r.err) {
    return { ...s, gates: r.gates, mark: {}, ghost: { tbl: 'c', cells }, result: r.err, ok: false, head: r.err, body: 'The whole statement rolled back. The attempted row is shown outlined; it is in no snapshot and has no visible effect.' };
  }
  const row: Cust = { id, email, region: String(cells[2]) };
  return {
    ...s,
    cust: [...s.cust, row],
    nextCust: id + 1,
    gates: r.gates,
    mark: { [`c${id}`]: 'added' },
    ghost: null,
    result: 'INSERT 0 1',
    ok: true,
    head: 'INSERT 0 1',
    body: email === null ? 'A NULL email passed the unique index. NULL is not a value, it is the absence of one, so it cannot be equal to anything — including another NULL.' : 'All five gates passed and the tuple is live.',
  };
}

function insertOrder(s: S, cfg: Cfg, customerId: number | null, total: number | null): S {
  const id = s.nextOrd;
  const r = runner();

  if (customerId === null && cfg.notNull) {
    r.add(
      'NOT NULL',
      'orders.customer_id NOT NULL',
      'fail',
      'NOT NULL is the one constraint that is a column property rather than a table predicate, and it is the only one that treats UNKNOWN as a rejection.',
      'ERROR 23502: null value in column "customer_id" of relation "orders" violates not-null constraint',
    );
  } else {
    r.add('NOT NULL', cfg.notNull ? 'orders.customer_id NOT NULL' : 'id only', 'pass', NN_OK);
  }

  if (total === null) {
    r.add(
      'CHECK',
      'orders_total_check (total >= 0)',
      'unknown',
      'total IS NULL, so total >= 0 evaluates to UNKNOWN. A CHECK constraint rejects only FALSE — UNKNOWN is accepted. This is the asymmetry that lets a NULL slip past a validation you thought was airtight.',
    );
  } else if (total < 0) {
    r.add(
      'CHECK',
      'orders_total_check (total >= 0)',
      'fail',
      `total = ${total.toFixed(2)} makes the predicate FALSE. CHECK runs in ExecConstraints, before the tuple is written, so nothing is inserted and no index is touched.`,
      'ERROR 23514: new row for relation "orders" violates check constraint "orders_total_check"',
    );
  } else {
    r.add('CHECK', 'orders_total_check (total >= 0)', 'pass', `total = ${total.toFixed(2)} makes the predicate TRUE.`);
  }

  r.add('PRIMARY KEY', 'orders_pkey (id)', 'pass', `id = ${id} comes from a sequence — a surrogate key, meaningless outside this database and therefore immune to the customer changing their email or the business changing its SKU format.`);
  r.add('UNIQUE', 'none beyond the primary key', 'skip', 'orders declares no secondary UNIQUE constraint.');

  if (customerId === null) {
    r.add(
      'FOREIGN KEY',
      `orders_customer_id_fkey (MATCH SIMPLE)`,
      'unknown',
      'The referencing column is NULL. Under MATCH SIMPLE — the SQL default, and the only mode PostgreSQL implements for partially-NULL keys beyond MATCH FULL — the constraint is satisfied without any lookup in customers. An orphan by omission is always allowed unless you add NOT NULL.',
    );
  } else if (s.cust.some((c) => c.id === customerId)) {
    r.add('FOREIGN KEY', 'orders_customer_id_fkey', 'pass', `customers.id = ${customerId} exists. The RI check is an after-row trigger that runs a SELECT ... FOR KEY SHARE on the parent, which is why it needs an index on the parent's key and why it takes a lock that blocks parent-key updates.`);
  } else {
    r.add(
      'FOREIGN KEY',
      'orders_customer_id_fkey',
      'fail',
      `No customers row has id = ${customerId}. Referential integrity is the invariant that every non-NULL foreign-key value appears as a key value in the referenced relation.`,
      'ERROR 23503: insert or update on table "orders" violates foreign key constraint "orders_customer_id_fkey"',
    );
  }

  const cells = [id, customerId, money(total), 'new'];
  if (r.err) {
    return { ...s, gates: r.gates, mark: {}, ghost: { tbl: 'o', cells }, result: r.err, ok: false, head: r.err, body: 'Statement rolled back. A failed constraint aborts the statement, not just the row — there is no partial INSERT.' };
  }
  return {
    ...s,
    ord: [...s.ord, { id, customer_id: customerId, total, status: 'new' }],
    nextOrd: id + 1,
    gates: r.gates,
    mark: { [`o${id}`]: 'added' },
    ghost: null,
    result: 'INSERT 0 1',
    ok: true,
    head: 'INSERT 0 1',
    body:
      total === null
        ? 'A NULL total was accepted by CHECK (total >= 0). Watch what the same row does to the UPDATE ... WHERE statement.'
        : customerId === null
          ? 'A NULL foreign key satisfies referential integrity by definition. The order now belongs to nobody.'
          : 'All gates passed.',
  };
}

function insertItem(s: S): S {
  const r = runner();
  const exists = s.item.some((i) => i.order_id === 5001 && i.line === 1);
  r.add('NOT NULL', 'order_id, line (implied by the primary key)', 'pass', 'Both primary-key columns have values. Every column of a PRIMARY KEY is NOT NULL whether you wrote it or not.');
  r.add('CHECK', 'order_items_qty_check (qty > 0)', 'pass', 'qty = 2 makes the predicate TRUE.');
  if (exists) {
    r.add(
      'PRIMARY KEY',
      'order_items_pkey (order_id, line)',
      'fail',
      'The key is the pair, not either column. (5001, 1) is already in the index, so the composite key collides even though line 2 for the same order is fine.',
      'ERROR 23505: duplicate key value violates unique constraint "order_items_pkey"',
    );
  } else {
    r.add('PRIMARY KEY', 'order_items_pkey (order_id, line)', 'pass', 'No row holds (5001, 1). {order_id, line} is a candidate key; {order_id, line, qty} is a superkey — unique, but not minimal, so not a candidate.');
  }
  r.add('UNIQUE', 'none beyond the primary key', 'skip', 'order_items declares no secondary UNIQUE constraint.');
  const parent = s.ord.some((o) => o.id === 5001);
  r.add(
    'FOREIGN KEY',
    'order_items_order_id_fkey',
    parent ? 'pass' : 'fail',
    parent ? 'orders.id = 5001 exists.' : 'Order 5001 is gone — you deleted its customer with ON DELETE CASCADE.',
    'ERROR 23503: insert or update on table "order_items" violates foreign key constraint "order_items_order_id_fkey"',
  );

  const cells = [5001, 1, 2];
  if (r.err) {
    return { ...s, gates: r.gates, mark: {}, ghost: { tbl: 'i', cells }, result: r.err, ok: false, head: r.err, body: 'Rolled back. Uniqueness was decided inside the index insert, not by anything the application could have checked first.' };
  }
  return { ...s, item: [...s.item, { order_id: 5001, line: 1, qty: 2 }], gates: r.gates, mark: { 'i5001-1': 'added' }, ghost: null, result: 'INSERT 0 1', ok: true, head: 'INSERT 0 1', body: 'The composite key was free.' };
}

function updatePaid(s: S): S {
  const r = runner();
  const truthy = s.ord.filter((o) => o.total !== null && o.total >= 0);
  const unknown = s.ord.filter((o) => o.total === null);
  const falsy = s.ord.filter((o) => o.total !== null && o.total < 0);

  r.add(
    'WHERE',
    'total >= 0',
    unknown.length ? 'unknown' : 'pass',
    `TRUE on ${truthy.length} row(s), FALSE on ${falsy.length}, UNKNOWN on ${unknown.length} (total IS NULL). WHERE keeps only TRUE — so the NULL rows silently vanish from the result. Note the asymmetry: the identical predicate in a CHECK constraint accepted those same rows.`,
  );
  r.add('NOT NULL', 'status', 'pass', "status is set to 'paid' on every matched row.");
  r.add('CHECK', 'orders_total_check (total >= 0)', 'pass', 'PostgreSQL re-evaluates every CHECK on the new version of each updated tuple. total is unchanged, so all of them hold.');
  r.add('PRIMARY KEY', 'orders_pkey (id)', 'skip', 'id is untouched, so no index entry changes.');
  r.add('UNIQUE', 'none beyond the primary key', 'skip', '—');
  r.add('FOREIGN KEY', 'orders_customer_id_fkey', 'skip', 'The RI after-row trigger fires only when a key column actually changes. customer_id was not written, so the check is skipped entirely.');

  const ids = new Set(truthy.map((o) => o.id));
  const mark: Record<string, Mark> = {};
  for (const o of truthy) mark[`o${o.id}`] = 'changed';
  for (const o of unknown) mark[`o${o.id}`] = 'unmatched';
  return {
    ...s,
    ord: s.ord.map((o) => (ids.has(o.id) ? { ...o, status: 'paid' } : o)),
    gates: r.gates,
    mark,
    ghost: null,
    result: `UPDATE ${truthy.length}`,
    ok: true,
    head: `UPDATE ${truthy.length}`,
    body: unknown.length
      ? `${unknown.length} row(s) with a NULL total were not matched — and they would not be matched by WHERE total < 0 either. NOT (UNKNOWN) is UNKNOWN. Only IS NULL finds them.`
      : 'Every row matched.',
  };
}

function deleteCustomer(s: S, cfg: Cfg): S {
  const r = runner();
  const parent = s.cust.find((c) => c.id === 1);
  if (!parent) {
    r.add('FOREIGN KEY', 'orders_customer_id_fkey', 'skip', 'No customer with id = 1 remains; the DELETE matches no rows and no referential action fires.');
    return { ...s, gates: r.gates, mark: {}, ghost: null, result: 'DELETE 0', ok: true, head: 'DELETE 0', body: 'Nothing matched the WHERE clause. Press Reset to start over.' };
  }
  const kids = s.ord.filter((o) => o.customer_id === 1);
  const act = cfg.fkOrd;
  const mark: Record<string, Mark> = {};
  let ord = s.ord;
  let item = s.item;
  let cascadedItems = 0;

  if (kids.length === 0) {
    r.add('FOREIGN KEY', `orders_customer_id_fkey ON DELETE ${FK_LABEL[act]}`, 'pass', 'No orders reference customer 1, so the action is a no-op whichever one you chose.');
  } else if (act === 'restrict' || act === 'noaction') {
    r.add(
      'FOREIGN KEY',
      `orders_customer_id_fkey ON DELETE ${FK_LABEL[act]}`,
      'fail',
      act === 'restrict'
        ? `RESTRICT fires immediately, inside the DELETE, and cannot be deferred. ${kids.length} order(s) still reference id = 1.`
        : `NO ACTION runs the same test but at the end of the statement, and it is the only one of the two that a DEFERRABLE INITIALLY DEFERRED constraint can postpone to COMMIT — which is how you delete a parent and re-parent its children inside one transaction.`,
      'ERROR 23503: update or delete on table "customers" violates foreign key constraint "orders_customer_id_fkey" on table "orders"',
    );
  } else if (act === 'setnull') {
    if (cfg.notNull) {
      r.add('FOREIGN KEY', 'orders_customer_id_fkey ON DELETE SET NULL', 'pass', `The action fires and writes NULL into customer_id on ${kids.length} order(s). Now the child rows have to survive their own constraints.`);
      r.add(
        'NOT NULL',
        'orders.customer_id NOT NULL',
        'fail',
        'SET NULL and NOT NULL on the same column is a contradiction that the DDL happily accepts and that only shows up the first time someone deletes a parent — often years later, in production.',
        'ERROR 23502: null value in column "customer_id" of relation "orders" violates not-null constraint',
      );
    } else {
      r.add('FOREIGN KEY', 'orders_customer_id_fkey ON DELETE SET NULL', 'pass', `customer_id is set to NULL on ${kids.length} order(s). The orders survive with no owner, which is exactly as much integrity as you asked for.`);
      ord = ord.map((o) => (o.customer_id === 1 ? { ...o, customer_id: null } : o));
      for (const o of kids) mark[`o${o.id}`] = 'changed';
    }
  } else {
    const doomed = new Set(kids.map((o) => o.id));
    const orphanItems = s.item.filter((i) => doomed.has(i.order_id));
    r.add('FOREIGN KEY', 'orders_customer_id_fkey ON DELETE CASCADE', 'pass', `CASCADE deletes ${kids.length} referencing order(s). Each of those deletes is itself a statement that must satisfy every constraint pointing at orders — the action recurses.`);
    if (orphanItems.length === 0) {
      r.add('FOREIGN KEY', `order_items_order_id_fkey ON DELETE ${FK_LABEL[cfg.fkItem]}`, 'pass', 'No order_items reference the cascaded orders.');
      ord = ord.filter((o) => !doomed.has(o.id));
      for (const o of kids) mark[`o${o.id}`] = 'deleted';
    } else if (cfg.fkItem === 'cascade') {
      r.add('FOREIGN KEY', 'order_items_order_id_fkey ON DELETE CASCADE', 'pass', `The cascade recurses one level further and removes ${orphanItems.length} item row(s). Two declarations, three tables, one DELETE.`);
      ord = ord.filter((o) => !doomed.has(o.id));
      item = item.filter((i) => !doomed.has(i.order_id));
      cascadedItems = orphanItems.length;
      for (const o of kids) mark[`o${o.id}`] = 'deleted';
      for (const i of orphanItems) mark[`i${i.order_id}-${i.line}`] = 'deleted';
    } else if (cfg.fkItem === 'setnull') {
      r.add(
        'FOREIGN KEY',
        'order_items_order_id_fkey ON DELETE SET NULL',
        'fail',
        'order_id is half of order_items_pkey, and every primary-key column is implicitly NOT NULL. SET NULL on a key column can never succeed.',
        'ERROR 23502: null value in column "order_id" of relation "order_items" violates not-null constraint',
      );
    } else {
      r.add(
        'FOREIGN KEY',
        `order_items_order_id_fkey ON DELETE ${FK_LABEL[cfg.fkItem]}`,
        'fail',
        `The cascaded delete of order 5001 is blocked by ${orphanItems.length} item row(s). A CASCADE above a ${FK_LABEL[cfg.fkItem]} is a delete that can never complete — and the error names a table nobody mentioned in the statement.`,
        'ERROR 23503: update or delete on table "orders" violates foreign key constraint "order_items_order_id_fkey" on table "order_items"',
      );
    }
  }

  if (r.err) {
    return { ...s, gates: r.gates, mark: {}, ghost: null, result: r.err, ok: false, head: r.err, body: 'The entire DELETE rolled back, cascaded rows included. Referential actions are part of the statement, so they are atomic with it.' };
  }
  mark['c1'] = 'deleted';
  const deletedOrders = s.ord.length - ord.length;
  return {
    ...s,
    cust: s.cust.filter((c) => c.id !== 1),
    ord,
    item,
    gates: r.gates,
    mark,
    ghost: null,
    result: 'DELETE 1',
    ok: true,
    head: 'DELETE 1',
    body: `Parent removed; ${deletedOrders} order(s) deleted, ${cascadedItems} item(s) deleted, ${kids.length - deletedOrders} order(s) orphaned by SET NULL. Every arrow in the schema still points at something that exists.`,
  };
}

function execute(s: S, cfg: Cfg, id: StmtId): S {
  const stmt = STMTS.find((x) => x.value === id)!;
  let next: S;
  switch (id) {
    case 'ins_cust_new':
      next = insertCustomer(s, cfg, 'ada@vine.io');
      break;
    case 'ins_cust_dup':
      next = insertCustomer(s, cfg, 'kay@vine.io');
      break;
    case 'ins_cust_null':
      next = insertCustomer(s, cfg, null);
      break;
    case 'ins_ord_ok':
      next = insertOrder(s, cfg, 1, 120);
      break;
    case 'ins_ord_orphan':
      next = insertOrder(s, cfg, 99, 120);
      break;
    case 'ins_ord_neg':
      next = insertOrder(s, cfg, 1, -5);
      break;
    case 'ins_ord_nulltotal':
      next = insertOrder(s, cfg, 1, null);
      break;
    case 'ins_ord_nullfk':
      next = insertOrder(s, cfg, null, 60);
      break;
    case 'ins_item_dup':
      next = insertItem(s);
      break;
    case 'upd_paid':
      next = updatePaid(s);
      break;
    default:
      next = deleteCustomer(s, cfg);
  }
  const failed = next.gates.find((g) => g.verdict === 'fail');
  return {
    ...next,
    log: [...s.log, { n: s.log.length + 1, stmt: stmt.sql, result: next.result, gate: failed ? `${failed.name} — ${failed.expr}` : '—' }].slice(-12),
  };
}

/* ----------------------------------------------------------------- drawing */

const VERDICT: Record<Verdict, { color: string; glyph: string; word: string }> = {
  pass: { color: 'var(--viz-good)', glyph: '✓', word: 'TRUE' },
  fail: { color: 'var(--viz-critical)', glyph: '✕', word: 'FALSE' },
  unknown: { color: 'var(--viz-warning)', glyph: '?', word: 'UNKNOWN' },
  skip: { color: 'var(--viz-ink-muted)', glyph: '–', word: 'n/a' },
};

const MARK_STYLE: Record<Mark, { color: string; glyph: string }> = {
  base: { color: 'var(--viz-border)', glyph: '' },
  added: { color: 'var(--viz-dirty)', glyph: '+' },
  rejected: { color: 'var(--viz-critical)', glyph: '✕' },
  deleted: { color: 'var(--viz-stale)', glyph: '✕' },
  changed: { color: 'var(--viz-7)', glyph: '~' },
  unmatched: { color: 'var(--viz-warning)', glyph: '?' },
};

type Row = { key: string; cells: (string | number | null)[]; mark: Mark };
type Panel = { name: string; decls: string[]; cols: { label: string; w: number }[]; rows: Row[]; w: number };

function panels(s: S, cfg: Cfg): Panel[] {
  const m = (k: string): Mark => s.mark[k] ?? 'base';
  const cust: Row[] = s.cust.map((c) => ({ key: `c${c.id}`, cells: [c.id, c.email, c.region], mark: m(`c${c.id}`) }));
  if (s.ghost?.tbl === 'c') cust.push({ key: 'gc', cells: s.ghost.cells, mark: 'rejected' });

  const ord: Row[] = s.ord.map((o) => ({ key: `o${o.id}`, cells: [o.id, o.customer_id, money(o.total), o.status], mark: m(`o${o.id}`) }));
  if (s.ghost?.tbl === 'o') ord.push({ key: 'go', cells: s.ghost.cells, mark: 'rejected' });

  const item: Row[] = s.item.map((i) => ({ key: `i${i.order_id}-${i.line}`, cells: [i.order_id, i.line, i.qty], mark: m(`i${i.order_id}-${i.line}`) }));
  if (s.ghost?.tbl === 'i') item.push({ key: 'gi', cells: s.ghost.cells, mark: 'rejected' });

  return [
    {
      name: 'customers',
      decls: ['PRIMARY KEY (id)', `UNIQUE (email) ${cfg.nullsDistinct ? 'NULLS DISTINCT' : 'NULLS NOT DISTINCT'}`, '—'],
      cols: [
        { label: 'id', w: 34 },
        { label: 'email', w: 132 },
        { label: 'region', w: 50 },
      ],
      rows: cust,
      w: 232,
    },
    {
      name: 'orders',
      decls: [
        'PRIMARY KEY (id)',
        `customer_id ${cfg.notNull ? 'NOT NULL ' : ''}REFERENCES customers ON DELETE ${FK_LABEL[cfg.fkOrd]}`,
        'CHECK (total >= 0)',
      ],
      cols: [
        { label: 'id', w: 44 },
        { label: 'customer_id', w: 74 },
        { label: 'total', w: 54 },
        { label: 'status', w: 50 },
      ],
      rows: ord,
      w: 238,
    },
    {
      name: 'order_items',
      decls: ['PRIMARY KEY (order_id, line)', `order_id REFERENCES orders ON DELETE ${FK_LABEL[cfg.fkItem]}`, 'CHECK (qty > 0)'],
      cols: [
        { label: 'order_id', w: 60 },
        { label: 'line', w: 36 },
        { label: 'qty', w: 34 },
      ],
      rows: item,
      w: 152,
    },
  ];
}

/* --------------------------------------------------------------- component */

export default function RelationalConstraintLab() {
  const [fkOrd, setFkOrd] = useState<FkAct>('restrict');
  const [fkItem, setFkItem] = useState<FkAct>('cascade');
  const [notNull, setNotNull] = useState(false);
  const [nullsDistinct, setNullsDistinct] = useState(true);
  const [stmt, setStmt] = useState<StmtId>('ins_ord_nulltotal');
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(780);
  const tip = useTip();

  const cfg: Cfg = { fkOrd, fkItem, notNull, nullsDistinct };
  const ps = panels(s, cfg);
  const current = STMTS.find((x) => x.value === stmt)!;

  const gap = 14;
  const pad = 4;
  const tableW = ps.reduce((a, p) => a + p.w, 0) + gap * 2 + pad * 2;
  const svgW = Math.max(width, Math.max(tableW, 720));

  const gates = s.gates;
  const gn = Math.max(gates.length, 1);
  const gGap = 8;
  const gw = (svgW - pad * 2 - gGap * (gn - 1)) / gn;
  const gh = 74;
  const gTop = 16;

  const rowH = 21;
  const maxRows = Math.max(...ps.map((p) => p.rows.length));
  const tTop = gTop + gh + 30;
  const headH = 64;
  const height = tTop + headH + 18 + maxRows * rowH + 14;

  const full = s.cust.length > 8 || s.ord.length > 8 || s.item.length > 8;

  return (
    <VizPanel
      title="Constraint enforcer: customers → orders → order_items"
      subtitle="Edit the DDL, fire a statement, and watch every gate evaluate in the order the engine evaluates it. The first FALSE aborts everything after it; UNKNOWN is accepted by some gates and rejected by others."
      controls={
        <>
          <Choice label="Statement" value={stmt} onChange={setStmt} options={STMTS.map((x) => ({ value: x.value, label: x.label }))} />
          <Choice
            label="orders → customers"
            value={fkOrd}
            onChange={setFkOrd}
            options={(['noaction', 'restrict', 'cascade', 'setnull'] as FkAct[]).map((a) => ({ value: a, label: `ON DELETE ${FK_LABEL[a]}` }))}
          />
          <Choice
            label="items → orders"
            value={fkItem}
            onChange={setFkItem}
            options={(['noaction', 'restrict', 'cascade', 'setnull'] as FkAct[]).map((a) => ({ value: a, label: `ON DELETE ${FK_LABEL[a]}` }))}
          />
          <Check label="orders.customer_id NOT NULL" checked={notNull} onChange={setNotNull} />
          <Segmented
            label="UNIQUE (email)"
            value={nullsDistinct ? 'distinct' : 'notdistinct'}
            onChange={(v) => setNullsDistinct(v === 'distinct')}
            options={[
              { value: 'distinct', label: 'NULLS DISTINCT', title: 'The SQL default: every NULL is its own value' },
              { value: 'notdistinct', label: 'NULLS NOT DISTINCT', title: 'PostgreSQL 15 and later: two NULLs collide' },
            ]}
          />
          <Button onClick={() => setS((cur) => execute(cur, cfg, stmt))} disabled={full} primary>
            Execute
          </Button>
          <Button onClick={() => setS(INITIAL)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: '✓ TRUE — gate satisfied', color: VERDICT.pass.color },
            { label: '✕ FALSE — statement aborts here', color: VERDICT.fail.color },
            { label: '? UNKNOWN — a NULL made it undecidable', color: VERDICT.unknown.color },
            { label: '– not applicable / not reached', color: VERDICT.skip.color },
            { label: '+ row inserted', color: MARK_STYLE.added.color },
            { label: '~ row updated or set to NULL', color: MARK_STYLE.changed.color },
            { label: '✕ row deleted by a referential action', color: MARK_STYLE.deleted.color },
            { label: '? row not matched by WHERE', color: MARK_STYLE.unmatched.color },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Result', value: s.ok ? s.result : s.result.split(':')[0], hint: s.result },
            { label: 'customers', value: s.cust.length },
            { label: 'orders', value: s.ord.length },
            { label: 'order_items', value: s.item.length },
            { label: 'Orphan orders', value: s.ord.filter((o) => o.customer_id === null).length, hint: 'Rows whose foreign key is NULL — legal under MATCH SIMPLE, invisible to every join' },
            { label: 'Statements run', value: s.log.length },
          ]}
        />
      }
      note={
        <Note>
          <code>{current.sql}</code> → <strong>{s.head}</strong> {s.body}
          {full ? ' The tables are full — press Reset.' : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Gate</th>
                <th>Evaluates</th>
                <th>Verdict</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {gates.length === 0 ? (
                <tr>
                  <td colSpan={4}>No statement executed yet.</td>
                </tr>
              ) : (
                gates.map((g, i) => (
                  <tr key={`${g.name}-${i}`}>
                    <td>{g.name}</td>
                    <td>{g.expr}</td>
                    <td>
                      {VERDICT[g.verdict].glyph} {VERDICT[g.verdict].word}
                    </td>
                    <td>{g.detail}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Relation</th>
                <th>Tuple</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {ps.flatMap((p) =>
                p.rows.map((r) => (
                  <tr key={`${p.name}-${r.key}`}>
                    <td>{p.name}</td>
                    <td>
                      ({p.cols.map((c, i) => `${c.label}=${r.cells[i] === null ? 'NULL' : r.cells[i]}`).join(', ')})
                    </td>
                    <td>{r.mark === 'base' ? 'unchanged' : r.mark}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Statement</th>
                <th>Result</th>
                <th>Failed gate</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={4}>Nothing run yet.</td>
                </tr>
              ) : (
                s.log.map((l) => (
                  <tr key={l.n}>
                    <td>{l.n}</td>
                    <td>{l.stmt}</td>
                    <td>{l.result}</td>
                    <td>{l.gate}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Constraint gates evaluating over three relations, with rows inserted, rejected, cascaded or set to NULL">
            {/* ------------------------------------------------ the gate pipeline */}
            <text x={pad} y={10} fill="var(--viz-ink-2)" fontWeight={600}>
              Constraint gates, in evaluation order
            </text>
            {gates.length === 0 ? (
              <rect x={pad} y={gTop} width={svgW - pad * 2} height={gh} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" strokeDasharray="4 4" />
            ) : null}
            {gates.length === 0 ? (
              <text x={svgW / 2} y={gTop + gh / 2 + 4} textAnchor="middle" fill="var(--viz-ink-muted)">
                Press Execute
              </text>
            ) : null}
            {gates.map((g, i) => {
              const x = pad + i * (gw + gGap);
              const v = VERDICT[g.verdict];
              return (
                <g key={`${g.name}-${i}`} {...tip(<><strong>{g.name}</strong> — {g.expr}<br />{v.glyph} {v.word}<br />{g.detail}</>)}>
                  <rect
                    x={x}
                    y={gTop}
                    width={gw}
                    height={gh}
                    rx={8}
                    fill="var(--viz-plane)"
                    stroke={g.verdict === 'skip' ? 'var(--viz-border)' : v.color}
                    strokeWidth={g.verdict === 'skip' ? 1 : 2}
                  />
                  <rect x={x} y={gTop} width={4} height={gh} rx={2} fill={v.color} />
                  <text x={x + 10} y={gTop + 18} fill="var(--viz-ink)" fontWeight={600}>
                    {g.name}
                  </text>
                  <text x={x + 10} y={gTop + 34} fill="var(--viz-ink-muted)">
                    {g.expr.length > Math.floor(gw / 5.6) ? `${g.expr.slice(0, Math.max(4, Math.floor(gw / 5.6) - 1))}…` : g.expr}
                  </text>
                  <text x={x + 10} y={gTop + 56} fill={v.color} fontWeight={600}>
                    {v.glyph} {v.word}
                  </text>
                  {i < gates.length - 1 ? (
                    <path d={`M ${x + gw + 1} ${gTop + gh / 2 - 4} L ${x + gw + gGap - 1} ${gTop + gh / 2} L ${x + gw + 1} ${gTop + gh / 2 + 4} Z`} fill="var(--viz-axis)" />
                  ) : null}
                </g>
              );
            })}

            {/* ----------------------------------------------------- the relations */}
            {ps.map((p, pi) => {
              const x = pad + ps.slice(0, pi).reduce((a, q) => a + q.w + gap, 0);
              const bodyY = tTop + headH;
              return (
                <g key={p.name}>
                  <rect x={x} y={tTop} width={p.w} height={headH + 18 + maxRows * rowH + 8} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                  <text x={x + 10} y={tTop + 17} fill="var(--viz-ink)" fontWeight={600}>
                    {p.name}
                  </text>
                  {p.decls.map((d, di) => (
                    <text key={d + di} x={x + 10} y={tTop + 32 + di * 13} fill="var(--viz-ink-muted)">
                      {d.length > Math.floor(p.w / 5.4) ? `${d.slice(0, Math.floor(p.w / 5.4) - 1)}…` : d}
                    </text>
                  ))}
                  {p.cols.map((c, ci) => (
                    <text key={c.label} x={x + 10 + p.cols.slice(0, ci).reduce((a, q) => a + q.w, 0)} y={bodyY + 10} fill="var(--viz-ink-2)" fontWeight={600}>
                      {c.label}
                    </text>
                  ))}
                  <line x1={x + 6} y1={bodyY + 16} x2={x + p.w - 6} y2={bodyY + 16} className="viz-grid-line" />
                  {p.rows.map((r, ri) => {
                    const y = bodyY + 20 + ri * rowH;
                    const ms = MARK_STYLE[r.mark];
                    const dead = r.mark === 'deleted' || r.mark === 'rejected';
                    return (
                      <g
                        key={r.key}
                        {...tip(
                          <>
                            <strong>{p.name}</strong>
                            <br />
                            ({p.cols.map((c, i) => `${c.label}=${r.cells[i] === null ? 'NULL' : r.cells[i]}`).join(', ')})
                            <br />
                            {r.mark === 'base'
                              ? 'Unchanged by the last statement.'
                              : r.mark === 'added'
                                ? 'Inserted by the last statement.'
                                : r.mark === 'rejected'
                                  ? 'Attempted and rolled back — this tuple never existed.'
                                  : r.mark === 'deleted'
                                    ? 'Removed by a referential action.'
                                    : r.mark === 'changed'
                                      ? 'Updated by the last statement.'
                                      : 'Not matched: the WHERE predicate was UNKNOWN on this row.'}
                          </>,
                        )}
                      >
                        <rect
                          x={x + 6}
                          y={y}
                          width={p.w - 12}
                          height={rowH - 3}
                          rx={4}
                          fill={r.mark === 'base' ? 'var(--viz-surface)' : 'var(--viz-plane)'}
                          stroke={r.mark === 'base' ? 'var(--viz-border)' : ms.color}
                          strokeWidth={r.mark === 'base' ? 1 : 2}
                          strokeDasharray={r.mark === 'rejected' ? '4 3' : undefined}
                          opacity={dead ? 0.55 : 1}
                        />
                        {r.mark !== 'base' ? (
                          <text x={x + p.w - 14} y={y + 13} textAnchor="end" fill={ms.color} fontWeight={600}>
                            {ms.glyph}
                          </text>
                        ) : null}
                        {p.cols.map((c, ci) => {
                          const val = r.cells[ci];
                          return (
                            <text
                              key={c.label}
                              x={x + 10 + p.cols.slice(0, ci).reduce((a, q) => a + q.w, 0)}
                              y={y + 13}
                              fill={val === null ? 'var(--viz-3)' : 'var(--viz-ink)'}
                              fontWeight={val === null ? 600 : 400}
                              textDecoration={dead ? 'line-through' : undefined}
                            >
                              {val === null ? 'NULL' : String(val)}
                            </text>
                          );
                        })}
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
