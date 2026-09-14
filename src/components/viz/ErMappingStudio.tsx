import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
} from './Viz';

/**
 * ER → tables, run as a generator.
 *
 * The learner edits the drawing (cardinality, participation, weak entities,
 * multi-valued attributes, whether the implied keys are actually declared) and the
 * component re-derives the physical schema from it: table boxes, DDL, the indexes
 * the mapping needs, and the sample result set the join now returns. "Violate it"
 * runs an INSERT the drawing forbids and reports which declared constraint — if any —
 * catches it. Reverse mode hands back an unfamiliar FK graph to read cardinalities out of.
 *
 * Everything is derived from one fixed dataset; the only randomness is the seeded
 * hire/birth years, so every render is identical.
 */

/* ------------------------------------------------------------------ domain */

const NAMES = ['Reyes', 'Nakamura', 'Okafor', 'Silva', 'Dubois', 'Petrov'];
const PROJECT_NAMES = ['Ledger rewrite', 'Billing API', 'Search index'];
const SKILL_POOL = ['sql', 'rust', 'go', 'kafka', 'k8s'];
const DEP_NAMES = ['Mira', 'Tomas', 'Ines', 'Ravi'];

type Emp = { id: number; name: string; hired: number };
type Proj = { id: number; name: string };
type Dep = { eid: number; name: string; born: number };

const BASE = (() => {
  const rng = makeRng(1976); // Chen, "The Entity-Relationship Model", 1976
  const employees: Emp[] = NAMES.map((name, i) => ({
    id: 101 + i,
    name,
    hired: 2016 + Math.floor(rng() * 9),
  }));
  const projects: Proj[] = PROJECT_NAMES.map((name, i) => ({ id: 10 + i * 10, name }));
  // What is actually true in the world. Each drawn cardinality keeps a different slice of it.
  const links: [number, number][] = [
    [101, 10],
    [102, 10],
    [102, 30],
    [103, 20],
    [104, 20],
    [104, 30],
    [105, 30],
  ];
  const oneOne: [number, number][] = [
    [101, 10],
    [103, 20],
    [105, 30],
  ];
  const deps: Dep[] = [101, 101, 104, 105].map((eid, i) => ({
    eid,
    name: DEP_NAMES[i],
    born: 2009 + Math.floor(rng() * 10),
  }));
  const skills: [number, string][] = employees.flatMap((e, i) =>
    (i % 3 === 0
      ? [SKILL_POOL[i % 5], SKILL_POOL[(i + 2) % 5]]
      : [SKILL_POOL[(i + 1) % 5]]
    ).map((s) => [e.id, s] as [number, string]),
  );
  return { employees, projects, links, oneOne, deps, skills };
})();

/* ------------------------------------------------------------------ schema */

type Card = '1:N' | 'M:N' | '1:1';
type Strategy = 'merge' | 'split';
type Part = 'total' | 'partial';
type Mode = 'forward' | 'reverse';

type Cfg = {
  card: Card;
  strategy: Strategy;
  part: Part;
  weak: boolean;
  mv: boolean;
  enforce: boolean;
};

type Col = {
  name: string;
  type: string;
  pk?: boolean;
  nn?: boolean;
  unique?: boolean;
  ref?: { t: string; c: string; onDelete?: string };
  tip: string;
};

type Row = { cells: (string | null)[]; bad?: boolean };

type Idx = { name: string; def: string; auto: boolean; why: string };

type Tbl = {
  id: string;
  name: string;
  cols: Col[];
  rows: Row[];
  col: number;
  band: number;
  tip: string;
  idx: Idx[];
};

type Edge = {
  from: string;
  fromCol: string;
  to: string;
  toCol: string;
  many: boolean;
  optional: boolean;
  label: string;
  tip: string;
};

type ResultSet = {
  sql: string[];
  head: string[];
  rows: { cells: (string | null)[]; dropped: boolean }[];
};

type Schema = {
  tables: Tbl[];
  edges: Edge[];
  result: ResultSet;
  enforced: string;
  rule: string;
};

function colKind(c: Col) {
  if (c.pk && c.ref) return 'pkfk';
  if (c.pk) return 'pk';
  if (c.ref) return 'fk';
  if (c.unique) return 'uniq';
  return 'plain';
}

const KIND_FILL: Record<string, string> = {
  pk: 'var(--viz-1)',
  pkfk: 'var(--viz-7)',
  fk: 'var(--viz-2)',
  uniq: 'var(--viz-7)',
  plain: 'var(--viz-ink-muted)',
};

/* -------------------------------------------------------- the mapping rules */

function build(cfg: Cfg, applied: string[]): Schema {
  const has = (k: string) => applied.includes(k);

  let employees = BASE.employees.slice();
  if (has('noLink')) employees = [...employees, { id: 107, name: 'Adeyemi', hired: 2025 }];
  if (has('cascade')) employees = employees.filter((e) => e.id !== 101);

  let projects = BASE.projects.slice();
  if (has('sharedLead')) projects = [...projects, { id: 40, name: 'Data platform' }];

  let links = BASE.links.slice();
  if (has('extraLink')) links = [...links, [101, 20], [101, 30]];
  if (has('dupLink')) links = [...links, [102, 30]];
  if (has('cascade')) links = links.filter(([e]) => e !== 101);

  const deps = has('cascade') ? BASE.deps.filter((d) => d.eid !== 101) : BASE.deps;
  const skills = has('cascade') ? BASE.skills.filter(([e]) => e !== 101) : BASE.skills;

  const firstProject = (eid: number) => BASE.links.find(([e]) => e === eid)?.[1] ?? null;
  const leadOf = (pid: number) =>
    pid === 40 ? 101 : (BASE.oneOne.find(([, p]) => p === pid)?.[0] ?? null);
  const soloProject = (eid: number) => BASE.oneOne.find(([e]) => e === eid)?.[1] ?? null;
  const pname = (pid: number | null) => projects.find((p) => p.id === pid)?.name ?? null;

  const tables: Tbl[] = [];
  const edges: Edge[] = [];
  let result: ResultSet;
  let enforced: string;
  let rule: string;

  const empCols: Col[] = [
    { name: 'id', type: 'bigint', pk: true, tip: 'The entity key. Every entity box becomes a table with a key of its own.' },
    { name: 'name', type: 'text', nn: true, tip: 'A single-valued attribute becomes one column.' },
    { name: 'hired', type: 'int', tip: 'A single-valued attribute becomes one column.' },
  ];

  const projCols: Col[] = [
    { name: 'id', type: 'bigint', pk: true, tip: 'The project entity key.' },
    { name: 'name', type: 'text', nn: true, tip: 'A single-valued attribute.' },
  ];

  if (cfg.card === '1:N') {
    const nn = cfg.part === 'total';
    empCols.push({
      name: 'project_id',
      type: 'bigint',
      nn,
      ref: { t: 'project', c: 'id' },
      tip: nn
        ? 'The FK lives on the many side. NOT NULL is the whole of what "total participation" maps to.'
        : 'Nullable FK: partial participation. Every query that must not lose these rows needs an OUTER join.',
    });
    const stored = employees.filter((e) => !nn || firstProject(e.id) !== null);
    tables.push({
      id: 'employee',
      name: 'employee',
      cols: empCols,
      col: 0,
      band: 0,
      tip: 'The many side. It carries the reference, because a column holds exactly one value.',
      rows: stored.map((e) => ({
        cells: [String(e.id), e.name, String(e.hired), firstProject(e.id) === null ? null : String(firstProject(e.id))],
      })),
      idx: [
        { name: 'employee_pkey', def: '-- employee_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' },
        {
          name: 'employee_project_id_idx',
          def: 'CREATE INDEX employee_project_id_idx ON employee (project_id);',
          auto: false,
          why: 'Postgres does not index a foreign key for you. Without this, deleting a project scans employee.',
        },
      ],
    });
    tables.push({
      id: 'project',
      name: 'project',
      cols: projCols,
      col: 2,
      band: 0,
      tip: 'The one side. It knows nothing about the relationship — no column changes here at all.',
      rows: projects.filter((p) => p.id !== 40).map((p) => ({ cells: [String(p.id), p.name] })),
      idx: [{ name: 'project_pkey', def: '-- project_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' }],
    });
    edges.push({
      from: 'employee',
      fromCol: 'project_id',
      to: 'project',
      toCol: 'id',
      many: true,
      optional: !nn,
      label: nn ? 'N:1 · NOT NULL' : 'N:1 · nullable',
      tip: nn
        ? 'Mandatory: an employee cannot be stored without a project. INNER and LEFT joins return the same rows.'
        : 'Optional: an employee may have no project. INNER JOIN silently drops those rows.',
    });

    const joined = tables[0].rows.map((r) => {
      const pid = r.cells[3];
      return {
        cells: [r.cells[1], pid === null ? null : pname(Number(pid))],
        dropped: pid === null,
      };
    });
    result = {
      sql: nn
        ? ['SELECT e.name AS employee, p.name AS project', '  FROM employee e', '  JOIN project  p ON p.id = e.project_id;']
        : [
            'SELECT e.name AS employee, p.name AS project',
            '  FROM employee e',
            '  LEFT JOIN project p ON p.id = e.project_id;  -- INNER would lose the unassigned',
          ],
      head: ['employee', 'project'],
      rows: joined,
    };
    enforced = nn
      ? 'at most one project per employee (structurally) + at least one (NOT NULL)'
      : 'at most one project per employee — structurally, for free';
    rule =
      'One-to-many maps to a foreign key on the many side. The column can hold one value, so the "at most one" half of the cardinality is enforced by the shape of the row rather than by any constraint — there is nowhere to put a second project id.';
  } else if (cfg.card === 'M:N') {
    tables.push({
      id: 'employee',
      name: 'employee',
      cols: empCols,
      col: 0,
      band: 0,
      tip: 'Unchanged. Neither entity table learns anything about an M:N relationship.',
      rows: employees.map((e) => ({ cells: [String(e.id), e.name, String(e.hired)], bad: e.id === 107 })),
      idx: [{ name: 'employee_pkey', def: '-- employee_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' }],
    });
    tables.push({
      id: 'employee_project',
      name: 'employee_project',
      cols: [
        {
          name: 'employee_id',
          type: 'bigint',
          pk: cfg.enforce,
          nn: !cfg.enforce,
          ref: { t: 'employee', c: 'id', onDelete: 'CASCADE' },
          tip: 'Half of the composite key, and an FK. The junction row IS the relationship instance.',
        },
        {
          name: 'project_id',
          type: 'bigint',
          pk: cfg.enforce,
          nn: !cfg.enforce,
          ref: { t: 'project', c: 'id', onDelete: 'CASCADE' },
          tip: 'The other half. Together they say: this pair appears at most once.',
        },
        { name: 'since', type: 'date', tip: 'The moment the junction grows an attribute of its own, it was an entity all along.' },
      ],
      col: 1,
      band: 0,
      tip: 'The junction (associative) table. It exists because neither side has room for a list.',
      rows: links.map(([e, p], i) => ({
        cells: [String(e), String(p), `20${20 + (i % 5)}-0${1 + (i % 9)}-01`],
        bad: has('dupLink') && i === links.length - 1,
      })),
      idx: cfg.enforce
        ? [
            {
              name: 'employee_project_pkey',
              def: '-- employee_project_pkey: btree (employee_id, project_id)',
              auto: true,
              why: 'Serves "which projects is this employee on?" — the leading column only.',
            },
            {
              name: 'employee_project_project_id_idx',
              def: 'CREATE INDEX employee_project_project_id_idx ON employee_project (project_id);',
              auto: false,
              why: 'The reverse direction: "who is on this project?" The composite PK cannot answer it.',
            },
          ]
        : [
            {
              name: 'employee_project_project_id_idx',
              def: 'CREATE INDEX employee_project_project_id_idx ON employee_project (project_id);',
              auto: false,
              why: 'The reverse direction. With no PK there is no index for the forward direction either.',
            },
          ],
    });
    tables.push({
      id: 'project',
      name: 'project',
      cols: projCols,
      col: 2,
      band: 0,
      tip: 'Also unchanged.',
      rows: projects.filter((p) => p.id !== 40).map((p) => ({ cells: [String(p.id), p.name] })),
      idx: [{ name: 'project_pkey', def: '-- project_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' }],
    });
    edges.push({
      from: 'employee_project',
      fromCol: 'employee_id',
      to: 'employee',
      toCol: 'id',
      many: true,
      optional: false,
      label: 'N:1',
      tip: 'Each junction row names exactly one employee; an employee may own many junction rows.',
    });
    edges.push({
      from: 'employee_project',
      fromCol: 'project_id',
      to: 'project',
      toCol: 'id',
      many: true,
      optional: false,
      label: 'N:1',
      tip: 'And exactly one project. Two N:1 edges is what an M:N looks like once it is physical.',
    });

    const joined = employees.flatMap((e) => {
      const mine = links.filter(([le]) => le === e.id);
      if (mine.length === 0) return [{ cells: [e.name, null], dropped: true }];
      return mine.map(([, p]) => ({ cells: [e.name, pname(p)], dropped: false }));
    });
    result = {
      sql: [
        'SELECT e.name AS employee, p.name AS project',
        '  FROM employee e',
        '  LEFT JOIN employee_project ep ON ep.employee_id = e.id',
        '  LEFT JOIN project          p  ON p.id = ep.project_id;',
      ],
      head: ['employee', 'project'],
      rows: joined,
    };
    enforced = cfg.enforce
      ? 'each pair at most once (composite PK). Nothing about how many.'
      : 'nothing. No PK on the junction means the same pair can appear any number of times.';
    rule =
      'Many-to-many maps to a junction table whose primary key is the pair of foreign keys. That composite key is the only cardinality it enforces — "this link exists at most once" — and it indexes one direction only, so the reverse lookup needs an index of its own.';
  } else if (cfg.strategy === 'merge') {
    const cols: Col[] = [
      ...empCols,
      {
        name: 'project_id',
        type: 'bigint',
        unique: cfg.enforce,
        tip: 'The absorbed entity keeps its own identity as a column. UNIQUE is what stops two employees claiming one project.',
      },
      { name: 'project_name', type: 'text', tip: 'Absorbed attribute. Nullable now, whatever the project table said.' },
    ];
    tables.push({
      id: 'employee',
      name: 'employee',
      cols,
      col: 0,
      band: 0,
      tip: 'One table for two entities. No join, no FK, and no way to store a project nobody works on.',
      rows: employees.map((e) => {
        const p = soloProject(e.id);
        return { cells: [String(e.id), e.name, String(e.hired), p === null ? null : String(p), pname(p)] };
      }),
      idx: [
        { name: 'employee_pkey', def: '-- employee_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' },
        ...(cfg.enforce
          ? [
              {
                name: 'employee_project_id_key',
                def: '-- employee_project_id_key: created by UNIQUE',
                auto: true,
                why: 'The other half of 1:1 — without it the merged table is a 1:N in disguise.',
              },
            ]
          : []),
      ],
    });
    result = {
      sql: ['SELECT name AS employee, project_name AS project', '  FROM employee;  -- no join at all'],
      head: ['employee', 'project'],
      rows: employees.map((e) => {
        const p = soloProject(e.id);
        return { cells: [e.name, pname(p)], dropped: false };
      }),
    };
    enforced = cfg.enforce
      ? 'one project per employee (row shape) and one employee per project (UNIQUE)'
      : 'one project per employee only — two rows may claim the same project_id';
    rule =
      'A 1:1 can be merged into one table. The join disappears and so does the null-free project row: every absorbed column becomes nullable, and a project cannot exist before the employee who holds it.';
  } else {
    const nn = cfg.part === 'total';
    tables.push({
      id: 'employee',
      name: 'employee',
      cols: empCols,
      col: 0,
      band: 0,
      tip: 'Untouched. The FK went to the other side, which is why employee-side totality is now unenforceable.',
      rows: employees.map((e) => ({ cells: [String(e.id), e.name, String(e.hired)] })),
      idx: [{ name: 'employee_pkey', def: '-- employee_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' }],
    });
    tables.push({
      id: 'project',
      name: 'project',
      cols: [
        ...projCols,
        {
          name: 'lead_employee_id',
          type: 'bigint',
          nn,
          unique: cfg.enforce,
          ref: { t: 'employee', c: 'id' },
          tip: 'A 1:1 is a 1:N with a UNIQUE on the FK. Drop the UNIQUE and it quietly becomes 1:N again.',
        },
      ],
      col: 2,
      band: 0,
      tip: 'The side that carries the FK. Whichever side that is, that is the side whose participation NOT NULL can police.',
      rows: projects.map((p) => ({
        cells: [String(p.id), p.name, String(leadOf(p.id))],
        bad: p.id === 40,
      })),
      idx: [
        { name: 'project_pkey', def: '-- project_pkey: created by PRIMARY KEY', auto: true, why: 'Implicit unique index behind the primary key.' },
        ...(cfg.enforce
          ? [
              {
                name: 'project_lead_employee_id_key',
                def: '-- project_lead_employee_id_key: created by UNIQUE',
                auto: true,
                why: 'Enforces the 1:1 AND indexes the FK — one object doing both jobs.',
              },
            ]
          : [
              {
                name: 'project_lead_employee_id_idx',
                def: 'CREATE INDEX project_lead_employee_id_idx ON project (lead_employee_id);',
                auto: false,
                why: 'Without UNIQUE you still need the FK index, and you have lost the cardinality.',
              },
            ]),
      ],
    });
    edges.push({
      from: 'project',
      fromCol: 'lead_employee_id',
      to: 'employee',
      toCol: 'id',
      many: !cfg.enforce,
      optional: !nn,
      label: cfg.enforce ? '1:1 · UNIQUE' : 'N:1 · no UNIQUE',
      tip: cfg.enforce
        ? 'UNIQUE turns the N side into a 1. This is the entire difference between a 1:1 and a 1:N in DDL.'
        : 'The drawing says 1:1; the schema says 1:N. Nothing will stop a second project pointing here.',
    });
    result = {
      sql: [
        'SELECT e.name AS employee, p.name AS project',
        '  FROM employee e',
        '  LEFT JOIN project p ON p.lead_employee_id = e.id;',
      ],
      head: ['employee', 'project'],
      rows: employees.map((e) => {
        const mine = projects.filter((p) => leadOf(p.id) === e.id);
        return mine.length > 0
          ? { cells: [e.name, mine.map((p) => p.name).join(' + ')], dropped: false }
          : { cells: [e.name, null], dropped: true };
      }),
    };
    enforced = cfg.enforce
      ? 'one lead per project' + (nn ? ' and every project has one' : '') + '; one project per employee (UNIQUE)'
      : 'one lead per project' + (nn ? ', mandatory' : '') + '. Nothing limits an employee to one project.';
    rule =
      'Split 1:1 keeps two tables and puts a UNIQUE foreign key on one of them. Which side carries it decides which side NOT NULL can make mandatory — the other side stays a convention.';
  }

  if (cfg.weak) {
    tables.push({
      id: 'dependent',
      name: 'dependent',
      cols: [
        {
          name: 'employee_id',
          type: 'bigint',
          pk: true,
          ref: { t: 'employee', c: 'id', onDelete: 'CASCADE' },
          tip: 'The owner key is part of the weak entity key. This is what "identifying relationship" means physically.',
        },
        { name: 'dep_name', type: 'text', pk: true, tip: 'The partial key — unique only within one employee.' },
        { name: 'born', type: 'int', tip: 'An ordinary attribute.' },
      ],
      col: 0,
      band: 1,
      tip: 'A weak entity: no key of its own, and no existence apart from its owner.',
      rows: deps.map((d) => ({ cells: [String(d.eid), d.name, String(d.born)] })),
      idx: [
        {
          name: 'dependent_pkey',
          def: '-- dependent_pkey: btree (employee_id, dep_name)',
          auto: true,
          why: 'Owner-first, so "all dependents of one employee" is a range scan — and the FK is indexed for free.',
        },
      ],
    });
    edges.push({
      from: 'dependent',
      fromCol: 'employee_id',
      to: 'employee',
      toCol: 'id',
      many: true,
      optional: false,
      label: 'identifying · CASCADE',
      tip: 'ON DELETE CASCADE is the existence dependency, made executable.',
    });
  }

  if (cfg.mv) {
    tables.push({
      id: 'employee_skill',
      name: 'employee_skill',
      cols: [
        {
          name: 'employee_id',
          type: 'bigint',
          pk: true,
          ref: { t: 'employee', c: 'id', onDelete: 'CASCADE' },
          tip: 'A multi-valued attribute becomes a child table keyed by owner plus value.',
        },
        { name: 'skill', type: 'text', pk: true, tip: 'The value itself is the rest of the key: the same skill twice is meaningless.' },
      ],
      col: 1,
      band: 1,
      tip: 'The multi-valued attribute "skills". First normal form is the reason this is not a comma-separated column.',
      rows: skills.map(([e, s]) => ({ cells: [String(e), s] })),
      idx: [
        {
          name: 'employee_skill_pkey',
          def: '-- employee_skill_pkey: btree (employee_id, skill)',
          auto: true,
          why: 'Dedupes and indexes the owner side at once.',
        },
        {
          name: 'employee_skill_skill_idx',
          def: 'CREATE INDEX employee_skill_skill_idx ON employee_skill (skill);',
          auto: false,
          why: '"Who knows Rust?" reads the key backwards and needs its own index.',
        },
      ],
    });
    edges.push({
      from: 'employee_skill',
      fromCol: 'employee_id',
      to: 'employee',
      toCol: 'id',
      many: true,
      optional: false,
      label: 'N:1 · CASCADE',
      tip: 'Same shape as a weak entity, because a multi-valued attribute is one.',
    });
  }

  return { tables, edges, result, enforced, rule };
}

/* ---------------------------------------------------------------- the DDL */

type Line = { t: string; tone?: 'pk' | 'fk' | 'uniq' | 'muted'; mark?: boolean };

function ddlFor(t: Tbl): Line[] {
  const w = Math.max(...t.cols.map((c) => c.name.length));
  const tw = Math.max(...t.cols.map((c) => c.type.length));
  const pks = t.cols.filter((c) => c.pk);
  const composite = pks.length > 1;
  const out: Line[] = [{ t: `CREATE TABLE ${t.name} (` }];
  t.cols.forEach((c, i) => {
    let s = `  ${c.name.padEnd(w)} ${c.type.padEnd(tw)}`;
    if (c.pk && !composite) s += ' PRIMARY KEY';
    if (c.nn && !c.pk) s += ' NOT NULL';
    if (c.unique) s += ' UNIQUE';
    if (c.ref) s += ` REFERENCES ${c.ref.t}(${c.ref.c})${c.ref.onDelete ? ` ON DELETE ${c.ref.onDelete}` : ''}`;
    const last = i === t.cols.length - 1 && !composite;
    out.push({ t: s + (last ? '' : ','), tone: c.pk ? 'pk' : c.ref ? 'fk' : c.unique ? 'uniq' : undefined });
  });
  if (composite) out.push({ t: `  PRIMARY KEY (${pks.map((c) => c.name).join(', ')})`, tone: 'pk' });
  if (!composite && pks.length === 0) out.push({ t: '  -- no PRIMARY KEY declared', tone: 'muted' });
  out.push({ t: ');' });
  t.idx
    .filter((i) => !i.auto)
    .forEach((i) => out.push({ t: i.def, tone: 'uniq' }));
  return out;
}

const TONE: Record<string, string> = {
  pk: 'var(--viz-1)',
  fk: 'var(--viz-2)',
  uniq: 'var(--viz-7)',
  muted: 'var(--viz-ink-muted)',
};

/* ------------------------------------------------------------- violations */

type Outcome = 'rejected' | 'accepted' | 'impossible';

type Violation = {
  id: string;
  label: string;
  sql: string;
  outcome: Outcome;
  code: string;
  msg: string;
  why: string;
  extra?: string;
};

function violationsFor(cfg: Cfg): Violation[] {
  const v: Violation[] = [];
  if (cfg.card === '1:N') {
    v.push({
      id: 'two',
      label: 'put one employee on two projects',
      sql: 'UPDATE employee SET project_id = 30 WHERE id = 101;',
      outcome: 'impossible',
      code: 'UPDATE 1',
      msg: 'The second value replaced the first. There was never anywhere to keep both.',
      why: 'The "at most one" half of a 1:N costs nothing to enforce: a column holds one value. No constraint is involved, which is why this half never breaks.',
    });
    v.push(
      cfg.part === 'total'
        ? {
            id: 'orphan',
            label: 'hire an employee with no project',
            sql: "INSERT INTO employee (id, name, hired) VALUES (107, 'Adeyemi', 2025);",
            outcome: 'rejected',
            code: '23502 not_null_violation',
            msg: 'ERROR:  null value in column "project_id" of relation "employee" violates not-null constraint',
            why: 'Total participation is one word of DDL — NOT NULL — and it is the only minimum cardinality SQL can declare without a trigger.',
          }
        : {
            id: 'orphan',
            label: 'hire an employee with no project',
            sql: "INSERT INTO employee (id, name, hired) VALUES (107, 'Adeyemi', 2025);",
            outcome: 'accepted',
            code: 'INSERT 0 1',
            msg: 'Accepted — and correctly so: you drew partial participation, so this row is legal.',
            why: 'The cost lands in the queries. Every INNER JOIN to project now drops this employee, and nothing warns you.',
            extra: 'noLink',
          },
    );
    v.push({
      id: 'fk',
      label: 'point at a project that does not exist',
      sql: "INSERT INTO employee (id, name, hired, project_id) VALUES (108, 'Haddad', 2025, 99);",
      outcome: 'rejected',
      code: '23503 foreign_key_violation',
      msg: 'ERROR:  insert or update on table "employee" violates foreign key constraint "employee_project_id_fkey"',
      why: 'Referential integrity is the one thing the FK genuinely buys. MySQL reports the same thing as error 1452; SQLite ignores it entirely unless PRAGMA foreign_keys = ON.',
    });
  } else if (cfg.card === 'M:N') {
    v.push(
      cfg.enforce
        ? {
            id: 'dup',
            label: 'record the same link twice',
            sql: 'INSERT INTO employee_project (employee_id, project_id) VALUES (102, 30);',
            outcome: 'rejected',
            code: '23505 unique_violation',
            msg: 'ERROR:  duplicate key value violates unique constraint "employee_project_pkey"',
            why: 'The composite PK is the junction table’s entire contribution to correctness: a relationship instance exists at most once.',
          }
        : {
            id: 'dup',
            label: 'record the same link twice',
            sql: 'INSERT INTO employee_project (employee_id, project_id) VALUES (102, 30);',
            outcome: 'accepted',
            code: 'INSERT 0 1',
            msg: 'Accepted. Nakamura is now on Search index twice.',
            why: 'This is the most common junction-table bug in the wild: a surrogate id column was added, the composite key was dropped, and every COUNT(*) over the join is now wrong.',
            extra: 'dupLink',
          },
    );
    v.push({
      id: 'min',
      label: 'hire an employee with no project at all',
      sql: "INSERT INTO employee (id, name, hired) VALUES (107, 'Adeyemi', 2025);",
      outcome: 'accepted',
      code: 'INSERT 0 1',
      msg: 'Accepted, even though you drew total participation.',
      why: 'Minimum cardinality across a junction table is not expressible in DDL. There is no column to make NOT NULL, and Postgres will not let you defer a CHECK — enforcing it needs a deferred constraint trigger, or acceptance that it is a convention.',
      extra: 'noLink',
    });
    v.push({
      id: 'max',
      label: 'put one employee on three projects',
      sql: 'INSERT INTO employee_project (employee_id, project_id) VALUES (101, 20), (101, 30);',
      outcome: 'accepted',
      code: 'INSERT 0 2',
      msg: 'Accepted. Reyes is now on every project.',
      why: 'Any maximum above one — "at most three projects" — is also beyond DDL. M:N means the schema enforces no count in either direction.',
      extra: 'extraLink',
    });
  } else if (cfg.strategy === 'merge') {
    v.push({
      id: 'two',
      label: 'give one employee two projects',
      sql: "UPDATE employee SET project_id = 30, project_name = 'Search index' WHERE id = 101;",
      outcome: 'impossible',
      code: 'UPDATE 1',
      msg: 'Overwritten, not added. One row has one set of project columns.',
      why: 'Merging enforces one direction of the 1:1 by construction. The other direction needs the UNIQUE.',
    });
    v.push(
      cfg.enforce
        ? {
            id: 'share',
            label: 'give two employees the same project',
            sql: 'UPDATE employee SET project_id = 10 WHERE id = 102;',
            outcome: 'rejected',
            code: '23505 unique_violation',
            msg: 'ERROR:  duplicate key value violates unique constraint "employee_project_id_key"',
            why: 'Without that UNIQUE the merged table is a 1:N wearing a 1:1 diagram.',
          }
        : {
            id: 'share',
            label: 'give two employees the same project',
            sql: 'UPDATE employee SET project_id = 10 WHERE id = 102;',
            outcome: 'accepted',
            code: 'UPDATE 1',
            msg: 'Accepted. Two employees now hold project 10.',
            why: 'The drawing said 1:1 and the schema says nothing. This is the failure the diagram cannot catch and the DDL was never asked to.',
          },
    );
    v.push({
      id: 'lonely',
      label: 'create a project nobody works on',
      sql: "INSERT INTO employee (id, project_id, project_name) VALUES (109, 40, 'Data platform');",
      outcome: 'rejected',
      code: '23502 not_null_violation',
      msg: 'ERROR:  null value in column "name" of relation "employee" violates not-null constraint',
      why: 'A merged table cannot hold half a row. The absorbed entity loses independent existence — which is exactly why you only merge when both sides are mandatory.',
    });
  } else {
    v.push(
      cfg.enforce
        ? {
            id: 'share',
            label: 'give two projects the same lead',
            sql: "INSERT INTO project (id, name, lead_employee_id) VALUES (40, 'Data platform', 101);",
            outcome: 'rejected',
            code: '23505 unique_violation',
            msg: 'ERROR:  duplicate key value violates unique constraint "project_lead_employee_id_key"',
            why: 'UNIQUE on the FK is the whole of 1:1. It also indexes the FK, so it costs you nothing extra.',
          }
        : {
            id: 'share',
            label: 'give two projects the same lead',
            sql: "INSERT INTO project (id, name, lead_employee_id) VALUES (40, 'Data platform', 101);",
            outcome: 'accepted',
            code: 'INSERT 0 1',
            msg: 'Accepted. Reyes now leads two projects, and the result set below returns both.',
            why: 'Drop the UNIQUE and a 1:1 silently degrades to 1:N. Every consumer that assumed one row now gets two — usually discovered by a report that doubled.',
            extra: 'sharedLead',
          },
    );
    v.push({
      id: 'min',
      label: 'hire an employee who leads nothing',
      sql: "INSERT INTO employee (id, name, hired) VALUES (107, 'Adeyemi', 2025);",
      outcome: 'accepted',
      code: 'INSERT 0 1',
      msg: 'Accepted. The NOT NULL you set is on project.lead_employee_id, not on employee.',
      why: 'Whichever side holds the FK is the side whose participation you can make mandatory. The other side is a convention until you add a trigger — or merge the tables.',
      extra: 'noLink',
    });
  }
  if (cfg.weak) {
    v.push({
      id: 'dep-dup',
      label: 'give one employee two dependents with the same name',
      sql: "INSERT INTO dependent (employee_id, dep_name, born) VALUES (101, 'Mira', 2018);",
      outcome: 'rejected',
      code: '23505 unique_violation',
      msg: 'ERROR:  duplicate key value violates unique constraint "dependent_pkey"',
      why: 'The partial key is unique only inside one owner. Another employee’s Mira is perfectly legal.',
    });
    v.push({
      id: 'cascade',
      label: 'delete the owner of a weak entity',
      sql: 'DELETE FROM employee WHERE id = 101;',
      outcome: 'accepted',
      code: 'DELETE 1',
      msg: 'Accepted, and the dependents went with it.',
      why: 'ON DELETE CASCADE is the existence dependency made executable. Without it you get 23503 and a delete you cannot perform.',
      extra: 'cascade',
    });
  }
  if (cfg.mv) {
    v.push({
      id: 'skill-dup',
      label: 'record the same skill twice',
      sql: "INSERT INTO employee_skill (employee_id, skill) VALUES (102, 'go');",
      outcome: 'rejected',
      code: '23505 unique_violation',
      msg: 'ERROR:  duplicate key value violates unique constraint "employee_skill_pkey"',
      why: 'A set-valued attribute is a set. Keying on (owner, value) is what makes it one.',
    });
  }
  return v;
}

/* ------------------------------------------------------- reverse-mode data */

type Puzzle = {
  id: string;
  label: string;
  answer: 'one-many' | 'many-many' | 'one-one' | 'self';
  tables: Tbl[];
  edges: Edge[];
  evidence: string;
  verdict: string;
};

const mkCol = (name: string, type: string, o: Partial<Col> = {}): Col => ({
  name,
  type,
  tip: o.tip ?? '',
  ...o,
});

const PUZZLES: Puzzle[] = [
  {
    id: 'token',
    label: 'account / api_token',
    answer: 'one-many',
    evidence: 'api_token.account_id: NOT NULL, a plain FK with no UNIQUE above it',
    verdict:
      'One-to-many, with a mandatory parent. Nothing stops two tokens carrying the same account_id, so the child side is "many"; NOT NULL says every token belongs to an account; nothing says an account must have a token, so the parent side is optional. ON DELETE CASCADE tells you the child has no life of its own — this is close to a weak entity with a surrogate key bolted on.',
    tables: [
      {
        id: 'account',
        name: 'account',
        col: 0,
        band: 0,
        tip: 'The parent.',
        rows: [],
        idx: [],
        cols: [mkCol('id', 'bigserial', { pk: true }), mkCol('email', 'text', { nn: true, unique: true })],
      },
      {
        id: 'api_token',
        name: 'api_token',
        col: 2,
        band: 0,
        tip: 'The child. Its own surrogate key, plus a reference.',
        rows: [],
        idx: [],
        cols: [
          mkCol('id', 'bigserial', { pk: true }),
          mkCol('account_id', 'bigint', { nn: true, ref: { t: 'account', c: 'id', onDelete: 'CASCADE' } }),
          mkCol('digest', 'bytea', { nn: true }),
          mkCol('expires_at', 'timestamptz'),
        ],
      },
    ],
    edges: [
      {
        from: 'api_token',
        fromCol: 'account_id',
        to: 'account',
        toCol: 'id',
        many: true,
        optional: false,
        label: 'N:1 · NOT NULL',
        tip: 'A non-unique, non-null FK is the signature of a mandatory-child 1:N.',
      },
    ],
  },
  {
    id: 'shipment',
    label: 'shipment / shipment_item / product',
    answer: 'many-many',
    evidence: 'PRIMARY KEY (shipment_id, sku) over exactly two FKs, plus an index on (sku)',
    verdict:
      'Many-to-many. A composite primary key made of exactly two foreign keys is the fingerprint of a junction table. The qty column means it is really an associative entity — the relationship carries data of its own — and the standalone index on (sku) exists because the composite PK only indexes shipment-first; somebody had to answer "which shipments carried this SKU".',
    tables: [
      {
        id: 'shipment',
        name: 'shipment',
        col: 0,
        band: 0,
        tip: 'One entity.',
        rows: [],
        idx: [],
        cols: [mkCol('id', 'bigserial', { pk: true }), mkCol('shipped_at', 'timestamptz')],
      },
      {
        id: 'shipment_item',
        name: 'shipment_item',
        col: 1,
        band: 0,
        tip: 'No key of its own: its key is the pair. That is a relationship, not an entity.',
        rows: [],
        idx: [
          {
            name: 'shipment_item_sku_idx',
            def: 'CREATE INDEX shipment_item_sku_idx ON shipment_item (sku);',
            auto: false,
            why: 'The reverse direction.',
          },
        ],
        cols: [
          mkCol('shipment_id', 'bigint', { pk: true, ref: { t: 'shipment', c: 'id' } }),
          mkCol('sku', 'text', { pk: true, ref: { t: 'product', c: 'sku' } }),
          mkCol('qty', 'int', { nn: true }),
        ],
      },
      {
        id: 'product',
        name: 'product',
        col: 2,
        band: 0,
        tip: 'The other entity, keyed on a natural key.',
        rows: [],
        idx: [],
        cols: [mkCol('sku', 'text', { pk: true }), mkCol('name', 'text', { nn: true })],
      },
    ],
    edges: [
      { from: 'shipment_item', fromCol: 'shipment_id', to: 'shipment', toCol: 'id', many: true, optional: false, label: 'N:1', tip: 'Half of the pair.' },
      { from: 'shipment_item', fromCol: 'sku', to: 'product', toCol: 'sku', many: true, optional: false, label: 'N:1', tip: 'The other half.' },
    ],
  },
  {
    id: 'subscription',
    label: 'account / subscription',
    answer: 'one-one',
    evidence: 'subscription.account_id is the PRIMARY KEY and the FOREIGN KEY at once',
    verdict:
      'One-to-one. When a table’s primary key is also its only foreign key, there can be at most one child row per parent — and it needs no extra UNIQUE and no extra index, which is why PK-as-FK is the better of the two split-1:1 mappings. Participation is total on the subscription side and partial on the account side: an account with no subscription is simply a missing row, so every query joining them must decide between INNER and LEFT.',
    tables: [
      {
        id: 'account',
        name: 'account',
        col: 0,
        band: 0,
        tip: 'The parent.',
        rows: [],
        idx: [],
        cols: [mkCol('id', 'bigserial', { pk: true }), mkCol('email', 'text', { nn: true, unique: true })],
      },
      {
        id: 'subscription',
        name: 'subscription',
        col: 2,
        band: 0,
        tip: 'Key and reference in one column.',
        rows: [],
        idx: [],
        cols: [
          mkCol('account_id', 'bigint', { pk: true, ref: { t: 'account', c: 'id', onDelete: 'CASCADE' } }),
          mkCol('plan', 'text', { nn: true }),
          mkCol('renews_at', 'date'),
        ],
      },
    ],
    edges: [
      {
        from: 'subscription',
        fromCol: 'account_id',
        to: 'account',
        toCol: 'id',
        many: false,
        optional: false,
        label: '1:1 · PK = FK',
        tip: 'PK-as-FK: one row at most, for free.',
      },
    ],
  },
  {
    id: 'org',
    label: 'employee (self-referencing)',
    answer: 'self',
    evidence: 'manager_id references the same table, and it is nullable',
    verdict:
      'A self-referencing 1:N — a hierarchy. One table plays both roles, the nullable FK marks the roots (the CEO has no manager), and the cycle it creates is why you reach for a recursive CTE rather than a join. Note what is still not enforced: nothing prevents a cycle of managers, because no declarative constraint can see more than one row at a time.',
    tables: [
      {
        id: 'employee',
        name: 'employee',
        col: 1,
        band: 0,
        tip: 'Both ends of the relationship live here.',
        rows: [],
        idx: [
          {
            name: 'employee_manager_id_idx',
            def: 'CREATE INDEX employee_manager_id_idx ON employee (manager_id);',
            auto: false,
            why: '"Who reports to me" needs it, and so does every delete of a manager.',
          },
        ],
        cols: [
          mkCol('id', 'bigserial', { pk: true }),
          mkCol('name', 'text', { nn: true }),
          mkCol('manager_id', 'bigint', { ref: { t: 'employee', c: 'id' } }),
        ],
      },
    ],
    edges: [
      {
        from: 'employee',
        fromCol: 'manager_id',
        to: 'employee',
        toCol: 'id',
        many: true,
        optional: true,
        label: 'N:1 · nullable, same table',
        tip: 'One table, both roles. The nullable FK is where the hierarchy stops.',
      },
    ],
  },
];

const READINGS = [
  { value: 'one-many' as const, label: '1:N' },
  { value: 'many-many' as const, label: 'M:N' },
  { value: 'one-one' as const, label: '1:1' },
  { value: 'self' as const, label: 'self-ref' },
];

/* --------------------------------------------------------------- geometry */

const BOXW = 192;
const HDR = 22;
const ROWH = 15;
const PADB = 7;
const COLX = [8, 240, 472];

function tblH(t: Tbl) {
  return HDR + t.cols.length * ROWH + PADB;
}

function layout(tables: Tbl[]) {
  const pos: Record<string, { x: number; y: number; h: number }> = {};
  const bandTop = [14, 0];
  const band0 = tables.filter((t) => t.band === 0);
  bandTop[1] = 14 + Math.max(0, ...band0.map(tblH)) + 34;
  for (const t of tables) {
    const stack = tables.filter((o) => o.col === t.col && o.band === t.band);
    const i = stack.indexOf(t);
    const y = bandTop[t.band] + stack.slice(0, i).reduce((a, o) => a + tblH(o) + 18, 0);
    pos[t.id] = { x: COLX[t.col], y, h: tblH(t) };
  }
  return pos;
}

function colY(t: Tbl, name: string, y: number) {
  const i = t.cols.findIndex((c) => c.name === name);
  return y + HDR + i * ROWH + ROWH / 2;
}

/* -------------------------------------------------------------- component */

export default function ErMappingStudio() {
  const [mode, setMode] = useState<Mode>('forward');
  const [card, setCard] = useState<Card>('1:N');
  const [strategy, setStrategy] = useState<Strategy>('split');
  const [part, setPart] = useState<Part>('partial');
  const [weak, setWeak] = useState(false);
  const [mv, setMv] = useState(false);
  const [enforce, setEnforce] = useState(true);
  const [applied, setApplied] = useState<string[]>([]);
  const [vIdx, setVIdx] = useState(0);
  const [last, setLast] = useState<Violation | null>(null);
  const [puzzleId, setPuzzleId] = useState(PUZZLES[0].id);
  const [reading, setReading] = useState<Puzzle['answer']>('one-many');
  const [guessed, setGuessed] = useState(false);
  const tip = useTip();

  const clear = () => {
    setApplied([]);
    setVIdx(0);
    setLast(null);
  };
  const on =
    <T,>(set: (v: T) => void) =>
    (v: T) => {
      set(v);
      clear();
    };

  const cfg: Cfg = { card, strategy, part, weak, mv, enforce };
  const schema = build(cfg, applied);
  const puzzle = PUZZLES.find((p) => p.id === puzzleId)!;

  const tables = mode === 'forward' ? schema.tables : puzzle.tables;
  const edges = mode === 'forward' ? schema.edges : puzzle.edges;
  const pos = layout(tables);
  const svgH = Math.max(...Object.values(pos).map((p) => p.y + p.h)) + 16;
  const svgW = COLX[2] + BOXW + 10;

  const vlist = violationsFor(cfg);
  const nextV = vlist[vIdx % vlist.length];
  const fire = () => {
    setLast(nextV);
    if (nextV.extra && !applied.includes(nextV.extra)) setApplied([...applied, nextV.extra]);
    setVIdx(vIdx + 1);
  };

  const rowsStored = schema.tables.reduce((a, t) => a + t.rows.length, 0);
  const kept = schema.result.rows.filter((r) => !r.dropped).length;
  const dropped = schema.result.rows.length - kept;
  const manualIdx = schema.tables.reduce((a, t) => a + t.idx.filter((i) => !i.auto).length, 0);

  const outcomeColor =
    last?.outcome === 'rejected'
      ? 'var(--viz-good)'
      : last?.outcome === 'accepted'
        ? 'var(--viz-critical)'
        : 'var(--viz-ink-2)';

  const correct = guessed && reading === puzzle.answer;

  return (
    <VizPanel
      title="ER → tables, generated live"
      subtitle="Edit the drawn model on the left of each control and watch the DDL, the table graph and the join result re-derive. Then try to insert a row the drawing forbids and see which declared constraint — if any — stops you."
      controls={
        <>
          <Segmented<Mode>
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'forward', label: 'Model → tables' },
              { value: 'reverse', label: 'Tables → model' },
            ]}
          />
          {mode === 'forward' ? (
            <>
              <Segmented<Card>
                label="Cardinality"
                value={card}
                onChange={on(setCard)}
                options={[
                  { value: '1:N', label: '1:N', title: 'A project has many employees; an employee works on one' },
                  { value: 'M:N', label: 'M:N', title: 'Employees split their time across projects' },
                  { value: '1:1', label: '1:1', title: 'One dedicated employee per project' },
                ]}
              />
              {card === '1:1' ? (
                <Segmented<Strategy>
                  label="1:1 strategy"
                  value={strategy}
                  onChange={on(setStrategy)}
                  options={[
                    { value: 'merge', label: 'merge', title: 'One table holding both entities' },
                    { value: 'split', label: 'split + UNIQUE FK', title: 'Two tables, a UNIQUE foreign key between them' },
                  ]}
                />
              ) : null}
              <Segmented<Part>
                label="Employee participation"
                value={part}
                onChange={on(setPart)}
                options={[
                  { value: 'total', label: 'total', title: 'Every employee must be in the relationship' },
                  { value: 'partial', label: 'partial', title: 'An employee may be in no relationship' },
                ]}
              />
              <Check label="Dependents (weak entity)" checked={weak} onChange={on(setWeak)} />
              <Check label="Skills (multi-valued)" checked={mv} onChange={on(setMv)} />
              <Check label="Declare the implied keys" checked={enforce} onChange={on(setEnforce)} />
              <Button onClick={fire} primary title={nextV.sql}>
                Violate it: {nextV.label}
              </Button>
              <Button onClick={clear} disabled={applied.length === 0 && !last}>
                Reset rows
              </Button>
            </>
          ) : (
            <>
              <Choice
                label="Unfamiliar schema"
                value={puzzleId}
                onChange={(v) => {
                  setPuzzleId(v);
                  setGuessed(false);
                }}
                options={PUZZLES.map((p) => ({ value: p.id, label: p.label }))}
              />
              <Segmented<Puzzle['answer']>
                label="Your reading"
                value={reading}
                onChange={(v) => {
                  setReading(v);
                  setGuessed(false);
                }}
                options={READINGS}
              />
              <Button onClick={() => setGuessed(true)} primary>
                Check my reading
              </Button>
            </>
          )}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'PK — declared key', color: 'var(--viz-1)' },
            { label: 'FK — reference', color: 'var(--viz-2)' },
            { label: 'PK ∩ FK / UNIQUE', color: 'var(--viz-7)' },
            { label: 'row the drawing forbids', color: 'var(--viz-critical)' },
            { label: 'NULL — outer-join territory', color: 'var(--viz-ink-muted)' },
          ]}
        />
      }
      stats={
        mode === 'forward' ? (
          <Stats
            items={[
              { label: 'Tables', value: schema.tables.length },
              { label: 'Rows stored', value: rowsStored, hint: 'Across every table the mapping produced' },
              { label: 'Rows the join returns', value: schema.result.rows.length, hint: 'With the OUTER join shown below' },
              { label: 'Dropped by INNER JOIN', value: dropped, hint: 'Rows an inner join would silently lose' },
              { label: 'Indexes you must create', value: manualIdx, hint: 'Postgres indexes primary and unique keys, never foreign keys' },
              { label: 'Cardinality the DDL enforces', value: <span style={{ fontSize: '.72rem', fontWeight: 500 }}>{schema.enforced}</span> },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Tables', value: puzzle.tables.length },
              { label: 'FK edges', value: puzzle.tables.reduce((a, t) => a + t.cols.filter((c) => c.ref).length, 0) },
              { label: 'Nullable FKs', value: puzzle.tables.reduce((a, t) => a + t.cols.filter((c) => c.ref && !c.nn && !c.pk).length, 0), hint: 'Optional participation' },
              { label: 'Junction tables', value: puzzle.tables.filter((t) => t.cols.filter((c) => c.pk).length > 1 && t.cols.filter((c) => c.pk && c.ref).length === 2).length, hint: 'Composite PK made of exactly two FKs' },
              { label: 'Your reading', value: guessed ? (correct ? 'correct' : 'not quite') : '—' },
            ]}
          />
        )
      }
      note={
        mode === 'forward' ? (
          <Note>
            {last ? (
              <>
                <strong style={{ color: outcomeColor }}>
                  {last.outcome === 'rejected'
                    ? 'Rejected by a declared constraint.'
                    : last.outcome === 'impossible'
                      ? 'Nothing to reject — the shape of the row already forbids it.'
                      : 'Accepted. The drawing said no; the schema said nothing.'}
                </strong>{' '}
                <code style={{ fontSize: '.75rem' }}>{last.sql}</code> → <code style={{ fontSize: '.75rem' }}>{last.code}</code>. {last.msg} {last.why}
              </>
            ) : (
              <>
                <strong>{schema.rule}</strong>
              </>
            )}
          </Note>
        ) : (
          <Note>
            <strong>{guessed ? (correct ? 'Correct.' : 'Not quite.') : 'Read the DDL, pick a cardinality, then check.'}</strong>{' '}
            {guessed ? (
              <>
                Decisive evidence: <code style={{ fontSize: '.75rem' }}>{puzzle.evidence}</code>. {puzzle.verdict}
              </>
            ) : (
              <>
                The foreign-key graph is the diagram. Three questions answer almost every edge: is the FK nullable, is
                it unique, and is it part of the primary key?
              </>
            )}
          </Note>
        )
      }
      table={
        mode === 'forward' ? (
          <>
            <table className="viz-table">
              <caption style={{ textAlign: 'left' }}>Every row the mapping stores</caption>
              <thead>
                <tr>
                  <th>Table</th>
                  <th>Columns</th>
                  <th>Row</th>
                </tr>
              </thead>
              <tbody>
                {schema.tables.flatMap((t) =>
                  t.rows.map((r, i) => (
                    <tr key={`${t.id}-${i}`}>
                      <td>{i === 0 ? t.name : ''}</td>
                      <td>{i === 0 ? t.cols.map((c) => c.name).join(', ') : ''}</td>
                      <td>
                        {r.cells.map((c) => (c === null ? 'NULL' : c)).join(' | ')}
                        {r.bad ? ' ← forbidden by the drawing, accepted by the schema' : ''}
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
            <table className="viz-table">
              <caption style={{ textAlign: 'left' }}>Indexes this mapping needs</caption>
              <thead>
                <tr>
                  <th>Index</th>
                  <th>Created by</th>
                  <th>Why it exists</th>
                </tr>
              </thead>
              <tbody>
                {schema.tables.flatMap((t) =>
                  t.idx.map((ix) => (
                    <tr key={ix.name}>
                      <td>{ix.name}</td>
                      <td>{ix.auto ? 'the key declaration' : 'you, by hand'}</td>
                      <td>{ix.why}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
            <table className="viz-table">
              <caption style={{ textAlign: 'left' }}>Violations available at this setting</caption>
              <thead>
                <tr>
                  <th>Try to</th>
                  <th>Outcome</th>
                  <th>What answers</th>
                </tr>
              </thead>
              <tbody>
                {vlist.map((v) => (
                  <tr key={v.id}>
                    <td>{v.label}</td>
                    <td>{v.outcome}</td>
                    <td>{v.code}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <table className="viz-table">
            <caption style={{ textAlign: 'left' }}>
              The FK graph as <code>pg_constraint</code> would hand it to you
            </caption>
            <thead>
              <tr>
                <th>Child column</th>
                <th>References</th>
                <th>Nullable</th>
                <th>Unique / PK</th>
                <th>Reads as</th>
              </tr>
            </thead>
            <tbody>
              {puzzle.tables.flatMap((t) =>
                t.cols
                  .filter((c) => c.ref)
                  .map((c) => (
                    <tr key={`${t.id}.${c.name}`}>
                      <td>
                        {t.name}.{c.name}
                      </td>
                      <td>
                        {c.ref!.t}.{c.ref!.c}
                      </td>
                      <td>{c.nn || c.pk ? 'no' : 'yes'}</td>
                      <td>{c.pk ? (t.cols.filter((k) => k.pk).length > 1 ? 'part of composite PK' : 'PK') : c.unique ? 'UNIQUE' : 'no'}</td>
                      <td>
                        {t.cols.filter((k) => k.pk && k.ref).length === 2
                          ? 'junction half → M:N'
                          : c.pk || c.unique
                            ? 'at most one child → 1:1'
                            : c.nn
                              ? 'many children, mandatory parent → 1:N'
                              : 'many children, optional parent → 1:N'}
                      </td>
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        )
      }
    >
      <div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <TooltipHost>
            <svg width={svgW} height={svgH} role="img" aria-label="The tables this mapping produces and the foreign keys between them">
              {edges.map((e, i) => {
                const s = tables.find((t) => t.id === e.from)!;
                const d = tables.find((t) => t.id === e.to)!;
                const sp = pos[s.id];
                const dp = pos[d.id];
                const y1 = colY(s, e.fromCol, sp.y);
                const y2 = colY(d, e.toCol, dp.y);
                let x1: number;
                let x2: number;
                let mx: number;
                if (s.col === d.col) {
                  x1 = sp.x + BOXW;
                  x2 = dp.x + BOXW;
                  mx = sp.x + BOXW + 16;
                } else if (s.col < d.col) {
                  x1 = sp.x + BOXW;
                  x2 = dp.x;
                  mx = (x1 + x2) / 2;
                } else {
                  x1 = sp.x;
                  x2 = dp.x + BOXW;
                  mx = (x1 + x2) / 2;
                }
                const dir = x1 <= mx ? 1 : -1;
                const dir2 = x2 >= mx ? 1 : -1;
                return (
                  <g key={`${e.from}-${e.fromCol}-${i}`} {...tip(<>{e.tip}</>)} style={{ cursor: 'help' }}>
                    <path
                      d={`M ${x1} ${y1} H ${mx} V ${y2} H ${x2}`}
                      fill="none"
                      stroke="var(--viz-2)"
                      strokeWidth={1.5}
                      strokeDasharray={e.optional ? '5 3' : undefined}
                    />
                    {/* crow's foot at the child end when the child side is "many" */}
                    {e.many ? (
                      <>
                        <line x1={x1} y1={y1} x2={x1 + dir * 9} y2={y1 - 5} stroke="var(--viz-2)" strokeWidth={1.5} />
                        <line x1={x1} y1={y1} x2={x1 + dir * 9} y2={y1 + 5} stroke="var(--viz-2)" strokeWidth={1.5} />
                      </>
                    ) : (
                      <line x1={x1 + dir * 6} y1={y1 - 5} x2={x1 + dir * 6} y2={y1 + 5} stroke="var(--viz-2)" strokeWidth={1.5} />
                    )}
                    {/* optionality ring, or the mandatory bar, at the child end */}
                    {e.optional ? (
                      <circle cx={x1 + dir * 15} cy={y1} r={3.5} fill="var(--viz-surface)" stroke="var(--viz-2)" strokeWidth={1.5} />
                    ) : (
                      <line x1={x1 + dir * 15} y1={y1 - 5} x2={x1 + dir * 15} y2={y1 + 5} stroke="var(--viz-2)" strokeWidth={1.5} />
                    )}
                    {/* the parent end is always exactly one */}
                    <line x1={x2 + dir2 * 7} y1={y2 - 5} x2={x2 + dir2 * 7} y2={y2 + 5} stroke="var(--viz-2)" strokeWidth={1.5} />
                    <text x={mx + 5} y={(y1 + y2) / 2 - 3} fill="var(--viz-ink-2)">
                      {e.label}
                    </text>
                  </g>
                );
              })}

              {tables.map((t) => {
                const p = pos[t.id];
                return (
                  <g key={t.id}>
                    <rect x={p.x} y={p.y} width={BOXW} height={p.h} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                    <rect x={p.x} y={p.y} width={BOXW} height={HDR} rx={8} fill="var(--viz-neutral)" stroke="var(--viz-border)" />
                    <g {...tip(<>{t.tip}</>)} style={{ cursor: 'help' }}>
                      <rect x={p.x} y={p.y} width={BOXW} height={HDR} fill="transparent" />
                      <text x={p.x + 9} y={p.y + 15} fill="var(--viz-ink)" fontWeight={600}>
                        {t.name}
                      </text>
                      <text x={p.x + BOXW - 9} y={p.y + 15} textAnchor="end" fill="var(--viz-ink-muted)">
                        {t.rows.length > 0 ? `${t.rows.length} rows` : ''}
                      </text>
                    </g>
                    {t.cols.map((c, i) => {
                      const k = colKind(c);
                      const cy = p.y + HDR + i * ROWH + ROWH / 2;
                      return (
                        <g key={c.name} {...tip(<><strong>{t.name}.{c.name}</strong><br />{c.tip}</>)} style={{ cursor: 'help' }}>
                          <rect x={p.x} y={cy - ROWH / 2} width={BOXW} height={ROWH} fill="transparent" />
                          <rect x={p.x + 8} y={cy - 3.5} width={7} height={7} rx={k === 'plain' ? 3.5 : 1.5} fill={KIND_FILL[k]} />
                          <text x={p.x + 21} y={cy + 4} fill="var(--viz-ink)">
                            {c.name}
                          </text>
                          <text x={p.x + BOXW - 9} y={cy + 4} textAnchor="end" fill="var(--viz-ink-muted)">
                            {k === 'pkfk' ? 'PK,FK' : k === 'pk' ? 'PK' : k === 'fk' ? 'FK' : c.unique ? 'UNQ' : c.nn ? 'NN' : ''}
                          </text>
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </svg>
          </TooltipHost>

          <div style={{ flex: '1 1 24rem', minWidth: '20rem' }}>
            <p className="viz-sub" style={{ margin: '0 0 .25rem' }}>
              {mode === 'forward' ? 'The DDL this drawing generates' : 'A schema you have never seen'}
            </p>
            <code style={{ display: 'block', fontSize: '.72rem', lineHeight: 1.5, whiteSpace: 'pre', overflowX: 'auto' }}>
              {tables.map((t) => (
                <span key={t.id} style={{ display: 'block' }}>
                  {ddlFor(t).map((l, i) => (
                    <span key={i} style={{ display: 'block', color: l.tone ? TONE[l.tone] : 'var(--viz-ink)' }}>
                      {l.t}
                    </span>
                  ))}
                  <span style={{ display: 'block' }}>{' '}</span>
                </span>
              ))}
            </code>
          </div>
        </div>

        {mode === 'forward' ? (
          <div style={{ marginTop: '.75rem' }}>
            <p className="viz-sub" style={{ margin: '0 0 .25rem' }}>
              The query that answers “who is on what”, and what it returns now
            </p>
            <code style={{ display: 'block', fontSize: '.72rem', lineHeight: 1.5, whiteSpace: 'pre', overflowX: 'auto', color: 'var(--viz-ink)' }}>
              {schema.result.sql.join('\n')}
            </code>
            <table className="viz-table">
              <thead>
                <tr>
                  {schema.result.head.map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                  <th>INNER JOIN</th>
                </tr>
              </thead>
              <tbody>
                {schema.result.rows.map((r, i) => (
                  <tr key={i}>
                    {r.cells.map((c, k) => (
                      <td key={k} style={c === null ? { color: 'var(--viz-ink-muted)' } : undefined}>
                        {c === null ? 'NULL' : c}
                      </td>
                    ))}
                    <td style={{ color: r.dropped ? 'var(--viz-critical)' : 'var(--viz-ink-muted)' }}>
                      {r.dropped ? 'drops this row' : 'keeps it'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </VizPanel>
  );
}
