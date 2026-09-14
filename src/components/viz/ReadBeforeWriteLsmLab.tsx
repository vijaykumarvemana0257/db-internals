import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Choice, Check, Button, Legend, Stats, Note, fmtNum, fmtTime } from './Viz';

type Scenario = 'row' | 'merge' | 'single';

/* ------------------------------------------------------------ row cost model */
// Stated assumptions, not measurements: the point is the ratio between blind and read-first writes.
const WRITE_NS = 10_000; // WAL append + memtable insert
const MEM_PROBE_NS = 1_000; // one memtable (skip list) search
const FILTER_NS = 1_000; // one filter check, filter block cached
const BLOCK_READ_NS = 100_000; // one data-block read that misses the block cache

type Op = 'insert' | 'update' | 'delete';

export function rowCost(op: Op, uniqueCheck: boolean, secondary: number, levels: number, l0: number, filters: boolean, immutables: number) {
  const fpr = 0.01; // ~10 bits per key
  // A point lookup probes: active + immutable memtables, every L0 file, then one file per deeper level.
  const memProbes = 1 + immutables;
  const files = l0 + levels;
  const absent = { mem: memProbes, filter: filters ? files : 0, blocks: filters ? files * fpr : files };
  // Present in the deepest level (the common case for an old row): every level above pays a filter check.
  const present = { mem: memProbes, filter: filters ? files : 0, blocks: filters ? 1 + (files - 1) * fpr : files };

  const reads: { why: string; kind: 'absent' | 'present' }[] = [];
  if (op === 'insert' && uniqueCheck) reads.push({ why: 'primary key must not exist', kind: 'absent' });
  if (op === 'update' || op === 'delete') reads.push({ why: 'read old row to find its index entries', kind: 'present' });
  // Secondary indexes are modeled as non-unique, so an INSERT writes their entries blind.
  const writes = 1 + secondary * (op === 'update' ? 2 : 1); // update deletes the old index entry and writes the new one

  let mem = 0;
  let filter = 0;
  let blocks = 0;
  for (const r of reads) {
    const c = r.kind === 'absent' ? absent : present;
    mem += c.mem;
    filter += c.filter;
    blocks += c.blocks;
  }
  const ns = writes * WRITE_NS + mem * MEM_PROBE_NS + filter * FILTER_NS + blocks * BLOCK_READ_NS;
  const blindNs = writes * WRITE_NS; // the same writes, if nothing had to be read first
  return { reads, writes, mem, filter, blocks, ns, blindNs };
}

/* ------------------------------------------------------------ merge operands */
type MergeState = { base: number; operands: number[]; rmwReads: number; mergeReads: number; lastRead: string };
const MERGE0: MergeState = { base: 0, operands: [], rmwReads: 0, mergeReads: 0, lastRead: '' };

/* ------------------------------------------------------------ single delete */
type Rec = { kind: 'put' | 'del' | 'sdel'; val?: string; seq: number };
const SD_STEPS: { title: string; body: string; mem: Rec[]; sst: Rec[][]; visible: string }[] = [
  { title: 'Put(k, v1)', body: 'The first write lands in the memtable.', mem: [{ kind: 'put', val: 'v1', seq: 1 }], sst: [], visible: 'v1' },
  { title: 'Flush', body: 'The memtable is flushed; v1 now lives in an older SSTable.', mem: [], sst: [[{ kind: 'put', val: 'v1', seq: 1 }]], visible: 'v1' },
  { title: 'Put(k, v2)', body: 'k is written a second time. This breaks the SingleDelete contract, which requires the key to have been Put exactly once since it was last deleted.', mem: [{ kind: 'put', val: 'v2', seq: 2 }], sst: [[{ kind: 'put', val: 'v1', seq: 1 }]], visible: 'v2' },
  { title: 'SingleDelete(k)', body: 'Reads now see k as deleted: the newest record for k is the SingleDelete.', mem: [{ kind: 'sdel', seq: 3 }, { kind: 'put', val: 'v2', seq: 2 }], sst: [[{ kind: 'put', val: 'v1', seq: 1 }]], visible: '(not found)' },
  { title: 'Compaction pairs them', body: 'A SingleDelete meeting a Put for the same key cancels both, and neither is written out. That is what makes SingleDelete cheap: no tombstone has to travel down to the bottom level. But it only cancels one Put — the one it meets.', mem: [], sst: [[{ kind: 'put', val: 'v1', seq: 1 }]], visible: 'v1' },
];

export default function ReadBeforeWriteLsmLab() {
  const [scenario, setScenario] = useState<Scenario>('row');

  // row
  const [op, setOp] = useState<Op>('update');
  const [unique, setUnique] = useState(true);
  const [secondary, setSecondary] = useState(3);
  const [levels, setLevels] = useState(5);
  const [l0, setL0] = useState(4);
  const [filters, setFilters] = useState(true);
  const immutables = 1;
  const cost = useMemo(() => rowCost(op, unique, secondary, levels, l0, filters, immutables), [op, unique, secondary, levels, l0, filters]);

  // merge
  const [m, setM] = useState<MergeState>(MERGE0);
  const incr = (times: number) =>
    setM((s) => ({ ...s, operands: [...s.operands, ...Array(times).fill(1)], rmwReads: s.rmwReads + times }));
  const readCounter = () =>
    setM((s) => ({ ...s, mergeReads: s.mergeReads + 1, lastRead: `Get folded the base value ${s.base} with ${s.operands.length} operand${s.operands.length === 1 ? '' : 's'} → ${s.base + s.operands.length}` }));
  const compact = () =>
    setM((s) => ({ ...s, base: s.base + s.operands.length, operands: [], lastRead: `Compaction collapsed ${s.operands.length} operands into the base value ${s.base + s.operands.length}` }));

  // single delete
  const [sd, setSd] = useState(0);
  const sdStep = SD_STEPS[sd];

  const legend =
    scenario === 'row' ? (
      <Legend
        items={[
          { label: 'Writes (clustered row + index entries)', color: 'var(--viz-1)' },
          { label: 'In-memory probes (memtables, filters)', color: 'var(--viz-3)' },
          { label: 'Data-block reads', color: 'var(--viz-2)' },
        ]}
      />
    ) : scenario === 'merge' ? (
      <Legend
        items={[
          { label: 'Base value', color: 'var(--viz-clean)' },
          { label: 'Merge operand (not yet folded)', color: 'var(--viz-dirty)' },
        ]}
      />
    ) : (
      <Legend
        items={[
          { label: 'Put', color: 'var(--viz-clean)' },
          { label: 'SingleDelete', color: 'var(--viz-critical)' },
          { label: 'What a read returns', color: 'var(--viz-ink)' },
        ]}
      />
    );

  const W = 680;
  return (
    <VizPanel
      title="When an LSM write has to read first"
      subtitle="A Put never reads. A unique check, an update of an indexed row, or an increment does — and a read in an LSM is the expensive half of the engine."
      controls={
        <Segmented
          label="Scenario"
          value={scenario}
          onChange={setScenario}
          options={[
            { value: 'row', label: 'One SQL row' },
            { value: 'merge', label: 'A counter' },
            { value: 'single', label: 'SingleDelete' },
          ]}
        />
      }
      legend={legend}
      stats={
        scenario === 'row' ? (
          <Stats
            items={[
              { label: 'Writes per row', value: fmtNum(cost.writes) },
              { label: 'Point reads per row', value: fmtNum(cost.reads.length) },
              { label: 'Data-block reads', value: fmtNum(cost.blocks, 2), hint: filters ? 'Filters skip ~99% of files that do not hold the key' : 'No filters: every probed file is read' },
              { label: 'Est. cost per row', value: fmtTime(cost.ns), hint: `Blind write of the same row: ${fmtTime(cost.blindNs)}` },
              { label: 'vs blind write', value: `${fmtNum(cost.ns / cost.blindNs, 1)}×` },
            ]}
          />
        ) : scenario === 'merge' ? (
          <Stats
            items={[
              { label: 'Counter value', value: fmtNum(m.base + m.operands.length) },
              { label: 'Reads paid, read-modify-write', value: fmtNum(m.rmwReads), hint: 'One Get before every Put' },
              { label: 'Reads paid, Merge', value: fmtNum(m.mergeReads), hint: 'Only when the application actually reads' },
              { label: 'Operands waiting to fold', value: fmtNum(m.operands.length) },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Step', value: `${sd + 1} / ${SD_STEPS.length}` },
              { label: 'Get(k) returns', value: sdStep.visible },
            ]}
          />
        )
      }
      note={
        <Note>
          {scenario === 'row' ? (
            <>
              <strong>
                {op.toUpperCase()} with {secondary} secondary index{secondary === 1 ? '' : 'es'}: {cost.writes} writes and {cost.reads.length} point read{cost.reads.length === 1 ? '' : 's'}.
              </strong>{' '}
              {cost.reads.length ? `The read${cost.reads.length === 1 ? '' : 's'} (${cost.reads.map((r) => r.why).join('; ')}) probe ${1 + immutables} memtables and ${l0 + levels} files; ` : 'Nothing is read, so the write stays blind; '}
              the row costs about {fmtNum(cost.ns / cost.blindNs, 1)}× a blind write of the same entries.{' '}
              {!filters && cost.reads.length ? 'Turn filters back on: without them every probed file is a block read.' : ''}
            </>
          ) : scenario === 'merge' ? (
            <>
              <strong>{m.lastRead || 'Increment the counter.'}</strong> With read-modify-write, every increment pays a Get before its Put. With a Merge operator, an increment appends a tiny operand and reads nothing; the cost moves to whoever reads the counter, until compaction folds the operands into one value.
            </>
          ) : (
            <>
              <strong>{sdStep.title}.</strong> {sdStep.body}
              {sd === SD_STEPS.length - 1 ? ' The SingleDelete cancelled v2 and the older v1 — which a regular Delete would have kept hidden until its tombstone reached the bottom level — is visible again.' : ''}
            </>
          )}
        </Note>
      }
      table={
        scenario === 'row' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Operation</th>
                <th>Writes</th>
                <th>Reads</th>
                <th>Block reads</th>
                <th>Est. cost</th>
              </tr>
            </thead>
            <tbody>
              {(['insert', 'update', 'delete'] as Op[]).map((o) => {
                const c = rowCost(o, unique, secondary, levels, l0, filters, immutables);
                return (
                  <tr key={o}>
                    <td>{o}</td>
                    <td>{c.writes}</td>
                    <td>{c.reads.length}</td>
                    <td>{fmtNum(c.blocks, 2)}</td>
                    <td>{fmtTime(c.ns)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : undefined
      }
    >
      {scenario === 'row' ? (
        <>
          <div className="viz-controls">
            <Choice label="Statement" value={op} onChange={setOp} options={[{ value: 'insert', label: 'INSERT' }, { value: 'update', label: 'UPDATE changing every indexed column' }, { value: 'delete', label: 'DELETE' }]} />
            <Check label="Enforce primary-key uniqueness on INSERT" checked={unique} onChange={setUnique} />
            <Slider label="Secondary indexes" min={0} max={8} value={secondary} onChange={setSecondary} />
            <Slider label="L0 files" min={0} max={20} value={l0} onChange={setL0} />
            <Slider label="Levels below L0" min={1} max={7} value={levels} onChange={setLevels} />
            <Check label="Bloom filters (~1% false positives)" checked={filters} onChange={setFilters} />
          </div>
          <svg width={W} height={70 + (l0 + levels + 2) * 18} role="img" aria-label="Probes a point read makes">
            <text x={0} y={16} fill="var(--viz-ink)">Per row</text>
            {Array.from({ length: cost.writes }, (_, i) => (
              <rect key={i} x={110 + i * 22} y={4} width={18} height={18} rx={3} fill="var(--viz-1)" />
            ))}
            <text x={110 + cost.writes * 22 + 6} y={17} fontSize={11}>
              {cost.writes} write{cost.writes === 1 ? '' : 's'}
            </text>
            <text x={0} y={52} fill="var(--viz-ink)">
              {cost.reads.length ? 'Each read probes:' : 'No reads — a blind write'}
            </text>
            {cost.reads.length
              ? [
                  ...Array.from({ length: 1 + immutables }, (_, i) => ({ label: i === 0 ? 'active memtable' : `immutable memtable ${i}`, kind: 'mem' as const })),
                  ...Array.from({ length: l0 }, (_, i) => ({ label: `L0 file ${i + 1}`, kind: 'file' as const })),
                  ...Array.from({ length: levels }, (_, i) => ({ label: `L${i + 1}`, kind: 'file' as const })),
                ].map((row, i) => {
                  const y = 66 + i * 18;
                  const deepest = i === 1 + immutables + l0 + levels - 1;
                  const blockRead = row.kind === 'file' && (!filters || (op !== 'insert' && deepest));
                  return (
                    <g key={row.label}>
                      <text x={20} y={y + 11} fontSize={10}>
                        {row.label}
                      </text>
                      <rect x={150} y={y} width={row.kind === 'mem' ? 60 : 30} height={13} rx={3} fill="var(--viz-3)" />
                      <text x={150 + (row.kind === 'mem' ? 66 : 36)} y={y + 10} fontSize={10}>
                        {row.kind === 'mem' ? 'skip-list search' : filters ? 'filter check' : 'no filter'}
                      </text>
                      {blockRead ? <rect x={300} y={y} width={120} height={13} rx={3} fill="var(--viz-2)" /> : null}
                      {blockRead ? (
                        <text x={426} y={y + 10} fontSize={10}>
                          {filters ? 'block read — the key is here' : 'block read'}
                        </text>
                      ) : null}
                    </g>
                  );
                })
              : null}
          </svg>
        </>
      ) : scenario === 'merge' ? (
        <>
          <div className="viz-controls">
            <Button primary onClick={() => incr(1)}>
              Increment
            </Button>
            <Button onClick={() => incr(10)}>Increment ×10</Button>
            <Button onClick={readCounter}>Read the counter</Button>
            <Button onClick={compact} disabled={!m.operands.length}>
              Compact
            </Button>
            <Button onClick={() => setM(MERGE0)}>Reset</Button>
          </div>
          <svg width={W} height={96} role="img" aria-label={`Counter: base ${m.base}, ${m.operands.length} operands`}>
            <text x={0} y={30} fill="var(--viz-ink)">On disk</text>
            <rect x={80} y={14} width={70} height={26} rx={5} fill="var(--viz-surface)" stroke="var(--viz-clean)" strokeWidth={3} />
            <text x={115} y={32} textAnchor="middle" fill="var(--viz-ink)" fontWeight={700}>
              {m.base}
            </text>
            {m.operands.slice(0, 40).map((_, i) => (
              <rect key={i} x={160 + i * 12} y={18} width={10} height={18} rx={2} fill="var(--viz-dirty)" />
            ))}
            {m.operands.length > 40 ? (
              <text x={160 + 40 * 12 + 4} y={32} fontSize={11}>
                +{m.operands.length - 40}
              </text>
            ) : null}
            <text x={80} y={62} fontSize={11}>
              base value, then {m.operands.length} “+1” operand{m.operands.length === 1 ? '' : 's'} appended with no read
            </text>
            <text x={80} y={82} fontSize={11}>
              read-modify-write would have paid {m.rmwReads} Get{m.rmwReads === 1 ? '' : 's'} for the same {m.rmwReads} increment{m.rmwReads === 1 ? '' : 's'}
            </text>
          </svg>
        </>
      ) : (
        <>
          <div className="viz-controls">
            <Button onClick={() => setSd((n) => Math.max(0, n - 1))} disabled={sd === 0}>
              Back
            </Button>
            <Button primary onClick={() => setSd((n) => Math.min(SD_STEPS.length - 1, n + 1))} disabled={sd === SD_STEPS.length - 1}>
              Next step
            </Button>
            <Button onClick={() => setSd(0)}>Reset</Button>
          </div>
          <svg width={W} height={130} role="img" aria-label={`SingleDelete scenario: ${sdStep.title}`}>
            {[
              { label: 'Memtable', recs: sdStep.mem, y: 10 },
              { label: 'Older SSTable', recs: sdStep.sst[0] ?? [], y: 56 },
            ].map((lane) => (
              <g key={lane.label}>
                <text x={0} y={lane.y + 20} fill="var(--viz-ink)">
                  {lane.label}
                </text>
                <rect x={120} y={lane.y} width={400} height={32} rx={6} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                {lane.recs.map((r, i) => (
                  <g key={r.seq}>
                    <rect x={130 + i * 130} y={lane.y + 5} width={120} height={22} rx={4} fill="var(--viz-surface)" stroke={r.kind === 'sdel' ? 'var(--viz-critical)' : 'var(--viz-clean)'} strokeWidth={3} />
                    <text x={190 + i * 130} y={lane.y + 20} textAnchor="middle" fill="var(--viz-ink)" fontSize={11} fontWeight={600}>
                      {r.kind === 'sdel' ? `SingleDelete k @${r.seq}` : `Put k=${r.val} @${r.seq}`}
                    </text>
                  </g>
                ))}
              </g>
            ))}
            <text x={120} y={118} fill="var(--viz-ink)" fontSize={12}>
              Get(k) → <tspan fontWeight={700}>{sdStep.visible}</tspan>
            </text>
          </svg>
        </>
      )}
    </VizPanel>
  );
}
