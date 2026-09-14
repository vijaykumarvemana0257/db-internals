import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Button, Legend, Stats, Note, fmtNum, makeRng } from './Viz';

type Panel = 'space' | 'cuckoo' | 'ops';

function hash32(s: string, seed = 0x811c9dc5) {
  let h = seed >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h ^= h >>> 12;
  return h >>> 0;
}

/* ------------------------------------------------------------------ space */
const log2 = (x: number) => Math.log(x) / Math.LN2;

/** Bits per key to reach false-positive rate eps. Formulas are the standard analytic results. */
export function designs(eps: number): { name: string; bits: number; hi?: number; note: string }[] {
  const bound = log2(1 / eps);
  return [
    { name: 'Information-theoretic floor', bits: bound, note: 'log₂(1/ε): no filter can do better' },
    { name: 'Bloom (optimal k)', bits: 1.44 * bound, note: 'about 44% above the floor' },
    { name: 'Cuckoo (4-slot buckets, 95% load)', bits: (bound + 3) / 0.95, note: 'fingerprint + 3 bits, at the load factor 4-slot buckets reach' },
    { name: 'Quotient (75% load)', bits: (bound + 3) / 0.75, note: 'remainder + 3 metadata bits per slot, at 75% load' },
    { name: 'Xor', bits: 1.23 * bound, note: 'about 23% above the floor' },
    { name: 'Binary fuse (4-wise)', bits: 1.075 * bound, note: 'about 7.5% above the floor; the 3-wise variant is about 12.5%' },
    { name: 'Ribbon (configurable)', bits: 1.01 * bound, hi: 1.1 * bound, note: 'roughly 1–10% above the floor, trading ribbon width against build time' },
  ];
}

/** Measured false-positive rate: standard Bloom vs cache-line-blocked Bloom, same bits per key. */
export function measureBlocked(bitsPerKey: number, n = 4000, trials = 40000) {
  const k = Math.max(1, Math.round(bitsPerKey * Math.LN2));
  const m = n * bitsPerKey;
  const std = new Uint8Array(m);
  const BLOCK = 512; // one 64-byte cache line
  const blocks = Math.max(1, Math.floor(m / BLOCK));
  const blk = new Uint8Array(blocks * BLOCK);
  const probe = (key: string, set: boolean) => {
    const h1 = hash32(key, 1);
    const h2 = hash32(key, 2) | 1;
    let s = true;
    let b = true;
    const base = (hash32(key, 3) % blocks) * BLOCK;
    for (let i = 0; i < k; i++) {
      const p = (h1 + Math.imul(i, h2)) >>> 0;
      const sp = p % m;
      const bp = base + (p % BLOCK);
      if (set) {
        std[sp] = 1;
        blk[bp] = 1;
      } else {
        s = s && std[sp] === 1;
        b = b && blk[bp] === 1;
      }
    }
    return { s, b };
  };
  for (let i = 0; i < n; i++) probe(`k${i}`, true);
  let fs = 0;
  let fb = 0;
  for (let i = 0; i < trials; i++) {
    const r = probe(`absent${i}`, false);
    if (r.s) fs++;
    if (r.b) fb++;
  }
  return { k, standard: fs / trials, blocked: fb / trials, predicted: Math.pow(1 - Math.exp(-k / bitsPerKey), k) };
}

/* ----------------------------------------------------------------- cuckoo */
const BUCKETS = 16; // power of two, so i2 = i1 XOR h(fp) stays in range
const SLOTS = 4;
const FP_BITS = 8;
const MAX_KICKS = 500;

type Cuckoo = { table: number[][]; inserted: string[]; lastChain: number[]; lastKicks: number; failed: string | null; log: string };
const emptyCuckoo = (): Cuckoo => ({ table: Array.from({ length: BUCKETS }, () => []), inserted: [], lastChain: [], lastKicks: 0, failed: null, log: '' });

const fingerprint = (key: string) => (hash32(key, 7) & ((1 << FP_BITS) - 1)) || 1;
const i1Of = (key: string) => hash32(key, 11) & (BUCKETS - 1);
const altIndex = (i: number, fp: number) => (i ^ (hash32(String(fp), 13) & (BUCKETS - 1))) & (BUCKETS - 1);

export function cuckooInsert(c: Cuckoo, key: string, seed: number): Cuckoo {
  const table = c.table.map((b) => [...b]);
  let fp = fingerprint(key);
  const i1 = i1Of(key);
  const i2 = altIndex(i1, fp);
  for (const i of [i1, i2]) {
    if (table[i].length < SLOTS) {
      table[i].push(fp);
      return { ...c, table, inserted: [...c.inserted, key], lastChain: [i], lastKicks: 0, failed: null, log: `${key}: fingerprint ${fp} went straight into bucket ${i} (candidates ${i1} and ${i2}).` };
    }
  }
  // Both buckets full: evict a random resident and move it to its alternate bucket, repeatedly.
  const rng = makeRng(seed);
  let i = rng() < 0.5 ? i1 : i2;
  const chain = [i];
  for (let kick = 1; kick <= MAX_KICKS; kick++) {
    const slot = Math.floor(rng() * SLOTS);
    const victim = table[i][slot];
    table[i][slot] = fp;
    fp = victim;
    i = altIndex(i, fp); // partial-key cuckoo hashing: the alternate bucket comes from the fingerprint alone
    chain.push(i);
    if (table[i].length < SLOTS) {
      table[i].push(fp);
      return { ...c, table, inserted: [...c.inserted, key], lastChain: chain, lastKicks: kick, failed: null, log: `${key}: both candidate buckets were full, so ${kick} resident fingerprint${kick === 1 ? ' was' : 's were'} kicked along a chain of buckets ${chain.join(' → ')} until one found a free slot.` };
    }
  }
  return { ...c, lastChain: chain.slice(0, 12), lastKicks: MAX_KICKS, failed: key, log: `${key}: gave up after ${MAX_KICKS} kicks. The filter is effectively full — this is where a cuckoo filter must be rebuilt larger.` };
}

export function cuckooDelete(c: Cuckoo, key: string): Cuckoo {
  const fp = fingerprint(key);
  const i1 = i1Of(key);
  const i2 = altIndex(i1, fp);
  const table = c.table.map((b) => [...b]);
  for (const i of [i1, i2]) {
    const at = table[i].indexOf(fp);
    if (at >= 0) {
      table[i].splice(at, 1);
      return { ...c, table, inserted: c.inserted.filter((k) => k !== key), lastChain: [i], lastKicks: 0, failed: null, log: `Deleted ${key}: removed one copy of fingerprint ${fp} from bucket ${i}. Safe only because ${key} really was inserted — deleting a key that was never added could remove another key’s matching fingerprint.` };
    }
  }
  return { ...c, log: `${key} is not in the filter.` };
}

const cuckooLookup = (c: Cuckoo, key: string) => {
  const fp = fingerprint(key);
  const i1 = i1Of(key);
  return c.table[i1].includes(fp) || c.table[altIndex(i1, fp)].includes(fp);
};

/* --------------------------------------------------------------------- ops */
type Verdict = 'yes' | 'no' | 'limited';
const OPS: { design: string; insertAfterBuild: [Verdict, string]; del: [Verdict, string]; merge: [Verdict, string]; streamBuild: [Verdict, string]; cacheLines: string }[] = [
  { design: 'Bloom', insertAfterBuild: ['yes', 'set k more bits'], del: ['no', 'clearing a bit could break other keys'], merge: ['yes', 'bitwise OR, if size and hashes match'], streamBuild: ['yes', 'set bits as each key arrives'], cacheLines: 'up to k' },
  { design: 'Blocked Bloom', insertAfterBuild: ['yes', 'set bits within one block'], del: ['no', 'same as Bloom'], merge: ['yes', 'bitwise OR, if size and hashes match'], streamBuild: ['yes', 'set bits as each key arrives'], cacheLines: '1' },
  { design: 'Cuckoo', insertAfterBuild: ['limited', 'can fail once load is high'], del: ['limited', 'only keys known to be present'], merge: ['no', 'no general merge of two tables'], streamBuild: ['yes', 'insert as keys arrive, if sized up front'], cacheLines: '2 buckets' },
  { design: 'Quotient', insertAfterBuild: ['yes', 'degrades as load rises'], del: ['yes', 'remove the remainder, fix metadata'], merge: ['yes', 'merge like sorted runs; can also resize'], streamBuild: ['yes', 'insert as keys arrive'], cacheLines: 'about 1 (contiguous clusters)' },
  { design: 'Xor / binary fuse', insertAfterBuild: ['no', 'static: the key set is fixed at build'], del: ['no', 'static'], merge: ['no', 'rebuild from all keys'], streamBuild: ['no', 'needs every key before solving'], cacheLines: '3 (xor) / 3–4 nearby (fuse)' },
  { design: 'Ribbon', insertAfterBuild: ['no', 'static: a solved linear system'], del: ['no', 'static'], merge: ['no', 'rebuild from all keys'], streamBuild: ['no', 'needs every key hash before solving'], cacheLines: 'about 1 (one band)' },
];

export default function FilterSpaceCuckooLab() {
  const [panel, setPanel] = useState<Panel>('space');

  const [fpPct, setFpPct] = useState(1);
  const eps = fpPct / 100;
  const rows = useMemo(() => designs(eps), [eps]);
  const [bpk, setBpk] = useState(10);
  const measured = useMemo(() => measureBlocked(bpk), [bpk]);

  const [cf, setCf] = useState<Cuckoo>(emptyCuckoo);
  const [nextId, setNextId] = useState(0);
  const load = cf.table.reduce((a, b) => a + b.length, 0) / (BUCKETS * SLOTS);
  const cuckooFp = useMemo(() => {
    let hits = 0;
    const trials = 4000;
    for (let i = 0; i < trials; i++) if (cuckooLookup(cf, `absent-${i}`)) hits++;
    return hits / trials;
  }, [cf]);

  const [opDesign, setOpDesign] = useState(OPS[0].design);
  const [opResult, setOpResult] = useState<string>('');

  const maxBits = Math.max(...rows.map((r) => r.hi ?? r.bits));
  const W = 660;

  return (
    <VizPanel
      title="Filters beyond Bloom"
      subtitle="Every approximate-membership filter pays at least log₂(1/ε) bits per key. The designs differ in how far above that floor they sit, how many cache lines a lookup touches, and whether you can insert, delete or merge after building — which decides where each one fits in an LSM."
      controls={
        <Segmented
          label="Panel"
          value={panel}
          onChange={setPanel}
          options={[
            { value: 'space', label: 'Space vs the floor' },
            { value: 'cuckoo', label: 'A cuckoo filter' },
            { value: 'ops', label: 'What each can do' },
          ]}
        />
      }
      legend={
        panel === 'space' ? (
          <Legend
            items={[
              { label: 'Bits per key needed', color: 'var(--viz-1)' },
              { label: 'Information-theoretic floor', color: 'var(--viz-ink)', shape: 'line' },
            ]}
          />
        ) : panel === 'cuckoo' ? (
          <Legend
            items={[
              { label: 'Stored fingerprint', color: 'var(--viz-1)' },
              { label: 'Bucket on the last eviction chain', color: 'var(--viz-2)' },
              { label: 'Empty slot', color: 'var(--viz-plane)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: '✓ supported', color: 'var(--viz-good)' },
              { label: '~ limited', color: 'var(--viz-warning)' },
              { label: '✕ refused', color: 'var(--viz-critical)' },
            ]}
          />
        )
      }
      stats={
        panel === 'space' ? (
          <Stats
            items={[
              { label: 'Target false-positive rate', value: `${fpPct}%` },
              { label: 'Floor', value: `${fmtNum(rows[0].bits, 2)} bits/key` },
              { label: `Measured at ${bpk} bits/key: Bloom`, value: `${fmtNum(measured.standard * 100, 2)}%` },
              { label: 'Blocked Bloom (one cache line)', value: `${fmtNum(measured.blocked * 100, 2)}%` },
            ]}
          />
        ) : panel === 'cuckoo' ? (
          <Stats
            items={[
              { label: 'Keys inserted', value: fmtNum(cf.inserted.length) },
              { label: 'Load factor', value: `${fmtNum(load * 100, 1)}%` },
              { label: 'Kicks on last insert', value: fmtNum(cf.lastKicks) },
              { label: 'Measured false positives', value: `${fmtNum(cuckooFp * 100, 2)}%`, hint: `Roughly 2 × ${SLOTS} × load / 2^${FP_BITS} for ${FP_BITS}-bit fingerprints` },
            ]}
          />
        ) : undefined
      }
      note={
        <Note>
          {panel === 'space' ? (
            <>
              <strong>
                At {fpPct}% false positives the floor is {fmtNum(rows[0].bits, 2)} bits per key; a Bloom filter needs {fmtNum(rows[1].bits, 2)}.
              </strong>{' '}
              At {bpk} bits per key, keeping all of a key’s probes inside one 512-bit cache line measured {fmtNum(measured.blocked * 100, 2)}% false positives against {fmtNum(measured.standard * 100, 2)}% for standard Bloom — a higher rate at the same memory, bought back with one cache miss per lookup instead of up to {measured.k}.
            </>
          ) : panel === 'cuckoo' ? (
            <>
              <strong>{cf.log || 'Insert keys and watch the table fill.'}</strong>
              {cf.failed ? ' Delete a few keys and insertion works again — something a Bloom filter cannot offer.' : ''}
            </>
          ) : (
            <>
              <strong>{opResult || 'Pick a design, then try an operation on it.'}</strong> In an LSM tree every SSTable is immutable, so its filter is built once during a flush or compaction and never changed: the static designs that cannot insert, delete or merge give up nothing that an LSM needs.
            </>
          )}
        </Note>
      }
      table={
        panel === 'space' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Design</th>
                <th>Bits per key at {fpPct}%</th>
                <th>× floor</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>{r.hi ? `${fmtNum(r.bits, 2)}–${fmtNum(r.hi, 2)}` : fmtNum(r.bits, 2)}</td>
                  <td>{r.hi ? `${fmtNum(r.bits / rows[0].bits, 2)}–${fmtNum(r.hi / rows[0].bits, 2)}` : fmtNum(r.bits / rows[0].bits, 2)}</td>
                  <td>{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Design</th>
                <th>Insert after build</th>
                <th>Delete</th>
                <th>Merge</th>
                <th>Build as keys stream in</th>
                <th>Cache lines per lookup</th>
              </tr>
            </thead>
            <tbody>
              {OPS.map((o) => (
                <tr key={o.design}>
                  <td>{o.design}</td>
                  <td>{o.insertAfterBuild[0]} — {o.insertAfterBuild[1]}</td>
                  <td>{o.del[0]} — {o.del[1]}</td>
                  <td>{o.merge[0]} — {o.merge[1]}</td>
                  <td>{o.streamBuild[0]} — {o.streamBuild[1]}</td>
                  <td>{o.cacheLines}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    >
      {panel === 'space' ? (
        <>
          <div className="viz-controls">
            <Slider label="Target false-positive rate" min={0.1} max={10} step={0.1} value={fpPct} onChange={setFpPct} format={(v) => `${v}%`} />
            <Slider label="Measure blocked Bloom at" min={4} max={20} value={bpk} onChange={setBpk} format={(v) => `${v} bits/key`} />
          </div>
          <svg width={W} height={rows.length * 30 + 20} role="img" aria-label="Bits per key by filter design">
            {rows.map((r, i) => {
              const y = 8 + i * 30;
              const w = (r.bits / maxBits) * (W - 330);
              const floorX = 250 + (rows[0].bits / maxBits) * (W - 330);
              return (
                <g key={r.name}>
                  <text x={0} y={y + 14} fontSize={11} fill="var(--viz-ink)">
                    {r.name}
                  </text>
                  <rect x={250} y={y + 2} width={w} height={18} rx={4} fill={i === 0 ? 'var(--viz-ink-muted)' : 'var(--viz-1)'} />
                  {r.hi ? <rect x={250 + w} y={y + 2} width={((r.hi - r.bits) / maxBits) * (W - 330)} height={18} rx={4} fill="var(--viz-1)" opacity={0.35} /> : null}
                  <text x={250 + (r.hi ? ((r.hi / maxBits) * (W - 330)) : w) + 6} y={y + 15} fontSize={11}>
                    {r.hi ? `${fmtNum(r.bits, 1)}–${fmtNum(r.hi, 1)}` : fmtNum(r.bits, 1)}
                  </text>
                  {i > 0 ? <line x1={floorX} x2={floorX} y1={y} y2={y + 22} stroke="var(--viz-ink)" strokeWidth={2} /> : null}
                </g>
              );
            })}
          </svg>
        </>
      ) : panel === 'cuckoo' ? (
        <>
          <div className="viz-controls">
            <Button
              primary
              onClick={() => {
                setCf((c) => cuckooInsert(c, `key-${nextId}`, nextId + 1));
                setNextId((n) => n + 1);
              }}
            >
              Insert next key
            </Button>
            <Button
              onClick={() => {
                let c = cf;
                let id = nextId;
                for (let t = 0; t < 10; t++) {
                  c = cuckooInsert(c, `key-${id}`, id + 1);
                  id++;
                  if (c.failed) break;
                }
                setCf(c);
                setNextId(id);
              }}
            >
              Insert 10
            </Button>
            <Button onClick={() => cf.inserted.length && setCf((c) => cuckooDelete(c, c.inserted[0]))} disabled={!cf.inserted.length}>
              Delete oldest key
            </Button>
            <Button
              onClick={() => {
                setCf(emptyCuckoo());
                setNextId(0);
              }}
            >
              Reset
            </Button>
          </div>
          <svg width={BUCKETS * 40 + 10} height={SLOTS * 26 + 40} role="img" aria-label={`Cuckoo filter at ${fmtNum(load * 100, 0)}% load`}>
            {cf.table.map((bucket, i) => {
              const onChain = cf.lastChain.includes(i);
              return (
                <g key={i}>
                  <rect x={4 + i * 40} y={4} width={36} height={SLOTS * 26 + 4} rx={5} fill="none" stroke={onChain ? 'var(--viz-2)' : 'var(--viz-border)'} strokeWidth={onChain ? 3 : 1} />
                  {Array.from({ length: SLOTS }, (_, s) => (
                    <g key={s}>
                      <rect x={8 + i * 40} y={8 + s * 26} width={28} height={22} rx={3} fill={bucket[s] !== undefined ? 'var(--viz-surface)' : 'var(--viz-plane)'} stroke={bucket[s] !== undefined ? 'var(--viz-1)' : 'none'} strokeWidth={2} />
                      {bucket[s] !== undefined ? (
                        <text x={22 + i * 40} y={23 + s * 26} textAnchor="middle" fontSize={9} fill="var(--viz-ink)">
                          {bucket[s]}
                        </text>
                      ) : null}
                    </g>
                  ))}
                  <text x={22 + i * 40} y={SLOTS * 26 + 24} textAnchor="middle" fontSize={9}>
                    {i}
                  </text>
                </g>
              );
            })}
          </svg>
        </>
      ) : (
        <div className="viz-controls">
          <label className="viz-control">
            <span>Design</span>
            <select value={opDesign} onChange={(e) => { setOpDesign(e.currentTarget.value); setOpResult(''); }}>
              {OPS.map((o) => (
                <option key={o.design} value={o.design}>
                  {o.design}
                </option>
              ))}
            </select>
          </label>
          {([
            ['insertAfterBuild', 'Insert a key after building'],
            ['del', 'Delete a key'],
            ['merge', 'Merge two filters'],
            ['streamBuild', 'Build during compaction, key by key'],
          ] as const).map(([field, label]) => (
            <Button
              key={field}
              onClick={() => {
                const o = OPS.find((x) => x.design === opDesign)!;
                const [v, why] = o[field];
                setOpResult(`${label} on ${opDesign}: ${v === 'yes' ? '✓ supported' : v === 'limited' ? '~ limited' : '✕ refused'} — ${why}.`);
              }}
            >
              {label}
            </Button>
          ))}
        </div>
      )}
    </VizPanel>
  );
}
