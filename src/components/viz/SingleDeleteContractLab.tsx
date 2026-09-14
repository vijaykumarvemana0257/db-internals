import { useState } from 'react';
import { VizPanel, Button, Legend, Stats, Note, TooltipHost, useTip } from './Viz';

/**
 * SingleDelete, and the one sentence of its contract.
 *
 * A Delete tombstone shadows *every* older version of the key it meets and may only be
 * dropped where nothing below overlaps. A SingleDelete annihilates exactly *one*
 * preceding Put and then both entries disappear — which is much cheaper, and is only
 * correct if there was exactly one Put to cancel. Put a key twice and SingleDelete it
 * once and the compaction iterator pairs the marker with the newer Put, leaving the
 * older one as the newest surviving version of the key: the value comes back.
 *
 * Reads are not the tell. Until a compaction pairs the marker off, a Get walks newest
 * first, sees the marker and correctly answers NotFound. The corruption appears at
 * compaction time, minutes or weeks later, in a process nobody was watching.
 */

type Kind = 'put' | 'del' | 'sdel';
type Place = 'mem' | 'l0' | 'l1' | 'bot';

type Ent = { seq: number; kind: Kind; val?: string; place: Place };

type LogRow = { n: number; op: string; result: string; entries: number };

type S = {
  ents: Ent[];
  seq: number;
  vals: number;
  putsSinceMarker: number;
  violated: boolean;
  log: LogRow[];
  head: string;
  body: string;
};

const START: S = {
  ents: [],
  seq: 0,
  vals: 0,
  putsSinceMarker: 0,
  violated: false,
  log: [],
  head: 'One key: the secondary-index entry idx_email(a@ex.com) → pk 7.',
  body:
    'Put it, delete it, and step the compactions by hand. SingleDelete is the marker MyRocks uses for ' +
    'secondary-index entries, because an index entry is written exactly once — which is exactly the ' +
    'condition under which it is defined.',
};

const LANES: { id: Place; label: string; sub: string }[] = [
  { id: 'mem', label: 'Memtable', sub: 'newest writes, not yet flushed' },
  { id: 'l0', label: 'L0', sub: 'flushed files, overlapping ranges' },
  { id: 'l1', label: 'L1', sub: 'first sorted level' },
  { id: 'bot', label: 'Bottommost level', sub: 'nothing below overlaps this key' },
];

const FILL: Record<Kind, string> = {
  put: 'var(--viz-clean)',
  del: 'var(--viz-stale)',
  sdel: 'var(--viz-2)',
};

const GLYPH: Record<Kind, string> = { put: 'PUT', del: 'DEL', sdel: 'SDEL' };

/** Newest first. Sequence numbers are the only ordering an LSM has. */
const byNewest = (a: Ent, b: Ent) => b.seq - a.seq;

/** What a Get returns right now: walk every source newest-first, first marker wins. */
function readNow(ents: Ent[]): string {
  const sorted = [...ents].sort(byNewest);
  const top = sorted[0];
  if (!top) return 'NotFound';
  if (top.kind === 'put') return top.val ?? 'NotFound';
  return 'NotFound';
}

/** What the sequence of logical operations says the answer should be. */
function expected(log: LogRow[]): string {
  for (let i = log.length - 1; i >= 0; i--) {
    const op = log[i].op;
    if (op.startsWith('Put')) return op.slice(op.indexOf('(') + 1, op.indexOf(')'));
    if (op === 'Delete' || op === 'SingleDelete') return 'NotFound';
  }
  return 'NotFound';
}

/**
 * One compaction over the entries in `input`, merged into `into`.
 *
 * `bottommost` means no file below overlaps this key, which is the only condition under
 * which a tombstone itself may be dropped.
 */
function compact(s: S, input: Place[], into: Place, bottommost: boolean): S {
  const inSet = s.ents.filter((e) => input.includes(e.place)).sort(byNewest);
  const rest = s.ents.filter((e) => !input.includes(e.place));
  if (inSet.length === 0) {
    return { ...s, head: 'Nothing to compact there.', body: 'That level pair holds no entry for this key.' };
  }

  const kept: Ent[] = [];
  let head = '';
  let body = '';
  let i = 0;
  let resurrected: Ent | null = null;

  while (i < inSet.length) {
    const e = inSet[i];
    if (e.kind === 'sdel') {
      const nxt = inSet[i + 1];
      if (nxt && nxt.kind === 'put') {
        // Annihilation: the marker cancels exactly one Put, and both vanish.
        i += 2;
        if (inSet[i] && inSet[i].kind === 'put') resurrected = inSet[i];
        head = 'SingleDelete and one Put annihilated each other.';
        body =
          `Sequence ${e.seq} cancelled sequence ${nxt.seq} and both entries left the LSM in this compaction — ` +
          'no tombstone to carry down, no work for any later compaction. That is the whole point of the marker: ' +
          'it is the cheapest possible delete.' +
          (resurrected
            ? ` But there was a second Put underneath it. Sequence ${resurrected.seq} is now the newest surviving ` +
              `version of the key, and the next Get returns ${resurrected.val} — a value that was deleted.`
            : '');
        continue;
      }
      if (!nxt) {
        if (!bottommost) kept.push(e);
        head = bottommost
          ? 'The stray SingleDelete was dropped at the bottommost level.'
          : 'The SingleDelete found no Put to cancel here, so it travels down.';
        body = bottommost
          ? 'Nothing below overlaps this key, so a marker with nothing left to shadow can finally be discarded.'
          : 'The Put it is meant to cancel is in a level this compaction did not read. The marker has to be ' +
            'carried into the output file and tried again later — which is why a delete-heavy workload keeps ' +
            'markers alive across several compactions.';
        i += 1;
        continue;
      }
      kept.push(e);
      head = 'Undefined behaviour: a SingleDelete sitting on another marker.';
      body =
        'RocksDB defines SingleDelete only against a key that was Put exactly once since the last delete of any ' +
        'kind. Two markers in a row is outside the contract, and nothing in the engine will tell you so.';
      i += 1;
      continue;
    }

    if (e.kind === 'del') {
      if (!bottommost) kept.push(e);
      const shadowed = inSet.length - i - 1;
      head = bottommost
        ? `Tombstone dropped, and ${shadowed} older version${shadowed === 1 ? '' : 's'} with it.`
        : `Tombstone kept; it shadowed ${shadowed} older version${shadowed === 1 ? '' : 's'} in this input.`;
      body = bottommost
        ? 'A Delete may only be discarded where nothing below can still hold an older version — the bottommost ' +
          'level, or a level whose key range overlaps no file beneath it. Only then is the space actually returned.'
        : 'A Delete shadows every older version of the key, not one. That is what makes it safe to issue against ' +
          'a key you have written any number of times, and what makes it expensive: the marker itself has to be ' +
          'carried all the way down before the space comes back.';
      break;
    }

    // A Put: it is the newest surviving version, and it shadows everything older here.
    kept.push(e);
    const dropped = inSet.length - i - 1;
    head = `Kept sequence ${e.seq}; dropped ${dropped} shadowed version${dropped === 1 ? '' : 's'}.`;
    body =
      'With no snapshot pinning them, every version older than the newest Put is unreachable, so the compaction ' +
      'writes out one entry and the rest of the key history disappears. Order matters: had this compaction run ' +
      'before the SingleDelete arrived, there would be nothing left underneath for the marker to uncover.';
    break;
  }

  const moved = kept.map((e) => ({ ...e, place: into }));
  return { ...s, ents: [...rest, ...moved], head, body };
}

export default function SingleDeleteContractLab() {
  const [s, setS] = useState<S>(START);
  const tip = useTip();

  const got = readNow(s.ents);
  const want = expected(s.log);
  const wrong = got !== want;

  const push = (kind: Kind) =>
    setS((c) => {
      const seq = c.seq + 1;
      const vals = kind === 'put' ? c.vals + 1 : c.vals;
      const val = kind === 'put' ? `v${vals}` : undefined;
      const ent: Ent = { seq, kind, val, place: 'mem' };
      const violated = c.violated || (kind === 'sdel' && c.putsSinceMarker !== 1);
      const head =
        kind === 'put'
          ? `Put ${val} at sequence ${seq}.`
          : kind === 'del'
            ? `Delete marker at sequence ${seq}.`
            : `SingleDelete marker at sequence ${seq}.`;
      const body =
        kind === 'put'
          ? 'A new version, appended in front of every older one. Nothing was read, nothing was overwritten.'
          : kind === 'del'
            ? 'A tombstone: a write like any other, and one that shadows every older version of this key it ever meets.'
            : c.putsSinceMarker === 1
              ? 'Inside the contract: exactly one Put since the last delete. This marker will cancel that Put and ' +
                'both will vanish at the first compaction that sees them together.'
              : c.putsSinceMarker === 0
                ? 'Outside the contract: this key has no live Put for the marker to cancel. RocksDB calls that ' +
                  'undefined, and the marker will wander down the levels looking for a victim.'
                : `Outside the contract: ${c.putsSinceMarker} Puts since the last delete. The marker can only ` +
                  'cancel one of them. Flush, then compact, and watch which version is left standing.';
      return {
        ...c,
        seq,
        vals,
        ents: [...c.ents, ent],
        putsSinceMarker: kind === 'put' ? c.putsSinceMarker + 1 : 0,
        violated,
        head,
        body,
        log: [
          ...c.log,
          {
            n: c.log.length + 1,
            op: kind === 'put' ? `Put(${val})` : kind === 'del' ? 'Delete' : 'SingleDelete',
            result: '—',
            entries: c.ents.length + 1,
          },
        ],
      };
    });

  const flush = () =>
    setS((c) => {
      const n = c.ents.filter((e) => e.place === 'mem').length;
      if (n === 0) return { ...c, head: 'The memtable is empty.', body: 'Nothing to flush for this key.' };
      return {
        ...c,
        ents: c.ents.map((e) => (e.place === 'mem' ? { ...e, place: 'l0' } : e)),
        head: `Flushed ${n} entr${n === 1 ? 'y' : 'ies'} into one L0 file.`,
        body:
          'A flush is not a compaction. It writes the memtable out verbatim — every version, every marker, in ' +
          'sequence order — so nothing has been resolved yet.',
        log: [...c.log, { n: c.log.length + 1, op: 'Flush', result: '—', entries: c.ents.length }],
      };
    });

  const step = (label: string, input: Place[], into: Place, bottommost: boolean) =>
    setS((c) => {
      const next = compact(c, input, into, bottommost);
      return {
        ...next,
        log: [...c.log, { n: c.log.length + 1, op: label, result: readNow(next.ents), entries: next.ents.length }],
      };
    });

  const doGet = () =>
    setS((c) => {
      const r = readNow(c.ents);
      const e = expected(c.log);
      return {
        ...c,
        head: `Get → ${r}`,
        body:
          r === e
            ? 'The read walked the sources newest-first and stopped at the first entry for the key. Correct — for now.'
            : `The last thing you asked for was ${e}. The read is answering with a version you deleted, because a ` +
              'compaction paired your SingleDelete with one Put and uncovered the one beneath it. Nothing logged an ' +
              'error; the index entry simply exists again.',
        log: [...c.log, { n: c.log.length + 1, op: 'Get', result: r, entries: c.ents.length }],
      };
    });

  const maxRow = Math.max(...LANES.map((l) => s.ents.filter((e) => e.place === l.id).length), 1);
  const labelW = 156;
  const chipW = 92;
  const laneH = 48;
  const width = labelW + maxRow * (chipW + 8) + 24;
  const height = LANES.length * laneH + 16;

  return (
    <VizPanel
      title="SingleDelete and the value that comes back"
      subtitle="One key, one version stack. Put it twice, SingleDelete it once, then step the compaction that pairs the marker with a Put and see what is left underneath."
      controls={
        <>
          <Button onClick={() => push('put')} primary>
            Put
          </Button>
          <Button onClick={() => push('del')} title="kTypeDeletion — shadows every older version">
            Delete
          </Button>
          <Button onClick={() => push('sdel')} title="kTypeSingleDeletion — cancels exactly one Put">
            SingleDelete
          </Button>
          <Button onClick={flush}>Flush memtable → L0</Button>
          <Button onClick={() => step('Compact L0→L1', ['l0', 'l1'], 'l1', false)}>Compact L0→L1</Button>
          <Button onClick={() => step('Compact L1→bottom', ['l1', 'bot'], 'bot', true)}>Compact L1→bottom</Button>
          <Button onClick={doGet}>Get</Button>
          <Button onClick={() => setS(START)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'PUT — a value version', color: FILL.put },
            { label: 'DEL — tombstone, shadows all older versions', color: FILL.del },
            { label: 'SDEL — SingleDelete, cancels exactly one Put', color: FILL.sdel },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Entries stored', value: s.ents.length, hint: 'Every version and marker still physically present' },
            { label: 'Puts since last marker', value: s.putsSinceMarker, hint: 'SingleDelete is defined only when this is exactly 1' },
            {
              label: 'Contract',
              value: s.violated ? 'violated' : 'held',
              hint: 'Nothing in RocksDB checks this. It is your job.',
            },
            { label: 'Get returns', value: got },
            { label: 'Should return', value: want },
            { label: 'Agreement', value: wrong ? 'resurrected' : 'correct' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Operation</th>
              <th>Get would return</th>
              <th>Entries stored</th>
            </tr>
          </thead>
          <tbody>
            {s.log.length === 0 ? (
              <tr>
                <td colSpan={4}>Nothing issued yet.</td>
              </tr>
            ) : (
              s.log.map((r) => (
                <tr key={r.n}>
                  <td>{r.n}</td>
                  <td>{r.op}</td>
                  <td>{r.result}</td>
                  <td>{r.entries}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <svg width={width} height={height} role="img" aria-label="Version stack of one key across the memtable, L0, L1 and the bottommost level">
          {LANES.map((ln, i) => {
            const y = i * laneH + 4;
            const rows = s.ents.filter((e) => e.place === ln.id).sort(byNewest);
            return (
              <g key={ln.id}>
                <rect
                  x={labelW}
                  y={y}
                  width={width - labelW - 16}
                  height={laneH - 10}
                  rx={7}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={0} y={y + 16} fill="var(--viz-ink)" fontWeight={600}>
                  {ln.label}
                </text>
                <text x={0} y={y + 29} fill="var(--viz-ink-muted)">
                  {ln.sub}
                </text>
                {rows.map((e, j) => {
                  const x = labelW + 10 + j * (chipW + 8);
                  return (
                    <g
                      key={e.seq}
                      {...tip(
                        <>
                          <strong>
                            seq {e.seq} · {GLYPH[e.kind]}
                            {e.val ? ` ${e.val}` : ''}
                          </strong>
                          <br />
                          {e.kind === 'put'
                            ? 'A value version. It is visible only while no newer entry for this key sits in front of it.'
                            : e.kind === 'del'
                              ? 'A tombstone. Reads stop here and answer NotFound; compaction drops every older version it meets, and drops itself only at the bottommost level.'
                              : 'A SingleDelete marker. Reads stop here too — but a compaction cancels it against exactly one Put and then both entries are gone.'}
                        </>,
                      )}
                      style={{ cursor: 'help' }}
                    >
                      <rect x={x} y={y + 7} width={chipW} height={24} rx={6} fill={FILL[e.kind]} />
                      <text x={x + chipW / 2} y={y + 23} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                        {e.seq} {GLYPH[e.kind]} {e.val ?? ''}
                      </text>
                    </g>
                  );
                })}
                {rows.length === 0 ? (
                  <text x={labelW + 12} y={y + 23} fill="var(--viz-ink-muted)">
                    empty
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </TooltipHost>
    </VizPanel>
  );
}
