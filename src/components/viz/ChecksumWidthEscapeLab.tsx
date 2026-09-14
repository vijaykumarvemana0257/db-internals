import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Check,
  Button,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  useSize,
  makeRng,
  fmtNum,
} from './Viz';
import { SCHEMES, crc32c, fnvBlockSum, foldToWidth } from './pageChecksumModel';

/**
 * How often does a corrupted page slip past the verifier? Replay thousands of
 * real corruptions against four real checksum fields and count the escapes.
 * The answer is set by the width of the stored field (2^-w), with two
 * structural exceptions the learner can trigger: CRC catches every single-bit
 * flip by construction, and no checksum catches a misdirected write unless the
 * block number is mixed into it.
 */

const WORDS = 32; // stand-in page: 4 sectors × 8 words
const SECTORS = 4;
const WPS = WORDS / SECTORS;
const BLK = 4711;
const OTHER_BLK = 8123;

const MODES = [
  { value: 'torn', label: 'Torn write' },
  { value: 'bitrot', label: 'Bit rot (1 flipped bit)' },
  { value: 'misdirected', label: 'Misdirected write' },
] as const;

const TRIALS = [
  { value: '5000', label: '5k' },
  { value: '25000', label: '25k' },
  { value: '100000', label: '100k' },
] as const;

type Mode = (typeof MODES)[number]['value'];

/**
 * The bar is the *width limit*: the escape rate a checksum of this width would
 * have against an adversary-free random corruption, 1/(2^w - 1). `guaranteed`
 * marks the cases where the structure of the scheme rules escapes out entirely
 * — those get no bar, because probability is the wrong model for them.
 */
function detectionFor(schemeId: string, bits: number, mode: Mode, mix: boolean) {
  const limit = schemeId === 'crc32c' ? 2 ** -32 : 1 / (2 ** bits - 1);
  if (mode === 'misdirected') {
    if (!mix) return { limit: 1, guaranteed: null as string | null };
    // Mixing the block number in only *guarantees* detection at full width:
    // the 32-bit sum xored with a different blkno cannot be equal, and a
    // CRC over a differing 4-byte prefix of a same-length message cannot
    // collide either (the difference polynomial has degree < 32). Fold that
    // sum into 8 or 16 bits and two block numbers can alias again, so those
    // rows keep their width limit.
    if (bits >= 32)
      return {
        limit,
        guaranteed:
          schemeId === 'crc32c'
            ? 'the block number is fed through the CRC at full width — a differing prefix cannot cancel'
            : 'the block number is xored into all 32 bits before storage — two blocks cannot match',
      };
    return { limit, guaranteed: null as string | null };
  }
  if (mode === 'bitrot' && schemeId === 'crc32c')
    return { limit, guaranteed: 'every 1-bit flip and every burst \u2264 32 bits — by construction' };
  return { limit, guaranteed: null as string | null };
}

function runSweep(mode: Mode, trials: number, mix: boolean, seed: number) {
  const rng = makeRng(seed);
  const w32 = () =>
    (Math.imul(Math.floor(rng() * 1e6) + 1, 0x9e3779b1) ^
      Math.imul(Math.floor(rng() * 1e6) + 1, 0x85ebca6b)) >>>
    0;

  const a = new Uint32Array(WORDS);
  const b = new Uint32Array(WORDS);
  const m = new Uint32Array(WORDS);
  const esc: Record<string, number> = { fnv8: 0, fnv16: 0, fnv32: 0, crc32c: 0 };

  for (let t = 0; t < trials; t++) {
    for (let i = 0; i < WORDS; i++) {
      a[i] = w32();
      b[i] = w32();
    }
    // Word 0 is the checksum field; the verifier zeroes it before hashing.
    a[0] = 0;
    b[0] = 0;

    let storedSum: number;
    let storedCrc: number;
    let compSum: number;
    let compCrc: number;

    if (mode === 'torn') {
      const mask = 1 + Math.floor(rng() * (2 ** SECTORS - 2)); // never 0, never all-landed
      for (let s = 0; s < SECTORS; s++) {
        const src = (mask >> s) & 1 ? b : a;
        m.set(src.subarray(s * WPS, (s + 1) * WPS), s * WPS);
      }
      const headerIsNew = (mask & 1) === 1;
      storedSum = fnvBlockSum(headerIsNew ? b : a, BLK, mix);
      storedCrc = crc32c(headerIsNew ? b : a, BLK, mix);
      compSum = fnvBlockSum(m, BLK, mix);
      compCrc = crc32c(m, BLK, mix);
    } else if (mode === 'bitrot') {
      m.set(b);
      const wi = 2 + Math.floor(rng() * (WORDS - 2));
      const bit = Math.floor(rng() * 32);
      m[wi] = (m[wi] ^ (1 << bit)) >>> 0;
      storedSum = fnvBlockSum(b, BLK, mix);
      storedCrc = crc32c(b, BLK, mix);
      compSum = fnvBlockSum(m, BLK, mix);
      compCrc = crc32c(m, BLK, mix);
    } else {
      // A self-consistent page that belongs to a different block landed here.
      storedSum = fnvBlockSum(b, OTHER_BLK, mix);
      storedCrc = crc32c(b, OTHER_BLK, mix);
      compSum = fnvBlockSum(b, BLK, mix);
      compCrc = crc32c(b, BLK, mix);
    }

    if (foldToWidth(storedSum, 8) === foldToWidth(compSum, 8)) esc.fnv8++;
    if (foldToWidth(storedSum, 16) === foldToWidth(compSum, 16)) esc.fnv16++;
    if (storedSum >>> 0 === compSum >>> 0) esc.fnv32++;
    if (storedCrc >>> 0 === compCrc >>> 0) esc.crc32c++;
  }
  return esc;
}

const LO = -10; // log10 lower bound of the escape-probability axis

type EscapeRow = {
  id: string;
  label: string;
  bits: number;
  engine: string;
  escapes: number;
  measured: number;
  limit: number;
  guaranteed: string | null;
};

const DECADES = [-10, -8, -6, -4, -2, 0];

/** Lives inside <TooltipHost> so useTip() sees the provider. */
function EscapeFigure({
  width,
  labelW,
  plotW,
  rowH,
  height,
  rows,
  trials,
}: {
  width: number;
  labelW: number;
  plotW: number;
  rowH: number;
  height: number;
  rows: EscapeRow[];
  trials: number;
}) {
  const tip = useTip();
  const x = (prob: number) => {
    if (prob <= 0) return 0;
    return Math.max(0, Math.min(1, (Math.log10(prob) - LO) / (0 - LO))) * plotW;
  };

  return (
    <svg
      width={Math.max(320, width)}
      height={height}
      role="img"
      aria-label="Escape probability by checksum width, theoretical versus measured"
    >
      {DECADES.map((d) => (
        <g key={d}>
          <line
            className="viz-grid-line"
            x1={labelW + x(10 ** d)}
            x2={labelW + x(10 ** d)}
            y1={16}
            y2={height - 20}
          />
          <text x={labelW + x(10 ** d)} y={12} textAnchor="middle" fill="var(--viz-ink-muted)">
            {d === 0 ? '1' : `1e${d}`}
          </text>
        </g>
      ))}
      <text x={labelW} y={height - 6} fill="var(--viz-ink-muted)">
        probability that a corrupted page verifies clean (log scale)
      </text>

      {rows.map((r, i) => {
        const y = 24 + i * rowH;
        const bw = Math.max(2, x(r.limit));
        return (
          <g
            key={r.id}
            {...tip(
              <>
                <strong>{r.label}</strong> · {r.bits} bits · {r.engine}
                <br />
                {fmtNum(r.escapes)} of {fmtNum(trials)} corruptions verified clean
                <br />
                {r.guaranteed
                  ? `detection is structural here: ${r.guaranteed}`
                  : `width limit 1 in ${fmtNum(Math.round(1 / r.limit))}`}
              </>,
            )}
            style={{ cursor: 'help' }}
          >
            <rect x={0} y={y - 4} width={Math.max(320, width)} height={rowH - 4} fill="transparent" />
            <text x={labelW - 10} y={y + 10} textAnchor="end" fill="var(--viz-ink)">
              {r.label}
            </text>
            <text x={labelW - 10} y={y + 23} textAnchor="end" fill="var(--viz-ink-muted)">
              {r.engine}
            </text>
            {r.guaranteed ? (
              <text x={labelW + 4} y={y + 12} fill="var(--viz-good)">
                detected by construction: {r.guaranteed}
              </text>
            ) : (
              <rect x={labelW} y={y + 2} width={bw} height={12} rx={3} fill="var(--viz-1)" />
            )}
            {r.escapes > 0 ? (
              <circle
                cx={labelW + x(r.measured)}
                cy={y + 8}
                r={5}
                fill="var(--viz-2)"
                stroke="var(--viz-surface)"
                strokeWidth={1.5}
              />
            ) : null}
            <text
              x={labelW + plotW + 8}
              y={y + 12}
              fill={r.escapes > 0 ? 'var(--viz-2)' : 'var(--viz-good)'}
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {r.escapes > 0 ? `${fmtNum(r.escapes)} escaped` : '0 escaped'}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function ChecksumWidthEscapeLab() {
  const [mode, setMode] = useState<Mode>('torn');
  const [trialsStr, setTrialsStr] = useState<string>('25000');
  const [mix, setMix] = useState(true);
  const [seed, setSeed] = useState(20250913);
  const [ref, width] = useSize(720);

  const trials = Number(trialsStr);
  const esc = useMemo(() => runSweep(mode, trials, mix, seed), [mode, trials, mix, seed]);

  const rows = SCHEMES.map((s) => ({
    ...s,
    escapes: esc[s.id],
    measured: esc[s.id] / trials,
    ...detectionFor(s.id, s.bits, mode, mix),
  }));

  const labelW = Math.min(230, Math.max(130, width * 0.3));
  const plotW = Math.max(180, width - labelW - 70);
  const rowH = 34;
  const height = rows.length * rowH + 44;

  const pg = rows.find((r) => r.id === 'fnv16')!;
  const innodb = rows.find((r) => r.id === 'crc32c')!;

  return (
    <VizPanel
      title="How often does a corrupted page slip past the verifier?"
      subtitle="Thousands of real corruptions, hashed with four real checksum fields. Escapes are counted, not estimated: the bar is the limit the field\u2019s width imposes, the dot is what this run actually measured."
      controls={
        <>
          <Segmented label="Corruption" value={mode} onChange={setMode} options={MODES} />
          <Segmented label="Trials" value={trialsStr} onChange={setTrialsStr} options={TRIALS} />
          <Check label="Mix the block number into the checksum (PostgreSQL does)" checked={mix} onChange={setMix} />
          <Button onClick={() => setSeed((s) => s + 1)}>Reseed</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Width limit, 1 in 2^w (bar)', color: 'var(--viz-1)' },
            { label: 'Measured escapes this run (dot)', color: 'var(--viz-2)', shape: 'dot' },
            { label: 'Structurally impossible to escape (no bar)', color: 'var(--viz-good)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Corruptions replayed', value: fmtNum(trials) },
            {
              label: 'Escaped 16-bit (PostgreSQL)',
              value: fmtNum(pg.escapes),
              hint: 'Undetected corrupt pages that would be handed to the executor',
            },
            {
              label: 'Escaped CRC-32C (InnoDB)',
              value: fmtNum(innodb.escapes),
            },
            {
              label: '16-bit width limit',
              value: pg.guaranteed ? 'n/a — structural' : `1 in ${fmtNum(Math.round(1 / pg.limit))}`,
              hint: 'pd_checksum is 1 / (2^16 − 1)',
            },
            {
              label: 'CRC-32C width limit',
              value: innodb.guaranteed ? 'n/a — structural' : `1 in ${fmtNum(Math.round(1 / innodb.limit))}`,
            },
          ]}
        />
      }
      note={
        mode === 'misdirected' && !mix ? (
          <>
            <strong>Every single misdirected write escapes.</strong> The page that landed here is
            internally perfect — it is just the wrong page. A checksum over the page bytes alone can
            never see that, which is why PostgreSQL xors the block number into{' '}
            <code>pd_checksum</code> and why InnoDB stores <code>FIL_PAGE_OFFSET</code> and the space
            id inside the page and compares them. Tick the box and every one of them is caught.
          </>
        ) : mode === 'misdirected' ? (
          <>
            <strong>Nothing escaped — but read the guarantee narrowly.</strong> The block number is
            xored into the 32-bit sum, so at full width two different blocks can never produce the
            same value. That guarantee survives the fold only in spirit: squeezing the sum into 16
            bits reintroduces the ordinary 1-in-65,535 aliasing, which is why those rows keep a bar
            rather than a certificate. And none of it covers a write that lands on the{' '}
            <em>same</em> block number of a different file: <code>pd_checksum</code> contains no
            relfilenode, so relation A block 5 arriving on relation B block 5 verifies perfectly
            clean. InnoDB is stricter — <code>FIL_PAGE_OFFSET</code> and <code>FIL_PAGE_SPACE_ID</code>
            live inside the page and are compared on read — and ZFS goes further still by storing the
            checksum in the parent block pointer rather than in the block itself.
          </>
        ) : mode === 'bitrot' ? (
          <>
            <strong>CRC-32C catches every single-bit flip — that is a theorem, not a sample.</strong> A
            one-bit error is the polynomial x<sup>i</sup>, which a degree-32 generator with a nonzero
            constant term cannot divide; the same argument covers every burst error up to 32 bits. The
            FNV folds have no such theorem at any width: a flipped bit changes one lane, the fold
            squeezes that lane back down, and whatever escape rate falls out is a property of these
            constants and this input — count the escapes, reseed, and watch the number move. That is
            why CRC is the standard choice for media errors and a truncated hash is merely adequate.
          </>
        ) : (
          <>
            <strong>Width is the whole story for torn pages.</strong> A tear always changes the bytes,
            so the only way past the verifier is a collision, and the measured dots sit right on the 2
            <sup>−w</sup> bars. At 16 bits that is 1 torn page in 65,535: a 1 TB PostgreSQL cluster
            holds 134 million 8 KB pages, so <code>pd_checksum</code> is a smoke alarm, not a proof.
            CRC-32C's 1 in 4.29 billion is why InnoDB, ZFS and every modern on-disk format spend the
            extra two bytes.
          </>
        )
      }
      table={
        <table className="viz-table">
          <caption style={{ captionSide: 'bottom', textAlign: 'left', paddingTop: '.4rem' }}>
            The simulator hashes a 32-word stand-in for the page; undetected-error probability is set
            by the width of the stored field, not by the length of the input.
          </caption>
          <thead>
            <tr>
              <th>Checksum field</th>
              <th>Bits</th>
              <th>Where it ships</th>
              <th>Escapes</th>
              <th>Measured rate</th>
              <th>Width limit</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.label}</td>
                <td>{r.bits}</td>
                <td>{r.engine}</td>
                <td>
                  {fmtNum(r.escapes)} / {fmtNum(trials)}
                </td>
                <td>{r.escapes === 0 ? `< 1 in ${fmtNum(trials)}` : `1 in ${fmtNum(Math.round(1 / r.measured))}`}</td>
                <td>
                  {r.guaranteed ? 'structural — cannot escape' : `1 in ${fmtNum(Math.round(1 / r.limit))}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <EscapeFigure
            width={width}
            labelW={labelW}
            plotW={plotW}
            rowH={rowH}
            height={height}
            rows={rows}
            trials={trials}
          />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
