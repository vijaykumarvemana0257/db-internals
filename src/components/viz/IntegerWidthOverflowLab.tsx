import { useState } from 'react';
import { VizPanel, Slider, Segmented, Choice, Check, Legend, Stats, Note, useSize } from './Viz';

/**
 * What a counter does when it runs out of bits — and how differently four engines answer.
 *
 * Everything here is exact: the value is a bigint, the stored result is computed with
 * BigInt.asIntN / asUintN for the wrapping case, and the bit strip is the real two's
 * complement pattern of the stored value. The only modelled thing is the engine's policy
 * on overflow, which is the whole point: the same INSERT raises, saturates, wraps or
 * silently turns into a float depending on who is running it.
 */

/* --------------------------------------------------------------- the widths */

type WidthId = 'i16' | 'i32' | 'i64';

const WIDTHS: { value: WidthId; label: string; bits: number; pg: string; my: string }[] = [
  { value: 'i16', label: '16-bit', bits: 16, pg: 'smallint', my: 'SMALLINT' },
  { value: 'i32', label: '32-bit', bits: 32, pg: 'integer', my: 'INT' },
  { value: 'i64', label: '64-bit', bits: 64, pg: 'bigint', my: 'BIGINT' },
];

function range(bits: number, unsigned: boolean) {
  const b = BigInt(bits);
  return unsigned ? { min: 0n, max: (1n << b) - 1n } : { min: -(1n << (b - 1n)), max: (1n << (b - 1n)) - 1n };
}

/* -------------------------------------------------------------- the engines */

type EngineId = 'pg' | 'mysql-strict' | 'mysql-lax' | 'sqlite' | 'app';

const ENGINES: { value: EngineId; label: string }[] = [
  { value: 'pg', label: 'PostgreSQL' },
  { value: 'mysql-strict', label: 'MySQL, strict mode' },
  { value: 'mysql-lax', label: 'MySQL, non-strict' },
  { value: 'sqlite', label: 'SQLite' },
  { value: 'app', label: 'Your application language' },
];

type Outcome = 'ok' | 'error' | 'saturate' | 'wrap' | 'float';

type Result = {
  outcome: Outcome;
  stored: bigint | null;
  approx: number | null; // for the float case
  message: string;
  colour: string;
};

function evaluate(engine: EngineId, bits: number, unsigned: boolean, v: bigint): Result {
  const { min, max } = range(bits, unsigned);
  const pgName = WIDTHS.find((w) => w.bits === bits)!.pg;
  const myName = WIDTHS.find((w) => w.bits === bits)!.my;

  // SQLite has one integer storage class; the declared width is advisory and ignored.
  if (engine === 'sqlite') {
    const i64 = range(64, false);
    if (v <= i64.max && v >= i64.min) {
      return {
        outcome: 'ok',
        stored: v,
        approx: null,
        message:
          bits === 64
            ? 'Stored. SQLite always uses a 64-bit signed integer.'
            : `Stored, in full. SQLite ignores the declared width — a column declared ${myName} holds a 64-bit integer, so this value fits even though it is out of range for ${pgName}.`,
        colour: 'var(--viz-good)',
      };
    }
    return {
      outcome: 'float',
      stored: null,
      approx: Number(v),
      message:
        'Past 2⁶³−1 SQLite converts the result of integer arithmetic to REAL — an IEEE-754 double. No error, no wrap, just a value that is now approximate. (sum() is the exception: over integers it raises "integer overflow" instead.)',
      colour: 'var(--viz-serious)',
    };
  }

  if (v >= min && v <= max) {
    return { outcome: 'ok', stored: v, approx: null, message: 'Stored exactly.', colour: 'var(--viz-good)' };
  }

  if (engine === 'pg') {
    return {
      outcome: 'error',
      stored: null,
      approx: null,
      message: `ERROR:  ${pgName} out of range   (SQLSTATE 22003). The statement aborts and the transaction is left in a failed state — nothing is written.`,
      colour: 'var(--viz-critical)',
    };
  }
  if (engine === 'mysql-strict') {
    return {
      outcome: 'error',
      stored: null,
      approx: null,
      message:
        "ERROR 1264 (22003): Out of range value for column 'n' at row 1. Strict mode has been the default since MySQL 5.7; the row is rejected.",
      colour: 'var(--viz-critical)',
    };
  }
  if (engine === 'mysql-lax') {
    return {
      outcome: 'saturate',
      stored: v > max ? max : min,
      approx: null,
      message:
        "Warning 1264: Out of range value for column 'n' at row 1 — and the row is inserted anyway, clamped to the column's limit. It saturates; it does not wrap. Without strict mode you get a plausible wrong number and a warning nobody reads.",
      colour: 'var(--viz-serious)',
    };
  }
  // Application-language arithmetic: two's complement wraparound, silently.
  const w = unsigned ? BigInt.asUintN(bits, v) : BigInt.asIntN(bits, v);
  return {
    outcome: 'wrap',
    stored: w,
    approx: null,
    message:
      'Two’s-complement wraparound, no signal at all. Java, Go, C and Rust in release mode all do this on a 32-bit counter, which is why the overflow usually happens before the INSERT — the database only sees the wrapped value and stores it happily.',
    colour: 'var(--viz-critical)',
  };
}

/* ------------------------------------------------------------- formatting */

function fmtBig(v: bigint): string {
  const neg = v < 0n;
  const s = (neg ? -v : v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return (neg ? '−' : '') + s;
}

function duration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 90) return `${seconds.toFixed(1)} s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(1)} hours`;
  if (seconds < 63_072_000) return `${(seconds / 86400).toFixed(1)} days`;
  const years = seconds / (86400 * 365.25);
  if (years < 1e4) return `${years.toFixed(0)} years`;
  return `${years.toExponential(1)} years`;
}

export default function IntegerWidthOverflowLab() {
  const [widthId, setWidthId] = useState<WidthId>('i32');
  const [engine, setEngine] = useState<EngineId>('pg');
  const [unsigned, setUnsigned] = useState(false);
  const [pct, setPct] = useState(98); // percent of the type's maximum
  const [rateExp, setRateExp] = useState(4); // 10^4 = 10 000 inserts/second
  const [ref, width] = useSize(720);

  const spec = WIDTHS.find((w) => w.value === widthId)!;
  const bits = spec.bits;
  const unsignedEffective = unsigned && engine !== 'pg';
  const { min, max } = range(bits, unsignedEffective);

  // Exact: value = floor(max * pct / 100), computed in bigint so 64-bit stays exact.
  const value = (max * BigInt(Math.round(pct * 100))) / 10_000n;
  const res = evaluate(engine, bits, unsignedEffective, value);

  const rate = 10 ** rateExp;
  const exhaustSeconds = Number(max) / rate;

  /* --------------------------------------------------------------- the bits */
  const shownBits = res.stored !== null ? BigInt.asUintN(bits, res.stored) : BigInt.asUintN(bits, value);
  const bitChars = shownBits.toString(2).padStart(bits, '0').split('');
  const perRow = bits === 64 ? 32 : bits;
  const W = Math.max(360, Math.min(width, 860));
  const cell = Math.min(22, Math.floor((W - 120) / perRow));
  const rows = Math.ceil(bits / perRow);
  const H = 78 + rows * (cell + 18);

  // The range bar: 0 … max, with the attempted value marked (clipped at 1.25× max).
  const barY = 26;
  const barW = W - 132;
  const frac = Math.min(1.25, Number(value) / Number(max));

  return (
    <VizPanel
      title="Running a counter past the end of its type"
      subtitle="Set the width, push the counter past the maximum, and change who is executing the INSERT. The bits are real two's complement; only the overflow policy differs by engine."
      controls={
        <>
          <Segmented
            label="Column width"
            value={widthId}
            onChange={setWidthId}
            options={WIDTHS.map((w) => ({ value: w.value, label: w.label, title: `${w.pg} / ${w.my}` }))}
          />
          <Choice label="Executed by" value={engine} onChange={setEngine} options={ENGINES} />
          <Check label="UNSIGNED" checked={unsigned} onChange={setUnsigned} />
          <Slider
            label="Counter"
            min={50}
            max={125}
            step={0.5}
            value={pct}
            onChange={setPct}
            format={(p) => `${p}% of max`}
          />
          <Slider
            label="Insert rate"
            min={0}
            max={7}
            value={rateExp}
            onChange={setRateExp}
            format={(e) => `10^${e}/s`}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'in range — stored exactly', color: 'var(--viz-good)' },
            { label: 'clamped or approximated — a wrong number, stored', color: 'var(--viz-serious)' },
            { label: 'rejected or wrapped', color: 'var(--viz-critical)' },
            { label: 'sign bit (heavy outline)', color: 'var(--viz-ink)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `${spec.pg} max`, value: fmtBig(max), hint: `min ${fmtBig(min)}` },
            { label: 'as money at 2dp', value: `$${fmtBig(max / 100n)}.${(max % 100n).toString().padStart(2, '0')}` },
            { label: 'counter value', value: fmtBig(value) },
            { label: 'stored', value: res.stored !== null ? fmtBig(res.stored) : res.approx !== null ? res.approx.toExponential(4) : 'nothing' },
            { label: `0 → max at 10^${rateExp}/s`, value: duration(exhaustSeconds) },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {res.outcome === 'ok'
              ? 'In range.'
              : res.outcome === 'error'
                ? 'Rejected.'
                : res.outcome === 'saturate'
                  ? 'Silently clamped.'
                  : res.outcome === 'wrap'
                    ? 'Silently wrapped.'
                    : 'Silently turned into a float.'}
          </strong>{' '}
          {res.message}
          {unsigned && engine === 'pg' ? ' PostgreSQL has no unsigned integer types at all, so the UNSIGNED toggle does nothing here — the range is still signed.' : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Type</th>
                <th>Bytes</th>
                <th>Maximum</th>
                <th>As minor units (2dp)</th>
                <th>0 → max at 10^{rateExp}/s</th>
              </tr>
            </thead>
            <tbody>
              {WIDTHS.map((w) => {
                const r = range(w.bits, unsignedEffective);
                return (
                  <tr key={w.value}>
                    <td>
                      {w.pg} / {w.my}
                      {unsignedEffective ? ' UNSIGNED' : ''}
                    </td>
                    <td>{w.bits / 8}</td>
                    <td>{fmtBig(r.max)}</td>
                    <td>${fmtBig(r.max / 100n)}</td>
                    <td>{duration(Number(r.max) / rate)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Executed by</th>
                <th>Outcome at {fmtBig(value)}</th>
                <th>Stored</th>
              </tr>
            </thead>
            <tbody>
              {ENGINES.map((e) => {
                const r = evaluate(e.value, bits, unsigned && e.value !== 'pg', value);
                return (
                  <tr key={e.value}>
                    <td>{e.label}</td>
                    <td>{r.outcome}</td>
                    <td>{r.stored !== null ? fmtBig(r.stored) : r.approx !== null ? r.approx.toExponential(4) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <svg width={W} height={H} role="img" aria-label="Counter position within the integer range, and the resulting bit pattern">
          {/* range bar */}
          <text x={0} y={barY - 8} fill="var(--viz-ink-2)">
            0
          </text>
          <text x={barW} y={barY - 8} textAnchor="middle" fill="var(--viz-ink-2)">
            max = {fmtBig(max)}
          </text>
          <rect x={0} y={barY} width={barW} height={16} rx={4} fill="var(--viz-neutral)" />
          <rect
            x={0}
            y={barY}
            width={Math.max(1, Math.min(barW, frac * barW))}
            height={16}
            rx={4}
            fill={frac > 1 ? 'var(--viz-2)' : 'var(--viz-1)'}
          />
          {/* the region past the maximum */}
          <rect x={barW} y={barY} width={barW * 0.25} height={16} rx={4} fill="var(--viz-neutral)" opacity={0.7} />
          {frac > 1 ? (
            <rect x={barW} y={barY} width={(frac - 1) * barW} height={16} rx={4} fill="var(--viz-critical)" />
          ) : null}
          <line className="viz-axis-line" x1={barW} x2={barW} y1={barY - 5} y2={barY + 21} stroke="var(--viz-ink)" />
          <text x={Math.min(barW * 1.25, frac * barW)} y={barY + 33} textAnchor="end" fill="var(--viz-ink)">
            {fmtBig(value)}
          </text>

          {/* bit strip of the value that actually ends up in the column */}
          <text x={0} y={barY + 58} fill="var(--viz-ink-2)">
            {res.stored !== null
              ? `stored bits (${bits}-bit two’s complement)`
              : res.outcome === 'float'
                ? 'bits of the attempted value — not what gets stored'
                : 'bits of the rejected value'}
          </text>
          {Array.from({ length: rows }, (_, r) => (
            <g key={r} transform={`translate(0, ${barY + 68 + r * (cell + 18)})`}>
              {bitChars.slice(r * perRow, (r + 1) * perRow).map((b, i) => {
                const idx = r * perRow + i;
                const isSign = idx === 0 && !unsignedEffective;
                return (
                  <g key={idx}>
                    <rect
                      x={i * cell}
                      y={0}
                      width={cell - 2}
                      height={cell - 2}
                      rx={2}
                      fill={b === '1' ? res.colour : 'var(--viz-neutral)'}
                      stroke={isSign ? 'var(--viz-ink)' : 'var(--viz-border)'}
                      strokeWidth={isSign ? 2 : 1}
                    />
                    {cell >= 14 ? (
                      <text
                        x={i * cell + (cell - 2) / 2}
                        y={(cell - 2) / 2 + 4}
                        textAnchor="middle"
                        fill={b === '1' ? 'var(--viz-surface)' : 'var(--viz-ink-muted)'}
                      >
                        {b}
                      </text>
                    ) : null}
                  </g>
                );
              })}
              <text x={perRow * cell + 8} y={(cell - 2) / 2 + 4} fill="var(--viz-ink-muted)">
                {`bit ${bits - 1 - r * perRow}…${bits - (r + 1) * perRow}`}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </VizPanel>
  );
}
