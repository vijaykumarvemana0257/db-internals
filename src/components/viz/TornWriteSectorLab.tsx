import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Check,
  Button,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  useSize,
  makeRng,
  fmtBytes,
} from './Viz';
import { BYTES_PER_WORD, checksumOf, hex, makePageWords } from './pageChecksumModel';

/**
 * A 16 KB engine page is not one write to the device — it is a burst of
 * sector-sized writes, and only a sector is atomic. Lose power mid-burst and
 * the platter holds a mixture. The learner picks how many sectors reached
 * media, then runs the verifier and watches the checksum flag (but not fix) it.
 */

const OLD_LSN = 0x1a2b3c40;
const NEW_LSN = 0x1a2b3c88;
const BLKNO = 4711;

const PAGE_SIZES = [
  { value: '4096', label: '4 KB (SQLite default)' },
  { value: '8192', label: '8 KB (PostgreSQL BLCKSZ)' },
  { value: '16384', label: '16 KB (InnoDB default)' },
] as const;

const SECTOR_OPTS = [
  { value: '4096', label: '4Kn (4096 B)', title: 'Native 4K drive: one 4 KB sector is the atomic unit' },
  { value: '512', label: '512e (512 B)', title: '512-byte emulation over 4K media: the OS sees 512 B sectors' },
] as const;

const SCHEME_OPTS = [
  { value: 'fnv16', label: 'PostgreSQL pd_checksum (16-bit)' },
  { value: 'crc32c', label: 'InnoDB CRC-32C (32-bit)' },
] as const;

type Row = { key: string; lines: string[]; y: number; kind: 'old' | 'new' | 'media' };

type FigureProps = {
  width: number;
  labelW: number;
  stripW: number;
  cellW: number;
  cellH: number;
  height: number;
  mediaY: number;
  axisY: number;
  vfyY: number;
  rows: Row[];
  nSectors: number;
  sectorBytes: number;
  pageBytes: number;
  order: number[];
  landedSet: Set<number>;
  boundaryX: number | null;
  reorder: boolean;
  torn: boolean;
  verified: boolean;
  detected: boolean;
  scheme: string;
  bits: number;
  storedCk: number;
  computedCk: number;
};

/** Lives inside <TooltipHost> so useTip() sees the provider. */
function TornFigure(p: FigureProps) {
  const tip = useTip();
  const fillFor = (kind: Row['kind'], s: number) =>
    kind === 'old'
      ? 'var(--viz-clean)'
      : kind === 'new'
        ? 'var(--viz-dirty)'
        : p.landedSet.has(s)
          ? 'var(--viz-dirty)'
          : 'var(--viz-clean)';

  return (
    <svg
      width={Math.max(320, p.width)}
      height={p.height}
      role="img"
      aria-label={`A ${fmtBytes(p.pageBytes)} page split into ${p.nSectors} sectors, of which ${p.landedSet.size} reached media before power loss`}
    >
      <text x={p.labelW} y={14} fill="var(--viz-ink-2)">
        {fmtBytes(p.pageBytes)} page · {p.nSectors} × {fmtBytes(p.sectorBytes)} sector
        {p.nSectors > 1 ? 's' : ''} · block {BLKNO}
      </text>

      {p.rows.map((row) => (
        <g key={row.key}>
          {row.lines.map((ln, i) => (
            <text
              key={ln}
              x={p.labelW - 10}
              y={row.y + 12 + i * 13}
              textAnchor="end"
              fill={i === 0 ? 'var(--viz-ink)' : 'var(--viz-ink-2)'}
            >
              {ln}
            </text>
          ))}
          {Array.from({ length: p.nSectors }, (_, s) => {
            const x = p.labelW + s * p.cellW;
            const isNew = row.kind === 'new' || (row.kind === 'media' && p.landedSet.has(s));
            return (
              <g
                key={s}
                {...(row.kind === 'media'
                  ? tip(
                      <>
                        <strong>Sector {s}</strong> · bytes {s * p.sectorBytes}–
                        {(s + 1) * p.sectorBytes - 1}
                        <br />
                        write order {p.order.indexOf(s) + 1} of {p.nSectors} ·{' '}
                        {p.landedSet.has(s) ? 'reached media' : 'never reached media'}
                        <br />
                        holds {p.landedSet.has(s) ? 'v2 (new)' : 'v1 (old)'} bytes
                        {s === 0 ? ' — including pd_checksum and the pageLSN' : ''}
                      </>,
                    )
                  : {})}
                style={row.kind === 'media' ? { cursor: 'help' } : undefined}
              >
                <rect
                  x={x + 1}
                  y={row.y}
                  width={Math.max(2, p.cellW - 2)}
                  height={p.cellH}
                  rx={3}
                  fill={fillFor(row.kind, s)}
                  opacity={row.kind === 'media' ? 1 : 0.7}
                  stroke={s === 0 ? 'var(--viz-ink-2)' : 'var(--viz-surface)'}
                  strokeWidth={s === 0 ? 2 : 1}
                />
                {p.cellW >= 30 ? (
                  <text
                    x={x + p.cellW / 2}
                    y={row.y + 19}
                    textAnchor="middle"
                    fill="var(--viz-surface)"
                    style={{ fontWeight: 600 }}
                  >
                    {isNew ? 'v2' : 'v1'}
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      ))}

      <text x={p.labelW + 2} y={p.axisY} fill="var(--viz-ink-2)">
        ↑ header sector: pd_checksum + pageLSN
      </text>
      <text x={p.labelW + p.stripW} y={p.axisY} textAnchor="end" fill="var(--viz-ink-muted)">
        byte {p.pageBytes}
      </text>

      {p.boundaryX !== null ? (
        <g>
          <line
            x1={p.boundaryX}
            x2={p.boundaryX}
            y1={p.mediaY - 26}
            y2={p.mediaY + p.cellH + 4}
            stroke="var(--viz-critical)"
            strokeWidth={2}
            strokeDasharray="4 3"
          />
          <text
            x={p.boundaryX > p.labelW + p.stripW * 0.7 ? p.boundaryX - 6 : p.boundaryX + 6}
            y={p.mediaY - 16}
            textAnchor={p.boundaryX > p.labelW + p.stripW * 0.7 ? 'end' : 'start'}
            fill="var(--viz-critical)"
            style={{ fontWeight: 600 }}
          >
            power fails here
          </text>
        </g>
      ) : null}

      {p.reorder && p.torn ? (
        <text x={p.labelW} y={p.mediaY - 16} fill="var(--viz-ink-2)">
          the device reordered the sectors — what survived is not even a prefix
        </text>
      ) : null}

      <g>
        <rect
          x={p.labelW}
          y={p.vfyY}
          width={p.stripW}
          height={44}
          rx={6}
          fill="var(--viz-plane)"
          stroke={
            !p.verified ? 'var(--viz-border)' : p.detected ? 'var(--viz-critical)' : 'var(--viz-good)'
          }
          strokeWidth={p.verified ? 2 : 1}
        />
        <text x={p.labelW - 10} y={p.vfyY + 20} textAnchor="end" fill="var(--viz-ink)">
          Verifier
        </text>
        <text x={p.labelW - 10} y={p.vfyY + 33} textAnchor="end" fill="var(--viz-ink-2)">
          runs on every read
        </text>
        {p.verified ? (
          <>
            <text x={p.labelW + 12} y={p.vfyY + 18} fill="var(--viz-ink-2)">
              stored <tspan fill="var(--viz-ink)">{hex(p.storedCk, p.bits)}</tspan>
              {'  '}
              {p.detected ? '≠' : '='}
              {'  '}
              recomputed <tspan fill="var(--viz-ink)">{hex(p.computedCk, p.bits)}</tspan>
            </text>
            <text
              x={p.labelW + 12}
              y={p.vfyY + 34}
              fill={p.detected ? 'var(--viz-critical)' : 'var(--viz-good)'}
              style={{ fontWeight: 600 }}
            >
              {p.detected
                ? p.scheme === 'crc32c'
                  ? '[ERROR] InnoDB: Database page corruption on disk — read aborted, nothing repaired'
                  : 'ERROR XX001: invalid page in block 4711 — read aborted, nothing repaired'
                : p.torn
                  ? 'checksum MATCHES a torn page — a collision, and the corruption is now silent'
                  : 'page verifies clean'}
            </text>
          </>
        ) : (
          <text x={p.labelW + 12} y={p.vfyY + 26} fill="var(--viz-ink-muted)">
            press “Run checksum verifier” to recompute over the bytes on media
          </text>
        )}
      </g>
    </svg>
  );
}

export default function TornWriteSectorLab() {
  const [pageSize, setPageSize] = useState<string>('16384');
  const [sectorSize, setSectorSize] = useState<string>('4096');
  const [scheme, setScheme] = useState<string>('fnv16');
  const [reorder, setReorder] = useState(false);
  const [landed, setLanded] = useState(1);
  const [verifiedKey, setVerifiedKey] = useState('');
  const [ref, width] = useSize(760);

  const pageBytes = Number(pageSize);
  const sectorBytes = Number(sectorSize);
  const nSectors = pageBytes / sectorBytes;
  const wordsPerSector = sectorBytes / BYTES_PER_WORD;
  const bits = scheme === 'crc32c' ? 32 : 16;
  const landedCount = Math.min(landed, nSectors);

  /** The order the device actually commits the sectors in. */
  const order = useMemo(() => {
    const ix = Array.from({ length: nSectors }, (_, i) => i);
    if (!reorder) return ix;
    const rng = makeRng(9001 + nSectors);
    for (let i = ix.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = ix[i];
      ix[i] = ix[j];
      ix[j] = t;
    }
    return ix;
  }, [nSectors, reorder]);

  const landedSet = useMemo(() => new Set(order.slice(0, landedCount)), [order, landedCount]);

  const model = useMemo(() => {
    const build = (version: number, lsn: number) => {
      const w = makePageWords(pageBytes, sectorBytes, version);
      w[0] = 0;
      w[1] = lsn;
      w[0] = checksumOf(scheme, w, BLKNO, true);
      return w;
    };
    const oldImg = build(0, OLD_LSN);
    const newImg = build(1, NEW_LSN);
    const media = new Uint32Array(oldImg.length);
    for (let s = 0; s < nSectors; s++) {
      const src = landedSet.has(s) ? newImg : oldImg;
      media.set(src.subarray(s * wordsPerSector, (s + 1) * wordsPerSector), s * wordsPerSector);
    }
    const storedCk = media[0] >>> 0;
    const probe = media.slice();
    probe[0] = 0;
    const computedCk = checksumOf(scheme, probe, BLKNO, true);
    return { storedCk, computedCk };
  }, [pageBytes, sectorBytes, scheme, nSectors, wordsPerSector, landedSet]);

  const torn = landedCount > 0 && landedCount < nSectors;
  const detected = model.storedCk !== model.computedCk;
  const headerLanded = landedSet.has(0);
  const key = `${pageSize}|${sectorSize}|${scheme}|${reorder}|${landedCount}`;
  const verified = verifiedKey === key;

  const labelW = Math.min(210, Math.max(118, width * 0.27));
  const stripW = Math.max(160, width - labelW - 10);
  const cellW = stripW / nSectors;
  const cellH = 30;
  const mediaY = 142;
  const axisY = mediaY + cellH + 15;
  const vfyY = mediaY + cellH + 36;

  const rows: Row[] = [
    { key: 'old', lines: ['Previous durable image', '(v1, what disk held)'], y: 30, kind: 'old' },
    { key: 'new', lines: ['Page in the buffer pool', '(v2, being written)'], y: 78, kind: 'new' },
    { key: 'media', lines: ['On the platter after', 'power loss'], y: mediaY, kind: 'media' },
  ];

  const boundaryX = !reorder && torn ? labelW + landedCount * cellW : null;
  const staleSectors = Array.from({ length: nSectors }, (_, i) => i).filter((s) => !landedSet.has(s));

  return (
    <VizPanel
      title="Torn-write simulator: a page is many sectors, and only a sector is atomic"
      subtitle="Cut power partway through a page write, then run the checksum verifier. Shrink the page or widen the sector until the page fits inside one atomic unit and the tear becomes impossible."
      controls={
        <>
          <Choice label="Engine page size" value={pageSize} onChange={setPageSize} options={PAGE_SIZES} />
          <Segmented label="Drive sector" value={sectorSize} onChange={setSectorSize} options={SECTOR_OPTS} />
          <Slider
            label="Sectors that reached media"
            min={0}
            max={nSectors}
            value={landedCount}
            onChange={setLanded}
            format={(n) => `${n} of ${nSectors}`}
          />
          <Check label="Device commits sectors out of order" checked={reorder} onChange={setReorder} />
          <Choice label="Checksum" value={scheme} onChange={setScheme} options={SCHEME_OPTS} />
          <Button primary onClick={() => setVerifiedKey(key)} disabled={verified}>
            {verified ? 'Verified' : 'Run checksum verifier'}
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'v1 bytes (previous image)', color: 'var(--viz-clean)' },
            { label: 'v2 bytes (new image)', color: 'var(--viz-dirty)' },
            { label: 'Header sector — outlined, carries pd_checksum + pageLSN', color: 'var(--viz-ink-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Sectors per page', value: `${nSectors} × ${fmtBytes(sectorBytes)}` },
            { label: 'Sectors landed', value: `${landedCount} / ${nSectors}` },
            {
              label: 'Stored checksum',
              value: verified ? hex(model.storedCk, bits) : '—',
              hint: 'The value sitting in the page header on media',
            },
            {
              label: 'Recomputed',
              value: verified ? hex(model.computedCk, bits) : '—',
              hint: 'Recomputed over the bytes actually on media, with the checksum field zeroed',
            },
            {
              label: 'Verdict',
              value: verified ? (detected ? 'MISMATCH' : 'OK') : 'not run',
              hint: 'A mismatch aborts the read. It does not repair anything.',
            },
          ]}
        />
      }
      note={
        !torn ? (
          nSectors === 1 ? (
            <>
              <strong>This page cannot tear.</strong> A {fmtBytes(pageBytes)} page on a{' '}
              {fmtBytes(sectorBytes)}-sector device is exactly one atomic write, so the platter ends up
              holding either the whole old image or the whole new one. That equivalence is the entire
              argument behind SQLite's <code>PRAGMA page_size=4096</code> with powersafe overwrite, and
              behind running InnoDB at <code>innodb_page_size=4k</code> on 4Kn media.
            </>
          ) : landedCount === 0 ? (
            <>
              <strong>Power failed before the first sector reached media.</strong> The page is the
              intact v1 image and its pageLSN is {hex(OLD_LSN, 32)}; redo replays cleanly on top of it.
              This is the case the WAL was designed for — and the only one it handles by itself.
            </>
          ) : (
            <>
              <strong>Every sector landed.</strong> The page is v2, pageLSN {hex(NEW_LSN, 32)}, checksum
              matches. Nothing about this outcome was promised by the device: the promise was made one
              sector at a time, and you happened to collect all {nSectors}.
            </>
          )
        ) : headerLanded ? (
          <>
            <strong>Torn, and the header sector landed.</strong> The page now advertises the{' '}
            <em>new</em> pageLSN {hex(NEW_LSN, 32)} while sector{staleSectors.length > 1 ? 's' : ''}{' '}
            {staleSectors.join(', ')} still hold v1 bytes. Redo compares each log record's LSN against
            the pageLSN and <em>skips</em> anything at or below it, so recovery would declare this page
            current and the stale bytes would live forever. The checksum is the only thing between you
            and silent corruption.
          </>
        ) : (
          <>
            <strong>Torn, and the header sector did not land.</strong> The page still advertises the old
            pageLSN {hex(OLD_LSN, 32)}, so redo will replay the records — but a physiological record
            says "insert this tuple into the free space at offset X", and it is about to be applied to a
            base image that is already half-new. The log describes a <em>delta</em> from an intact
            predecessor state, and this page is not one.
          </>
        )
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Sector</th>
              <th>Byte range in page</th>
              <th>Commit order</th>
              <th>Reached media</th>
              <th>Holds</th>
              <th>Carries</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: nSectors }, (_, s) => (
              <tr key={s}>
                <td>{s}</td>
                <td>
                  {s * sectorBytes}–{(s + 1) * sectorBytes - 1}
                </td>
                <td>{order.indexOf(s) + 1}</td>
                <td>{landedSet.has(s) ? 'yes' : 'no'}</td>
                <td>{landedSet.has(s) ? 'v2 (new)' : 'v1 (old)'}</td>
                <td>{s === 0 ? 'pd_checksum, pageLSN' : 'tuple data'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <TornFigure
            width={width}
            labelW={labelW}
            stripW={stripW}
            cellW={cellW}
            cellH={cellH}
            height={vfyY + 56}
            mediaY={mediaY}
            axisY={axisY}
            vfyY={vfyY}
            rows={rows}
            nSectors={nSectors}
            sectorBytes={sectorBytes}
            pageBytes={pageBytes}
            order={order}
            landedSet={landedSet}
            boundaryX={boundaryX}
            reorder={reorder}
            torn={torn}
            verified={verified}
            detected={detected}
            scheme={scheme}
            bits={bits}
            storedCk={model.storedCk}
            computedCk={model.computedCk}
          />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
