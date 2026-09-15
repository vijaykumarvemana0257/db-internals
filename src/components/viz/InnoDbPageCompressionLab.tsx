import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Button, Legend, Stats, Note, fmtNum, useTicker } from './Viz';
import {
  PAGE_SIZE,
  FIL_PAGE_DATA,
  PAGE_DATA,
  zipLoad,
  zipUpdate,
  zipMEnd,
  zipTrailer,
  punchLoad,
  loadRows,
  punchUpdate,
  type InnoCfg,
  type ZipPage,
  type PunchPage,
} from './blockCompressionModel';

/**
 * InnoDB's two page-compression designs on the same synthetic table.
 *
 * ROW_FORMAT=COMPRESSED: each 16 KB page is stored as a KEY_BLOCK_SIZE image laid out as in page0zip.ic —
 * uncompressed header, compressed stream, modification log (m_start..m_end), free space, and a trailer of
 * 15 bytes per record (dense directory slot + trx_id + roll_ptr). An update goes to the log if
 * page_zip_available() says it fits; otherwise the page is reorganized (recompressed from its uncompressed frame)
 * and checked again (btr_cur_update_alloc_zip in btr0cur.cc); otherwise the pessimistic path recompresses the page
 * with the record inside the stream (page_cur_insert_rec_zip in page0cur.cc); only if that fails does it split.
 *
 * COMPRESSION='lz4': os_file_compress_page() compresses the bytes after the 38-byte FIL header with
 * LZ4_compress_default, keeps the result only if it saves at least one filesystem block, rounds it up to whole
 * blocks, and punches a hole (FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE) over the rest of the 16 KB slot.
 *
 * Model assumptions, shown on screen: synthetic JSON-like rows; LZ4 sizes stand in for zlib in the
 * COMPRESSED mode; no padding; every update is flushed; split-off pages go to the end of the file.
 */

type Mode = 'zip' | 'punch';
type UpdateKind = 'same' | 'token';
type Event = { text: string; tone: 'ok' | 'work' | 'bad' };

const N_ROWS = 8 * 111;
const MAX_PAGES = 30;

export const cfgFor = (zipKB: 4 | 8, fsKB: 1 | 4, upd: UpdateKind): InnoCfg => ({ loadTok: 0, updateTok: upd === 'token' ? 32 : 0, zipSize: zipKB * 1024, fsBlock: fsKB * 1024 });
const LOGICAL_PAGES = loadRows(N_ROWS, cfgFor(8, 4, 'same')).length;

export default function InnoDbPageCompressionLab() {
  const [mode, setMode] = useState<Mode>('zip');
  const [zipKB, setZipKB] = useState<4 | 8>(8);
  const [fsKB, setFsKB] = useState<1 | 4>(4);
  const [upd, setUpd] = useState<UpdateKind>('same');
  const cfg = useMemo(() => cfgFor(zipKB, fsKB, upd), [zipKB, fsKB, upd]);

  const [zip, setZip] = useState<ZipPage[]>(() => zipLoad(N_ROWS, cfgFor(8, 4, 'same')));
  const [punch, setPunch] = useState<PunchPage[]>(() => punchLoad(N_ROWS, cfgFor(8, 4, 'same')));
  const [sel, setSel] = useState(0);
  const [cursor, setCursor] = useState<Record<number, number>>({});
  const [counts, setCounts] = useState({ mlog: 0, recompress: 0, split: 0, writes: 0, refills: 0, freed: 0 });
  const [events, setEvents] = useState<Event[]>([]);
  const [touched, setTouched] = useState<number[]>([]);
  const [grown, setGrown] = useState<{ page: number; from: number; to: number } | null>(null);
  const [reveal, setReveal] = useState(1e9);

  const animating = mode === 'punch' && reveal < punch.length + 1;
  useTicker((dt) => setReveal((r) => Math.min(punch.length + 1, r + dt / 110)), animating);

  const reset = (nextMode: Mode, c: InnoCfg) => {
    setZip(zipLoad(N_ROWS, c));
    setPunch(punchLoad(N_ROWS, c));
    setSel(0);
    setCursor({});
    setCounts({ mlog: 0, recompress: 0, split: 0, writes: 0, refills: 0, freed: 0 });
    setEvents([]);
    setTouched([]);
    setGrown(null);
    setReveal(nextMode === 'punch' ? 0 : 1e9);
  };

  const pagesNow = mode === 'zip' ? zip.length : punch.length;
  const selPage = Math.min(sel, pagesNow - 1);

  const updateOnce = (state: { zip: ZipPage[]; punch: PunchPage[]; cursor: Record<number, number>; counts: typeof counts; events: Event[] }) => {
    const p = selPage;
    const rowsOnPage = mode === 'zip' ? state.zip[p].rows.length : state.punch[p].rows.length;
    const ri = (state.cursor[p] ?? 0) % rowsOnPage;
    const nextCursor = { ...state.cursor, [p]: (state.cursor[p] ?? 0) + 1 };
    if (mode === 'zip') {
      const before = state.zip[p];
      const r = zipUpdate(state.zip, p, ri, cfg);
      const after = r.pages[p];
      const c = { ...state.counts, writes: state.counts.writes + 1 };
      let ev: Event;
      if (r.outcome === 'mlog') {
        c.mlog++;
        ev = { tone: 'ok', text: `Row ${before.rows[ri].id}: the ${r.entry}-byte record is appended to page ${p}'s modification log (m_end ${fmtNum(zipMEnd(before))} → ${fmtNum(zipMEnd(after))}; the trailer starts at ${fmtNum(cfg.zipSize - zipTrailer(after))}). No recompression.` };
      } else if (r.outcome === 'reorganize') {
        c.recompress += r.recompressions;
        ev = { tone: 'work', text: `Row ${before.rows[ri].id}: the log has no room, so page ${p} is reorganized: rebuilt from its uncompressed copy and recompressed with an empty log (${fmtNum(before.compressed)} → ${fmtNum(after.compressed)} bytes). The ${r.entry}-byte record then fits the log. With innodb_log_compressed_pages=ON the recompressed image also goes to the redo log.` };
      } else if (r.outcome === 'repack') {
        c.recompress += r.recompressions;
        ev = { tone: 'work', text: `Row ${before.rows[ri].id}: even an empty log cannot take the ${r.entry}-byte uncompressed record, so the update takes the pessimistic path: the record goes into the uncompressed page and page ${p} is compressed again with the record inside the stream (${fmtNum(before.compressed)} → ${fmtNum(after.compressed)} bytes). That fits; the log stays empty.` };
      } else {
        c.split++;
        c.recompress += r.recompressions;
        ev = {
          tone: 'bad',
          text:
            r.reason === 'compression-failure'
              ? `Row ${before.rows[ri].id}: page ${p} does not compress into ${zipKB} KB even with the record inside the stream — a compression failure. The B-tree page splits; the right half becomes page ${r.touched.slice(1).join(', ')} at the end of the file.`
              : `Row ${before.rows[ri].id}: the records no longer fit a 16 KB uncompressed page, so page ${p} splits like any B-tree page; the right half becomes page ${r.touched.slice(1).join(', ')}.`,
        };
      }
      return { zip: r.pages, punch: state.punch, cursor: nextCursor, counts: c, events: [ev, ...state.events].slice(0, 5), touched: r.touched, grown: null };
    }
    const before = state.punch[p];
    const r = punchUpdate(state.punch, p, ri, cfg);
    const after = r.pages[p];
    const c = { ...state.counts, writes: state.counts.writes + 1 };
    let ev: Event;
    let grownNow: { page: number; from: number; to: number } | null = null;
    if (r.split) {
      c.split++;
      ev = { tone: 'bad', text: `Row ${before.rows[ri].id}: page ${p} is full and splits; both halves are compressed and written, the right half as page ${r.touched.slice(1).join(', ')}. Page ${p} now needs ${after.blocks} block${after.blocks === 1 ? '' : 's'} instead of ${before.blocks}.` };
    } else if (after.blocks > before.blocks) {
      c.refills += after.blocks - before.blocks;
      grownNow = { page: p, from: before.blocks, to: after.blocks };
      ev = { tone: 'bad', text: `Row ${before.rows[ri].id}: page ${p} recompresses to ${fmtNum(after.lz4 + FIL_PAGE_DATA)} bytes and now needs ${after.blocks} blocks, not ${before.blocks}. The filesystem must allocate ${after.blocks - before.blocks} new block${after.blocks - before.blocks === 1 ? '' : 's'} inside its punched range, placed wherever the filesystem has free space rather than next to the page's other blocks.` };
    } else if (after.blocks < before.blocks) {
      c.freed += before.blocks - after.blocks;
      ev = { tone: 'work', text: `Row ${before.rows[ri].id}: page ${p} recompresses smaller and fits ${after.blocks} blocks; the write punches a larger hole.` };
    } else {
      ev = { tone: 'work', text: `Row ${before.rows[ri].id}: page ${p} is recompressed whole (${fmtNum(before.lz4)} → ${fmtNum(after.lz4)} bytes of LZ4) and rewritten into the same ${after.blocks} block${after.blocks === 1 ? '' : 's'}; the rest of the slot is punched again.` };
    }
    return { zip: state.zip, punch: r.pages, cursor: nextCursor, counts: c, events: [ev, ...state.events].slice(0, 5), touched: r.touched, grown: grownNow };
  };

  const doUpdates = (n: number) => {
    let s = { zip, punch, cursor, counts, events };
    let t: number[] = [];
    let g: { page: number; from: number; to: number } | null = null;
    for (let i = 0; i < n; i++) {
      if ((mode === 'zip' ? s.zip.length : s.punch.length) >= MAX_PAGES) break;
      const r = updateOnce(s);
      s = r;
      t = r.touched;
      g = r.grown ?? g;
    }
    setZip(s.zip);
    setPunch(s.punch);
    setCursor(s.cursor);
    setCounts(s.counts);
    setEvents(s.events);
    setTouched(t);
    setGrown(g);
    setReveal(1e9);
  };

  /* ------------------------------------------------ geometry */
  const W = 680;
  const LEFT = 64;
  const RIGHT = 12;
  const bytePx = (W - LEFT - RIGHT) / PAGE_SIZE;
  const ROW_H = 13;
  const GAP = 3;
  const rowY = (i: number) => 22 + i * (ROW_H + GAP);
  const H = rowY(pagesNow) + 10;

  const zipFileBytes = zip.length * cfg.zipSize;
  const punchFile = punch.length * PAGE_SIZE;
  const punchAlloc = punch.reduce((s, p) => s + p.stored, 0);
  const holes = punch.reduce((s, p) => s + (PAGE_SIZE - p.stored) / cfg.fsBlock, 0);
  const rawPages = punch.filter((p) => p.raw).length;
  const sp = mode === 'zip' ? zip[selPage] : null;
  const pp = mode === 'punch' ? punch[selPage] : null;
  const atCap = pagesNow >= MAX_PAGES;
  const last = events[0];

  const blockTicks = (y: number, width: number, step: number) =>
    Array.from({ length: Math.floor(width / step) - 1 }, (_, k) => (
      <line key={k} x1={LEFT + (k + 1) * step * bytePx} x2={LEFT + (k + 1) * step * bytePx} y1={y} y2={y + ROW_H} stroke="var(--viz-surface)" strokeWidth={1.2} />
    ));

  return (
    <VizPanel
      title="Two ways InnoDB compresses a 16 KB page"
      subtitle="The same 888-row table stored with ROW_FORMAT=COMPRESSED or with transparent page compression. Pick a page, update rows on it, and watch where the bytes go: into the modification log, a recompressed image, a split — or a different number of filesystem blocks around a punched hole."
      controls={
        <>
          <Segmented
            label="Storage"
            value={mode}
            onChange={(m) => {
              setMode(m);
              reset(m, cfg);
            }}
            options={[
              { value: 'zip', label: 'COMPRESSED pages', title: 'ROW_FORMAT=COMPRESSED' },
              { value: 'punch', label: "Hole punching, COMPRESSION='lz4'", title: "Transparent page compression: COMPRESSION='lz4'" },
            ]}
          />
          {mode === 'zip' ? (
            <Segmented
              label="KEY_BLOCK_SIZE"
              value={String(zipKB)}
              onChange={(v) => {
                const z = Number(v) as 4 | 8;
                setZipKB(z);
                reset(mode, cfgFor(z, fsKB, upd));
              }}
              options={[
                { value: '8', label: '8 KB (default)' },
                { value: '4', label: '4 KB' },
              ]}
            />
          ) : (
            <Segmented
              label="Filesystem block"
              value={String(fsKB)}
              onChange={(v) => {
                const f = Number(v) as 1 | 4;
                setFsKB(f);
                reset(mode, cfgFor(zipKB, f, upd));
              }}
              options={[
                { value: '4', label: '4 KB' },
                { value: '1', label: '1 KB' },
              ]}
            />
          )}
          <Choice
            label="Each update"
            value={upd}
            onChange={(v) => {
              setUpd(v);
              reset(mode, cfgFor(zipKB, fsKB, v));
            }}
            options={[
              { value: 'same', label: 'changes a number (same size)' },
              { value: 'token', label: 'adds a 32-char random session token' },
            ]}
          />
          <Choice label="Page to update" value={String(selPage)} onChange={(v) => setSel(Number(v))} options={Array.from({ length: pagesNow }, (_, i) => ({ value: String(i), label: `page ${i}` }))} />
        </>
      }
      legend={
        mode === 'zip' ? (
          <Legend
            items={[
              { label: 'Compressed stream', color: 'var(--viz-1)' },
              { label: 'Modification log', color: 'var(--viz-2)' },
              { label: 'Trailer: directory + trx_id + roll_ptr', color: 'var(--viz-3)' },
              { label: 'Uncompressed page header', color: 'var(--viz-4)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Compressed page data', color: 'var(--viz-1)' },
              { label: 'FIL header (38 B, uncompressed)', color: 'var(--viz-4)' },
              { label: 'Punched hole (no blocks allocated)', color: 'var(--viz-ink-muted)', shape: 'line' },
              { label: 'Written raw: compression saved < 1 block', color: 'var(--viz-serious)' },
              { label: 'Block allocated back into a hole', color: 'var(--viz-critical)', shape: 'line' },
            ]}
          />
        )
      }
      stats={
        mode === 'zip' ? (
          <Stats
            items={[
              { label: 'File size', value: `${fmtNum(zipFileBytes / 1024)} KB`, hint: 'Pages × KEY_BLOCK_SIZE. The file is dense: no holes.' },
              { label: 'Pages', value: fmtNum(zip.length) },
              { label: 'Updates into the log', value: fmtNum(counts.mlog) },
              { label: 'Recompressions', value: fmtNum(counts.recompress), hint: 'Each one costs zlib CPU and, with innodb_log_compressed_pages=ON, a full page image in the redo log.' },
              { label: 'Splits', value: fmtNum(counts.split) },
              { label: `Page ${selPage}: log room`, value: sp ? `${fmtNum(Math.max(0, cfg.zipSize - zipMEnd(sp) - zipTrailer(sp)))} B` : '—', hint: 'Free bytes between m_end and the trailer.' },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'FILE_SIZE (ls -l)', value: `${fmtNum(punchFile / 1024)} KB` },
              { label: 'ALLOCATED_SIZE (du)', value: `${fmtNum(punchAlloc / 1024)} KB (${fmtNum((100 * punchAlloc) / punchFile, 1)}%)` },
              { label: 'Punched blocks', value: fmtNum(holes) },
              { label: 'Page writes (each recompresses)', value: fmtNum(counts.writes) },
              { label: 'Blocks allocated into holes', value: fmtNum(counts.refills), hint: 'Each is a new extent the filesystem hands out from wherever it has space.' },
              { label: 'Pages written raw', value: fmtNum(rawPages) },
            ]}
          />
        )
      }
      note={
        <Note>
          {last ? (
            <>
              <strong>{last.tone === 'ok' ? 'Fits the modification log.' : last.tone === 'work' ? (mode === 'zip' ? 'Recompressed.' : 'Recompressed and rewritten.') : mode === 'zip' ? 'Split.' : grown ? 'The page grew into its hole.' : 'Split.'}</strong> {last.text}
            </>
          ) : mode === 'zip' ? (
            <>
              <strong>
                {zip.length} pages × {zipKB} KB = {fmtNum(zipFileBytes / 1024)} KB for data that fills {LOGICAL_PAGES} uncompressed 16 KB pages ({fmtNum((LOGICAL_PAGES * 16384) / 1024)} KB).
              </strong>{' '}
              Page {selPage} holds a {fmtNum(sp?.compressed ?? 0)}-byte compressed stream and a {fmtNum(sp ? zipTrailer(sp) : 0)}-byte trailer, leaving {fmtNum(sp ? Math.max(0, cfg.zipSize - zipMEnd(sp) - zipTrailer(sp)) : 0)} bytes for the modification log. Press <em>Update a row</em>.
            </>
          ) : (
            <>
              <strong>
                ls -l says {fmtNum(punchFile / 1024)} KB; du says {fmtNum(punchAlloc / 1024)} KB.
              </strong>{' '}
              Each page was compressed on write, rounded up to whole {fsKB} KB blocks, and the rest of its 16 KB slot punched out. Page {selPage} compresses to {fmtNum((pp?.lz4 ?? 0) + FIL_PAGE_DATA)} bytes and occupies {pp?.blocks} block{pp?.blocks === 1 ? '' : 's'}.
            </>
          )}
          {atCap ? ' The table has reached the lab’s page limit; press Reset.' : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            {mode === 'zip' ? (
              <tr>
                <th>Page</th>
                <th>Rows</th>
                <th>Compressed stream</th>
                <th>Modification log</th>
                <th>Trailer</th>
                <th>Free in {zipKB} KB</th>
              </tr>
            ) : (
              <tr>
                <th>Page</th>
                <th>Rows</th>
                <th>LZ4 bytes</th>
                <th>Written</th>
                <th>Blocks</th>
                <th>Punched</th>
              </tr>
            )}
          </thead>
          <tbody>
            {mode === 'zip'
              ? zip.map((p, i) => (
                  <tr key={i}>
                    <td>{i}</td>
                    <td>{p.rows.length}</td>
                    <td>{fmtNum(p.compressed)}</td>
                    <td>{fmtNum(p.mlog)}</td>
                    <td>{fmtNum(zipTrailer(p))}</td>
                    <td>{fmtNum(cfg.zipSize - zipMEnd(p) - zipTrailer(p))}</td>
                  </tr>
                ))
              : punch.map((p, i) => (
                  <tr key={i}>
                    <td>{i}</td>
                    <td>{p.rows.length}</td>
                    <td>{fmtNum(p.lz4)}</td>
                    <td>{p.raw ? 'raw 16 KB' : `${fmtNum(p.stored)} B`}</td>
                    <td>{p.blocks}</td>
                    <td>{fmtNum(PAGE_SIZE - p.stored)} B</td>
                  </tr>
                ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => doUpdates(1)} disabled={atCap}>
          Update a row on page {selPage}
        </Button>
        <Button onClick={() => doUpdates(10)} disabled={atCap}>
          Update 10 rows
        </Button>
        <Button onClick={() => reset(mode, cfg)}>Reset</Button>
        {mode === 'punch' ? <Button onClick={() => setReveal(0)}>Replay the initial writes</Button> : null}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={mode === 'zip' ? `Compressed pages: ${zip.length} pages of ${zipKB} KB` : `Sparse data file: ${fmtNum(punchAlloc / 1024)} KB allocated of ${fmtNum(punchFile / 1024)} KB`}>
        <text x={LEFT} y={12} fontSize={10}>
          0
        </text>
        <text x={LEFT + (PAGE_SIZE / 2) * bytePx} y={12} fontSize={10} textAnchor="middle">
          8 KB
        </text>
        <text x={W - RIGHT} y={12} fontSize={10} textAnchor="end">
          16 KB of logical page
        </text>
        {mode === 'zip'
          ? zip.map((p, i) => {
              const y = rowY(i);
              const zw = cfg.zipSize * bytePx;
              const hdr = PAGE_DATA * bytePx;
              const comp = p.compressed * bytePx;
              const log = p.mlog * bytePx;
              const trl = zipTrailer(p) * bytePx;
              const isSel = i === selPage;
              return (
                <g key={i} onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>
                  <text x={LEFT - 8} y={y + 10} textAnchor="end" fontSize={10} fill={isSel ? 'var(--viz-ink)' : 'var(--viz-ink-2)'} fontWeight={isSel ? 700 : 400}>
                    page {i}
                  </text>
                  <rect x={LEFT} y={y} width={zw} height={ROW_H} fill="var(--viz-plane)" stroke="var(--viz-axis)" />
                  <rect x={LEFT} y={y} width={hdr} height={ROW_H} fill="var(--viz-4)" />
                  <rect x={LEFT + hdr} y={y} width={comp} height={ROW_H} fill="var(--viz-1)" />
                  <rect x={LEFT + hdr + comp} y={y} width={log} height={ROW_H} fill="var(--viz-2)" />
                  <rect x={LEFT + zw - trl} y={y} width={trl} height={ROW_H} fill="var(--viz-3)" />
                  {blockTicks(y, cfg.zipSize, 4096)}
                  {touched.includes(i) || isSel ? <rect x={LEFT - 2} y={y - 2} width={zw + 4} height={ROW_H + 4} fill="none" stroke={touched.includes(i) && last?.tone === 'bad' ? 'var(--viz-critical)' : 'var(--viz-ink)'} strokeWidth={isSel ? 1.8 : 1.2} /> : null}
                  {i === 0 ? (
                    <text x={LEFT + zw + 8} y={y + 10} fontSize={10}>
                      ← {zipKB} KB on disk; nothing past it
                    </text>
                  ) : null}
                </g>
              );
            })
          : punch.map((p, i) => {
              const y = rowY(i);
              const written = i < reveal;
              const blocks = written ? p.blocks : PAGE_SIZE / cfg.fsBlock;
              const allocW = blocks * cfg.fsBlock * bytePx;
              const dataW = written ? (p.raw ? PAGE_SIZE : p.lz4 + FIL_PAGE_DATA) * bytePx : PAGE_SIZE * bytePx;
              const isSel = i === selPage;
              const g = grown && grown.page === i ? grown : null;
              return (
                <g key={i} onClick={() => setSel(i)} style={{ cursor: 'pointer' }}>
                  <text x={LEFT - 8} y={y + 10} textAnchor="end" fontSize={10} fill={isSel ? 'var(--viz-ink)' : 'var(--viz-ink-2)'} fontWeight={isSel ? 700 : 400}>
                    page {i}
                  </text>
                  <rect x={LEFT} y={y} width={allocW} height={ROW_H} fill="var(--viz-plane)" stroke="var(--viz-axis)" />
                  {written && !p.raw ? (
                    <>
                      <rect x={LEFT} y={y} width={FIL_PAGE_DATA * bytePx} height={ROW_H} fill="var(--viz-4)" />
                      <rect x={LEFT + FIL_PAGE_DATA * bytePx} y={y} width={Math.max(0, dataW - FIL_PAGE_DATA * bytePx)} height={ROW_H} fill="var(--viz-1)" />
                    </>
                  ) : (
                    <rect x={LEFT} y={y} width={dataW} height={ROW_H} fill={written ? 'var(--viz-serious)' : 'var(--viz-1)'} opacity={written ? 1 : 0.35} />
                  )}
                  {Array.from({ length: PAGE_SIZE / cfg.fsBlock - blocks }, (_, k) => {
                    const x = LEFT + (blocks + k) * cfg.fsBlock * bytePx;
                    return <rect key={k} x={x + 1} y={y + 1} width={cfg.fsBlock * bytePx - 2} height={ROW_H - 2} fill="none" stroke="var(--viz-ink-muted)" strokeDasharray="3 2" />;
                  })}
                  {blockTicks(y, blocks * cfg.fsBlock, cfg.fsBlock)}
                  {g
                    ? Array.from({ length: g.to - g.from }, (_, k) => (
                        <rect key={`g${k}`} x={LEFT + (g.from + k) * cfg.fsBlock * bytePx} y={y - 2} width={cfg.fsBlock * bytePx} height={ROW_H + 4} fill="none" stroke="var(--viz-critical)" strokeWidth={2} />
                      ))
                    : null}
                  {isSel || touched.includes(i) ? <rect x={LEFT - 2} y={y - 2} width={W - LEFT - RIGHT + 4} height={ROW_H + 4} fill="none" stroke={touched.includes(i) && last?.tone === 'bad' && !g ? 'var(--viz-critical)' : 'var(--viz-ink)'} strokeWidth={isSel ? 1.4 : 1} strokeOpacity={0.7} /> : null}
                </g>
              );
            })}
      </svg>
      {events.length > 1 ? (
        <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.2rem', fontSize: '0.78rem', display: 'grid', gap: 3 }}>
          {events.slice(1).map((e, i) => (
            <li key={i} style={{ color: 'var(--viz-ink-2)', borderLeft: `3px solid ${e.tone === 'bad' ? 'var(--viz-critical)' : e.tone === 'work' ? 'var(--viz-warning)' : 'transparent'}`, paddingLeft: 6 }}>
              {e.text}
            </li>
          ))}
        </ol>
      ) : null}
      <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '6px 0 0' }}>
        Model: synthetic JSON-like rows loaded to 15/16 of each 16 KB page; compressed sizes come from this page’s LZ4 implementation (exactly what COMPRESSION='lz4' calls; a stand-in for zlib under ROW_FORMAT=COMPRESSED, where real pages would be smaller); no compression padding; every update is flushed before the next.
      </p>
    </VizPanel>
  );
}
