import { useMemo, useState } from 'react';
import { VizPanel, Slider, Check, Button, Legend, Stats, Note, fmtBytes, fmtNum } from './Viz';

/**
 * Key-value separation as a byte model with stated assumptions:
 * - keys are 24 B; a blob reference in the LSM is about 16 B (file number, offset, size);
 * - the LSM's compactions rewrite each byte in it about LSM_WA times (a leveled tree);
 * - values at or above min_blob_size go to an append-only blob log and are written once, unless
 *   garbage collection relocates them;
 * - a range scan reads sorted data blocks sequentially, but each separated value is one random read.
 */
const KEY = 24;
const REF = 16;
const LSM_WA = 10;
const BLOCK = 4096;
const RECORDS = 1_000_000;

export function model(valueSize: number, minBlob: number, separation: boolean, overwritePct: number, gcRuns: number, scanKeys: number) {
  const separated = separation && valueSize >= minBlob;
  const userBytes = RECORDS * (KEY + valueSize);
  const lsmEntry = separated ? KEY + REF : KEY + valueSize;
  const lsmBytes = RECORDS * lsmEntry;
  const compactionBytes = lsmBytes * (LSM_WA - 1); // rewrites beyond the flush
  const blobBytesWritten = separated ? RECORDS * valueSize : 0;

  // Overwrites leave dead blobs behind; each GC run relocates live blobs out of the oldest files.
  const liveValueBytes = RECORDS * (1 - overwritePct / 100) * valueSize;
  const deadBlobBytes0 = separated ? RECORDS * (overwritePct / 100) * valueSize : 0;
  const reclaimPerRun = 0.25; // fraction of blob files old enough to collect (RocksDB's default age cutoff)
  let dead = deadBlobBytes0;
  let relocated = 0;
  for (let i = 0; i < gcRuns; i++) {
    const collectDead = dead * reclaimPerRun;
    const collectLive = liveValueBytes * reclaimPerRun * 0.5; // live blobs sharing those files get rewritten
    dead -= collectDead;
    relocated += collectLive;
  }
  const gcPointerUpdates = separated ? (relocated / Math.max(1, valueSize)) * (KEY + REF) : 0;

  const totalWritten = userBytes + compactionBytes + blobBytesWritten + relocated + gcPointerUpdates;
  const writeAmp = totalWritten / userBytes;

  const onDisk = lsmBytes + (separated ? liveValueBytes + dead : 0);
  const liveBytes = RECORDS * (1 - overwritePct / 100) * (KEY + valueSize);
  const spaceAmp = separated ? onDisk / Math.max(1, liveBytes) : 1; // LSM-side space amp handled by compaction

  // Range scan of scanKeys consecutive keys.
  const scanSeqBlocks = Math.ceil((scanKeys * lsmEntry) / BLOCK);
  const scanRandomReads = separated ? scanKeys : 0;
  return { separated, lsmEntry, compactionBytes, writeAmp, spaceAmp, dead, relocated, scanSeqBlocks, scanRandomReads, blobBytesWritten };
}

export default function KvSeparationLab() {
  const [valueSize, setValueSize] = useState(4096);
  const [minBlob, setMinBlob] = useState(1024);
  const [separation, setSeparation] = useState(true);
  const [overwrite, setOverwrite] = useState(40);
  const [gcRuns, setGcRuns] = useState(0);
  const [scanKeys, setScanKeys] = useState(100);

  const on = useMemo(() => model(valueSize, minBlob, separation, overwrite, gcRuns, scanKeys), [valueSize, minBlob, separation, overwrite, gcRuns, scanKeys]);
  const off = useMemo(() => model(valueSize, minBlob, false, overwrite, 0, scanKeys), [valueSize, minBlob, overwrite, scanKeys]);

  const W = 660;
  const maxComp = Math.max(on.compactionBytes, off.compactionBytes, 1);

  return (
    <VizPanel
      title="Key-value separation: moving values out of the LSM"
      subtitle="A million records. With separation on, values at or above min_blob_size go to an append-only blob log and the LSM keeps only keys and small references — so compaction stops rewriting the values."
      controls={
        <>
          <Check label="Separate values into blob files" checked={separation} onChange={(v) => { setSeparation(v); setGcRuns(0); }} />
          <Slider label="Value size" min={64} max={65536} step={64} value={valueSize} onChange={(v) => { setValueSize(v); setGcRuns(0); }} format={fmtBytes} />
          <Slider label="min_blob_size" min={0} max={65536} step={256} value={minBlob} onChange={(v) => { setMinBlob(v); setGcRuns(0); }} format={fmtBytes} />
          <Slider label="Values overwritten" min={0} max={90} step={10} value={overwrite} onChange={(v) => { setOverwrite(v); setGcRuns(0); }} format={(v) => `${v}%`} />
          <Slider label="Range scan length" min={10} max={1000} step={10} value={scanKeys} onChange={setScanKeys} format={(v) => `${v} keys`} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Compaction bytes', color: 'var(--viz-1)' },
            { label: 'Sequential block reads', color: 'var(--viz-3)' },
            { label: 'Random blob reads', color: 'var(--viz-2)' },
            { label: 'Live blob bytes', color: 'var(--viz-4)' },
            { label: 'Dead blob bytes', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Values in', value: on.separated ? 'blob log' : 'the LSM' },
            { label: 'LSM entry size', value: `${fmtNum(on.lsmEntry)} B` },
            { label: 'Compaction bytes', value: fmtBytes(on.compactionBytes), hint: `Without separation: ${fmtBytes(off.compactionBytes)}` },
            { label: 'Write amp (whole system)', value: `${fmtNum(on.writeAmp, 1)}×`, hint: `Without separation: ${fmtNum(off.writeAmp, 1)}×` },
            { label: `Scan of ${scanKeys} keys`, value: on.separated ? `${on.scanSeqBlocks} seq + ${on.scanRandomReads} random` : `${on.scanSeqBlocks} sequential` },
            { label: 'Blob space amp', value: on.separated ? `${fmtNum(on.spaceAmp, 2)}×` : '—' },
          ]}
        />
      }
      note={
        <Note>
          {!on.separated ? (
            <>
              <strong>{separation ? `Values of ${fmtBytes(valueSize)} are below min_blob_size (${fmtBytes(minBlob)}), so they stay in the LSM.` : 'Separation is off: values live in the LSM and move with every compaction.'}</strong> Compaction rewrites {fmtBytes(on.compactionBytes)} of data, and a scan of {scanKeys} keys reads {on.scanSeqBlocks} sequential blocks.
            </>
          ) : (
            <>
              <strong>
                Compaction now rewrites {fmtBytes(on.compactionBytes)} instead of {fmtBytes(off.compactionBytes)} — only keys and references move.
              </strong>{' '}
              The cost moved elsewhere: a scan of {scanKeys} keys reads {on.scanSeqBlocks} sequential block{on.scanSeqBlocks === 1 ? '' : 's'} of keys but then {on.scanRandomReads} random blob reads, where it used to read {off.scanSeqBlocks} sequential blocks.{' '}
              {overwrite > 0 ? `Overwrites have left ${fmtBytes(on.dead)} of dead blobs${gcRuns ? ` after ${gcRuns} garbage-collection run${gcRuns === 1 ? '' : 's'} that relocated ${fmtBytes(on.relocated)} of live values` : ' that no compaction of the LSM can reclaim — run garbage collection'}.` : ''}
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Value size</th>
              <th>Compaction bytes, in LSM</th>
              <th>Compaction bytes, separated</th>
              <th>Scan I/O, in LSM</th>
              <th>Scan I/O, separated</th>
            </tr>
          </thead>
          <tbody>
            {[128, 1024, 4096, 16384, 65536].map((v) => {
              const a = model(v, 0, false, overwrite, 0, scanKeys);
              const b = model(v, 0, true, overwrite, 0, scanKeys);
              return (
                <tr key={v}>
                  <td>{fmtBytes(v)}</td>
                  <td>{fmtBytes(a.compactionBytes)}</td>
                  <td>{fmtBytes(b.compactionBytes)}</td>
                  <td>{a.scanSeqBlocks} seq</td>
                  <td>
                    {b.scanSeqBlocks} seq + {b.scanRandomReads} random
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => setGcRuns((n) => n + 1)} disabled={!on.separated || overwrite === 0}>
          Run blob garbage collection
        </Button>
        <Button onClick={() => setGcRuns(0)} disabled={gcRuns === 0}>
          Reset GC
        </Button>
      </div>
      <svg width={W} height={210} role="img" aria-label="Compaction bytes, scan I/O and blob space">
        <text x={0} y={18} fill="var(--viz-ink)" fontSize={12}>Compaction rewrites</text>
        {[
          { label: 'values in the LSM', v: off.compactionBytes, y: 28 },
          { label: 'separated', v: on.compactionBytes, y: 52 },
        ].map((r) => (
          <g key={r.label}>
            <text x={0} y={r.y + 13} fontSize={11}>{r.label}</text>
            <rect x={130} y={r.y} width={Math.max(2, (r.v / maxComp) * (W - 260))} height={18} rx={4} fill="var(--viz-1)" />
            <text x={130 + Math.max(2, (r.v / maxComp) * (W - 260)) + 6} y={r.y + 13} fontSize={11}>{fmtBytes(r.v)}</text>
          </g>
        ))}

        <text x={0} y={104} fill="var(--viz-ink)" fontSize={12}>Range scan of {scanKeys} keys</text>
        {(() => {
          const y = 114;
          const unit = Math.max(2, Math.min(14, (W - 150) / Math.max(1, off.scanSeqBlocks, on.scanSeqBlocks + Math.min(on.scanRandomReads, 60))));
          return (
            <g>
              <text x={0} y={y + 12} fontSize={11}>values in the LSM</text>
              {Array.from({ length: Math.min(off.scanSeqBlocks, 60) }, (_, i) => (
                <rect key={i} x={130 + i * unit} y={y} width={unit - 1} height={14} fill="var(--viz-3)" />
              ))}
              <text x={0} y={y + 36} fontSize={11}>separated</text>
              {on.separated
                ? [
                    ...Array.from({ length: Math.min(on.scanSeqBlocks, 60) }, (_, i) => <rect key={`s${i}`} x={130 + i * unit} y={y + 24} width={unit - 1} height={14} fill="var(--viz-3)" />),
                    ...Array.from({ length: Math.min(on.scanRandomReads, 60) }, (_, i) => (
                      <rect key={`r${i}`} x={130 + (Math.min(on.scanSeqBlocks, 60) + i) * unit} y={y + 24 + (i % 3) * 3} width={unit - 1} height={8} fill="var(--viz-2)" />
                    )),
                  ]
                : null}
              {on.separated && on.scanRandomReads > 60 ? (
                <text x={130 + (Math.min(on.scanSeqBlocks, 60) + 60) * unit + 4} y={y + 36} fontSize={10}>
                  +{on.scanRandomReads - 60} more
                </text>
              ) : null}
            </g>
          );
        })()}

        <text x={0} y={188} fill="var(--viz-ink)" fontSize={12}>Blob log</text>
        {on.separated ? (
          (() => {
            const live = (1 - overwrite / 100) * valueSize * RECORDS;
            const total = live + on.dead;
            const w = W - 140;
            return (
              <g>
                <rect x={130} y={176} width={(live / total) * w} height={16} fill="var(--viz-4)" />
                <rect x={130 + (live / total) * w} y={176} width={(on.dead / total) * w} height={16} fill="var(--viz-stale)" />
                <text x={130} y={206} fontSize={10}>
                  live {fmtBytes(live)} · dead {fmtBytes(on.dead)}
                </text>
              </g>
            );
          })()
        ) : (
          <text x={130} y={188} fontSize={11}>not used</text>
        )}
      </svg>
    </VizPanel>
  );
}
