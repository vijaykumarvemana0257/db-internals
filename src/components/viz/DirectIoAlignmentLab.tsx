import { useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Button,
  Legend,
  Stats,
  Note,
  fmtBytes,
  useSize,
} from './Viz';

/**
 * The O_DIRECT kernel contract, run as a checker.
 *
 * Three things must be multiples of the device's logical block size: the user
 * buffer's virtual address, the file offset, and the length. Miss any one and
 * the syscall returns -1/EINVAL before a single byte moves. Satisfy all three
 * and the request becomes a DMA transfer straight into user memory, skipping
 * the page cache (and skipping readahead and cross-process page sharing with it).
 *
 * The 512e case is the subtle one: the syscall succeeds at 512-byte alignment
 * because the *logical* block is 512, but the NAND's *physical* block is 4 KB,
 * so the drive does a read-modify-write you never see in errno.
 */

type LbaMode = '512n' | '512e' | '4kn';

const LBA: Record<LbaMode, { label: string; logical: number; physical: number; note: string }> = {
  '512n': {
    label: '512n — logical 512 / physical 512',
    logical: 512,
    physical: 512,
    note: 'Legacy drives and most virtual disks. BLKSSZGET and BLKPBSZGET both report 512.',
  },
  '512e': {
    label: '512e — logical 512 / physical 4096',
    logical: 512,
    physical: 4096,
    note: 'The common SATA/SAS shipping default: the drive accepts 512-byte I/O but its media is 4 KB.',
  },
  '4kn': {
    label: '4Kn — logical 4096 / physical 4096',
    logical: 4096,
    physical: 4096,
    note: 'Native 4 KB NVMe namespaces (nvme format --lbaf). Anything not 4 KB-aligned is rejected outright.',
  },
};

const WINDOW = 16384; // bytes of address space drawn on each ruler
const BASE_ADDR = 0x7f9c4a200000; // a plausible mmap/posix_memalign base

const hex = (n: number) => '0x' + n.toString(16);

export default function DirectIoAlignmentLab() {
  const [mode, setMode] = useState<'direct' | 'buffered'>('direct');
  const [lba, setLba] = useState<LbaMode>('4kn');
  const [bufAddr, setBufAddr] = useState(64); // low bits of the buffer address
  const [fileOff, setFileOff] = useState(4096);
  const [len, setLen] = useState(4096);
  const [ref, width] = useSize(760);

  const dev = LBA[lba];
  const align = dev.logical;

  const fields = [
    { key: 'buf', label: 'Buffer address', value: bufAddr, shown: hex(BASE_ADDR + bufAddr), rem: bufAddr % align },
    { key: 'off', label: 'File offset', value: fileOff, shown: fmtBytes(fileOff), rem: fileOff % align },
    { key: 'len', label: 'Length', value: len, shown: fmtBytes(len), rem: len % align },
  ] as const;

  const bad = fields.filter((f) => f.rem !== 0);
  const einval = mode === 'direct' && bad.length > 0;
  const rmw =
    mode === 'direct' &&
    !einval &&
    dev.physical > dev.logical &&
    (bufAddr % dev.physical !== 0 || fileOff % dev.physical !== 0 || len % dev.physical !== 0);

  const status: 'einval' | 'rmw' | 'dma' | 'buffered' = einval
    ? 'einval'
    : mode === 'buffered'
      ? 'buffered'
      : rmw
        ? 'rmw'
        : 'dma';

  const statusColor =
    status === 'einval'
      ? 'var(--viz-critical)'
      : status === 'rmw'
        ? 'var(--viz-warning)'
        : status === 'dma'
          ? 'var(--viz-good)'
          : 'var(--viz-2)';

  /* ------------------------------------------------------------ geometry */
  const svgW = Math.max(660, width);
  const labelW = 152;
  const plotW = svgW - labelW - 96;
  const px = (bytes: number) => (bytes / WINDOW) * plotW;
  const height = 306;

  const ticks: number[] = [];
  const tickStep = align < 1024 && plotW / (WINDOW / align) < 14 ? align * 4 : align;
  for (let b = 0; b <= WINDOW; b += tickStep) ticks.push(b);

  function Ruler({
    y,
    label,
    start,
    startBad,
    lenBad,
    unit,
  }: {
    y: number;
    label: string;
    start: number;
    startBad: boolean;
    lenBad: boolean;
    unit: string;
  }) {
    const x0 = labelW + px(start);
    const w = Math.max(2, px(len));
    const failing = startBad || lenBad;
    return (
      <g>
        <text x={labelW - 10} y={y + 14} textAnchor="end" fill="var(--viz-ink)">
          {label}
        </text>
        <text x={labelW - 10} y={y + 27} textAnchor="end" fill="var(--viz-ink-muted)" fontSize={10}>
          {unit}
        </text>
        <line className="viz-axis-line" x1={labelW} x2={labelW + plotW} y1={y + 24} y2={y + 24} />
        {ticks.map((b) => (
          <line
            key={b}
            className="viz-grid-line"
            x1={labelW + px(b)}
            x2={labelW + px(b)}
            y1={y - 4}
            y2={y + 29}
          />
        ))}
        <rect
          x={x0}
          y={y}
          width={w}
          height={24}
          rx={3}
          fill={failing ? 'var(--viz-critical)' : 'var(--viz-1)'}
          opacity={0.85}
          stroke={failing ? 'var(--viz-critical)' : 'var(--viz-1)'}
          strokeWidth={1}
        />
        {startBad ? (
          <>
            <line
              x1={labelW + px(start - (start % align))}
              x2={x0}
              y1={y + 36}
              y2={y + 36}
              stroke="var(--viz-critical)"
              strokeWidth={2}
            />
            <text
              x={x0 + 4}
              y={y + 40}
              fill="var(--viz-critical)"
              fontSize={10}
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              starts {start % align} B past a block boundary
            </text>
          </>
        ) : null}
        {lenBad ? (
          <>
            <line
              x1={x0 + w}
              x2={x0 + w}
              y1={y - 6}
              y2={y + 32}
              stroke="var(--viz-critical)"
              strokeWidth={2}
            />
            <text
              x={Math.min(x0 + w + 6, labelW + plotW - 150)}
              y={y - 8}
              fill="var(--viz-critical)"
              fontSize={10}
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              length {len} % {align} = {len % align}
            </text>
          </>
        ) : null}
        <text
          x={labelW + plotW + 8}
          y={y + 16}
          fill="var(--viz-ink-2)"
          fontSize={10}
          style={{ fontVariantNumeric: 'tabular-nums' }}
        >
          {fmtBytes(len)}
        </text>
      </g>
    );
  }

  /* ---------------------------------------------------------- data path */
  const pathY = 214;
  const boxes = [
    { x: labelW, label: 'NVMe device', sub: 'DMA engine' },
    { x: labelW + 168, label: 'Page cache', sub: 'struct page, 4 KB' },
    { x: labelW + 348, label: 'User buffer', sub: 'buffer pool frame' },
  ];
  const boxW = 128;
  const boxH = 46;

  return (
    <VizPanel
      title="The O_DIRECT alignment contract"
      subtitle="Three things must be multiples of the logical block size: the buffer address, the file offset and the length. Break one and the syscall never reaches the device."
      controls={
        <>
          <Segmented
            label="open() flags"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'direct', label: 'O_DIRECT', title: 'Bypass the page cache; alignment is enforced' },
              { value: 'buffered', label: 'Buffered', title: 'Normal read()/pread(); the kernel copies through the page cache' },
            ]}
          />
          <Choice
            label="Device geometry"
            value={lba}
            onChange={setLba}
            options={(Object.keys(LBA) as LbaMode[]).map((k) => ({ value: k, label: LBA[k].label }))}
          />
          <Slider
            label="Buffer address"
            min={0}
            max={4096}
            step={64}
            value={bufAddr}
            onChange={setBufAddr}
            format={(n) => `…${(BASE_ADDR + n).toString(16).slice(-5)}`}
          />
          <Slider
            label="File offset"
            min={0}
            max={8192}
            step={64}
            value={fileOff}
            onChange={setFileOff}
            format={(n) => fmtBytes(n)}
          />
          <Slider
            label="Length"
            min={512}
            max={8192}
            step={64}
            value={len}
            onChange={setLen}
            format={(n) => fmtBytes(n)}
          />
          <Button
            onClick={() => {
              setBufAddr(Math.round(bufAddr / dev.physical) * dev.physical);
              setFileOff(Math.round(fileOff / dev.physical) * dev.physical);
              setLen(Math.max(dev.physical, Math.round(len / dev.physical) * dev.physical));
            }}
            title="What posix_memalign() and a page-sized I/O unit buy you"
          >
            Snap all three to {fmtBytes(dev.physical)}
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Request extent (aligned)', color: 'var(--viz-1)' },
            { label: 'Request extent (rejected)', color: 'var(--viz-critical)' },
            { label: 'Block boundaries', color: 'var(--viz-grid)', shape: 'line' },
            { label: 'Copy through the page cache', color: 'var(--viz-2)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'pread() returns',
              value: (
                <span style={{ color: statusColor }}>
                  {status === 'einval' ? '-1 · EINVAL' : `${len}`}
                </span>
              ),
              hint: 'Bytes read, or -1 with errno set',
            },
            {
              label: 'Alignment required',
              value: fmtBytes(align),
              hint: 'statx STATX_DIOALIGN reports this as dio_mem_align / dio_offset_align',
            },
            {
              label: 'Copies in RAM',
              value: mode === 'buffered' ? '2 (cache + pool)' : '1 (DMA)',
            },
            {
              label: 'Kernel readahead',
              value: mode === 'buffered' ? 'on, 128 KB' : 'none',
            },
          ]}
        />
      }
      note={
        <Note>
          {status === 'einval' ? (
            <>
              <strong>EINVAL.</strong> {bad.map((f) => f.label.toLowerCase()).join(' and ')}{' '}
              {bad.length > 1 ? 'are' : 'is'} not a multiple of {fmtBytes(align)} (
              {bad.map((f) => `${f.value} % ${align} = ${f.rem}`).join(', ')}). The kernel rejects the
              request in <code>iomap_dio_rw()</code> before any I/O is issued — nothing partial
              happens, and the error carries no hint about which field was wrong.
            </>
          ) : status === 'rmw' ? (
            <>
              <strong>Accepted, but the drive pays.</strong> Everything is 512-byte aligned, so the
              logical contract is satisfied — but this is a 512e device whose media block is 4 KB, so
              the controller must read a 4 KB physical block, merge your bytes and write it back. No
              errno, just latency and wear. This is why engines align to 4 KB even when 512 is legal.
            </>
          ) : status === 'dma' ? (
            <>
              <strong>Issued as DMA.</strong> The kernel pins your pages with{' '}
              <code>get_user_pages()</code>, builds a bio, and the controller writes straight into
              your buffer pool frame. Zero copies, zero page-cache pollution — and also zero
              readahead, so the engine now owns prefetch. It is still not durable: a write that
              lands in the drive's volatile cache needs <code>fdatasync()</code> or a FUA flag.
            </>
          ) : (
            <>
              <strong>Buffered: alignment is irrelevant.</strong> Any address, any offset, any length
              works — the kernel reads whole 4 KB pages into the page cache and{' '}
              <code>copy_to_user()</code>s the slice you asked for. You get readahead and sharing
              across processes, and you pay a second copy of every hot page in RAM.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Field</th>
              <th>Value</th>
              <th>Required multiple</th>
              <th>Remainder</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => (
              <tr key={f.key}>
                <td>{f.label}</td>
                <td>{f.shown}</td>
                <td>{fmtBytes(align)}</td>
                <td>{f.rem}</td>
                <td>
                  {mode === 'buffered'
                    ? 'not checked'
                    : f.rem === 0
                      ? f.value % dev.physical === 0
                        ? 'aligned'
                        : 'legal, but sub-physical-block'
                      : 'EINVAL'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <svg
          width={svgW}
          height={height}
          role="img"
          aria-label="O_DIRECT alignment checker showing buffer, offset and length against block boundaries"
        >
          <Ruler
            y={22}
            label="User buffer"
            unit={`virtual address · ${hex(BASE_ADDR + bufAddr)}`}
            start={bufAddr}
            startBad={mode === 'direct' && bufAddr % align !== 0}
            lenBad={mode === 'direct' && len % align !== 0}
          />
          <Ruler
            y={110}
            label="File / device LBA"
            unit={`offset ${fileOff} · LBA ${Math.floor(fileOff / align)}`}
            start={fileOff}
            startBad={mode === 'direct' && fileOff % align !== 0}
            lenBad={mode === 'direct' && len % align !== 0}
          />

          {/* data path */}
          <text x={labelW - 10} y={pathY + 28} textAnchor="end" fill="var(--viz-ink)">
            Data path
          </text>
          {boxes.map((b, i) => {
            const bypassed = mode === 'direct' && i === 1;
            return (
              <g key={b.label}>
                <rect
                  x={b.x}
                  y={pathY}
                  width={boxW}
                  height={boxH}
                  rx={6}
                  fill="var(--viz-plane)"
                  stroke={bypassed ? 'var(--viz-axis)' : 'var(--viz-ink-muted)'}
                  strokeWidth={1}
                  strokeDasharray={bypassed ? '4 3' : undefined}
                  opacity={bypassed ? 0.5 : 1}
                />
                <text
                  x={b.x + boxW / 2}
                  y={pathY + 20}
                  textAnchor="middle"
                  fill={bypassed ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}
                >
                  {b.label}
                </text>
                <text
                  x={b.x + boxW / 2}
                  y={pathY + 35}
                  textAnchor="middle"
                  fill="var(--viz-ink-muted)"
                  fontSize={10}
                >
                  {bypassed ? 'bypassed' : b.sub}
                </text>
              </g>
            );
          })}

          {mode === 'buffered' ? (
            <>
              <line
                x1={boxes[0].x + boxW}
                x2={boxes[1].x}
                y1={pathY + boxH / 2}
                y2={pathY + boxH / 2}
                stroke="var(--viz-2)"
                strokeWidth={2}
              />
              <line
                x1={boxes[1].x + boxW}
                x2={boxes[2].x}
                y1={pathY + boxH / 2}
                y2={pathY + boxH / 2}
                stroke="var(--viz-2)"
                strokeWidth={2}
              />
              <text x={boxes[0].x + boxW + 6} y={pathY + boxH / 2 - 6} fill="var(--viz-2)" fontSize={10}>
                DMA
              </text>
              <text x={boxes[1].x + boxW + 6} y={pathY + boxH / 2 - 6} fill="var(--viz-2)" fontSize={10}>
                copy_to_user
              </text>
            </>
          ) : (
            <path
              d={`M ${boxes[0].x + boxW} ${pathY + 12} Q ${boxes[1].x + boxW / 2} ${pathY - 34} ${boxes[2].x} ${pathY + 12}`}
              fill="none"
              stroke={einval ? 'var(--viz-critical)' : rmw ? 'var(--viz-warning)' : 'var(--viz-good)'}
              strokeWidth={2}
              strokeDasharray={einval ? '5 4' : undefined}
            />
          )}

          {mode === 'direct' ? (
            <text
              x={boxes[1].x + boxW / 2}
              y={pathY - 38}
              textAnchor="middle"
              fill={einval ? 'var(--viz-critical)' : rmw ? 'var(--viz-warning)' : 'var(--viz-good)'}
              fontSize={11}
            >
              {einval
                ? 'request rejected — EINVAL, no bio built'
                : rmw
                  ? 'DMA, but the drive read-modify-writes a 4 KB physical block'
                  : 'DMA straight into the pinned user buffer'}
            </text>
          ) : null}
        </svg>
      </div>
    </VizPanel>
  );
}
