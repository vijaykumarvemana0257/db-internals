import { useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Check,
  Button,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtTime,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * The durability contract, one layer at a time.
 *
 * Four layers can hold a copy of the same 8 KB record: the process's own buffer, the OS
 * page cache, the drive's DRAM write cache, and the media. Every syscall moves bytes
 * between exactly two of them, and a power cut deletes exactly the layers that are
 * volatile. The fsyncgate path is the one place the model breaks its own invariant: a
 * failed writeback clears PG_dirty and discards the page, so the *retried* fsync() finds
 * nothing left to do and returns 0 with the data gone.
 */

const PAGE_BYTES = 8192;
const MAX_RECS = 8;

/* --------------------------------------------------------------- the device */

type DevId = 'hdd' | 'sata' | 'nvme';

type Device = {
  id: DevId;
  label: string;
  writeNs: number; // buffered write(): a memcpy into a page-cache page
  wbNs: number; // writeback: DMA to the device plus the device's own acknowledgement
  flushVolNs: number; // REQ_OP_FLUSH with a volatile cache: the drive must reach media
  flushNvNs: number; // REQ_OP_FLUSH with a power-safe cache: the drive can ack at once
};

const DEVICES: Device[] = [
  { id: 'hdd', label: 'HDD, 7200 rpm', writeNs: 1_600, wbNs: 220_000, flushVolNs: 10_000_000, flushNvNs: 150_000 },
  { id: 'sata', label: 'SATA SSD (consumer)', writeNs: 1_300, wbNs: 90_000, flushVolNs: 900_000, flushNvNs: 60_000 },
  { id: 'nvme', label: 'NVMe SSD', writeNs: 900, wbNs: 20_000, flushVolNs: 250_000, flushNvNs: 12_000 },
];

/** Deterministic per-step jitter, so the latency log looks like a device and not a spreadsheet. */
const JITTER = (() => {
  const rng = makeRng(20180328); // the day fsyncgate landed on pgsql-hackers
  return Array.from({ length: 64 }, () => 0.88 + rng() * 0.3);
})();

/* ---------------------------------------------------------------- the state */

type Stage = 1 | 2 | 3; // 1 = page cache, 2 = drive write cache, 3 = media

type Rec = {
  id: number;
  stage: Stage;
  dirty: boolean; // the page-cache copy is dirty (PG_dirty set)
  cached: boolean; // a page-cache copy exists at all (false for O_DIRECT)
  dropped: boolean; // writeback failed: page marked clean and discarded
  lost: boolean; // vanished in a power cut
};

type LogRow = {
  n: number;
  op: string;
  ret: string;
  ns: number;
  dirty: number;
  dwc: number;
  media: number;
  gone: number;
};

type S = {
  recs: Rec[];
  nextId: number;
  procDead: boolean;
  sizeDirty: boolean; // i_size changed and is not on media
  sizeOnMedia: boolean;
  mtimeDirty: boolean; // mtime/ctime changed and is not on media
  mtimeOnMedia: boolean;
  errArmed: boolean; // the next writeback will fail
  errPending: boolean; // an EIO is recorded on the inode and not yet reported to anyone
  errSeen: boolean;
  panicked: boolean;
  crashed: boolean;
  head: string;
  body: string;
  log: LogRow[];
};

const INITIAL: S = {
  recs: [],
  nextId: 1,
  procDead: false,
  sizeDirty: false,
  sizeOnMedia: true,
  mtimeDirty: false,
  mtimeOnMedia: true,
  errArmed: false,
  errPending: false,
  errSeen: false,
  panicked: false,
  crashed: false,
  head: 'Nothing written yet.',
  body:
    'Press write() to copy an 8 KB WAL record into the page cache, then walk it down the stack one ' +
    'syscall at a time. Only what sits below the dashed boundary survives a power cut.',
  log: [],
};

type Cfg = { dev: Device; plp: boolean; append: boolean; onError: 'panic' | 'retry' };

type Op = 'write' | 'writeback' | 'flush' | 'fdatasync' | 'fsync' | 'fua' | 'inject' | 'kill' | 'power';

const LABEL: Record<Op, string> = {
  write: 'write()',
  writeback: 'writeback',
  flush: 'FLUSH',
  fdatasync: 'fdatasync()',
  fsync: 'fsync()',
  fua: 'O_DIRECT|O_DSYNC',
  inject: 'inject EIO',
  kill: 'SIGKILL',
  power: 'power cut',
};

function tally(recs: Rec[]) {
  let dirty = 0;
  let dwc = 0;
  let media = 0;
  let gone = 0;
  for (const r of recs) {
    if (r.dropped || r.lost) gone++;
    else if (r.stage === 3) media++;
    else if (r.stage === 2) dwc++;
    else dirty++;
  }
  return { dirty, dwc, media, gone };
}

function flushCost(cfg: Cfg) {
  return cfg.plp ? cfg.dev.flushNvNs : cfg.dev.flushVolNs;
}

/** What one durable commit costs at this config: data out, journal if the file grew, barrier. */
function syncCost(cfg: Cfg, kind: 'fdatasync' | 'fsync') {
  const f = flushCost(cfg);
  const journal = kind === 'fsync' ? f : cfg.append ? f : 0;
  return cfg.dev.wbNs + journal + f;
}

function apply(s: S, op: Op, cfg: Cfg): S {
  const step = s.log.length;
  const j = JITTER[step % JITTER.length];
  const f = flushCost(cfg);
  const live = s.recs.filter((r) => !r.dropped).length;

  let recs = s.recs;
  let ret = '—';
  let ns = 0;
  let head = '';
  let body = '';
  const next = { ...s };

  /** Hand every dirty page to the device. Returns true if the device rejected them. */
  const doWriteback = (): boolean => {
    const targets = recs.filter((r) => r.stage === 1 && r.dirty && !r.dropped && !r.lost);
    if (targets.length === 0) return false;
    if (next.errArmed) {
      recs = recs.map((r) =>
        r.stage === 1 && r.dirty && !r.dropped && !r.lost ? { ...r, dirty: false, dropped: true } : r,
      );
      next.errArmed = false;
      next.errPending = true;
      return true;
    }
    recs = recs.map((r) =>
      r.stage === 1 && r.dirty && !r.dropped && !r.lost ? { ...r, stage: 2 as Stage, dirty: false } : r,
    );
    return false;
  };

  switch (op) {
    case 'write': {
      if (live >= MAX_RECS) return s;
      recs = [...recs, { id: s.nextId, stage: 1, dirty: true, cached: true, dropped: false, lost: false }];
      next.nextId = s.nextId + 1;
      next.procDead = false;
      next.mtimeDirty = true;
      next.mtimeOnMedia = false;
      if (cfg.append) {
        next.sizeDirty = true;
        next.sizeOnMedia = false;
      }
      ns = cfg.dev.writeNs * j;
      ret = String(PAGE_BYTES);
      head = 'write() returned — and promised almost nothing.';
      body =
        'The kernel copied 8 KB into a page-cache page, set PG_dirty, bumped i_size and mtime in the ' +
        'in-memory inode, and returned the byte count. No byte has left DRAM. The only failure this ' +
        'survives is a crash of your own process.';
      break;
    }

    case 'writeback': {
      const failed = doWriteback();
      ns = cfg.dev.wbNs * j;
      ret = 'no caller';
      if (failed) {
        head = 'Writeback failed — and the page is now clean and empty.';
        body =
          'The device rejected the I/O. Linux marked those pages clean, dropped them, and recorded one ' +
          'EIO on the inode. Nobody has been told: writeback is asynchronous, so there is no caller to ' +
          'return an error to. The next fsync() on each open fd gets that error exactly once.';
      } else {
        head = 'The bytes reached the drive. They are still not durable.';
        body =
          'The flusher thread (or your own sync_file_range) DMA-ed the dirty pages to the device and ' +
          'cleared PG_dirty. The drive acknowledged them into its own DRAM. A power cut here loses them ' +
          'unless that cache is power-loss protected — and the page cache no longer holds a dirty copy to retry with.';
      }
      break;
    }

    case 'flush': {
      const moved = recs.filter((r) => r.stage === 2 && !r.lost).length;
      recs = recs.map((r) => (r.stage === 2 && !r.lost ? { ...r, stage: 3 as Stage } : r));
      ns = (moved > 0 ? f : 2_000) * j;
      ret = '0';
      if (moved > 0) {
        head = 'FLUSH: the drive emptied its write cache onto the media.';
        body =
          `A REQ_OP_FLUSH — SATA FLUSH CACHE, NVMe Flush, SCSI SYNCHRONIZE CACHE — tells the drive not to ` +
          `acknowledge until every cached write is on the media. ${moved} record${moved === 1 ? '' : 's'} ` +
          `crossed the power-loss boundary, and it cost ${fmtTime(f)}. This barrier is what fsync() ends with; ` +
          `issue it by hand and you can see exactly where the time goes.`;
      } else {
        head = 'A FLUSH with an empty cache is nearly free.';
        body =
          'Nothing was queued in the drive, so the command returned immediately. ext4 skips the barrier ' +
          'entirely when no data and no journal commit are pending, which is why fsync() on an untouched ' +
          'file costs microseconds and tells you nothing about your real commit cost.';
      }
      break;
    }

    case 'fdatasync':
    case 'fsync': {
      const isFsync = op === 'fsync';
      const hadDirty = recs.some((r) => r.stage === 1 && r.dirty && !r.dropped && !r.lost);
      const failed = doWriteback();

      if (failed || next.errPending) {
        next.errPending = false;
        next.errSeen = true;
        ns = cfg.dev.wbNs * j;
        ret = '-1 EIO';
        head = `${LABEL[op]} returned -1, errno = EIO.`;
        body =
          (failed
            ? 'Those pages are gone. The kernel cleaned and discarded them before reporting, so there is ' +
              'nothing left to retry and no way to reconstruct them from the page cache. '
            : 'This error was recorded on the inode by a flusher thread you never called. errseq_t hands it ' +
              'to the next fsync() on each fd that had the file open — once each — and the bytes are already gone. ') +
          (cfg.onError === 'panic'
            ? 'Postgres 12+ treats this as unrecoverable: PANIC, "could not fdatasync file", the server restarts ' +
              'and replays the WAL from the last checkpoint that actually made it to disk.'
            : 'With data_sync_retry = on the engine retries. Press fsync() again and read what it says.');
        if (cfg.onError === 'panic') next.panicked = true;
        break;
      }

      const droppedCount = recs.filter((r) => r.dropped).length;
      const pendingMeta = isFsync ? next.sizeDirty || next.mtimeDirty : next.sizeDirty;
      const inDrive = recs.filter((r) => r.stage === 2 && !r.lost).length;
      recs = recs.map((r) => (r.stage === 2 && !r.lost ? { ...r, stage: 3 as Stage } : r));

      if (next.sizeDirty) {
        next.sizeDirty = false;
        next.sizeOnMedia = true;
      }
      if (isFsync && next.mtimeDirty) {
        next.mtimeDirty = false;
        next.mtimeOnMedia = true;
      }

      const work = hadDirty || inDrive > 0 || pendingMeta;
      ns = work ? (cfg.dev.wbNs + (pendingMeta ? f : 0) + f) * j : 2_400 * j;
      ret = '0';

      if (droppedCount > 0) {
        head = `${LABEL[op]} returned 0 — and it is lying to you.`;
        body =
          `The pages that failed are not dirty any more, so this call had nothing to write and reported ` +
          `success. It is telling the truth about its own bookkeeping and nothing at all about your data: ` +
          `${fmtBytes(droppedCount * PAGE_BYTES)} are permanently gone, and no future fsync() will ever say ` +
          `so again. This is fsyncgate. A pre-12 Postgres read this 0 as "checkpoint complete", advanced the ` +
          `redo pointer past those pages, and recycled the WAL that could have rebuilt them.`;
      } else if (!work) {
        head = `${LABEL[op]} had nothing to do.`;
        body = 'No dirty pages and no pending journal commit, so ext4 skipped the barrier and the call was nearly free.';
      } else if (isFsync) {
        head = 'fsync(): data, then the whole inode, then a barrier.';
        body =
          `The dirty pages went to the device, the inode — i_size *and* mtime/ctime — was journaled, and a ` +
          `device FLUSH made both durable. On ext4 that is two ordered flushes, ${fmtTime(syncCost(cfg, 'fsync'))} ` +
          `here. Everything below the boundary now survives a power cut.`;
      } else {
        head = 'fdatasync(): data, only the metadata you need, then a barrier.';
        body = cfg.append
          ? `The write extended the file, so i_size had to be journaled too: fdatasync() may skip metadata, but ` +
            `never metadata required to find the data again. mtime stayed dirty — nobody needs it to read your ` +
            `bytes back. Cost here: ${fmtTime(syncCost(cfg, 'fdatasync'))}.`
          : `The write landed inside the existing file, i_size did not change, and fdatasync() skipped the journal ` +
            `commit entirely — one flush instead of two, ${fmtTime(syncCost(cfg, 'fdatasync'))}. This is exactly why ` +
            `Postgres recycles pre-allocated 16 MB WAL segments instead of extending a file on every commit.`;
      }
      break;
    }

    case 'fua': {
      if (live >= MAX_RECS) return s;
      recs = [...recs, { id: s.nextId, stage: 3, dirty: false, cached: false, dropped: false, lost: false }];
      next.nextId = s.nextId + 1;
      next.procDead = false;
      next.mtimeDirty = true;
      next.mtimeOnMedia = false;
      const journal = cfg.append || next.sizeDirty;
      if (journal) {
        next.sizeDirty = false;
        next.sizeOnMedia = true;
      }
      ns = (cfg.dev.wbNs + f * 0.5 + (journal ? f : 0)) * j;
      ret = String(PAGE_BYTES);
      head = 'pwrite(O_DIRECT | O_DSYNC): straight to the media in one round trip.';
      body =
        'O_DIRECT skips the page cache, so there is no DRAM copy to lose, and O_DSYNC makes the block layer ' +
        'tag the command FUA — Force Unit Access — which forbids the drive from acknowledging out of its ' +
        'volatile cache. One command does what write() + fdatasync() needed three for, which is why InnoDB ' +
        'defaults to O_DIRECT for data files. Drop the O_DSYNC and the write lands in the drive cache: ' +
        'acknowledged, and still volatile.';
      break;
    }

    case 'inject': {
      next.errArmed = true;
      head = 'Armed: the next writeback to this device will fail.';
      body =
        'A sector that could not be remapped, a thin-provisioned LUN that ran out of space, an NFS server ' +
        'that went away, a USB disk somebody unplugged. Now press writeback, fdatasync() or fsync() and ' +
        'watch what the page cache does with the pages it could not write.';
      break;
    }

    case 'kill': {
      next.procDead = true;
      const k = tally(recs);
      head = 'SIGKILL — and every byte in the page cache is still there.';
      body =
        `Only the record being assembled in your own WAL buffer died. The ${k.dirty} dirty page` +
        `${k.dirty === 1 ? '' : 's'} belong to the kernel, not to your process, and the flusher will write ` +
        `them out on its own schedule — dirty_expire_centisecs defaults to 30 seconds. That is the entire ` +
        `promise of write(): durability against a process crash, never against a machine crash.`;
      break;
    }

    case 'power': {
      let saved = 0;
      let lost = 0;
      let rescued = 0;
      recs = recs.map((r) => {
        if (r.dropped) return r;
        if (r.stage === 3) {
          saved++;
          return { ...r, cached: false };
        }
        if (r.stage === 2 && cfg.plp) {
          saved++;
          rescued++;
          return { ...r, stage: 3 as Stage, cached: false };
        }
        lost++;
        return { ...r, lost: true, cached: false };
      });
      next.crashed = true;
      next.procDead = true;
      next.mtimeDirty = false;
      head = `Power cut: ${fmtBytes(saved * PAGE_BYTES)} survived, ${fmtBytes(lost * PAGE_BYTES)} did not.`;
      body =
        (rescued > 0
          ? `The drive still had ${rescued} record${rescued === 1 ? '' : 's'} in its cache and pushed them to ` +
            `NAND on capacitor power on the way down — that is the whole of what power-loss protection buys. `
          : '') +
        (lost > 0
          ? `Everything in DRAM — your buffer, the page cache${cfg.plp ? '' : ', the drive write cache'} — is ` +
            `gone, whether or not a syscall had already returned success for it. `
          : 'Every byte had crossed the power-loss boundary before the lights went out. ') +
        (next.sizeDirty && saved > 0
          ? 'Worse: i_size never reached the journal, so on remount the file is its old length and the blocks ' +
            'that did land on media are unreachable.'
          : '');
      break;
    }
  }

  const t = tally(recs);
  return {
    ...next,
    recs,
    head,
    body,
    log: [
      ...s.log,
      { n: step + 1, op: LABEL[op], ret, ns, dirty: t.dirty, dwc: t.dwc, media: t.media, gone: t.gone },
    ],
  };
}

/* -------------------------------------------------------------- the drawing */

type ChipKind = 'dirty' | 'copy' | 'volatile' | 'durable' | 'gone';

const CHIP: Record<ChipKind, { fill: string; stroke: string; sw: number; glyph: string }> = {
  dirty: { fill: 'var(--viz-dirty)', stroke: 'var(--viz-surface)', sw: 2, glyph: 'D' },
  copy: { fill: 'var(--viz-plane)', stroke: 'var(--viz-ink-2)', sw: 1.5, glyph: 'c' },
  volatile: { fill: 'var(--viz-warning)', stroke: 'var(--viz-surface)', sw: 2, glyph: 'V' },
  durable: { fill: 'var(--viz-clean)', stroke: 'var(--viz-surface)', sw: 2, glyph: '✓' },
  gone: { fill: 'var(--viz-stale)', stroke: 'var(--viz-surface)', sw: 2, glyph: '✕' },
};

type Chip = { key: string; kind: ChipKind; label: string; hint: string; meta?: boolean };

const LANES = [
  { name: 'User buffer', sub: 'your process — WAL buffer' },
  { name: 'OS page cache', sub: 'kernel DRAM — PG_dirty' },
  { name: 'Drive write cache', sub: 'device DRAM' },
  { name: 'Media', sub: 'NAND / platter' },
];

function laneChips(s: S, cfg: Cfg): Chip[][] {
  const out: Chip[][] = [[], [], [], []];

  if (!s.procDead && !s.crashed && s.recs.filter((r) => !r.dropped).length < MAX_RECS) {
    out[0].push({
      key: 'pending',
      kind: 'dirty',
      label: `#${s.nextId}`,
      hint: 'Being assembled in your process. A SIGKILL loses exactly this record and nothing else.',
      meta: true,
    });
  }

  for (const r of s.recs) {
    if (r.dropped) {
      out[1].push({
        key: `r${r.id}x`,
        kind: 'gone',
        label: `#${r.id}`,
        hint: 'Writeback failed: the kernel cleared PG_dirty and discarded the page. These bytes exist nowhere.',
      });
      continue;
    }
    if (r.lost) {
      out[r.stage === 2 ? 2 : 1].push({
        key: `r${r.id}l`,
        kind: 'gone',
        label: `#${r.id}`,
        hint: 'Was in volatile memory when the power went. Gone.',
      });
      continue;
    }
    if (r.cached && !s.crashed) {
      out[1].push({
        key: `r${r.id}c`,
        kind: r.dirty ? 'dirty' : 'copy',
        label: `#${r.id}`,
        hint: r.dirty
          ? 'Dirty page: the only copy of these bytes is in kernel DRAM.'
          : 'Clean cached copy. Good for reads, worth nothing for durability — and nothing to retry with if the device rejects the write.',
      });
    }
    if (r.stage === 2) {
      out[2].push({
        key: `r${r.id}d`,
        kind: 'volatile',
        label: `#${r.id}`,
        hint: cfg.plp
          ? 'In the drive cache, which is capacitor-backed: the drive finishes this write on its own power.'
          : 'Acknowledged by the drive and sitting in volatile DRAM. A power cut loses it silently.',
      });
    }
    if (r.stage === 3) {
      out[3].push({ key: `r${r.id}m`, kind: 'durable', label: `#${r.id}`, hint: 'On the media. This one survives a power cut.' });
    }
  }

  if (s.sizeDirty && !s.crashed) {
    out[1].push({
      key: 'size-d',
      kind: 'dirty',
      label: 'i_size',
      hint: 'The file grew. Until this is journaled, blocks that reached the media are unreachable.',
      meta: true,
    });
  }
  if (s.mtimeDirty && !s.crashed) {
    out[1].push({
      key: 'mtime-d',
      kind: 'dirty',
      label: 'mtime',
      hint: 'fdatasync() deliberately leaves this dirty: you do not need mtime to read your data back.',
      meta: true,
    });
  }
  if (s.sizeOnMedia) {
    out[3].push({ key: 'size-m', kind: 'durable', label: 'i_size', hint: 'Journaled and flushed: the on-disk length matches the data.', meta: true });
  }
  if (s.mtimeOnMedia) {
    out[3].push({ key: 'mtime-m', kind: 'durable', label: 'mtime', hint: 'Persisted by fsync(); fdatasync() would have skipped it.', meta: true });
  }
  return out;
}

/* ------------------------------------------------------------- the component */

export default function DurabilityStackLab() {
  const [devId, setDevId] = useState<DevId>('nvme');
  const [plp, setPlp] = useState(false);
  const [append, setAppend] = useState(true);
  const [onError, setOnError] = useState<'panic' | 'retry'>('retry');
  const [group, setGroup] = useState(1);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const dev = DEVICES.find((d) => d.id === devId)!;
  const cfg: Cfg = { dev, plp, append, onError };
  const t = tally(s.recs);
  const chips = laneChips(s, cfg);

  const run = (op: Op) => setS((cur) => apply(cur, op, cfg));
  const frozen = s.crashed || s.panicked;
  const full = s.recs.filter((r) => !r.dropped).length >= MAX_RECS;

  const labelW = 168;
  const chipW = 50;
  const chipGap = 8;
  const slots = MAX_RECS + 2;
  const svgW = Math.max(width, labelW + slots * (chipW + chipGap) + 24);
  const laneH = 62;
  const laneGap = 12;
  const top = 18;
  const height = top + LANES.length * laneH + (LANES.length - 1) * laneGap + 10;
  const laneY = (i: number) => top + i * (laneH + laneGap);
  const boundaryY = laneY(plp ? 2 : 3) - laneGap / 2;

  const atRisk = t.dirty + (plp ? 0 : t.dwc);
  const fdCost = syncCost(cfg, 'fdatasync');

  return (
    <VizPanel
      title="The four-layer durability stack"
      subtitle="Walk an 8 KB WAL record from your buffer down to the media, then pull the plug at any point. Only what sits below the dashed boundary is still there after the reboot."
      controls={
        <>
          <Choice
            label="Device"
            value={devId}
            onChange={(v) => setDevId(v)}
            options={DEVICES.map((d) => ({ value: d.id, label: d.label }))}
          />
          <Check label="Power-loss-protected cache" checked={plp} onChange={setPlp} />
          <Check label="Write extends the file" checked={append} onChange={setAppend} />
          <Segmented
            label="On fsync error"
            value={onError}
            onChange={setOnError}
            options={[
              { value: 'retry', label: 'retry', title: 'data_sync_retry = on — the pre-2018 behaviour' },
              { value: 'panic', label: 'PANIC', title: 'Postgres 12+ default: data_sync_retry = off' },
            ]}
          />
          <Slider label="Group commit" min={1} max={64} value={group} onChange={setGroup} format={(n) => `${n} txn/flush`} />
          <Button onClick={() => run('write')} disabled={frozen || full} primary>
            write()
          </Button>
          <Button onClick={() => run('writeback')} disabled={frozen} title="What the kernel flusher thread does on its own schedule">
            writeback
          </Button>
          <Button onClick={() => run('flush')} disabled={frozen} title="REQ_OP_FLUSH — the device cache-flush command">
            FLUSH
          </Button>
          <Button onClick={() => run('fdatasync')} disabled={frozen}>
            fdatasync()
          </Button>
          <Button onClick={() => run('fsync')} disabled={frozen}>
            fsync()
          </Button>
          <Button onClick={() => run('fua')} disabled={frozen || full} title="pwrite() with O_DIRECT | O_DSYNC — the block layer tags it FUA">
            O_DIRECT|O_DSYNC
          </Button>
          <Button onClick={() => run('inject')} disabled={frozen || s.errArmed} title="Make the next writeback return EIO">
            Inject write error
          </Button>
          <Button onClick={() => run('kill')} disabled={frozen}>
            SIGKILL
          </Button>
          <Button onClick={() => run('power')} disabled={frozen}>
            Power cut
          </Button>
          <Button onClick={() => setS(INITIAL)}>{frozen ? 'Reboot' : 'Reset'}</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'D — dirty in the page cache', color: CHIP.dirty.fill },
            { label: 'c — clean cached copy (outlined)', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'V — in a volatile drive cache', color: CHIP.volatile.fill },
            { label: '✓ — on media', color: CHIP.durable.fill },
            { label: '✕ — gone', color: CHIP.gone.fill },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Durable on media', value: fmtBytes(t.media * PAGE_BYTES), hint: 'Survives a power cut right now' },
            {
              label: 'Acked but volatile',
              value: fmtBytes(atRisk * PAGE_BYTES),
              hint: 'Bytes a syscall has already returned success for that a power cut would still lose',
            },
            { label: 'Lost', value: fmtBytes(t.gone * PAGE_BYTES), hint: 'Dropped by a failed writeback, or by a power cut' },
            { label: 'Last call returned', value: s.log.length ? s.log[s.log.length - 1].ret : '—' },
            { label: 'fdatasync() cost', value: fmtTime(fdCost), hint: 'writeback + journal commit (only if the file grew) + device FLUSH' },
            {
              label: `Durable commits/s @ ${group}/flush`,
              value: fmtNum((1e9 / fdCost) * group),
              hint: 'One flush amortized over a batch of transactions — this is why group commit exists',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
          {s.panicked ? ' The server is down — press Reboot to replay the WAL.' : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Call</th>
                <th>Returns</th>
                <th>Cost</th>
                <th>Dirty pages</th>
                <th>Drive cache</th>
                <th>On media</th>
                <th>Gone</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={8}>No calls yet.</td>
                </tr>
              ) : (
                s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.ret}</td>
                    <td>{r.ns > 0 ? fmtTime(r.ns) : '—'}</td>
                    <td>{r.dirty}</td>
                    <td>{r.dwc}</td>
                    <td>{r.media}</td>
                    <td>{r.gone}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Device</th>
                <th>write()</th>
                <th>writeback</th>
                <th>FLUSH, volatile</th>
                <th>FLUSH, power-safe</th>
                <th>fdatasync(), in place</th>
                <th>fsync(), appending</th>
              </tr>
            </thead>
            <tbody>
              {DEVICES.map((d) => (
                <tr key={d.id}>
                  <td>{d.label}</td>
                  <td>{fmtTime(d.writeNs)}</td>
                  <td>{fmtTime(d.wbNs)}</td>
                  <td>{fmtTime(d.flushVolNs)}</td>
                  <td>{fmtTime(d.flushNvNs)}</td>
                  <td>{fmtTime(syncCost({ dev: d, plp, append: false, onError }, 'fdatasync'))}</td>
                  <td>{fmtTime(syncCost({ dev: d, plp, append: true, onError }, 'fsync'))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label="Four layers of the write path and which of them survive a power cut"
          >
            {LANES.map((ln, i) => (
              <g key={ln.name}>
                <rect
                  x={labelW}
                  y={laneY(i)}
                  width={svgW - labelW - 8}
                  height={laneH}
                  rx={8}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={0} y={laneY(i) + 26} fill="var(--viz-ink)" fontWeight={600}>
                  {ln.name}
                </text>
                <text x={0} y={laneY(i) + 42} fill="var(--viz-ink-muted)">
                  {ln.sub}
                </text>
                {i === 2 ? (
                  <text
                    x={svgW - 16}
                    y={laneY(i) + 18}
                    textAnchor="end"
                    fill={plp ? 'var(--viz-good)' : 'var(--viz-critical)'}
                  >
                    {plp ? 'power-loss protected' : 'volatile'}
                  </text>
                ) : null}
                {i === 2 && s.errArmed ? (
                  <text x={svgW - 16} y={laneY(i) + 54} textAnchor="end" fill="var(--viz-critical)">
                    EIO armed
                  </text>
                ) : null}
                {chips[i].map((c, k) => {
                  const cx = labelW + 16 + k * (chipW + chipGap);
                  const cy = laneY(i) + 10;
                  const style = CHIP[c.kind];
                  return (
                    <g
                      key={c.key}
                      {...tip(
                        <>
                          <strong>{c.label}</strong>
                          <br />
                          {c.hint}
                        </>,
                      )}
                      style={{ cursor: 'help' }}
                    >
                      <rect
                        x={cx}
                        y={cy}
                        width={chipW}
                        height={22}
                        rx={5}
                        fill={style.fill}
                        stroke={c.meta ? 'var(--viz-ink-2)' : style.stroke}
                        strokeWidth={c.meta ? 1 : style.sw}
                        strokeDasharray={c.meta ? '3 2' : undefined}
                        opacity={c.kind === 'gone' ? 0.6 : 1}
                      />
                      <text x={cx + chipW / 2} y={cy + 44} textAnchor="middle" fill="var(--viz-ink)">
                        {c.label} {style.glyph}
                      </text>
                    </g>
                  );
                })}
              </g>
            ))}

            <line
              x1={labelW}
              x2={svgW - 8}
              y1={boundaryY}
              y2={boundaryY}
              stroke="var(--viz-critical)"
              strokeWidth={1.5}
              strokeDasharray="6 4"
            />
            <text x={labelW - 8} y={boundaryY + 4} textAnchor="end" fill="var(--viz-critical)">
              power-loss boundary
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
