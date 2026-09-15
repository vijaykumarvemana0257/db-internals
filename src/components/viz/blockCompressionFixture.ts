/**
 * Measured compressed sizes for the block-compression bench. Generated offline with the zstd and lz4
 * command-line tools and Node's zlib, not modelled. Every block is the first N bytes of the same
 * deterministic generators the page uses (blockCompressionModel.ts), so the 16 KB rows are exactly the
 * unedited blocks the lab shows.
 *
 *   lz4, lz4hc   lz4 1.10.0 CLI, -1 and -9, independent block; bytes of the compressed block. raw=true
 *                means the frame stored the block uncompressed because LZ4 would have expanded it.
 *                For blocks under 64 KB, lz4 equals the page's own LZ4 port byte for byte.
 *   zstd*        zstd 1.5.6 CLI, levels 1/3/9/19, --no-check: bytes of the whole frame (header included).
 *   zstd3dict    zstd -3 -D with a dictionary from `zstd --train --maxdict=16384` over 400 held-out
 *                4 KB samples of the same kind (other generator seeds).
 *   zlib6        Node zlib deflateSync level 6 (zlib wrapper: 2-byte header + Adler-32).
 */
export type FixtureKind = 'timestamps' | 'timestamps-delta' | 'json' | 'random';
export type FixtureCodec = 'lz4' | 'lz4hc' | 'zstd1' | 'zstd3' | 'zstd9' | 'zstd19' | 'zstd3dict' | 'zlib6';
export const FIXTURE_BLOCK_SIZES = [1024, 4096, 16384, 65536, 262144] as const;
export type FixtureBlockSize = (typeof FIXTURE_BLOCK_SIZES)[number];
export type FixtureCell = { bytes: number; raw?: boolean };
export const FIXTURE_SIZES: Record<FixtureKind, Record<FixtureBlockSize, Record<FixtureCodec, FixtureCell>>> = {
  'timestamps': {
    1024: { lz4: { bytes: 650 }, lz4hc: { bytes: 643 }, zstd1: { bytes: 305 }, zstd3: { bytes: 305 }, zstd9: { bytes: 305 }, zstd19: { bytes: 308 }, zstd3dict: { bytes: 304 }, zlib6: { bytes: 418 } },
    4096: { lz4: { bytes: 2581 }, lz4hc: { bytes: 2550 }, zstd1: { bytes: 1106 }, zstd3: { bytes: 1106 }, zstd9: { bytes: 1106 }, zstd19: { bytes: 1105 }, zstd3dict: { bytes: 1114 }, zlib6: { bytes: 1448 } },
    16384: { lz4: { bytes: 10301 }, lz4hc: { bytes: 10114 }, zstd1: { bytes: 4289 }, zstd3: { bytes: 4289 }, zstd9: { bytes: 4289 }, zstd19: { bytes: 4287 }, zstd3dict: { bytes: 4293 }, zlib6: { bytes: 5562 } },
    65536: { lz4: { bytes: 41184 }, lz4hc: { bytes: 40421 }, zstd1: { bytes: 17009 }, zstd3: { bytes: 17009 }, zstd9: { bytes: 17015 }, zstd19: { bytes: 17008 }, zstd3dict: { bytes: 17013 }, zlib6: { bytes: 22120 } },
    262144: { lz4: { bytes: 164779 }, lz4hc: { bytes: 160777 }, zstd1: { bytes: 67943 }, zstd3: { bytes: 67943 }, zstd9: { bytes: 67955 }, zstd19: { bytes: 67941 }, zstd3dict: { bytes: 67947 }, zlib6: { bytes: 88366 } },
  },
  'timestamps-delta': {
    1024: { lz4: { bytes: 379 }, lz4hc: { bytes: 212 }, zstd1: { bytes: 143 }, zstd3: { bytes: 149 }, zstd9: { bytes: 173 }, zstd19: { bytes: 153 }, zstd3dict: { bytes: 113 }, zlib6: { bytes: 162 } },
    4096: { lz4: { bytes: 1328 }, lz4hc: { bytes: 662 }, zstd1: { bytes: 408 }, zstd3: { bytes: 436 }, zstd9: { bytes: 476 }, zstd19: { bytes: 420 }, zstd3dict: { bytes: 381 }, zlib6: { bytes: 472 } },
    16384: { lz4: { bytes: 5122 }, lz4hc: { bytes: 2217 }, zstd1: { bytes: 1330 }, zstd3: { bytes: 1412 }, zstd9: { bytes: 1503 }, zstd19: { bytes: 1323 }, zstd3dict: { bytes: 1354 }, zlib6: { bytes: 1506 } },
    65536: { lz4: { bytes: 20234 }, lz4hc: { bytes: 8065 }, zstd1: { bytes: 4806 }, zstd3: { bytes: 5171 }, zstd9: { bytes: 5743 }, zstd19: { bytes: 4575 }, zstd3dict: { bytes: 5097 }, zlib6: { bytes: 5473 } },
    262144: { lz4: { bytes: 81384 }, lz4hc: { bytes: 31283 }, zstd1: { bytes: 19226 }, zstd3: { bytes: 20143 }, zstd9: { bytes: 22194 }, zstd19: { bytes: 17254 }, zstd3dict: { bytes: 20074 }, zlib6: { bytes: 21398 } },
  },
  'json': {
    1024: { lz4: { bytes: 428 }, lz4hc: { bytes: 388 }, zstd1: { bytes: 303 }, zstd3: { bytes: 297 }, zstd9: { bytes: 286 }, zstd19: { bytes: 293 }, zstd3dict: { bytes: 192 }, zlib6: { bytes: 296 } },
    4096: { lz4: { bytes: 1260 }, lz4hc: { bytes: 1066 }, zstd1: { bytes: 729 }, zstd3: { bytes: 720 }, zstd9: { bytes: 662 }, zstd19: { bytes: 652 }, zstd3dict: { bytes: 647 }, zlib6: { bytes: 740 } },
    16384: { lz4: { bytes: 4467 }, lz4hc: { bytes: 3504 }, zstd1: { bytes: 2437 }, zstd3: { bytes: 2392 }, zstd9: { bytes: 2013 }, zstd19: { bytes: 1910 }, zstd3dict: { bytes: 2428 }, zlib6: { bytes: 2376 } },
    65536: { lz4: { bytes: 16672 }, lz4hc: { bytes: 12607 }, zstd1: { bytes: 9427 }, zstd3: { bytes: 9711 }, zstd9: { bytes: 7693 }, zstd19: { bytes: 6880 }, zstd3dict: { bytes: 9864 }, zlib6: { bytes: 8960 } },
    262144: { lz4: { bytes: 65633 }, lz4hc: { bytes: 47853 }, zstd1: { bytes: 39102 }, zstd3: { bytes: 39996 }, zstd9: { bytes: 30680 }, zstd19: { bytes: 26613 }, zstd3dict: { bytes: 39650 }, zlib6: { bytes: 35000 } },
  },
  'random': {
    1024: { lz4: { bytes: 1024, raw: true }, lz4hc: { bytes: 1024, raw: true }, zstd1: { bytes: 1034 }, zstd3: { bytes: 1034 }, zstd9: { bytes: 1034 }, zstd19: { bytes: 1034 }, zstd3dict: { bytes: 1038 }, zlib6: { bytes: 1035 } },
    4096: { lz4: { bytes: 4096, raw: true }, lz4hc: { bytes: 4096, raw: true }, zstd1: { bytes: 4106 }, zstd3: { bytes: 4106 }, zstd9: { bytes: 4106 }, zstd19: { bytes: 4106 }, zstd3dict: { bytes: 4110 }, zlib6: { bytes: 4107 } },
    16384: { lz4: { bytes: 16384, raw: true }, lz4hc: { bytes: 16384, raw: true }, zstd1: { bytes: 16394 }, zstd3: { bytes: 16394 }, zstd9: { bytes: 16394 }, zstd19: { bytes: 16394 }, zstd3dict: { bytes: 16398 }, zlib6: { bytes: 16397 } },
    65536: { lz4: { bytes: 65536, raw: true }, lz4hc: { bytes: 65536, raw: true }, zstd1: { bytes: 65546 }, zstd3: { bytes: 65546 }, zstd9: { bytes: 65546 }, zstd19: { bytes: 65546 }, zstd3dict: { bytes: 65550 }, zlib6: { bytes: 65564 } },
    262144: { lz4: { bytes: 262144, raw: true }, lz4hc: { bytes: 262144, raw: true }, zstd1: { bytes: 262159 }, zstd3: { bytes: 262159 }, zstd9: { bytes: 262159 }, zstd19: { bytes: 262159 }, zstd3dict: { bytes: 262163 }, zlib6: { bytes: 262233 } },
  },
};

/**
 * Default throughput constants, MB/s (MB = 10^6 bytes), compress / decompress. Measured once on an Apple M3 Pro
 * over 4 MB of the JSON sample cut into independent 16 KB chunks: `zstd -b<level> -i2 -B16384` (zstd 1.5.6;
 * level 19 with -i1), `lz4 -b1|-b9 -i2 -B16384` (lz4 1.10.0), and a deflateSync/inflateSync loop in Node 22
 * (zlib 1.3.0.1) for zlib level 6. They are defaults to edit, not facts about your hardware or data.
 */
export const THROUGHPUT_DEFAULTS: Record<FixtureCodec, { c: number; d: number }> = {
  lz4: { c: 1654, d: 7290 },
  lz4hc: { c: 258, d: 10511 },
  zstd1: { c: 908, d: 1735 },
  zstd3: { c: 790, d: 1628 },
  zstd9: { c: 101, d: 2050 },
  zstd19: { c: 5.8, d: 2039 },
  zstd3dict: { c: 591, d: 1685 },
  zlib6: { c: 178, d: 1256 },
};
