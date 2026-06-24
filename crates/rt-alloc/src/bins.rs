//! Segregated free lists - scsynth's binned allocator. A chunk's size maps to one of [`NUM_BINS`]
//! bins. The low [`NUM_SMALL_BINS`] are "small": each is a single exact size class spaced 16 bytes
//! apart. The rest cover larger sizes logarithmically. A 128-bit `binmap` flags which bins are
//! non-empty so allocation skips empties with a trailing-zero scan instead of walking every bin
//! (scsynth's count-leading/trailing-zeros trick).

/// Total number of bins (scsynth's `kNumAllocBins`).
pub const NUM_BINS: usize = 128;

/// Number of small (exact-size-class) bins (scsynth's `kNumSmallBins`). Bins below this index are
/// kept unordered; larger bins are kept sorted by size, largest first, for best-fit.
pub const NUM_SMALL_BINS: usize = 64;

/// Map a chunk size to its bin index (scsynth's `BinIndex`). Sizes below 1024 use 16-byte
/// granularity; larger sizes use a logarithmic mapping, saturating at the final bin for `>= 256` KiB.
pub fn bin_index(size: usize) -> usize {
    if size < 1024 {
        size >> 4
    } else if size >= 262_144 {
        NUM_BINS - 1
    } else {
        // `size` here is in [1024, 262144), so it fits in u32 and `leading_zeros` is exact.
        let bits = 28 - (size as u32).leading_zeros() as usize;
        (bits << 3) + (size >> bits)
    }
}

/// Flag bin `index` as non-empty.
pub fn mark_bin(binmap: &mut [u32; 4], index: usize) {
    binmap[index >> 5] |= 1 << (index & 31);
}

/// Flag bin `index` as empty.
pub fn clear_bin(binmap: &mut [u32; 4], index: usize) {
    binmap[index >> 5] &= !(1 << (index & 31));
}

/// The lowest non-empty bin at or above `from`, if any (scsynth's `NextFullBin`). Skips empty bins a
/// 32-bit word at a time, then a trailing-zero count finds the exact bin.
pub fn next_full_bin(binmap: &[u32; 4], from: usize) -> Option<usize> {
    if from >= NUM_BINS {
        return None;
    }
    let mut word = from >> 5;
    // Mask off the bins below `from` within the first word.
    let mut bits = binmap[word] & (!0u32 << (from & 31));
    while bits == 0 {
        word += 1;
        if word >= binmap.len() {
            return None;
        }
        bits = binmap[word];
    }
    Some((word << 5) + bits.trailing_zeros() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{MIN_CHUNK, request_to_size};

    #[test]
    fn request_to_size_boundaries() {
        // Anything that fits in the minimum chunk rounds up to it.
        assert_eq!(request_to_size(0), MIN_CHUNK);
        assert_eq!(request_to_size(1), MIN_CHUNK);
        assert_eq!(request_to_size(MIN_CHUNK - 16), MIN_CHUNK);
        // Above that, round (req + 16) up to the next multiple of 64.
        assert_eq!(request_to_size(200), 256); // 216 -> 256
        assert_eq!(request_to_size(240), 256); // 256 -> 256
        assert_eq!(request_to_size(241), 320); // 257 -> 320
        // Every chunk size is a multiple of the alignment.
        for req in 0..4096 {
            assert_eq!(request_to_size(req) % 64, 0);
            assert!(request_to_size(req) >= req + 16 || request_to_size(req) == MIN_CHUNK);
        }
    }

    #[test]
    fn bin_index_is_monotonic_and_bounded() {
        let mut last = 0;
        let mut size = MIN_CHUNK;
        while size <= 1 << 20 {
            let bin = bin_index(size);
            assert!(bin < NUM_BINS);
            assert!(bin >= last, "bin must not decrease as size grows");
            last = bin;
            size += 64;
        }
        assert_eq!(bin_index(262_144), NUM_BINS - 1);
        assert_eq!(bin_index(usize::MAX), NUM_BINS - 1);
    }

    #[test]
    fn bin_index_small_classes_are_exact() {
        // Two sizes a full 16-byte class apart land in different small bins.
        assert_eq!(bin_index(128), 8);
        assert_eq!(bin_index(192), 12);
        assert_eq!(bin_index(1008), 63);
        assert_eq!(bin_index(1024), 64); // first large bin
    }

    #[test]
    fn next_full_bin_scans_across_words() {
        let mut map = [0u32; 4];
        assert_eq!(next_full_bin(&map, 0), None);
        mark_bin(&mut map, 8);
        mark_bin(&mut map, 70);
        mark_bin(&mut map, 127);
        assert_eq!(next_full_bin(&map, 0), Some(8));
        assert_eq!(next_full_bin(&map, 8), Some(8));
        assert_eq!(next_full_bin(&map, 9), Some(70));
        assert_eq!(next_full_bin(&map, 71), Some(127));
        assert_eq!(next_full_bin(&map, 127), Some(127));
        assert_eq!(next_full_bin(&map, 128), None);
        clear_bin(&mut map, 70);
        assert_eq!(next_full_bin(&map, 9), Some(127));
    }
}
