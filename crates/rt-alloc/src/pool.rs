//! [`RtPool`] - the allocator itself - and [`Region`], the owned handle it hands out.
//!
//! The pool carves a single fixed `[Align64]` backing buffer into boundary-tag chunks (see
//! [`layout`](crate::layout)) linked into [`bins`](crate::bins). It mirrors scsynth's `AllocPool`:
//! best-fit binned allocation, splitting, and backward/forward coalescing on free - but pointers are
//! byte offsets and every payload borrow goes through `&self`/`&mut self`, so the borrow checker
//! forbids aliasing the bookkeeping and a payload at once. Allocation never panics; exhaustion is
//! [`None`].
//!
//! The internal chunk surgery is written as free functions over `(bytes, bins, binmap)` rather than
//! methods, so each step is small and independently testable; the public methods just destructure
//! the pool into those disjoint borrows and orchestrate.

use core::ops::Range;

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use crate::bins::{NUM_SMALL_BINS, bin_index, clear_bin, mark_bin, next_full_bin};
use crate::layout::{
    Align64, FreeLinks, HEADER, Header, INUSE, LINKS, MIN_CHUNK, NIL, PROLOGUE, request_to_size,
};

/// A fixed-size real-time memory pool over a `[Align64]` backing buffer.
///
/// Construct with [`RtPool::with_capacity_bytes`] (heap-backed, needs the `alloc` feature) or
/// [`RtPool::from_blocks`] (over any caller-owned `[Align64]` store, e.g. a `static` array - fully
/// `no_std`). The type parameter `S` is the backing store and is normally inferred.
///
/// The pool is fully self-contained - no globals, statics, or thread-local state - so any number of
/// independent pools can coexist. Allocation and freeing take `&mut self`, so a single pool is driven
/// by one thread at a time; a parallel engine gives each DSP thread its own pool.
pub struct RtPool<S> {
    buf: S,
    /// Free-list head (a chunk offset, or [`NIL`]) per bin.
    bins: [u64; 128],
    /// One bit per bin: set while the bin is non-empty.
    binmap: [u32; 4],
}

/// An owned handle to one allocation: exclusive access to a byte sub-range of the pool's buffer.
///
/// Obtain a payload slice via [`RtPool::slice`]/[`RtPool::slice_mut`] (or a typed
/// [view](RtPool::view)). Return the memory with [`RtPool::dealloc`], which *consumes* the handle -
/// so a freed region is unnameable, making use-after-free a compile error rather than UB.
///
/// # Pool affinity
///
/// A handle belongs to the [`RtPool`] that produced it. Using it with a *different* pool is a logic
/// error, never a soundness one: it may read or free the wrong bytes and corrupt that pool's
/// free lists - breaking its later behaviour (leaks, overlapping regions) or panicking on an
/// out-of-range index - but it can **never** violate memory safety. The crate is
/// `#![forbid(unsafe_code)]`, so every access is bounds-checked, and exclusive `&mut` access stays
/// mediated by the pool (and by [`slice::get_disjoint_mut`] in [`slices_mut`](RtPool::slices_mut)),
/// regardless of allocator state. Keep each pool's handles with their pool.
#[derive(Debug)]
#[must_use = "dropping a Region leaks its allocation; pass it to RtPool::dealloc to reclaim it"]
pub struct Region {
    user_offset: u64,
    len: u32,
}

impl Region {
    /// Length in bytes of the payload this handle owns (the originally requested size).
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the payload is zero-length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The payload's byte range within the backing buffer.
    fn range(&self) -> Range<usize> {
        let start = self.user_offset as usize;
        start..start + self.len as usize
    }
}

impl<S: AsRef<[Align64]> + AsMut<[Align64]>> RtPool<S> {
    /// Build a pool over a caller-owned `[Align64]` backing store. Works with an owned array, a
    /// `Box<[Align64]>`, or a `&mut [Align64]` (e.g. backed by a `static`), so it needs no allocator.
    ///
    /// Panics if the backing is too small to hold a single minimum allocation (a boot-time
    /// precondition, never reached on the audio thread).
    pub fn from_blocks(buf: S) -> Self {
        let mut pool = RtPool {
            buf,
            bins: [NIL; 128],
            binmap: [0; 4],
        };
        pool.init();
        pool
    }

    /// Lay out the initial single free chunk spanning the whole arena, bracketed by virtual in-use
    /// fences so coalescing never walks past either end.
    fn init(&mut self) {
        let Self {
            buf, bins, binmap, ..
        } = self;
        *bins = [NIL; 128];
        *binmap = [0; 4];
        let bytes: &mut [u8] = cast_slice_mut(buf.as_mut());
        let len = bytes.len();
        assert!(
            len >= PROLOGUE + MIN_CHUNK + HEADER,
            "rt-alloc backing buffer too small for a single allocation",
        );
        let first = PROLOGUE;
        let fence = len - HEADER;
        let size = fence - first; // multiple of ALIGN by construction
        debug_assert_eq!(size % crate::layout::ALIGN, 0);
        // Right fence: a perpetual in-use chunk so forward coalescing stops here.
        write_header(
            bytes,
            fence,
            Header {
                prev_size: 0,
                size: INUSE,
            },
        );
        // Left fence: encode "previous chunk in use" in the first chunk's prev tag.
        write_header(
            bytes,
            first,
            Header {
                prev_size: INUSE,
                size: 0,
            },
        );
        // Mark the whole arena free (sets first.size and the fence's prev tag) and bin it.
        set_free(bytes, first, size);
        link_free(bytes, bins, binmap, first);
    }

    /// Allocate `bytes` of payload, returning an owned [`Region`], or [`None`] if the pool can't
    /// satisfy the request. Never panics. Best-fit over the bins, splitting an oversized chunk and
    /// returning the remainder to the free lists (scsynth's `Alloc`).
    pub fn alloc(&mut self, bytes: usize) -> Option<Region> {
        let need = request_to_size(bytes);
        let Self { buf, bins, binmap } = self;
        let buf: &mut [u8] = cast_slice_mut(buf.as_mut());

        let chunk = find_fit(buf, bins, binmap, need)?;
        let chunk_size = read_header(buf, chunk).chunk_size();
        unlink_free(buf, bins, binmap, chunk);

        let remainder = chunk_size - need;
        if remainder >= MIN_CHUNK {
            set_in_use(buf, chunk, need);
            free_chunk(buf, bins, binmap, chunk + need, remainder);
        } else {
            set_in_use(buf, chunk, chunk_size);
        }

        Some(Region {
            user_offset: (chunk + HEADER) as u64,
            len: bytes as u32,
        })
    }

    /// Allocate room for `count` values of type `T`. A subsequent [`view`](Self::view)`::<T>` then
    /// yields exactly `count` elements. Returns [`None`] on overflow or exhaustion.
    pub fn alloc_for<T: Pod>(&mut self, count: usize) -> Option<Region> {
        self.alloc(count.checked_mul(core::mem::size_of::<T>())?)
    }

    /// Return a region's memory to the pool, coalescing with free neighbours (scsynth's `Free`).
    /// Consumes the handle. `region` must belong to this pool (see [`Region`]).
    pub fn dealloc(&mut self, region: Region) {
        let Self { buf, bins, binmap } = self;
        let buf: &mut [u8] = cast_slice_mut(buf.as_mut());
        let chunk = region.user_offset as usize - HEADER;
        let size = read_header(buf, chunk).chunk_size();
        free_chunk(buf, bins, binmap, chunk, size);
    }

    /// Resize a region. On success returns the new handle and consumes the old; on failure (only
    /// when growing would exhaust the pool) returns the original handle unchanged in `Err`.
    ///
    /// Shrinks and grows in place where it can (splitting, or absorbing a free following chunk);
    /// otherwise allocates fresh, copies the payload, and frees the old region. (scsynth also shifts
    /// back into a free *preceding* chunk; that extra case is left to the copy path here.)
    ///
    /// `region` must belong to this pool (see [`Region`]).
    pub fn realloc(&mut self, region: Region, bytes: usize) -> Result<Region, Region> {
        let need = request_to_size(bytes);

        // Phase 1: attempt in place. Scoped so the disjoint borrows end before phase 2 needs `self`.
        let in_place = {
            let Self { buf, bins, binmap } = self;
            let buf: &mut [u8] = cast_slice_mut(buf.as_mut());
            let chunk = region.user_offset as usize - HEADER;
            let old_size = read_header(buf, chunk).chunk_size();

            let kept: Option<u64> = if need <= old_size {
                split_tail(buf, bins, binmap, chunk, old_size, need);
                Some(region.user_offset)
            } else {
                let next = chunk + old_size;
                let next_hdr = read_header(buf, next);
                if !next_hdr.in_use() && old_size + next_hdr.chunk_size() >= need {
                    unlink_free(buf, bins, binmap, next);
                    split_tail(
                        buf,
                        bins,
                        binmap,
                        chunk,
                        old_size + next_hdr.chunk_size(),
                        need,
                    );
                    Some(region.user_offset)
                } else {
                    None
                }
            };
            kept.map(|user_offset| Region {
                user_offset,
                len: bytes as u32,
            })
        };
        if let Some(grown) = in_place {
            return Ok(grown);
        }

        // Phase 2: relocate. Allocate fresh, copy the payload, free the old region.
        match self.alloc(bytes) {
            Some(new_region) => {
                let copy_len = region.len.min(new_region.len) as usize;
                let src = region.user_offset as usize;
                let dst = new_region.user_offset as usize;
                let buf: &mut [u8] = cast_slice_mut(self.buf.as_mut());
                buf.copy_within(src..src + copy_len, dst);
                self.dealloc(region);
                Ok(new_region)
            }
            None => Err(region),
        }
    }

    /// Shared read-only access to a region's payload. `region` must belong to this pool (see
    /// [`Region`]).
    pub fn slice(&self, region: &Region) -> &[u8] {
        let buf: &[u8] = cast_slice(self.buf.as_ref());
        &buf[region.range()]
    }

    /// Exclusive mutable access to a region's payload. `region` must belong to this pool (see
    /// [`Region`]).
    pub fn slice_mut(&mut self, region: &Region) -> &mut [u8] {
        let buf: &mut [u8] = cast_slice_mut(self.buf.as_mut());
        &mut buf[region.range()]
    }

    /// A typed read-only view of a region, or [`None`] if its byte length isn't a whole number of
    /// `T`s. Alignment always holds (payloads are 64-byte aligned), so this only ever fails on the
    /// length check. `region` must belong to this pool (see [`Region`]).
    pub fn view<T: Pod>(&self, region: &Region) -> Option<&[T]> {
        bytemuck::try_cast_slice(self.slice(region)).ok()
    }

    /// A typed mutable view of a region; see [`view`](Self::view).
    pub fn view_mut<T: Pod>(&mut self, region: &Region) -> Option<&mut [T]> {
        bytemuck::try_cast_slice_mut(self.slice_mut(region)).ok()
    }

    /// Exclusive mutable access to several distinct regions at once (e.g. to copy between two
    /// ugens' buffers). Returns [`None`] if any two regions overlap - which never happens for
    /// distinct live allocations, so this is really a guard against passing the same region twice.
    ///
    /// Fully safe: distinct allocations occupy disjoint ranges, handed out via the standard library's
    /// [`slice::get_disjoint_mut`]. Every `region` must belong to this pool (see [`Region`]).
    pub fn slices_mut<'a, const N: usize>(
        &'a mut self,
        regions: [&Region; N],
    ) -> Option<[&'a mut [u8]; N]> {
        let ranges = regions.map(Region::range);
        let buf: &mut [u8] = cast_slice_mut(self.buf.as_mut());
        buf.get_disjoint_mut(ranges).ok()
    }

    /// Total bytes the arena can hold across all chunks (capacity minus the fixed fences). Equal to
    /// [`used_bytes`](Self::used_bytes)` + `[`free_bytes`](Self::free_bytes).
    pub fn total_bytes(&self) -> usize {
        let buf: &[u8] = cast_slice(self.buf.as_ref());
        buf.len() - PROLOGUE - HEADER
    }

    /// Bytes currently handed out (summed over in-use chunks, headers included). Walks the heap, so
    /// it's `O(chunks)` - intended for tests and introspection, not the hot path.
    pub fn used_bytes(&self) -> usize {
        let mut sum = 0;
        self.for_each_chunk(|size, in_use| {
            if in_use {
                sum += size;
            }
        });
        sum
    }

    /// Bytes currently free (summed over free chunks, headers included). See [`used_bytes`](Self::used_bytes).
    pub fn free_bytes(&self) -> usize {
        let mut sum = 0;
        self.for_each_chunk(|size, in_use| {
            if !in_use {
                sum += size;
            }
        });
        sum
    }

    /// The largest free chunk's size (header included), or 0 if none is free - a fragmentation gauge.
    /// `alloc(n)` succeeds exactly when `request_to_size(n) <= largest_free_block()`. Walks the heap,
    /// so it's `O(chunks)`; intended for introspection, not the hot path.
    pub fn largest_free_block(&self) -> usize {
        let mut max = 0;
        self.for_each_chunk(|size, in_use| {
            if !in_use && size > max {
                max = size;
            }
        });
        max
    }

    /// Visit each chunk in physical order as `(size, in_use)`.
    fn for_each_chunk(&self, mut f: impl FnMut(usize, bool)) {
        let buf: &[u8] = cast_slice(self.buf.as_ref());
        let fence = buf.len() - HEADER;
        let mut off = PROLOGUE;
        while off < fence {
            let header = read_header(buf, off);
            let size = header.chunk_size();
            if size == 0 {
                break; // defensive: a corrupt zero size would otherwise loop forever
            }
            f(size, header.in_use());
            off += size;
        }
    }
}

#[cfg(feature = "alloc")]
impl RtPool<alloc::boxed::Box<[Align64]>> {
    /// Build a heap-backed pool with at least `bytes` of backing (rounded up to whole 64-byte
    /// blocks, and to the minimum the layout needs).
    pub fn with_capacity_bytes(bytes: usize) -> Self {
        let blocks = bytes.div_ceil(crate::layout::ALIGN).max(3);
        let mut buf = alloc::vec::Vec::with_capacity(blocks);
        buf.resize(blocks, Align64::ZERO);
        Self::from_blocks(buf.into_boxed_slice())
    }
}

// --- chunk surgery (free functions over the destructured pool) ---

/// Read a chunk header at `off`. Alignment-independent (so it never panics on a misaligned access).
fn read_header(buf: &[u8], off: usize) -> Header {
    bytemuck::pod_read_unaligned(&buf[off..off + HEADER])
}

/// Write a chunk header at `off`.
fn write_header(buf: &mut [u8], off: usize, header: Header) {
    buf[off..off + HEADER].copy_from_slice(bytemuck::bytes_of(&header));
}

/// Read the free-list links stored in a free chunk's body.
fn read_links(buf: &[u8], off: usize) -> FreeLinks {
    bytemuck::pod_read_unaligned(&buf[off + HEADER..off + HEADER + LINKS])
}

/// Write the free-list links into a free chunk's body.
fn write_links(buf: &mut [u8], off: usize, links: FreeLinks) {
    buf[off + HEADER..off + HEADER + LINKS].copy_from_slice(bytemuck::bytes_of(&links));
}

/// Tag chunk `[off, off+size)` as free: clear its size flag and mirror the size (flag clear) into the
/// next chunk's `prev_size`. Leaves this chunk's own `prev_size` untouched (scsynth's `SetSizeFree`).
fn set_free(buf: &mut [u8], off: usize, size: usize) {
    let mut header = read_header(buf, off);
    header.size = size as u64;
    write_header(buf, off, header);
    let mut next = read_header(buf, off + size);
    next.prev_size = size as u64;
    write_header(buf, off + size, next);
}

/// Tag chunk `[off, off+size)` as in use, mirroring the tagged size into the next chunk's
/// `prev_size` (scsynth's `SetSizeInUse`).
fn set_in_use(buf: &mut [u8], off: usize, size: usize) {
    let mut header = read_header(buf, off);
    header.size = size as u64 | INUSE;
    write_header(buf, off, header);
    let mut next = read_header(buf, off + size);
    next.prev_size = size as u64 | INUSE;
    write_header(buf, off + size, next);
}

/// Find a free chunk that fits `need` bytes, best-fit, scanning from its bin upward. Returns the
/// chunk offset without unlinking it.
fn find_fit(buf: &[u8], bins: &[u64; 128], binmap: &[u32; 4], need: usize) -> Option<usize> {
    let mut index = bin_index(need);
    while let Some(bin) = next_full_bin(binmap, index) {
        let mut cur = bins[bin];
        let mut pick = NIL;
        while cur != NIL {
            let size = read_header(buf, cur as usize).chunk_size();
            if size >= need {
                pick = cur;
                // Small bins are exact-size: the first fit is the best. Large bins are sorted
                // largest-first, so keep going to find the tightest fit.
                if bin < NUM_SMALL_BINS {
                    break;
                }
                cur = read_links(buf, cur as usize).next;
            } else {
                // Large bin sorted descending: nothing further down will fit.
                break;
            }
        }
        if pick != NIL {
            return Some(pick as usize);
        }
        index = bin + 1;
    }
    None
}

/// Free chunk `[chunk, chunk+size)`: coalesce with a free previous and/or next chunk, then tag the
/// merged span free and link it into its bin. The chunk's `prev_size` tag must already be correct.
fn free_chunk(
    buf: &mut [u8],
    bins: &mut [u64; 128],
    binmap: &mut [u32; 4],
    chunk: usize,
    size: usize,
) {
    let mut off = chunk;
    let mut size = size;

    let header = read_header(buf, off);
    if !header.prev_in_use() {
        let prev = off - header.prev_chunk_size();
        unlink_free(buf, bins, binmap, prev);
        size += header.prev_chunk_size();
        off = prev;
    }

    let next = off + size;
    let next_hdr = read_header(buf, next);
    if !next_hdr.in_use() {
        unlink_free(buf, bins, binmap, next);
        size += next_hdr.chunk_size();
    }

    set_free(buf, off, size);
    link_free(buf, bins, binmap, off);
}

/// Tag `[chunk, chunk+need)` in use and, if the `total - need` tail is worth its own chunk, free it;
/// otherwise keep the slack inside the allocation (scsynth's split-or-not on alloc/realloc).
fn split_tail(
    buf: &mut [u8],
    bins: &mut [u64; 128],
    binmap: &mut [u32; 4],
    chunk: usize,
    total: usize,
    need: usize,
) {
    if total - need >= MIN_CHUNK {
        set_in_use(buf, chunk, need);
        free_chunk(buf, bins, binmap, chunk + need, total - need);
    } else {
        set_in_use(buf, chunk, total);
    }
}

/// Insert a free chunk into its bin: push-front for small/empty bins, sorted (largest first) for
/// non-empty large bins (scsynth's `LinkFree`).
fn link_free(buf: &mut [u8], bins: &mut [u64; 128], binmap: &mut [u32; 4], off: usize) {
    let size = read_header(buf, off).chunk_size();
    let index = bin_index(size);
    let head = bins[index];

    if index < NUM_SMALL_BINS || head == NIL {
        write_links(
            buf,
            off,
            FreeLinks {
                next: head,
                prev: NIL,
            },
        );
        if head != NIL {
            let mut head_links = read_links(buf, head as usize);
            head_links.prev = off as u64;
            write_links(buf, head as usize, head_links);
        }
        bins[index] = off as u64;
    } else {
        // Walk the descending list to the first chunk no larger than `size`, insert before it.
        let mut cur = head;
        let mut prev = NIL;
        while cur != NIL && size < read_header(buf, cur as usize).chunk_size() {
            prev = cur;
            cur = read_links(buf, cur as usize).next;
        }
        write_links(buf, off, FreeLinks { next: cur, prev });
        if prev == NIL {
            bins[index] = off as u64;
        } else {
            let mut prev_links = read_links(buf, prev as usize);
            prev_links.next = off as u64;
            write_links(buf, prev as usize, prev_links);
        }
        if cur != NIL {
            let mut cur_links = read_links(buf, cur as usize);
            cur_links.prev = off as u64;
            write_links(buf, cur as usize, cur_links);
        }
    }
    mark_bin(binmap, index);
}

/// Remove a free chunk from its bin (scsynth's `UnlinkFree`).
fn unlink_free(buf: &mut [u8], bins: &mut [u64; 128], binmap: &mut [u32; 4], off: usize) {
    let size = read_header(buf, off).chunk_size();
    let index = bin_index(size);
    let links = read_links(buf, off);

    if links.prev == NIL {
        bins[index] = links.next;
    } else {
        let mut prev_links = read_links(buf, links.prev as usize);
        prev_links.next = links.next;
        write_links(buf, links.prev as usize, prev_links);
    }
    if links.next != NIL {
        let mut next_links = read_links(buf, links.next as usize);
        next_links.prev = links.prev;
        write_links(buf, links.next as usize, next_links);
    }
    if bins[index] == NIL {
        clear_bin(binmap, index);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::layout::ALIGN;

    fn pool(bytes: usize) -> RtPool<alloc::boxed::Box<[Align64]>> {
        RtPool::with_capacity_bytes(bytes)
    }

    /// A pool packed into exactly four equal 128-byte (`MIN_CHUNK`) chunks with no trailing free
    /// space, so coalescing has unambiguous neighbours and no confounding remainder. Returns the
    /// four regions in physical order.
    fn filled_quad() -> (RtPool<alloc::boxed::Box<[Align64]>>, [Region; 4]) {
        // 9 blocks = 576 bytes backing -> 512-byte arena -> four 128-byte chunks, exactly.
        let mut p = pool(576);
        let regions: alloc::vec::Vec<Region> =
            (0..4).map(|_| p.alloc(100).expect("alloc")).collect();
        assert_eq!(p.free_bytes(), 0, "quad should fill the arena exactly");
        let quad: [Region; 4] = regions.try_into().expect("four regions");
        (p, quad)
    }

    #[test]
    fn fresh_pool_is_one_free_chunk() {
        let p = pool(8 * 1024);
        assert_eq!(p.used_bytes(), 0);
        assert_eq!(p.free_bytes(), p.total_bytes());
    }

    #[test]
    fn payloads_are_aligned_and_usable() {
        let mut p = pool(256 * 1024);
        let mut regions = alloc::vec::Vec::new();
        for i in 1..200usize {
            let r = p.alloc(i * 3).expect("alloc");
            // Payload start is 64-byte aligned.
            assert_eq!(r.user_offset as usize % ALIGN, 0);
            let s = p.slice_mut(&r);
            assert_eq!(s.len(), i * 3);
            s.fill(i as u8);
            regions.push(r);
        }
        // Every payload still holds its pattern (no overlap / corruption).
        for (i, r) in regions.iter().enumerate() {
            assert!(p.slice(r).iter().all(|&b| b == (i + 1) as u8));
        }
        for r in regions {
            p.dealloc(r);
        }
        // Everything coalesced back to a single free arena.
        assert_eq!(p.used_bytes(), 0);
        assert_eq!(p.free_bytes(), p.total_bytes());
    }

    #[test]
    fn used_plus_free_is_invariant() {
        let mut p = pool(32 * 1024);
        let a = p.alloc(100).unwrap();
        let b = p.alloc(2000).unwrap();
        let c = p.alloc(50).unwrap();
        assert_eq!(p.used_bytes() + p.free_bytes(), p.total_bytes());
        p.dealloc(b);
        assert_eq!(p.used_bytes() + p.free_bytes(), p.total_bytes());
        p.dealloc(a);
        p.dealloc(c);
        assert_eq!(p.free_bytes(), p.total_bytes());
    }

    #[test]
    fn exhaustion_returns_none_without_panic() {
        let mut p = pool(512); // tiny
        let mut held = alloc::vec::Vec::new();
        while let Some(r) = p.alloc(64) {
            held.push(r);
        }
        assert!(!held.is_empty(), "should fit at least one allocation");
        assert!(p.alloc(64).is_none());
        // Freeing one makes room again.
        let r = held.pop().unwrap();
        p.dealloc(r);
        let r = p.alloc(64).expect("room after free");
        p.dealloc(r);
    }

    #[test]
    fn realloc_grows_shrinks_and_relocates() {
        let mut p = pool(16 * 1024);
        let r = p.alloc_for::<f32>(16).unwrap();
        p.view_mut::<f32>(&r).unwrap().fill(1.5);
        // Grow in place into the trailing free space.
        let r = p.realloc(r, 64 * 4).unwrap();
        assert_eq!(r.len(), 64 * 4);
        assert!(p.view::<f32>(&r).unwrap()[..16].iter().all(|&x| x == 1.5));
        // Shrink.
        let r = p.realloc(r, 8 * 4).unwrap();
        assert_eq!(r.len(), 8 * 4);
        // Force relocation: wall off the chunk so it can't grow in place.
        let wall = p.alloc(16).unwrap();
        let r = p.realloc(r, 4096).unwrap();
        assert!(p.view::<f32>(&r).unwrap()[..8].iter().all(|&x| x == 1.5));
        p.dealloc(wall);
        p.dealloc(r);
    }

    #[test]
    fn disjoint_slices_allow_copy_between_regions() {
        let mut p = pool(8 * 1024);
        let src = p.alloc(32).unwrap();
        let dst = p.alloc(32).unwrap();
        p.slice_mut(&src).fill(7);
        let [s, d] = p.slices_mut([&src, &dst]).unwrap();
        d.copy_from_slice(s);
        assert!(p.slice(&dst).iter().all(|&b| b == 7));
        // Passing the same region twice is rejected, not UB.
        assert!(p.slices_mut([&src, &src]).is_none());
        p.dealloc(src);
        p.dealloc(dst);
    }

    #[test]
    fn free_with_no_free_neighbour_does_not_coalesce() {
        let (mut p, [a, b, c, d]) = filled_quad();
        p.dealloc(b); // both neighbours (a, c) are in use
        assert_eq!(p.largest_free_block(), 128);
        assert_eq!(p.free_bytes(), 128);
        p.dealloc(a);
        p.dealloc(c);
        p.dealloc(d);
    }

    #[test]
    fn free_coalesces_forward_only() {
        let (mut p, [a, b, c, d]) = filled_quad();
        p.dealloc(b); // standalone 128 (neighbours in use)
        p.dealloc(a); // a's next (b) is free -> merge forward into 256
        assert_eq!(p.largest_free_block(), 256);
        p.dealloc(c);
        p.dealloc(d);
    }

    #[test]
    fn free_coalesces_backward_only() {
        let (mut p, [a, b, c, d]) = filled_quad();
        p.dealloc(a); // standalone 128 (prev is the left fence)
        p.dealloc(b); // b's prev (a) is free -> merge backward into 256
        assert_eq!(p.largest_free_block(), 256);
        p.dealloc(c);
        p.dealloc(d);
    }

    #[test]
    fn free_coalesces_both_sides() {
        let (mut p, [a, b, c, d]) = filled_quad();
        p.dealloc(a);
        p.dealloc(c); // two separate 128 holes, b still in use between them
        assert_eq!(p.largest_free_block(), 128);
        p.dealloc(b); // merges with a (prev) and c (next) into one 384 chunk
        assert_eq!(p.largest_free_block(), 384);
        assert_eq!(p.free_bytes(), 384);
        p.dealloc(d);
    }

    #[test]
    fn frees_at_arena_boundaries_do_not_cross_fences() {
        // Freeing the last chunk must not coalesce past the right fence, nor the first past the left.
        let (mut p, [a, b, c, d]) = filled_quad();
        let last_addr = p.slice(&d).as_ptr();
        p.dealloc(d); // adjacent to the right fence
        assert_eq!(
            p.largest_free_block(),
            128,
            "must not merge into the right fence"
        );
        // The freed slot is reusable and uncorrupted.
        let reused = p.alloc(100).unwrap();
        assert_eq!(p.slice(&reused).as_ptr(), last_addr);
        p.dealloc(reused);
        p.dealloc(a); // adjacent to the left fence
        assert_eq!(
            p.largest_free_block(),
            128,
            "must not merge below the left fence"
        );
        p.dealloc(b);
        p.dealloc(c);
    }

    #[test]
    fn large_bin_best_fit_picks_tightest_chunk() {
        // Two free chunks in the same large bin (chunk sizes 1024 and 1088 both map to bin 64); a
        // request needing a 1024 chunk must take the tighter one, leaving the 1088 untouched.
        let mut p = pool(64 * 1024);
        let a = p.alloc(1008).unwrap(); // chunk 1024
        let g1 = p.alloc(16).unwrap(); // guard, prevents coalescing
        let b = p.alloc(1072).unwrap(); // chunk 1088
        let g2 = p.alloc(16).unwrap(); // guard
        let a_addr = p.slice(&a).as_ptr();
        let b_addr = p.slice(&b).as_ptr();
        p.dealloc(a);
        p.dealloc(b); // both standalone in bin 64 (sorted 1088 -> 1024)
        let tight = p.alloc(1008).unwrap();
        assert_eq!(
            p.slice(&tight).as_ptr(),
            a_addr,
            "best-fit should pick the 1024 chunk"
        );
        // The 1088 chunk is now the only fit for a second 1024 request.
        let next = p.alloc(1008).unwrap();
        assert_eq!(p.slice(&next).as_ptr(), b_addr);
        p.dealloc(tight);
        p.dealloc(next);
        p.dealloc(g1);
        p.dealloc(g2);
    }

    #[test]
    fn exact_fit_reuses_slot_without_splitting() {
        let mut p = pool(8 * 1024);
        let r = p.alloc(100).unwrap(); // chunk 128
        let addr = p.slice(&r).as_ptr();
        let used = p.used_bytes();
        p.dealloc(r);
        let r = p.alloc(112).unwrap(); // also chunk 128 -> exact fit on the freed slot
        assert_eq!(
            p.slice(&r).as_ptr(),
            addr,
            "exact fit should reuse the slot"
        );
        assert_eq!(p.used_bytes(), used, "no split: used bytes unchanged");
        p.dealloc(r);
    }

    #[test]
    fn small_remainder_is_kept_not_split() {
        let mut p = pool(8 * 1024);
        let filler = p.alloc(176).unwrap(); // chunk 192
        let guard = p.alloc(16).unwrap(); // pins the slot so freeing leaves a 192 hole
        p.dealloc(filler);
        // A 128-chunk request from a 192 hole leaves 64 slack (< MIN_CHUNK) -> no split.
        let before = p.used_bytes();
        let r = p.alloc(100).unwrap();
        assert_eq!(
            p.used_bytes() - before,
            192,
            "sub-MIN_CHUNK slack stays inside the allocation"
        );
        p.dealloc(r);
        p.dealloc(guard);
    }

    #[test]
    fn fills_entire_arena_exactly() {
        let mut p = pool(8 * 1024);
        let cap = p.total_bytes();
        // The biggest single payload is one chunk spanning the arena: cap - HEADER.
        assert!(p.alloc(cap - HEADER + 1).is_none());
        let r = p.alloc(cap - HEADER).expect("largest single allocation");
        assert_eq!(p.free_bytes(), 0);
        assert!(p.alloc(1).is_none());
        p.dealloc(r);
        assert_eq!(p.free_bytes(), p.total_bytes());
    }

    #[test]
    fn realloc_same_size_is_a_noop() {
        let mut p = pool(8 * 1024);
        let r = p.alloc(100).unwrap();
        let addr = p.slice(&r).as_ptr();
        p.slice_mut(&r).fill(9);
        let r = p.realloc(r, 110).unwrap(); // still chunk 128 -> in place, same slot
        assert_eq!(p.slice(&r).as_ptr(), addr);
        assert_eq!(r.len(), 110);
        assert!(p.slice(&r)[..100].iter().all(|&b| b == 9));
        p.dealloc(r);
    }

    #[test]
    fn realloc_grows_into_free_neighbour_in_place() {
        let mut p = pool(8 * 1024);
        let a = p.alloc(100).unwrap(); // chunk 128
        let b = p.alloc(100).unwrap(); // chunk 128
        let c = p.alloc(100).unwrap(); // pins b's slot so it stays standalone when freed
        let a_addr = p.slice(&a).as_ptr();
        p.slice_mut(&a).fill(3);
        p.dealloc(b); // a's forward neighbour is now a free 128 chunk
        let free_before = p.free_bytes();
        let a = p.realloc(a, 200).unwrap(); // needs 256 = 128 + 128 -> absorbs b in place
        assert_eq!(p.slice(&a).as_ptr(), a_addr, "grew in place, no relocation");
        assert_eq!(p.free_bytes(), free_before - 128, "absorbed the neighbour");
        assert!(p.slice(&a)[..100].iter().all(|&x| x == 3));
        p.dealloc(a);
        p.dealloc(c);
    }

    #[test]
    fn realloc_failure_returns_original_intact() {
        // A full pool: a grow can neither expand in place (neighbour in use) nor relocate (no room).
        let (mut p, [a, b, c, d]) = filled_quad();
        p.slice_mut(&a).fill(5);
        let a = match p.realloc(a, 200) {
            Ok(_) => panic!("realloc should fail on a full pool"),
            Err(orig) => orig,
        };
        assert_eq!(a.len(), 100, "original length preserved");
        assert!(
            p.slice(&a).iter().all(|&b| b == 5),
            "original payload intact"
        );
        p.dealloc(a);
        p.dealloc(b);
        p.dealloc(c);
        p.dealloc(d);
    }

    #[test]
    fn zero_sized_alloc_yields_empty_region() {
        let mut p = pool(4 * 1024);
        let r = p.alloc(0).unwrap();
        assert!(r.is_empty());
        assert!(p.slice(&r).is_empty());
        assert_eq!(p.used_bytes(), MIN_CHUNK, "still occupies a minimum chunk");
        p.dealloc(r);
        assert_eq!(p.free_bytes(), p.total_bytes());
    }

    #[test]
    fn typed_views_roundtrip_and_reject_partial_lengths() {
        let mut p = pool(8 * 1024);
        // f32: the view exposes exactly the requested count.
        let r = p.alloc_for::<f32>(10).unwrap();
        assert_eq!(r.len(), 40);
        p.view_mut::<f32>(&r)
            .unwrap()
            .copy_from_slice(&[2.5f32; 10]);
        assert_eq!(p.view::<f32>(&r).unwrap(), &[2.5f32; 10]);
        p.dealloc(r);
        // f64 too.
        let r = p.alloc_for::<f64>(4).unwrap();
        assert_eq!(p.view::<f64>(&r).unwrap().len(), 4);
        p.dealloc(r);
        // A byte length that isn't a whole number of f32s can't be viewed as f32, but the raw bytes
        // are still accessible.
        let r = p.alloc(10).unwrap();
        assert!(p.view::<f32>(&r).is_none());
        assert_eq!(p.slice(&r).len(), 10);
        p.dealloc(r);
        // `alloc_for` reports overflow instead of wrapping.
        assert!(p.alloc_for::<f32>(usize::MAX).is_none());
    }
}

#[cfg(test)]
mod backing_tests {
    use super::*;
    use crate::layout::Align64;

    // Exercises the no-alloc constructor over caller-owned `[Align64]` storage - the path a
    // `#![no_std]` build with no allocator uses. Compiles and runs even with `--no-default-features`.
    #[test]
    fn from_blocks_over_owned_array_works() {
        // 9 blocks = 576 bytes -> 512-byte arena.
        let mut p = RtPool::from_blocks([Align64::ZERO; 9]);
        assert_eq!(p.free_bytes(), p.total_bytes());
        let a = p.alloc(100).unwrap();
        let b = p.alloc(100).unwrap();
        p.slice_mut(&a).fill(1);
        p.slice_mut(&b).fill(2);
        assert!(p.slice(&a).iter().all(|&x| x == 1));
        assert!(p.slice(&b).iter().all(|&x| x == 2));
        assert_eq!(p.used_bytes() + p.free_bytes(), p.total_bytes());
        p.dealloc(a);
        p.dealloc(b);
        assert_eq!(p.free_bytes(), p.total_bytes());
    }
}
