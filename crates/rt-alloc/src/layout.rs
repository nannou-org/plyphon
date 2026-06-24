//! On-buffer layout: the boundary-tag chunk header, free-list links, and the size arithmetic - the
//! in-memory shapes scsynth's `AllocPool` stores *inline*, translated from raw pointers to byte
//! offsets read and written through [`bytemuck`].
//!
//! A chunk is a 16-byte [`Header`] immediately followed by its payload. The header is a boundary
//! tag: it records this chunk's size and the previous chunk's size, with the low bit of each size
//! reused as an in-use flag (sizes are multiples of [`ALIGN`], so the low bits are free). The
//! previous-size tag lets [`free`](crate::RtPool::dealloc) jump backwards to coalesce. While a chunk
//! is free its payload's first 16 bytes hold [`FreeLinks`] instead of user data.

use bytemuck::{Pod, Zeroable};

/// Every payload is aligned to this many bytes (scsynth's `kAlign`). Cache-line sized, and large
/// enough that any [`bytemuck`] view (`f32`, `f64`, 128-bit SIMD) over a region is alignment-valid.
pub const ALIGN: usize = 64;

/// Size of a chunk's boundary-tag header in bytes (scsynth's `sizeof(AllocChunk)`).
pub const HEADER: usize = core::mem::size_of::<Header>();

/// Size of the free-list links stored in a free chunk's body, in bytes.
pub const LINKS: usize = core::mem::size_of::<FreeLinks>();

/// The smallest chunk we ever create, header included (scsynth's `kMinAllocSize`, `2 * kAlign`).
pub const MIN_CHUNK: usize = 2 * ALIGN;

/// Bytes reserved before the first chunk header so the first payload lands on an [`ALIGN`] boundary
/// (the header sits at `ALIGN - HEADER`; every later payload stays aligned because chunk sizes are
/// multiples of [`ALIGN`]).
pub const PROLOGUE: usize = ALIGN - HEADER;

/// Low bit of a tagged size field: set when the corresponding chunk is in use (scsynth's
/// `kChunkInUse`).
pub const INUSE: u64 = 1;

/// Mask selecting the size out of a tagged size field.
pub const SIZE_BITS: u64 = !INUSE;

/// Free-list sentinel meaning "no chunk".
pub const NIL: u64 = u64::MAX;

/// A 64-byte-aligned block. The pool's backing buffer is `[Align64]`, so its bytes are [`ALIGN`]-
/// aligned wherever the buffer lives (heap, `static`, or stack) with no runtime alignment fix-up.
#[repr(C, align(64))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Align64([u8; ALIGN]);

impl Align64 {
    /// A zeroed block, for sizing a fresh backing buffer.
    pub const ZERO: Align64 = Align64([0; ALIGN]);
}

/// A chunk's boundary-tag header.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Header {
    /// Previous chunk's size, with [`INUSE`] in the low bit reflecting whether *that* chunk is in
    /// use. Lets a free jump backwards (`prev = off - prev_chunk_size`) to coalesce.
    pub prev_size: u64,
    /// This chunk's size, with [`INUSE`] in the low bit. scsynth mirrors this into the next chunk's
    /// `prev_size`, so the tag is readable from either neighbour.
    pub size: u64,
}

impl Header {
    /// This chunk's size in bytes (header included), flag masked off.
    #[inline]
    pub fn chunk_size(&self) -> usize {
        (self.size & SIZE_BITS) as usize
    }

    /// Whether this chunk is in use.
    #[inline]
    pub fn in_use(&self) -> bool {
        self.size & INUSE != 0
    }

    /// The previous (physically-adjacent) chunk's size in bytes, flag masked off.
    #[inline]
    pub fn prev_chunk_size(&self) -> usize {
        (self.prev_size & SIZE_BITS) as usize
    }

    /// Whether the previous (physically-adjacent) chunk is in use.
    #[inline]
    pub fn prev_in_use(&self) -> bool {
        self.prev_size & INUSE != 0
    }
}

/// Stored in the body of a *free* chunk (its first [`LINKS`] bytes); meaningless while in use.
/// Offsets into the backing buffer, or [`NIL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FreeLinks {
    pub next: u64,
    pub prev: u64,
}

/// Round a user request up to an actual chunk size (payload + header, [`ALIGN`]-rounded), floored at
/// [`MIN_CHUNK`]. scsynth's `RequestToSize`.
pub fn request_to_size(req: usize) -> usize {
    let with_header = req.saturating_add(HEADER);
    if with_header <= MIN_CHUNK {
        MIN_CHUNK
    } else {
        with_header.next_multiple_of(ALIGN)
    }
}
