//! `rt-alloc`: a safe, `no_std` real-time memory pool - a faithful port of
//! scsynth's `AllocPool`.
//!
//! scsynth gives its audio thread a single fixed arena (`World_Alloc`, default
//! 8 MB) carved up by a readability rewrite of Doug Lea's malloc: 16-byte
//! boundary-tag chunk headers, 128 binned free lists scanned through a `binmap`
//! bitvector, and backward/forward coalescing on free.
//!
//! This crate ports that allocator with one change that buys a memory-safe
//! implementation: **pointers become byte offsets into one aligned backing
//! buffer**, and the pool *mediates* access. You need `&mut RtPool` to obtain a
//! `&mut [u8]`, so the borrow checker forbids aliasing the bookkeeping and a
//! payload at once. Headers and free-list links still live inline, exactly as
//! scsynth lays them out, read and written through [`bytemuck`] instead of raw
//! pointer casts.
//!
//! [`RtPool::alloc`] hands back an owned [`Region`] - a handle granting
//! exclusive access to a byte sub-range. [`RtPool::dealloc`] *consumes* the
//! handle, so a freed region is unnameable at compile time (use-after-free
//! becomes a borrow error, not UB). Allocation never panics: exhaustion is
//! [`None`], and because every allocation is 64-byte aligned by construction,
//! typed [views](RtPool::view) over a region are always alignment-valid.
//!
//! A [`Region`] belongs to the pool that made it; using one with a *different*
//! pool can corrupt that pool's bookkeeping but never memory safety - see
//! [`Region`]'s "Pool affinity" note.
//!
//! The pool is single-threaded by design, mirroring scsynth: only the audio
//! thread ever calls into it. The crate is `#![no_std]` and
//! `#![forbid(unsafe_code)]`; the heap-owning constructor sits behind the
//! default `alloc` feature, and the core algorithm uses only [`core`].

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod bins;
mod layout;
mod pool;

pub use layout::Align64;
pub use pool::{Region, RtPool};
