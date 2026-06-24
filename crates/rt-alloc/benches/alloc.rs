//! Allocation benchmarks, with a side-by-side comparison against `offset-allocator` (a proven O(1)
//! allocator) over the same workloads - the data that would justify swapping engines if the faithful
//! scsynth port ever underperforms. Setup (building the arena, pre-filling) is excluded from timing
//! via `iter_batched_ref`.
//!
//! Two groups model how plyphon is expected to drive the pool, the way scsynth uses `World_Alloc`:
//!
//! - `synth_lifecycle` - the primary workload. Like scsynth, a synth's wire buffers and per-ugen
//!   working buffers (delay lines, etc.) are allocated from the shared RT pool at instantiation and
//!   freed at node death, giving buffer reuse across synths. Medium, variable sizes (KB-range wire
//!   arenas plus occasional tens-of-KB delay buffers), churned at voice rate, freed in non-LIFO
//!   order - real fragmentation pressure, mostly large bins.
//! - `scratch_churn` - per audio block, a handful of ugens borrow block-sized scratch (256 B = a
//!   64-sample `f32` block) plus a few control values (4 B), released LIFO. Small, uniform, very
//!   short-lived - the small-bin exact-size fast path, near-zero fragmentation.
//!
//! Two more are synthetic: `fill_drain` (raw alloc/free throughput) and `fragmenting` (a worst-case
//! fragmentation pattern). They bound the extremes rather than model real use.

use std::collections::VecDeque;
use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use offset_allocator::{Allocation, Allocator};
use rt_alloc::{Align64, Region, RtPool};

const POOL_BYTES: usize = 8 * 1024 * 1024;

/// A deterministic xorshift sequence of request sizes in `[16, 1040)`.
fn sizes(n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    let mut x: u32 = 0x9e37_79b9;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        out.push((x as usize % 1024) + 16);
    }
    out
}

/// Buffer-size groups for a few representative synth shapes. Each inner slice is one synth's set of
/// RT-pool allocations: a wire arena (`n_wires * block_size * 4` bytes), optionally plus a delay-line
/// buffer. Mostly large-bin sized, matching scsynth's per-graph allocations.
const VOICE_PROFILES: [&[usize]; 4] = [
    &[2_048],         // ~8-wire synth
    &[4_096],         // ~16-wire synth
    &[8_192],         // ~32-wire synth
    &[8_192, 38_400], // ~32 wires + a ~0.2 s delay line
];

fn voice_profile(seed: u32) -> &'static [usize] {
    VOICE_PROFILES[(seed as usize) % VOICE_PROFILES.len()]
}

fn synth_lifecycle(c: &mut Criterion) {
    const LIVE: usize = 64; // steady-state voice population
    const CYCLES: usize = 64; // voices replaced per timed run
    let mut group = c.benchmark_group("synth_lifecycle");
    group.throughput(Throughput::Elements(CYCLES as u64));

    group.bench_function("rt_alloc", |b| {
        b.iter_batched_ref(
            || {
                let mut pool = RtPool::<Box<[Align64]>>::with_capacity_bytes(POOL_BYTES);
                let mut voices: VecDeque<Vec<Region>> = VecDeque::with_capacity(LIVE + 1);
                for i in 0..LIVE as u32 {
                    voices.push_back(
                        voice_profile(i)
                            .iter()
                            .filter_map(|&s| pool.alloc(s))
                            .collect(),
                    );
                }
                (pool, voices, LIVE as u32)
            },
            |(pool, voices, next)| {
                for _ in 0..CYCLES {
                    if let Some(voice) = voices.pop_front() {
                        for r in voice {
                            pool.dealloc(r);
                        }
                    }
                    voices.push_back(
                        voice_profile(*next)
                            .iter()
                            .filter_map(|&s| pool.alloc(s))
                            .collect(),
                    );
                    *next += 1;
                }
                black_box(voices.len());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("offset_allocator", |b| {
        b.iter_batched_ref(
            || {
                let mut alloc = Allocator::<u32>::new(POOL_BYTES as u32);
                let mut voices: VecDeque<Vec<Allocation>> = VecDeque::with_capacity(LIVE + 1);
                for i in 0..LIVE as u32 {
                    voices.push_back(
                        voice_profile(i)
                            .iter()
                            .filter_map(|&s| alloc.allocate(s as u32))
                            .collect(),
                    );
                }
                (alloc, voices, LIVE as u32)
            },
            |(alloc, voices, next)| {
                for _ in 0..CYCLES {
                    if let Some(voice) = voices.pop_front() {
                        for a in voice {
                            alloc.free(a);
                        }
                    }
                    voices.push_back(
                        voice_profile(*next)
                            .iter()
                            .filter_map(|&s| alloc.allocate(s as u32))
                            .collect(),
                    );
                    *next += 1;
                }
                black_box(voices.len());
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn scratch_churn(c: &mut Criterion) {
    const UGENS: usize = 16; // ugens borrowing scratch per block
    const BLOCKS: usize = 64;
    const AUDIO: usize = 64 * 4; // a 64-sample f32 block = 256 B
    let mut group = c.benchmark_group("scratch_churn");
    group.throughput(Throughput::Elements((UGENS * BLOCKS) as u64));

    group.bench_function("rt_alloc", |b| {
        b.iter_batched_ref(
            || RtPool::<Box<[Align64]>>::with_capacity_bytes(256 * 1024),
            |pool| {
                let mut held = Vec::with_capacity(UGENS);
                for _ in 0..BLOCKS {
                    for i in 0..UGENS {
                        let size = if i % 4 == 0 { 4 } else { AUDIO }; // mix in control-rate values
                        if let Some(r) = pool.alloc(size) {
                            held.push(r);
                        }
                    }
                    while let Some(r) = held.pop() {
                        pool.dealloc(r); // LIFO release, as a block's scratch unwinds
                    }
                }
                black_box(&held);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("offset_allocator", |b| {
        b.iter_batched_ref(
            || Allocator::<u32>::new(256 * 1024),
            |alloc| {
                let mut held: Vec<Allocation> = Vec::with_capacity(UGENS);
                for _ in 0..BLOCKS {
                    for i in 0..UGENS {
                        let size = if i % 4 == 0 { 4 } else { AUDIO as u32 };
                        if let Some(a) = alloc.allocate(size) {
                            held.push(a);
                        }
                    }
                    while let Some(a) = held.pop() {
                        alloc.free(a);
                    }
                }
                black_box(&held);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn fill_drain(c: &mut Criterion) {
    let mut group = c.benchmark_group("fill_drain");
    for &n in &[64usize, 256, 1024] {
        let szs = sizes(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("rt_alloc", n), &szs, |b, szs| {
            b.iter_batched_ref(
                || RtPool::<Box<[Align64]>>::with_capacity_bytes(POOL_BYTES),
                |pool| {
                    let mut held = Vec::with_capacity(szs.len());
                    for &s in szs {
                        if let Some(r) = pool.alloc(s) {
                            held.push(r);
                        }
                    }
                    for r in held.drain(..) {
                        pool.dealloc(r);
                    }
                    black_box(&mut held);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("offset_allocator", n), &szs, |b, szs| {
            b.iter_batched_ref(
                || Allocator::<u32>::new(POOL_BYTES as u32),
                |alloc| {
                    let mut held = Vec::with_capacity(szs.len());
                    for &s in szs {
                        if let Some(a) = alloc.allocate(s as u32) {
                            held.push(a);
                        }
                    }
                    for a in held.drain(..) {
                        alloc.free(a);
                    }
                    black_box(&mut held);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn fragmenting(c: &mut Criterion) {
    let n = 1024usize;
    let szs = sizes(n);
    let mut group = c.benchmark_group("fragmenting");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("rt_alloc", |b| {
        b.iter_batched_ref(
            || RtPool::<Box<[Align64]>>::with_capacity_bytes(POOL_BYTES),
            |pool| {
                let mut held: Vec<Option<_>> = szs.iter().map(|&s| pool.alloc(s)).collect();
                for slot in held.iter_mut().step_by(2) {
                    if let Some(r) = slot.take() {
                        pool.dealloc(r);
                    }
                }
                for slot in held.iter_mut().step_by(2) {
                    *slot = pool.alloc(48);
                }
                for r in held.into_iter().flatten() {
                    pool.dealloc(r);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("offset_allocator", |b| {
        b.iter_batched_ref(
            || Allocator::<u32>::new(POOL_BYTES as u32),
            |alloc| {
                let mut held: Vec<Option<_>> =
                    szs.iter().map(|&s| alloc.allocate(s as u32)).collect();
                for slot in held.iter_mut().step_by(2) {
                    if let Some(a) = slot.take() {
                        alloc.free(a);
                    }
                }
                for slot in held.iter_mut().step_by(2) {
                    *slot = alloc.allocate(48);
                }
                for a in held.into_iter().flatten() {
                    alloc.free(a);
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    synth_lifecycle,
    scratch_churn,
    fill_drain,
    fragmenting
);
criterion_main!(benches);
