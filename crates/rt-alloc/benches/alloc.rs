//! Allocation latency benchmarks, and a side-by-side comparison against `offset-allocator` (a
//! proven O(1) allocator) over the same workload - the data that would justify swapping engines if
//! the faithful scsynth port ever underperforms.
//!
//! Both allocators run a "fill then drain" churn over a deterministic sequence of mixed sizes, and
//! a "fragmenting" pass that frees every other block then refills the gaps. Setup (building the
//! arena) is excluded from timing via `iter_batched_ref`.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use offset_allocator::Allocator;
use rt_alloc::{Align64, RtPool};

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
                // Free every other allocation, opening holes.
                for slot in held.iter_mut().step_by(2) {
                    if let Some(r) = slot.take() {
                        pool.dealloc(r);
                    }
                }
                // Refill the holes with small blocks.
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

criterion_group!(benches, fill_drain, fragmenting);
criterion_main!(benches);
