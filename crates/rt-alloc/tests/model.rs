//! Property test: drive random `alloc`/`dealloc`/`realloc` sequences and check the pool against a
//! reference model held alongside it. Each live region is filled with a unique, position-dependent
//! byte pattern; re-reading every pattern after every operation catches any overlap or corruption
//! (an overlap would scribble one region's bytes over another's). We also assert the accounting
//! balances after every step and that draining everything coalesces back to a single free arena.

use proptest::collection::vec;
use proptest::prelude::*;
use rt_alloc::{Align64, Region, RtPool};

type Pool = RtPool<Box<[Align64]>>;

const POOL_BYTES: usize = 64 * 1024;
const MAX_REQUEST: usize = 2048;

#[derive(Debug, Clone)]
enum Op {
    Alloc(usize),
    Free(usize),
    Realloc(usize, usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (1usize..MAX_REQUEST).prop_map(Op::Alloc),
        (0usize..256).prop_map(Op::Free),
        (0usize..256, 1usize..MAX_REQUEST).prop_map(|(i, s)| Op::Realloc(i, s)),
    ]
}

/// A unique, position-dependent pattern keyed by a per-allocation id.
fn fill(slice: &mut [u8], id: u32) {
    for (i, b) in slice.iter_mut().enumerate() {
        *b = id.wrapping_add(i as u32) as u8;
    }
}

fn intact(slice: &[u8], id: u32) -> bool {
    slice
        .iter()
        .enumerate()
        .all(|(i, &b)| b == id.wrapping_add(i as u32) as u8)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn pool_matches_reference(ops in vec(op(), 0..200)) {
        let mut pool: Pool = RtPool::with_capacity_bytes(POOL_BYTES);
        let mut live: Vec<(Region, u32)> = Vec::new();
        let mut next_id: u32 = 1;

        for op in ops {
            match op {
                Op::Alloc(size) => {
                    if let Some(r) = pool.alloc(size) {
                        prop_assert_eq!(r.len(), size);
                        // Payload is 64-byte aligned (the backing base is, and offsets are multiples).
                        prop_assert_eq!(pool.slice(&r).as_ptr() as usize % 64, 0, "payload must be aligned");
                        let id = next_id;
                        next_id += 1;
                        fill(pool.slice_mut(&r), id);
                        live.push((r, id));
                    }
                }
                Op::Free(i) => {
                    if !live.is_empty() {
                        let (r, _) = live.swap_remove(i % live.len());
                        pool.dealloc(r);
                    }
                }
                Op::Realloc(i, size) => {
                    if !live.is_empty() {
                        let (r, id) = live.swap_remove(i % live.len());
                        let old_len = r.len();
                        match pool.realloc(r, size) {
                            Ok(nr) => {
                                prop_assert_eq!(nr.len(), size);
                                // The preserved prefix must still hold the old pattern.
                                let keep = old_len.min(size);
                                prop_assert!(intact(&pool.slice(&nr)[..keep], id));
                                // Re-stamp the whole new region with a fresh id.
                                let new_id = next_id;
                                next_id += 1;
                                fill(pool.slice_mut(&nr), new_id);
                                live.push((nr, new_id));
                            }
                            Err(orig) => {
                                // Grow failed (out of memory); the original is unchanged.
                                prop_assert!(intact(pool.slice(&orig), id));
                                live.push((orig, id));
                            }
                        }
                    }
                }
            }

            // Accounting balances every step.
            prop_assert_eq!(pool.used_bytes() + pool.free_bytes(), pool.total_bytes());
            // No live region's bytes were disturbed by any other operation.
            for (r, id) in &live {
                prop_assert!(intact(pool.slice(r), *id), "a live region was corrupted");
            }
        }

        // Draining everything must coalesce back to one fully-free arena.
        for (r, _) in live {
            pool.dealloc(r);
        }
        prop_assert_eq!(pool.used_bytes(), 0);
        prop_assert_eq!(pool.free_bytes(), pool.total_bytes());
    }
}
