use std::sync::Arc;
use std::thread;

use crvecdb::{DistanceMetric, Index};

fn gen_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut rng = seed;
    (0..dim)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / (u32::MAX as f32)
        })
        .collect()
}

fn run_stress(seed_count: usize, per_writer: usize, num_writers: usize, num_readers: usize, searches_per_reader: usize) {
    let dim = 16;
    let capacity = seed_count + per_writer * num_writers + 128;

    let index = Arc::new(
        Index::builder(dim)
            .metric(DistanceMetric::Euclidean)
            .m(16)
            .ef_construction(100)
            .capacity(capacity)
            .build()
            .unwrap(),
    );

    // Seed so search has an entry point.
    for i in 0..seed_count as u64 {
        index.insert(i, &gen_vec(i, dim)).unwrap();
    }

    let mut handles = Vec::new();

    for w in 0..num_writers {
        let index = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let base = seed_count as u64 + (w as u64) * (per_writer as u64);
            for i in 0..per_writer {
                let id = base + i as u64;
                let v = gen_vec(id, dim);
                index.insert(id, &v).unwrap();
            }
        }));
    }

    for r in 0..num_readers {
        let index = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 0..searches_per_reader as u64 {
                let q = gen_vec(i.wrapping_add(1_000_000 * r as u64 + 7), dim);
                let _ = index.search(&q, 10).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn concurrent_insert_and_search_stress() {
    for _ in 0..20 {
        run_stress(4, 500, 8, 8, 2_000);
    }
}

#[test]
fn concurrent_insert_and_search_tiny_seed() {
    // Start with very few seeds — maximizes contention during graph construction.
    for _ in 0..50 {
        run_stress(1, 200, 8, 4, 1_000);
    }
}

#[test]
fn zero_seed_then_bombard() {
    // Start with NO seeds at all. Many threads race on the first insert.
    let dim = 8;
    let total = 2_000;

    for _ in 0..30 {
        let index = Arc::new(
            Index::builder(dim)
                .metric(DistanceMetric::Euclidean)
                .m(8)
                .ef_construction(50)
                .capacity(total + 128)
                .build()
                .unwrap(),
        );

        let mut handles = Vec::new();
        for w in 0..12 {
            let index = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for i in 0..(total / 12) as u64 {
                    let id = w as u64 * 100_000 + i;
                    let v = gen_vec(id, dim);
                    index.insert(id, &v).unwrap();
                }
            }));
        }
        for r in 0..8 {
            let index = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for i in 0..5_000u64 {
                    let q = gen_vec(i.wrapping_add(r as u64 * 10101), dim);
                    let _ = index.search(&q, 5).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}

#[test]
fn parallel_insert_and_concurrent_search() {
    let dim = 16;
    let total = 4_000;
    let capacity = total + 128;

    for _ in 0..10 {
        let index = Arc::new(
            Index::builder(dim)
                .metric(DistanceMetric::Euclidean)
                .m(16)
                .ef_construction(100)
                .capacity(capacity)
                .build()
                .unwrap(),
        );

        // Seed so search has entry
        for i in 0..4u64 {
            index.insert(i, &gen_vec(i, dim)).unwrap();
        }

        let vectors: Vec<(u64, Vec<f32>)> = (4..total as u64)
            .map(|i| (i, gen_vec(i, dim)))
            .collect();

        let idx2 = Arc::clone(&index);
        let reader = thread::spawn(move || {
            for i in 0..20_000u64 {
                let q = gen_vec(i.wrapping_add(7), dim);
                let _ = idx2.search(&q, 10).unwrap();
            }
        });

        index.insert_parallel(&vectors).unwrap();
        reader.join().unwrap();
    }
}
