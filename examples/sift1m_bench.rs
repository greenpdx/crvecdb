//! SIFT1M Benchmark - Standard ANN benchmark
//!
//! Tests recall and QPS against the SIFT1M dataset.
//! Download: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

use crvecdb::{DistanceMetric, Index};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const DATA_DIR: &str = "data/sift";

fn main() {
    println!("=== SIFT1M Benchmark ===\n");

    // Load data
    println!("[1/4] Loading dataset...");
    let start = Instant::now();
    let base = read_fvecs(&format!("{}/sift_base.fvecs", DATA_DIR));
    let queries = read_fvecs(&format!("{}/sift_query.fvecs", DATA_DIR));
    let groundtruth = read_ivecs(&format!("{}/sift_groundtruth.ivecs", DATA_DIR));
    println!("  Base vectors:  {} x {}", base.len(), base[0].len());
    println!("  Query vectors: {} x {}", queries.len(), queries[0].len());
    println!("  Ground truth:  {} x {}", groundtruth.len(), groundtruth[0].len());
    println!("  Load time:     {:?}\n", start.elapsed());

    // Build index
    println!("[2/4] Building index (parallel)...");
    let start = Instant::now();
    let index = Index::builder(128)
        .metric(DistanceMetric::Euclidean)
        .m(16)
        .ef_construction(200)
        .capacity(base.len())
        .build()
        .unwrap();

    // Prepare batch for parallel insert
    let batch: Vec<_> = base.iter().enumerate()
        .map(|(i, vec)| (i as u64, vec.clone()))
        .collect();

    index.insert_parallel(&batch).unwrap();
    let build_time = start.elapsed();
    println!("  Build time:    {:?}", build_time);
    println!("  Vectors/sec:   {:.0}\n", base.len() as f64 / build_time.as_secs_f64());

    // Benchmark search (parallel)
    println!("[3/4] Benchmarking search (parallel)...\n");

    for &k in &[1, 10, 100] {
        let start = Instant::now();
        let correct = AtomicU64::new(0);

        queries.par_iter().enumerate().for_each(|(i, query)| {
            let results = index.search(query, k).unwrap();
            let gt_set: std::collections::HashSet<_> = groundtruth[i][..k].iter().copied().collect();

            let mut local_correct = 0u64;
            for result in &results {
                if gt_set.contains(&(result.id as i32)) {
                    local_correct += 1;
                }
            }
            correct.fetch_add(local_correct, Ordering::Relaxed);
        });

        let elapsed = start.elapsed();
        let total = queries.len() as u64 * k as u64;
        let recall = correct.load(Ordering::Relaxed) as f64 / total as f64 * 100.0;
        let qps = queries.len() as f64 / elapsed.as_secs_f64();

        println!("  Recall@{:<3} {:.2}%  |  QPS: {:.0}  |  Time: {:?}", k, recall, qps, elapsed);
    }

    // Latency distribution (single-threaded for accurate per-query timing)
    println!("\n[4/4] Latency distribution (k=10, single-threaded)...\n");
    let mut latencies: Vec<f64> = Vec::with_capacity(queries.len());

    for query in &queries {
        let start = Instant::now();
        let _ = index.search(query, 10).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0); // ms
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[latencies.len() * 95 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];
    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;

    println!("  Avg:  {:.3} ms", avg);
    println!("  P50:  {:.3} ms", p50);
    println!("  P95:  {:.3} ms", p95);
    println!("  P99:  {:.3} ms", p99);

    println!("\n=== Benchmark Complete ===");
}

/// Read fvecs format (float vectors)
fn read_fvecs(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).expect(&format!("Cannot open {}", path));
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        // Read dimension (4 bytes, little endian i32)
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        // Read vector data
        let mut data = vec![0u8; dim * 4];
        reader.read_exact(&mut data).expect("Failed to read vector");

        let vec: Vec<f32> = data
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        vectors.push(vec);
    }

    vectors
}

/// Read ivecs format (integer vectors for ground truth)
fn read_ivecs(path: &str) -> Vec<Vec<i32>> {
    let file = File::open(path).expect(&format!("Cannot open {}", path));
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        // Read vector data
        let mut data = vec![0u8; dim * 4];
        reader.read_exact(&mut data).expect("Failed to read vector");

        let vec: Vec<i32> = data
            .chunks(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        vectors.push(vec);
    }

    vectors
}
