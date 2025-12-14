# crvecdb

A fast vector database library with HNSW indexing for Rust.

## Features

- **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast approximate nearest neighbor search
- **Multiple Distance Metrics** - Cosine, Euclidean (L2), Dot Product
- **SIMD Acceleration** - Cross-platform support for ARM NEON and x86 SSE/AVX2
- **Memory-Mapped Storage** - Persistent indexes with automatic memory mapping
- **Parallel Operations** - Rayon-powered parallel insert and search

## Installation

```toml
[dependencies]
crvecdb = "0.1"
```

## Quick Start

```rust
use crvecdb::{Index, DistanceMetric};

// Create an in-memory index
let index = Index::builder(128)  // 128 dimensions
    .metric(DistanceMetric::Cosine)
    .m(16)                           // HNSW connections per node
    .ef_construction(200)            // Build-time search width
    .capacity(10_000)
    .build()
    .unwrap();

// Insert vectors
index.insert(1, &vec![0.1; 128]).unwrap();
index.insert(2, &vec![0.2; 128]).unwrap();

// Search for nearest neighbors
let results = index.search(&vec![0.15; 128], 10).unwrap();
for result in results {
    println!("ID: {}, Distance: {:.4}", result.id, result.distance);
}
```

## Parallel Bulk Insert

```rust
use crvecdb::{Index, DistanceMetric};

let index = Index::builder(128)
    .metric(DistanceMetric::Euclidean)
    .capacity(1_000_000)
    .build()
    .unwrap();

// Prepare batch
let vectors: Vec<_> = (0..1_000_000)
    .map(|i| (i as u64, vec![0.1; 128]))
    .collect();

// Parallel insert - uses all CPU cores
index.insert_parallel(&vectors).unwrap();
```

## Persistent Storage

```rust
use crvecdb::{Index, DistanceMetric};

// Create a memory-mapped index
let index = Index::builder(768)
    .metric(DistanceMetric::DotProduct)
    .capacity(1_000_000)
    .build_mmap("/path/to/index.db")
    .unwrap();

// Data persists automatically
index.insert(1, &vec![0.1; 768]).unwrap();
index.flush().unwrap();  // Ensure durability
```

## Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `Cosine` | Normalized angular distance | Text embeddings, semantic search |
| `Euclidean` | L2 distance | Image features, spatial data |
| `DotProduct` | Inner product | Recommendation systems |

## HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Max connections per node. Higher = better recall, more memory |
| `ef_construction` | 200 | Search width during build. Higher = better graph, slower insert |
| `ef_search` | 50 | Search width at query time. Higher = better recall, slower search |

## Feature Flags

```toml
[features]
default = ["simd", "parallel"]
simd = ["simdeez"]      # SIMD acceleration
parallel = ["rayon"]    # Parallel insert and search
```

The `parallel` feature enables multi-threaded operations:
- `insert_parallel()` uses all CPU cores for bulk loading
- Search benchmarks run queries in parallel

Disable for single-threaded builds:

```toml
[dependencies]
crvecdb = { version = "0.1", default-features = false, features = ["simd"] }
```

## Performance

SIFT1M benchmark (1M vectors, 128 dimensions, Euclidean distance):

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Parallel Insert | 4,000 vectors/sec | m=16, ef_construction=200 |
| Parallel Search (k=10) | 4,000 QPS | 97% recall@10 |
| Single Query Latency | ~1ms p50 | |

## Benchmarks

### SIFT1M Benchmark

Download the dataset (not included in repo):

```bash
mkdir -p data/sift
cd data/sift
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* .
rmdir sift
rm sift.tar.gz
cd ../..
```

Run the benchmark:

```bash
cargo run --release --example sift1m_bench
```

Expected output:
```
=== SIFT1M Benchmark ===

[1/4] Loading dataset...
  Base vectors:  1000000 x 128
  Query vectors: 10000 x 128
  Ground truth:  10000 x 100

[2/4] Building index (parallel)...
  Build time:    ~4 minutes
  Vectors/sec:   ~4000

[3/4] Benchmarking search (parallel)...
  Recall@1   96.7%  |  QPS: ~4000
  Recall@10  97.1%  |  QPS: ~4000
  Recall@100 94.0%  |  QPS: ~4000

[4/4] Latency distribution (k=10, single-threaded)...
  Avg:  ~1.0 ms
  P50:  ~1.0 ms
  P95:  ~1.5 ms
  P99:  ~1.7 ms
```

## License

MIT OR Apache-2.0
