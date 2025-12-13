# crvecdb

A fast vector database library with HNSW indexing for Rust.

## Features

- **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast approximate nearest neighbor search
- **Multiple Distance Metrics** - Cosine, Euclidean (L2), Dot Product
- **SIMD Acceleration** - Cross-platform support for ARM NEON and x86 SSE/AVX2
- **Memory-Mapped Storage** - Persistent indexes with automatic memory mapping
- **Parallel Operations** - Optional Rayon integration for concurrent insertions

## Installation

```toml
[dependencies]
crvecdb = "0.1"
```

## Quick Start

```rust
use crvecdb::{Index, DistanceMetric};

// Create an in-memory index
let mut index = Index::builder(128)  // 128 dimensions
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

## Persistent Storage

```rust
use crvecdb::{Index, DistanceMetric};

// Create a memory-mapped index
let mut index = Index::builder(768)
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
parallel = ["rayon"]    # Parallel operations
```

Disable default features for minimal builds:

```toml
[dependencies]
crvecdb = { version = "0.1", default-features = false }
```

## Performance

Typical performance on modern hardware (measured with 256-dim vectors):

| Operation | Throughput |
|-----------|------------|
| Insert | ~50,000 vectors/sec |
| Search (k=10) | ~100,000 queries/sec |

## License

MIT OR Apache-2.0
