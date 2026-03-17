# crvecdb API Reference

## Types

### `VectorId`
```rust
pub type VectorId = u64;
```
User-provided identifier for a vector. Must be unique within an index.

### `InternalId`
```rust
pub type InternalId = u32;
```
Internal storage slot index. Not exposed in the public API except via `SearchResult`.

### `DistanceMetric`
```rust
pub enum DistanceMetric {
    Cosine,     // 1 - cos(a,b). Range [0,2]. 0 = identical.
    Euclidean,  // ||a-b||^2 (squared L2). 0 = identical.
    DotProduct, // -dot(a,b). Lower = more similar.
}
```

### `SearchResult`
```rust
pub struct SearchResult {
    pub id: VectorId,    // User-provided ID
    pub distance: f32,   // Lower = more similar
}
```

### `CrvecError`
```rust
pub enum CrvecError {
    DimensionMismatch { expected: usize, got: usize },
    EmptyIndex,
    InvalidFormat(String),
    Io(std::io::Error),
    CapacityExceeded { capacity: usize },
    NotFound(VectorId),
    InvalidParameter(String),
}
```

## Index

### Builder

```rust
let index = Index::builder(dimension: usize)
    .metric(DistanceMetric::Cosine)    // default: Cosine
    .m(16)                              // default: 16
    .ef_construction(200)               // default: 200
    .capacity(10_000)                   // default: 10_000
    .build()?;                          // in-memory

let index = Index::builder(768)
    .build_mmap("/path/to/index.db")?;  // persistent
```

### Open Existing

```rust
let index = Index::open_mmap("/path/to/index.db")?;
```
Reopens a previously flushed index. Both the `.db` (vectors) and `.graph` (HNSW) files must exist.

### Insert

```rust
index.insert(id: u64, vector: &[f32]) -> Result<(), CrvecError>
```
Insert a single vector. Dimension must match the index dimension.

```rust
index.insert_batch(vectors: &[(u64, Vec<f32>)]) -> Result<(), CrvecError>
```
Insert multiple vectors sequentially.

```rust
// Requires `parallel` feature
index.insert_parallel(vectors: &[(u64, Vec<f32>)]) -> Result<(), CrvecError>
```
Insert using all CPU cores via rayon.

### Search

```rust
index.search(query: &[f32], k: usize) -> Result<Vec<SearchResult>, CrvecError>
```
Find k nearest neighbors using default ef_search.

```rust
index.search_with_ef(query: &[f32], k: usize, ef_search: usize)
    -> Result<Vec<SearchResult>, CrvecError>
```
Search with custom ef_search (higher = better recall, slower).

### Metadata

```rust
index.len() -> usize           // Number of vectors stored
index.is_empty() -> bool       // True if no vectors
index.dimension() -> usize     // Vector dimension
```

### Persistence

```rust
index.flush() -> Result<(), CrvecError>
```
Persist vectors (via mmap sync) and save the HNSW graph to `.graph` file. Only meaningful for mmap-backed indexes.

## Distance Trait

For custom distance metrics:

```rust
pub trait Distance: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn distance_with_norm(&self, a: &[f32], b: &[f32], norm_a: f32) -> f32 {
        self.distance(a, b)  // default: ignore norm
    }
    fn uses_norm(&self) -> bool { false }
}
```

## Feature Flags

| Feature | Default | Dependencies | Effect |
|---------|---------|-------------|--------|
| `simd` | Yes | `simdeez` | ARM NEON + x86 SSE/AVX2 acceleration |
| `parallel` | Yes | `rayon` | `insert_parallel()` and parallel benchmarks |

## Performance Tuning

| Scenario | m | ef_construction | ef_search | Notes |
|----------|---|-----------------|-----------|-------|
| RPi (low memory) | 8 | 100 | 30 | Halves graph memory |
| Balanced | 16 | 200 | 50 | Default, good recall/speed |
| High recall | 32 | 400 | 200 | 99%+ recall, 4x memory |
| Max speed | 8 | 100 | 20 | Lower recall (~90%) |
