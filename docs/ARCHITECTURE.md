# crvecdb Architecture

## Overview

crvecdb is a pure Rust HNSW vector index library with SIMD acceleration and memory-mapped persistence. This document covers the internal architecture, data structures, file formats, and extension points.

## Crate Structure

```
src/
├── lib.rs              # Public API re-exports
├── index.rs            # Index type: builder, insert, search (450 LOC)
├── config.rs           # HnswConfig, IndexConfig, DistanceMetric
├── error.rs            # CrvecError enum
├── hnsw/
│   ├── mod.rs          # Module root
│   ├── graph.rs        # HnswGraph: nodes, layers, neighbors (492 LOC)
│   └── search.rs       # Search algorithm: greedy + beam search
├── storage/
│   ├── mod.rs          # VectorStorage trait
│   ├── memory.rs       # In-memory storage (UnsafeCell + atomics)
│   ├── mmap.rs         # Memory-mapped file storage
│   └── layout.rs       # File format: header, vector stride
└── distance/
    ├── mod.rs          # Distance trait, metric dispatch
    ├── simd.rs         # SIMD-accelerated: dot, euclidean, norm (simdeez)
    └── scalar.rs       # Scalar fallback implementations
```

## Data Flow

```
insert(id, vector)
    ↓
VectorStorage::push(id, vector)     ← atomic slot allocation
    ↓
HnswGraph::insert(internal_id)
    ↓
  1. Random layer assignment: floor(-ln(random) / ln(M))
  2. Greedy descent from top layer to insertion layer
  3. Beam search at each insertion layer (ef_construction width)
  4. Bidirectional neighbor connections + pruning
  5. Atomic CAS entry point update (if new max layer)

search(query, k)
    ↓
  1. Greedy descent from max_layer to layer 1
  2. Beam search at layer 0 (ef_search width)
  3. Return top-k by distance
```

## HNSW Graph

Each vector occupies a node in a multi-layer skip-list graph:

```
Layer 3:  [N7] ──────────────── [N42]
Layer 2:  [N7] ── [N15] ────── [N42] ── [N91]
Layer 1:  [N7] ── [N15] ── [N23] ── [N42] ── [N67] ── [N91]
Layer 0:  [N1] [N3] [N7] [N12] [N15] [N23] [N31] [N42] [N55] [N67] [N78] [N91]
```

- MAX_LAYERS = 16
- m (default 16): max connections per layer > 0
- m0 (default 2*m = 32): max connections at layer 0 (denser for recall)
- ml = 1/ln(M): layer probability factor
- ef_construction (default 200): search width during insert
- ef_search (default 50): search width during query

**Node Structure:**
```rust
HnswNode {
    max_layer: usize,
    neighbors: Vec<RwLock<Vec<InternalId>>>,  // per-layer neighbor lists
}
```

## Storage Backends

### VectorStorage Trait

```rust
trait VectorStorage: Send + Sync {
    fn get(&self, id: InternalId) -> &[f32];
    fn get_id(&self, id: InternalId) -> VectorId;
    fn get_norm(&self, id: InternalId) -> f32;
    fn push(&self, id: VectorId, vector: &[f32]) -> InternalId;
    fn len(&self) -> usize;
    fn dimension(&self) -> usize;
}
```

### MemoryStorage

- Flat arrays: `ids: Vec<u64>`, `norms: Vec<f32>`, `vectors: Vec<f32>`
- Pre-allocated to capacity
- Thread-safe via `UnsafeCell` + `AtomicUsize` slot counter
- Each `push()` atomically claims a slot with `fetch_add(AcqRel)`

### MmapStorage

- Single file with 64-byte header + vector data
- Vector stride aligned to 32 bytes (SIMD-friendly)
- `parking_lot::RwLock<MmapMut>` for concurrent access
- Persistence: OS dirty-page writeback + explicit `flush()`

## File Format

### Storage File (.db)

```
Offset  Size   Field
0       8      Magic: "CRVECDB\0"
8       4      Version: 1
12      4      Dimension (u32)
16      8      Count (u64, vectors stored)
24      8      Capacity (u64)
32      4      Metric (u32: 0=Cosine, 1=Euclidean, 2=DotProduct)
36      28     Reserved (zero-filled)
64      ...    Vector data (stride-aligned entries)

Per vector entry (stride = align_to_32(16 + dimension * 4)):
  0     8      VectorId (u64)
  8     4      L2 Norm (f32, precomputed)
  12    4      Padding
  16    dim*4  Vector data ([f32; dimension])
  ...   pad    Zero padding to 32-byte boundary
```

### Graph File (.graph)

```
Offset  Size   Field
0       4      Magic: "HNSW"
4       4      Version: 1
8       4      Entry point (u32, MAX = empty)
12      4      Max level (u32)
16      4      M parameter (u32)
20      4      M0 parameter (u32)
24      4      Num nodes (u32)
28      ...    Node data

Per node:
  0     1      max_layer (u8)
  Per layer 0..=max_layer:
    0   2      num_neighbors (u16)
    2   n*4    neighbor IDs ([u32; num_neighbors])
```

## Distance Computation

### SIMD Dispatch (via simdeez)

Runtime detection selects the best instruction set:
- ARM: NEON (128-bit, 4x f32)
- x86: AVX2 > AVX > SSE > Scalar

Three accelerated operations:
- `dot_product(a, b)` - used by Cosine and DotProduct metrics
- `squared_euclidean(a, b)` - used by Euclidean metric
- `l2_norm(v)` - precomputed at insert time for Cosine

### Distance Trait

```rust
trait Distance: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
    fn distance_with_norm(&self, a: &[f32], b: &[f32], norm_a: f32) -> f32;
    fn uses_norm(&self) -> bool;
}
```

**Cosine:** `1.0 - dot(a,b) / (norm_a * norm_b)` (uses precomputed norms)
**Euclidean:** `sum((a-b)^2)` (squared L2, no sqrt)
**DotProduct:** `-dot(a,b)` (negated so lower = more similar)

## Thread Safety

| Component | Mechanism | Contention |
|-----------|-----------|------------|
| Storage slot allocation | `AtomicUsize::fetch_add` | None (CAS-free) |
| Neighbor lists | `parking_lot::RwLock<Vec<InternalId>>` | Per-node, low |
| Entry point | `AtomicU32` with CAS | Only on new max layer |
| Max level | `AtomicUsize` | Only on new max layer |
| Mmap access | `parking_lot::RwLock<MmapMut>` | Read-heavy, low |

Parallel insert (`rayon`): each thread independently allocates slots and builds connections. Write contention only occurs on shared neighbor lists of adjacent nodes.

## Current Limitations

1. **Insert-only** - No update or delete operations
2. **Fixed capacity** - Must specify at creation, no dynamic growth
3. **No filtering** - Pure vector similarity, no metadata predicates
4. **No quantization** - All vectors stored as full f32
5. **Simple neighbor selection** - Closest-K heuristic (no diversity-aware)
6. **Single dimension** - All vectors must be same size
7. **No compaction** - Deleted space not reclaimed (delete not supported)
8. **Graph in memory** - Full HNSW graph must fit in RAM

## Memory Usage

```
Vectors: count * (16 + dimension * 4) bytes  (+ padding to 32B)
Graph:   count * (1 + avg_layers * (2 + avg_neighbors * 4)) bytes
Norms:   count * 4 bytes (included in vector entry)

Example: 10K vectors, 768 dimensions
  Vectors: 10K * (16 + 768*4) = ~30 MB
  Graph:   10K * ~200 bytes   = ~2 MB
  Total:   ~32 MB
```
