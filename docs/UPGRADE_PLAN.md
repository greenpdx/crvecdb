# crvecdb Upgrade Plan: Embedded Vector Database for IronClaw

## Goal

Evolve crvecdb from a vector index library into a lightweight embedded database that can replace PostgreSQL+pgvector for ironclaw on Raspberry Pi, by combining crvecdb's HNSW vector search with a pure Rust relational storage engine.

## Gap Analysis

### What ironclaw needs (78+ Database trait methods)

| Capability | crvecdb today | Required |
|-----------|--------------|----------|
| Vector similarity search (HNSW) | Yes | Yes |
| Relational tables (20+) | No | Yes |
| SQL or structured queries | No | Yes |
| ACID transactions | No | Yes |
| Full-text search | No | Yes |
| JSON/document storage | No | Yes |
| Upserts (ON CONFLICT) | No | Yes |
| Foreign keys / cascades | No | Nice-to-have |
| Aggregation (COUNT, SUM) | No | Yes |
| Vector insert/update/delete | Insert only | Yes |
| Metadata filtering on search | No | Yes |
| Dynamic capacity (no fixed limit) | No | Yes |

### Recommended Architecture: redb + Tantivy + crvecdb

Rather than building SQL from scratch, combine three battle-tested pure Rust libraries:

```
┌─────────────────────────────────────────────┐
│              crdb (new crate)               │
│         Unified embedded database           │
├─────────────┬──────────────┬────────────────┤
│   redb      │   Tantivy    │   crvecdb      │
│  (storage)  │   (FTS)      │  (vectors)     │
│             │              │                │
│ Tables      │ Inverted     │ HNSW graph     │
│ Indexes     │ index        │ SIMD distance  │
│ Transactions│ Tokenizers   │ Mmap persist   │
│ K-V + range │ BM25 ranking │ ANN search     │
└─────────────┴──────────────┴────────────────┘
        │              │              │
        └──── Single data directory ──┘
              ~/.ironclaw/crdb/
```

**Why this combination:**
- **redb**: Pure Rust, ACID transactions, MVCC, key-value with range queries, ~1MB binary overhead, designed for embedded. File format is stable.
- **Tantivy**: Pure Rust, full-text search with BM25 ranking, tokenizers, field-level search. Used by Meilisearch in production.
- **crvecdb**: Already built, SIMD-accelerated HNSW, ARM NEON support, mmap persistence.

**Why NOT other options:**
- **Fjall**: Great K-V but LSM write amplification wastes RPi SD card life
- **SurrealDB**: Too heavy for embedded RPi (pulls in entire query engine)
- **Limbo**: Not production-ready (alpha)
- **sled**: Stalled rewrite, uncertain future
- **PoloDB**: MongoDB-like, not relational enough for ironclaw's 20+ table schema

## Phase 1: crvecdb Core Upgrades

### 1.1 Vector Delete + Update

**Current:** Insert-only
**Needed:** Delete vectors, update embeddings (ironclaw re-embeds chunks)

```rust
// New API
index.delete(vector_id) -> Result<(), CrvecError>
index.update(vector_id, new_vector) -> Result<(), CrvecError>
```

**Implementation:**
- Add a `deleted` bitset (`Vec<AtomicBool>`) to storage
- `delete()` marks the bit, search skips deleted nodes
- `update()` = soft-delete old + insert new
- Graph connections to deleted nodes: lazy cleanup during search (remove dead neighbors when encountered)
- Mmap: mark slot as tombstone (zero the VectorId)

**Files:** `src/index.rs`, `src/storage/memory.rs`, `src/storage/mmap.rs`, `src/hnsw/search.rs`

### 1.2 Dynamic Capacity

**Current:** Fixed capacity at creation
**Needed:** Grow as vectors are added

**Implementation:**
- MemoryStorage: double the backing arrays when full (amortized O(1))
- MmapStorage: `mremap()` or reopen with larger file + copy header
- Graph: `Vec<HnswNode>` already grows dynamically

**Files:** `src/storage/memory.rs`, `src/storage/mmap.rs`, `src/index.rs`

### 1.3 Filtered Search

**Current:** Pure vector similarity
**Needed:** `search_with_filter(query, k, |id| predicate)` for metadata-aware search

```rust
index.search_filtered(
    &query_vec,
    k,
    |id: VectorId| -> bool { /* user-defined predicate */ }
) -> Vec<SearchResult>
```

**Implementation:**
- Modify search beam to skip candidates where `filter(id)` returns false
- Increase ef_search dynamically when many candidates are filtered
- Post-filter approach as fallback for very selective filters

**Files:** `src/hnsw/search.rs`, `src/index.rs`

### 1.4 Batch Operations

```rust
index.insert_batch(&[(id, vec), ...]) -> Result<Vec<InternalId>>
index.delete_batch(&[id, ...]) -> Result<()>
```

**Files:** `src/index.rs`

## Phase 2: crdb — The Unified Database Layer

New crate `crdb` that wraps redb + Tantivy + crvecdb into a single embedded database.

### 2.1 Dependencies

```toml
[package]
name = "crdb"

[dependencies]
redb = "2"
tantivy = "0.22"
crvecdb = { path = "../crvecdb" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = { version = "1", features = ["v4"] }
thiserror = "2"
```

### 2.2 Core Types

```rust
/// A row in a table, stored as JSON in redb.
pub type Row = serde_json::Value;

/// Table definition with optional FTS and vector fields.
pub struct TableDef {
    pub name: &'static str,
    pub primary_key: &'static str,
    pub fts_fields: &'static [&'static str],      // Tantivy-indexed
    pub vector_field: Option<VectorFieldDef>,       // crvecdb-indexed
    pub indexes: &'static [IndexDef],               // redb secondary indexes
}

pub struct VectorFieldDef {
    pub field_name: &'static str,
    pub dimensions: usize,
    pub metric: DistanceMetric,
}

pub struct IndexDef {
    pub name: &'static str,
    pub fields: &'static [&'static str],
    pub unique: bool,
}
```

### 2.3 Database API

```rust
pub struct Database {
    redb: redb::Database,
    tantivy: tantivy::Index,
    vectors: HashMap<String, crvecdb::Index>,  // per-table vector indexes
    path: PathBuf,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self>;

    // CRUD
    pub fn insert(&self, table: &str, row: &Row) -> Result<()>;
    pub fn get(&self, table: &str, key: &str) -> Result<Option<Row>>;
    pub fn update(&self, table: &str, key: &str, row: &Row) -> Result<()>;
    pub fn delete(&self, table: &str, key: &str) -> Result<()>;

    // Queries
    pub fn query(&self, table: &str) -> QueryBuilder;
    pub fn count(&self, table: &str, filter: &Filter) -> Result<u64>;

    // Full-text search
    pub fn search_text(&self, table: &str, query: &str, limit: usize) -> Result<Vec<Row>>;

    // Vector search
    pub fn search_vector(
        &self, table: &str, vector: &[f32], k: usize
    ) -> Result<Vec<(Row, f32)>>;

    // Hybrid search (FTS + vector with RRF fusion)
    pub fn search_hybrid(
        &self, table: &str, text: &str, vector: &[f32], k: usize
    ) -> Result<Vec<(Row, f32)>>;

    // Transactions
    pub fn transaction<F, R>(&self, f: F) -> Result<R>
    where F: FnOnce(&Transaction) -> Result<R>;
}
```

### 2.4 Query Builder

```rust
pub struct QueryBuilder<'a> {
    db: &'a Database,
    table: &'a str,
    filters: Vec<Filter>,
    order_by: Option<(String, Order)>,
    limit: Option<usize>,
    offset: Option<usize>,
}

pub enum Filter {
    Eq(String, serde_json::Value),
    Ne(String, serde_json::Value),
    Gt(String, serde_json::Value),
    Lt(String, serde_json::Value),
    In(String, Vec<serde_json::Value>),
    Like(String, String),
    IsNull(String),
    IsNotNull(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
}

// Usage:
let rows = db.query("workspace_chunks")
    .filter(Filter::Eq("user_id".into(), json!("default")))
    .filter(Filter::IsNotNull("embedding".into()))
    .order_by("updated_at", Order::Desc)
    .limit(50)
    .execute()?;
```

### 2.5 Storage Layout

```
~/.ironclaw/crdb/
├── data.redb              # All table data (redb file)
├── tantivy/               # Full-text search indexes
│   ├── workspace_files/   # Per-table FTS index
│   └── ...
└── vectors/               # Per-table vector indexes
    ├── workspace_chunks.db    # crvecdb storage
    ├── workspace_chunks.graph # crvecdb HNSW graph
    └── ...
```

## Phase 3: ironclaw Database Backend

### 3.1 New Backend: `crdb`

Implement ironclaw's `Database` supertrait using crdb:

```rust
// src/db/crdb/mod.rs
pub struct CrdbBackend {
    db: crdb::Database,
}

impl Database for CrdbBackend { ... }
impl SettingsStore for CrdbBackend { ... }
impl WorkspaceStore for CrdbBackend { ... }
impl JobStore for CrdbBackend { ... }
impl SecretsStore for CrdbBackend { ... }
// ... all 7 sub-traits
```

### 3.2 Schema Mapping

ironclaw's 20+ tables map to redb tables with JSON rows:

| ironclaw table | redb table key | FTS fields | Vector field |
|---------------|----------------|------------|--------------|
| `workspace_files` | `(user_id, path)` | `content` | - |
| `workspace_chunks` | `chunk_id` | `content` | `embedding` (via crvecdb) |
| `settings` | `(user_id, key)` | - | - |
| `secrets` | `(user_id, name)` | - | - |
| `sandbox_jobs` | `job_id` | - | - |
| `wasm_tools` | `(user_id, name)` | - | - |
| `routines` | `routine_id` | - | - |
| `routine_runs` | `run_id` | - | - |
| `leak_detection_events` | `event_id` | - | - |
| ... | ... | ... | ... |

### 3.3 Feature Flag

```toml
# ironclaw/Cargo.toml
[features]
crdb = ["dep:crdb"]

[dependencies]
crdb = { path = "../crdb", optional = true }
```

### 3.4 Migration Path

```bash
# RPi deployment
cargo build --no-default-features --features crdb
DATABASE_BACKEND=crdb CRDB_PATH=~/.ironclaw/crdb ./ironclaw run
```

## Phase 4: Optimizations for RPi

### 4.1 Memory Budget

Target: 256 MB total (RPi with 1GB, leaving room for OS + Ollama)

| Component | Budget |
|-----------|--------|
| ironclaw binary + heap | 50 MB |
| redb (mmap, paged) | 20 MB active |
| Tantivy (mmap segments) | 20 MB active |
| crvecdb vectors (mmap) | 30 MB (10K chunks * 768d) |
| crvecdb graph (in-memory) | 5 MB |
| Ollama + model | 100-500 MB (separate) |

### 4.2 crvecdb RPi Tuning

```rust
// Lower HNSW parameters for RPi
Index::builder(768)
    .metric(DistanceMetric::Cosine)
    .m(8)                    // 8 instead of 16 (halves graph memory)
    .ef_construction(100)    // 100 instead of 200 (faster inserts)
    .capacity(10_000)        // Personal assistant scale
    .build_mmap(path)
```

### 4.3 Quantization (Future)

For further memory reduction, add int8 quantization:
- 4x memory reduction on vectors
- Minimal recall loss (~1-2%)
- Requires SIMD int8 dot product (ARM has `sdot` instruction on v8.2+)

## Implementation Order

| Phase | Effort | Deliverable |
|-------|--------|-------------|
| 1.1 Delete/Update | 2-3 days | crvecdb v0.2 |
| 1.2 Dynamic capacity | 1-2 days | crvecdb v0.2 |
| 1.3 Filtered search | 1-2 days | crvecdb v0.2 |
| 1.4 Batch operations | 1 day | crvecdb v0.2 |
| 2.1-2.5 crdb crate | 1-2 weeks | crdb v0.1 |
| 3.1-3.4 ironclaw backend | 1 week | ironclaw crdb feature |
| 4.1-4.2 RPi tuning | 2-3 days | Deployment guide |
| 4.3 Quantization | 1 week | crvecdb v0.3 (optional) |

## Alternative: Quick Path (No New Crate)

If the full crdb crate is too much, the fastest path to RPi deployment is:

1. Use ironclaw's existing `--features libsql` backend (zero code changes)
2. Replace libSQL's brute-force vector search with crvecdb (Phase 1 upgrades + ~200 lines in ironclaw)
3. Total effort: ~1 week

This gets you 90% of the benefit with 10% of the work.
