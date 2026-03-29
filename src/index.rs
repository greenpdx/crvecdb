use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use parking_lot::RwLock;

use crate::config::{DistanceMetric, HnswConfig, IndexConfig, InternalId, VectorId};
use crate::distance::{create_distance, Distance};
use crate::error::{CrvecError, Result};
use crate::hnsw::{HnswGraph, SearchResult};
use crate::storage::{MemoryStorage, MmapStorage, VectorStorage};

/// Builder for creating indexes
pub struct IndexBuilder {
    config: IndexConfig,
}

impl IndexBuilder {
    /// Create builder with dimension
    pub fn new(dimension: usize) -> Self {
        Self {
            config: IndexConfig::new(dimension),
        }
    }

    /// Set distance metric
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.config.metric = metric;
        self
    }

    /// Set M parameter (max connections per layer)
    pub fn m(mut self, m: usize) -> Self {
        self.config.hnsw = HnswConfig::with_m(m);
        self
    }

    /// Set ef_construction parameter
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.config.hnsw.ef_construction = ef;
        self
    }

    /// Set ef_search parameter (search width at query time, default 64)
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.config.hnsw.ef_search = ef;
        self
    }

    /// Set initial capacity
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.config.capacity = capacity;
        self
    }

    /// Build in-memory index
    pub fn build(self) -> Result<Index> {
        Index::new_memory(self.config)
    }

    /// Build memory-mapped index at path
    pub fn build_mmap<P: AsRef<Path>>(self, path: P) -> Result<Index> {
        Index::new_mmap(self.config, path.as_ref())
    }
}

enum StorageBackend {
    Memory(MemoryStorage),
    Mmap { storage: MmapStorage, path: std::path::PathBuf },
}

impl StorageBackend {
    /// Thread-safe push
    fn push_sync(&self, id: VectorId, vector: &[f32]) -> Result<crate::config::InternalId> {
        match self {
            Self::Memory(s) => s.push_sync(id, vector),
            Self::Mmap { storage, .. } => storage.push_sync(id, vector),
        }
    }

    /// Get the graph file path (only for mmap backend)
    fn graph_path(&self) -> Option<std::path::PathBuf> {
        match self {
            Self::Memory(_) => None,
            Self::Mmap { path, .. } => {
                let mut graph_path = path.clone();
                graph_path.set_extension("graph");
                Some(graph_path)
            }
        }
    }
}

impl VectorStorage for StorageBackend {
    fn dimension(&self) -> usize {
        match self {
            Self::Memory(s) => s.dimension(),
            Self::Mmap { storage, .. } => storage.dimension(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Memory(s) => s.len(),
            Self::Mmap { storage, .. } => storage.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Memory(s) => s.capacity(),
            Self::Mmap { storage, .. } => storage.capacity(),
        }
    }

    fn push(&mut self, id: VectorId, vector: &[f32]) -> Result<crate::config::InternalId> {
        match self {
            Self::Memory(s) => s.push(id, vector),
            Self::Mmap { storage, .. } => storage.push(id, vector),
        }
    }

    fn get_vector(&self, internal_id: crate::config::InternalId) -> &[f32] {
        match self {
            Self::Memory(s) => s.get_vector(internal_id),
            Self::Mmap { storage, .. } => storage.get_vector(internal_id),
        }
    }

    fn get_id(&self, internal_id: crate::config::InternalId) -> VectorId {
        match self {
            Self::Memory(s) => s.get_id(internal_id),
            Self::Mmap { storage, .. } => storage.get_id(internal_id),
        }
    }

    fn get_norm(&self, internal_id: crate::config::InternalId) -> f32 {
        match self {
            Self::Memory(s) => s.get_norm(internal_id),
            Self::Mmap { storage, .. } => storage.get_norm(internal_id),
        }
    }

    fn flush(&self) -> Result<()> {
        match self {
            Self::Memory(s) => s.flush(),
            Self::Mmap { storage, .. } => storage.flush(),
        }
    }
}

/// The main vector index type
pub struct Index {
    storage: StorageBackend,
    graph: HnswGraph,
    distance: Arc<dyn Distance>,
    config: IndexConfig,
    /// Reverse mapping from external VectorId to InternalId
    id_map: RwLock<HashMap<VectorId, InternalId>>,
    /// Set of soft-deleted internal IDs
    deleted: RwLock<HashSet<InternalId>>,
}

impl Index {
    /// Create index using builder pattern
    pub fn builder(dimension: usize) -> IndexBuilder {
        IndexBuilder::new(dimension)
    }

    /// Create in-memory index
    pub fn new_memory(config: IndexConfig) -> Result<Self> {
        let storage = MemoryStorage::new(&config);
        let graph = HnswGraph::new(config.hnsw);
        let distance: Arc<dyn Distance> = Arc::from(create_distance(config.metric));

        Ok(Self {
            storage: StorageBackend::Memory(storage),
            graph,
            distance,
            config,
            id_map: RwLock::new(HashMap::new()),
            deleted: RwLock::new(HashSet::new()),
        })
    }

    /// Create memory-mapped index
    pub fn new_mmap(config: IndexConfig, path: &Path) -> Result<Self> {
        let storage = MmapStorage::create(path, &config)?;
        let graph = HnswGraph::new(config.hnsw);
        let distance: Arc<dyn Distance> = Arc::from(create_distance(config.metric));

        Ok(Self {
            storage: StorageBackend::Mmap { storage, path: path.to_path_buf() },
            graph,
            distance,
            config,
            id_map: RwLock::new(HashMap::new()),
            deleted: RwLock::new(HashSet::new()),
        })
    }

    /// Open existing memory-mapped index
    pub fn open_mmap(path: &Path) -> Result<Self> {
        let storage = MmapStorage::open(path)?;

        // Read config from header
        let header = crate::storage::mmap::validate_header(path)?;
        let metric = DistanceMetric::from_u32(header.metric)
            .ok_or_else(|| CrvecError::InvalidFormat("unknown metric".into()))?;

        // Try to load graph from file
        let mut graph_path = path.to_path_buf();
        graph_path.set_extension("graph");

        let (graph, hnsw_config) = if graph_path.exists() {
            let loaded_graph = HnswGraph::load_from_file(&graph_path)?;
            let config = loaded_graph.config;
            (loaded_graph, config)
        } else {
            let config = HnswConfig::default();
            (HnswGraph::new(config), config)
        };

        let config = IndexConfig {
            dimension: header.dimension as usize,
            metric,
            hnsw: hnsw_config,
            capacity: header.capacity as usize,
        };

        let distance: Arc<dyn Distance> = Arc::from(create_distance(config.metric));

        // Rebuild id_map from storage
        let count = storage.len();
        let mut id_map = HashMap::with_capacity(count);
        for i in 0..count {
            let internal_id = i as InternalId;
            let vector_id = storage.get_id(internal_id);
            id_map.insert(vector_id, internal_id);
        }

        Ok(Self {
            storage: StorageBackend::Mmap { storage, path: path.to_path_buf() },
            graph,
            distance,
            config,
            id_map: RwLock::new(id_map),
            deleted: RwLock::new(HashSet::new()),
        })
    }

    /// Insert a single vector
    pub fn insert(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(CrvecError::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        let internal_id = self.storage.push_sync(id, vector)?;
        self.id_map.write().insert(id, internal_id);
        self.graph
            .insert(internal_id, &self.storage, self.distance.as_ref());

        Ok(())
    }

    /// Insert multiple vectors sequentially
    pub fn insert_batch(&self, vectors: &[(VectorId, Vec<f32>)]) -> Result<()> {
        for (id, vector) in vectors {
            self.insert(*id, vector)?;
        }
        Ok(())
    }

    /// Insert multiple vectors in parallel (when `parallel` feature enabled)
    /// or sequentially (when disabled). Both storage writes and graph building
    /// happen concurrently when parallel.
    pub fn insert_parallel(&self, vectors: &[(VectorId, Vec<f32>)]) -> Result<()> {
        // Validate all vectors first
        for (_, vector) in vectors {
            if vector.len() != self.config.dimension {
                return Err(CrvecError::DimensionMismatch {
                    expected: self.config.dimension,
                    got: vector.len(),
                });
            }
        }

        #[cfg(feature = "parallel")]
        {
            // Parallel: Push to storage and build graph for each vector
            // Storage uses atomic slot allocation, so each thread gets unique slots
            vectors.par_iter().try_for_each(|(id, vector)| {
                let internal_id = self.storage.push_sync(*id, vector)?;
                self.id_map.write().insert(*id, internal_id);
                self.graph
                    .insert(internal_id, &self.storage, self.distance.as_ref());
                Ok::<_, CrvecError>(())
            })?;
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Sequential fallback
            for (id, vector) in vectors {
                let internal_id = self.storage.push_sync(*id, vector)?;
                self.id_map.write().insert(*id, internal_id);
                self.graph
                    .insert(internal_id, &self.storage, self.distance.as_ref());
            }
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.config.hnsw.ef_search)
    }

    /// Search with custom ef_search parameter
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(CrvecError::DimensionMismatch {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        // Over-fetch to account for deleted nodes, then filter
        let deleted = self.deleted.read();
        if deleted.is_empty() {
            let results =
                self.graph
                    .search(query, k, ef_search, &self.storage, self.distance.as_ref());
            return Ok(results);
        }

        // Request more candidates to compensate for filtered deletions
        let fetch_k = k + deleted.len().min(k);
        let results =
            self.graph
                .search(query, fetch_k, ef_search.max(fetch_k), &self.storage, self.distance.as_ref());

        Ok(results
            .into_iter()
            .filter(|r| {
                // Look up internal ID to check deletion status
                if let Some(&internal_id) = self.id_map.read().get(&r.id) {
                    !deleted.contains(&internal_id)
                } else {
                    true
                }
            })
            .take(k)
            .collect())
    }

    /// Get number of vectors in index
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get vector dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get the index configuration
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Check if a vector ID exists in the index (and is not deleted)
    pub fn contains(&self, id: VectorId) -> bool {
        let id_map = self.id_map.read();
        match id_map.get(&id) {
            Some(&internal_id) => !self.deleted.read().contains(&internal_id),
            None => false,
        }
    }

    /// Get the vector data for a given vector ID
    pub fn get_vector(&self, id: VectorId) -> Result<Vec<f32>> {
        let id_map = self.id_map.read();
        match id_map.get(&id) {
            Some(&internal_id) => {
                if self.deleted.read().contains(&internal_id) {
                    return Err(CrvecError::NotFound(id));
                }
                Ok(self.storage.get_vector(internal_id).to_vec())
            }
            None => Err(CrvecError::NotFound(id)),
        }
    }

    /// Soft-delete a vector by ID
    ///
    /// The vector remains in the graph for traversal but is excluded from
    /// search results. This is the standard HNSW approach to deletion.
    pub fn delete(&self, id: VectorId) -> Result<()> {
        let id_map = self.id_map.read();
        match id_map.get(&id) {
            Some(&internal_id) => {
                self.deleted.write().insert(internal_id);
                Ok(())
            }
            None => Err(CrvecError::NotFound(id)),
        }
    }

    /// Flush to disk (for mmap storage)
    ///
    /// This saves both the vector data and the HNSW graph structure.
    pub fn flush(&self) -> Result<()> {
        self.storage.flush()?;

        // Save graph for mmap storage
        if let Some(graph_path) = self.storage.graph_path() {
            self.graph.save_to_file(&graph_path)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_basic() {
        let index = Index::builder(3)
            .metric(DistanceMetric::Euclidean)
            .m(8)
            .ef_construction(50)
            .capacity(100)
            .build()
            .unwrap();

        index.insert(1, &[1.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0]).unwrap();
        index.insert(3, &[0.0, 0.0, 1.0]).unwrap();
        index.insert(4, &[1.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.len(), 4);

        let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Exact match
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_index_cosine() {
        let index = Index::builder(3)
            .metric(DistanceMetric::Cosine)
            .build()
            .unwrap();

        index.insert(1, &[1.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0]).unwrap();
        index.insert(3, &[0.707, 0.707, 0.0]).unwrap(); // 45 degrees

        let results = index.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results[0].id, 1); // Exact match has distance 0
    }

    #[test]
    fn test_larger_index() {
        let index = Index::builder(128)
            .metric(DistanceMetric::Euclidean)
            .m(16)
            .ef_construction(100)
            .capacity(1000)
            .build()
            .unwrap();

        // Insert 500 random vectors
        for i in 0..500 {
            let vector: Vec<f32> = (0..128).map(|j| ((i * 17 + j) % 100) as f32 / 100.0).collect();
            index.insert(i as u64, &vector).unwrap();
        }

        assert_eq!(index.len(), 500);

        // Search should return k results
        let results = index.search(&vec![0.5; 128], 10).unwrap();
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }

    #[test]
    fn test_mmap_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Create and populate an index
        {
            let index = Index::builder(4)
                .metric(DistanceMetric::Euclidean)
                .m(8)
                .capacity(100)
                .build_mmap(&db_path)
                .unwrap();

            index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
            index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
            index.insert(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();
            index.insert(4, &[0.0, 0.0, 0.0, 1.0]).unwrap();
            index.flush().unwrap();
        }

        // Reopen and verify
        {
            let index = Index::open_mmap(&db_path).unwrap();
            assert_eq!(index.len(), 4);

            // Search should work immediately (graph was persisted)
            let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].id, 1);
            assert!(results[0].distance < 0.001);
        }
    }
}
