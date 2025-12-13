use std::path::Path;

use crate::config::{DistanceMetric, HnswConfig, IndexConfig, VectorId};
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
    Mmap(MmapStorage),
}

impl VectorStorage for StorageBackend {
    fn dimension(&self) -> usize {
        match self {
            Self::Memory(s) => s.dimension(),
            Self::Mmap(s) => s.dimension(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Memory(s) => s.len(),
            Self::Mmap(s) => s.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Memory(s) => s.capacity(),
            Self::Mmap(s) => s.capacity(),
        }
    }

    fn push(&mut self, id: VectorId, vector: &[f32]) -> Result<crate::config::InternalId> {
        match self {
            Self::Memory(s) => s.push(id, vector),
            Self::Mmap(s) => s.push(id, vector),
        }
    }

    fn get_vector(&self, internal_id: crate::config::InternalId) -> &[f32] {
        match self {
            Self::Memory(s) => s.get_vector(internal_id),
            Self::Mmap(s) => s.get_vector(internal_id),
        }
    }

    fn get_id(&self, internal_id: crate::config::InternalId) -> VectorId {
        match self {
            Self::Memory(s) => s.get_id(internal_id),
            Self::Mmap(s) => s.get_id(internal_id),
        }
    }

    fn get_norm(&self, internal_id: crate::config::InternalId) -> f32 {
        match self {
            Self::Memory(s) => s.get_norm(internal_id),
            Self::Mmap(s) => s.get_norm(internal_id),
        }
    }

    fn flush(&self) -> Result<()> {
        match self {
            Self::Memory(s) => s.flush(),
            Self::Mmap(s) => s.flush(),
        }
    }
}

/// The main vector index type
pub struct Index {
    storage: StorageBackend,
    graph: HnswGraph,
    distance: Box<dyn Distance>,
    config: IndexConfig,
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
        let distance = create_distance(config.metric);

        Ok(Self {
            storage: StorageBackend::Memory(storage),
            graph,
            distance,
            config,
        })
    }

    /// Create memory-mapped index
    pub fn new_mmap(config: IndexConfig, path: &Path) -> Result<Self> {
        let storage = MmapStorage::create(path, &config)?;
        let graph = HnswGraph::new(config.hnsw);
        let distance = create_distance(config.metric);

        Ok(Self {
            storage: StorageBackend::Mmap(storage),
            graph,
            distance,
            config,
        })
    }

    /// Open existing memory-mapped index
    pub fn open_mmap(path: &Path) -> Result<Self> {
        let storage = MmapStorage::open(path)?;

        // Read config from header
        let header = crate::storage::mmap::validate_header(path)?;
        let metric = DistanceMetric::from_u32(header.metric)
            .ok_or_else(|| CrvecError::InvalidFormat("unknown metric".into()))?;

        let config = IndexConfig {
            dimension: header.dimension as usize,
            metric,
            hnsw: HnswConfig::default(),
            capacity: header.capacity as usize,
        };

        let graph = HnswGraph::new(config.hnsw);
        let distance = create_distance(config.metric);

        // TODO: Load graph from separate file

        Ok(Self {
            storage: StorageBackend::Mmap(storage),
            graph,
            distance,
            config,
        })
    }

    /// Insert a single vector
    pub fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(CrvecError::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        let internal_id = self.storage.push(id, vector)?;
        self.graph
            .insert(internal_id, &self.storage, self.distance.as_ref());

        Ok(())
    }

    /// Insert multiple vectors
    pub fn insert_batch(&mut self, vectors: &[(VectorId, Vec<f32>)]) -> Result<()> {
        for (id, vector) in vectors {
            self.insert(*id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_ef(query, k, self.config.hnsw.ef_construction)
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

        let results =
            self.graph
                .search(query, k, ef_search, &self.storage, self.distance.as_ref());

        Ok(results)
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

    /// Flush to disk (for mmap storage)
    pub fn flush(&self) -> Result<()> {
        self.storage.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_basic() {
        let mut index = Index::builder(3)
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
        let mut index = Index::builder(3)
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
        let mut index = Index::builder(128)
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
}
