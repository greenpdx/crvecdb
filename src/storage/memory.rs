use crate::config::{IndexConfig, InternalId, VectorId};
use crate::distance::l2_norm;
use crate::error::{CrvecError, Result};
use crate::storage::VectorStorage;

/// In-memory vector storage
#[allow(dead_code)]
pub struct MemoryStorage {
    dimension: usize,
    ids: Vec<VectorId>,
    norms: Vec<f32>,
    vectors: Vec<f32>, // Flat array: vectors[i*dim..(i+1)*dim]
    capacity: usize,
}

#[allow(dead_code)]
impl MemoryStorage {
    pub fn new(config: &IndexConfig) -> Self {
        let capacity = config.capacity;
        Self {
            dimension: config.dimension,
            ids: Vec::with_capacity(capacity),
            norms: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * config.dimension),
            capacity,
        }
    }

    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        Self {
            dimension,
            ids: Vec::with_capacity(capacity),
            norms: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * dimension),
            capacity,
        }
    }
}

impl VectorStorage for MemoryStorage {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn push(&mut self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        if vector.len() != self.dimension {
            return Err(CrvecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let internal_id = self.ids.len() as InternalId;
        let norm = l2_norm(vector);

        self.ids.push(id);
        self.norms.push(norm);
        self.vectors.extend_from_slice(vector);

        Ok(internal_id)
    }

    fn get_vector(&self, internal_id: InternalId) -> &[f32] {
        let idx = internal_id as usize;
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    fn get_id(&self, internal_id: InternalId) -> VectorId {
        self.ids[internal_id as usize]
    }

    fn get_norm(&self, internal_id: InternalId) -> f32 {
        self.norms[internal_id as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage() {
        let mut storage = MemoryStorage::with_capacity(3, 100);

        let id0 = storage.push(100, &[1.0, 2.0, 3.0]).unwrap();
        let id1 = storage.push(200, &[4.0, 5.0, 6.0]).unwrap();

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.get_id(0), 100);
        assert_eq!(storage.get_id(1), 200);
        assert_eq!(storage.get_vector(0), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.get_vector(1), &[4.0, 5.0, 6.0]);
    }
}
