use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

use crate::config::{IndexConfig, InternalId, VectorId};
use crate::distance::l2_norm;
use crate::error::{CrvecError, Result};
use crate::storage::VectorStorage;

/// In-memory vector storage optimized for parallel bulk loading
pub struct MemoryStorage {
    dimension: usize,
    capacity: usize,
    /// Number of vectors stored
    len: AtomicUsize,
    /// Storage arrays (pre-allocated for parallel access)
    ids: UnsafeCell<Vec<VectorId>>,
    norms: UnsafeCell<Vec<f32>>,
    vectors: UnsafeCell<Vec<f32>>, // Flat array: vectors[i*dim..(i+1)*dim]
}

// SAFETY: We ensure thread safety through careful slot assignment
// Each thread writes to its own unique slot
unsafe impl Send for MemoryStorage {}
unsafe impl Sync for MemoryStorage {}

impl MemoryStorage {
    pub fn new(config: &IndexConfig) -> Self {
        let capacity = config.capacity;
        Self {
            dimension: config.dimension,
            capacity,
            len: AtomicUsize::new(0),
            ids: UnsafeCell::new(vec![0; capacity]),
            norms: UnsafeCell::new(vec![0.0; capacity]),
            vectors: UnsafeCell::new(vec![0.0; capacity * config.dimension]),
        }
    }

    /// Thread-safe push using atomic increment for slot allocation
    pub fn push_sync(&self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        if vector.len() != self.dimension {
            return Err(CrvecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        // Atomically allocate a slot
        let slot = self.len.fetch_add(1, Ordering::AcqRel) as InternalId;

        if slot as usize >= self.capacity {
            // Roll back the increment to avoid corrupting the counter
            self.len.fetch_sub(1, Ordering::AcqRel);
            return Err(CrvecError::CapacityExceeded {
                capacity: self.capacity,
            });
        }

        let norm = l2_norm(vector);

        // SAFETY: Each slot is written exactly once, no concurrent access
        unsafe {
            let ids = &mut *self.ids.get();
            let norms = &mut *self.norms.get();
            let vectors = &mut *self.vectors.get();

            ids[slot as usize] = id;
            norms[slot as usize] = norm;
            let start = slot as usize * self.dimension;
            vectors[start..start + self.dimension].copy_from_slice(vector);
        }

        Ok(slot)
    }

}

impl VectorStorage for MemoryStorage {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn push(&mut self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        // Delegate to thread-safe version
        self.push_sync(id, vector)
    }

    fn get_vector(&self, internal_id: InternalId) -> &[f32] {
        let idx = internal_id as usize;
        let start = idx * self.dimension;
        // SAFETY: Pre-allocated storage, reading from already-written slot
        unsafe {
            let vectors = &*self.vectors.get();
            &vectors[start..start + self.dimension]
        }
    }

    fn get_id(&self, internal_id: InternalId) -> VectorId {
        // SAFETY: Pre-allocated storage, reading from already-written slot
        unsafe {
            let ids = &*self.ids.get();
            ids[internal_id as usize]
        }
    }

    fn get_norm(&self, internal_id: InternalId) -> f32 {
        // SAFETY: Pre-allocated storage, reading from already-written slot
        unsafe {
            let norms = &*self.norms.get();
            norms[internal_id as usize]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage() {
        let config = crate::config::IndexConfig {
            dimension: 3,
            capacity: 100,
            ..crate::config::IndexConfig::new(3)
        };
        let mut storage = MemoryStorage::new(&config);

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
