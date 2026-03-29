pub mod layout;
pub mod memory;
pub mod mmap;

use crate::config::{InternalId, VectorId};
use crate::error::Result;

pub use memory::MemoryStorage;
pub use mmap::MmapStorage;

/// Trait for vector storage backends
#[allow(dead_code)] // Methods used through dynamic dispatch (dyn VectorStorage)
pub trait VectorStorage: Send + Sync {
    /// Get vector dimensionality
    fn dimension(&self) -> usize;

    /// Get number of stored vectors
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity
    fn capacity(&self) -> usize;

    /// Add a vector, returns internal ID
    fn push(&mut self, id: VectorId, vector: &[f32]) -> Result<InternalId>;

    /// Get vector data by internal ID
    fn get_vector(&self, internal_id: InternalId) -> &[f32];

    /// Get external ID by internal ID
    fn get_id(&self, internal_id: InternalId) -> VectorId;

    /// Get precomputed norm by internal ID
    fn get_norm(&self, internal_id: InternalId) -> f32;

    /// Flush to disk (if applicable)
    fn flush(&self) -> Result<()> {
        Ok(())
    }
}
