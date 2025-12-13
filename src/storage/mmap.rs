use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::MmapMut;
use parking_lot::RwLock;

use crate::config::{IndexConfig, InternalId, VectorId};
use crate::distance::l2_norm;
use crate::error::{CrvecError, Result};
use crate::storage::layout::{file_size, vector_stride, StorageHeader, VectorMeta};
use crate::storage::VectorStorage;

/// Memory-mapped file storage (thread-safe)
pub struct MmapStorage {
    #[allow(dead_code)]
    file: File,
    mmap: RwLock<MmapMut>,
    dimension: usize,
    stride: usize,
    count: AtomicUsize,
    capacity: AtomicUsize,
}

impl MmapStorage {
    /// Create new mmap storage at path
    pub fn create(path: &Path, config: &IndexConfig) -> Result<Self> {
        let dimension = config.dimension;
        let capacity = config.capacity;
        let stride = vector_stride(dimension);
        let size = file_size(dimension, capacity);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        let header = StorageHeader::new(dimension, capacity, config.metric);
        mmap[0..64].copy_from_slice(&header.to_bytes());
        mmap.flush()?;

        Ok(Self {
            file,
            mmap: RwLock::new(mmap),
            dimension,
            stride,
            count: AtomicUsize::new(0),
            capacity: AtomicUsize::new(capacity),
        })
    }

    /// Open existing mmap storage
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Read and validate header
        let mut header_bytes = [0u8; 64];
        header_bytes.copy_from_slice(&mmap[0..64]);
        let header = StorageHeader::from_bytes(&header_bytes)?;

        let dimension = header.dimension as usize;
        let stride = vector_stride(dimension);

        Ok(Self {
            file,
            mmap: RwLock::new(mmap),
            dimension,
            stride,
            count: AtomicUsize::new(header.count as usize),
            capacity: AtomicUsize::new(header.capacity as usize),
        })
    }

    /// Thread-safe push
    pub fn push_sync(&self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        if vector.len() != self.dimension {
            return Err(CrvecError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let mut mmap = self.mmap.write();
        let count = self.count.load(Ordering::Acquire);
        let capacity = self.capacity.load(Ordering::Acquire);

        // For now, don't support growing in thread-safe mode
        if count >= capacity {
            return Err(CrvecError::InvalidFormat("capacity exceeded".into()));
        }

        let internal_id = count as InternalId;
        let offset = 64 + count * self.stride;

        // Write metadata
        let norm = l2_norm(vector);
        let meta = VectorMeta::new(id, norm);
        mmap[offset..offset + 16].copy_from_slice(&meta.to_bytes());

        // Write vector data
        let data_offset = offset + 16;
        for (i, &val) in vector.iter().enumerate() {
            let byte_offset = data_offset + i * 4;
            mmap[byte_offset..byte_offset + 4].copy_from_slice(&val.to_le_bytes());
        }

        let new_count = count + 1;
        self.count.store(new_count, Ordering::Release);
        mmap[16..24].copy_from_slice(&(new_count as u64).to_le_bytes());

        Ok(internal_id)
    }
}

impl VectorStorage for MmapStorage {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    fn push(&mut self, id: VectorId, vector: &[f32]) -> Result<InternalId> {
        // Delegate to thread-safe version
        self.push_sync(id, vector)
    }

    fn get_vector(&self, internal_id: InternalId) -> &[f32] {
        let offset = 64 + (internal_id as usize) * self.stride + 16;
        let mmap = self.mmap.read();
        // SAFETY: mmap is pinned in memory as long as file exists
        unsafe {
            let ptr = mmap[offset..].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.dimension)
        }
    }

    fn get_id(&self, internal_id: InternalId) -> VectorId {
        let offset = 64 + (internal_id as usize) * self.stride;
        let mmap = self.mmap.read();
        u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap())
    }

    fn get_norm(&self, internal_id: InternalId) -> f32 {
        let offset = 64 + (internal_id as usize) * self.stride + 8;
        let mmap = self.mmap.read();
        f32::from_le_bytes(mmap[offset..offset + 4].try_into().unwrap())
    }

    fn flush(&self) -> Result<()> {
        self.mmap.read().flush()?;
        Ok(())
    }
}

/// Validate file header without opening for writes
pub fn validate_header(path: &Path) -> Result<StorageHeader> {
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    if mmap.len() < 64 {
        return Err(CrvecError::InvalidFormat("file too small".into()));
    }

    let mut header_bytes = [0u8; 64];
    header_bytes.copy_from_slice(&mmap[0..64]);
    StorageHeader::from_bytes(&header_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let config = IndexConfig::new(3);

        {
            let mut storage = MmapStorage::create(&path, &config).unwrap();

            let id0 = storage.push(100, &[1.0, 2.0, 3.0]).unwrap();
            let id1 = storage.push(200, &[4.0, 5.0, 6.0]).unwrap();

            assert_eq!(id0, 0);
            assert_eq!(id1, 1);
            assert_eq!(storage.len(), 2);
            storage.flush().unwrap();
        }

        // Reopen and verify
        {
            let storage = MmapStorage::open(&path).unwrap();
            assert_eq!(storage.len(), 2);
            assert_eq!(storage.get_id(0), 100);
            assert_eq!(storage.get_id(1), 200);
            assert_eq!(storage.get_vector(0), &[1.0, 2.0, 3.0]);
            assert_eq!(storage.get_vector(1), &[4.0, 5.0, 6.0]);
        }
    }
}
