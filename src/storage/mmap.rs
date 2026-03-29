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
///
/// SAFETY: The mmap is created with a fixed size at construction time and is never
/// remapped. The `base_ptr` is stable for the lifetime of the `MmapStorage` because
/// the `MmapMut` (held by `mmap`) keeps the mapping alive. Reads through `base_ptr`
/// are safe as long as we only read slots that have already been written (guaranteed
/// by `count` being updated after the write completes).
pub struct MmapStorage {
    _file: File,
    mmap: RwLock<MmapMut>,
    /// Stable pointer to the start of the mmap region, used for lock-free reads.
    base_ptr: *const u8,
    dimension: usize,
    stride: usize,
    count: AtomicUsize,
    capacity: AtomicUsize,
}

// SAFETY: base_ptr points into the mmap which is pinned for the lifetime of this struct.
// Concurrent reads of already-written slots are safe. Writes go through the RwLock.
unsafe impl Send for MmapStorage {}
unsafe impl Sync for MmapStorage {}

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

        let base_ptr = mmap.as_ptr();

        Ok(Self {
            _file: file,
            mmap: RwLock::new(mmap),
            base_ptr,
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
        let base_ptr = mmap.as_ptr();

        Ok(Self {
            _file: file,
            mmap: RwLock::new(mmap),
            base_ptr,
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

        if count >= capacity {
            return Err(CrvecError::CapacityExceeded { capacity });
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
        self.push_sync(id, vector)
    }

    fn get_vector(&self, internal_id: InternalId) -> &[f32] {
        let offset = 64 + (internal_id as usize) * self.stride + 16;
        // SAFETY: base_ptr is stable (mmap is never remapped), and this slot
        // has been written before count was incremented, so the data is valid.
        unsafe {
            let ptr = self.base_ptr.add(offset) as *const f32;
            std::slice::from_raw_parts(ptr, self.dimension)
        }
    }

    fn get_id(&self, internal_id: InternalId) -> VectorId {
        let offset = 64 + (internal_id as usize) * self.stride;
        // SAFETY: same as get_vector
        unsafe {
            let ptr = self.base_ptr.add(offset) as *const u64;
            ptr.read_unaligned()
        }
    }

    fn get_norm(&self, internal_id: InternalId) -> f32 {
        let offset = 64 + (internal_id as usize) * self.stride + 8;
        // SAFETY: same as get_vector
        unsafe {
            let ptr = self.base_ptr.add(offset) as *const f32;
            ptr.read_unaligned()
        }
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
