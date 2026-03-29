use crate::config::DistanceMetric;
use crate::error::{CrvecError, Result};

/// File magic bytes
pub const MAGIC: [u8; 8] = *b"CRVECDB\0";

/// Current format version
pub const VERSION: u32 = 1;

/// Storage file header (64 bytes, cache-line aligned)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StorageHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub dimension: u32,
    pub count: u64,
    pub capacity: u64,
    pub metric: u32,
    pub _reserved: [u8; 28],
}

impl StorageHeader {
    pub fn new(dimension: usize, capacity: usize, metric: DistanceMetric) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            dimension: dimension as u32,
            count: 0,
            capacity: capacity as u64,
            metric: metric as u32,
            _reserved: [0; 28],
        }
    }

    pub fn to_bytes(self) -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.dimension.to_le_bytes());
        buf[16..24].copy_from_slice(&self.count.to_le_bytes());
        buf[24..32].copy_from_slice(&self.capacity.to_le_bytes());
        buf[32..36].copy_from_slice(&self.metric.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; 64]) -> Result<Self> {
        let magic: [u8; 8] = buf[0..8].try_into().unwrap();
        if magic != MAGIC {
            return Err(CrvecError::InvalidFormat("invalid magic bytes".into()));
        }

        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(CrvecError::InvalidFormat(format!(
                "unsupported version: {version}"
            )));
        }

        Ok(Self {
            magic,
            version,
            dimension: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
            count: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            capacity: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
            metric: u32::from_le_bytes(buf[32..36].try_into().unwrap()),
            _reserved: [0; 28],
        })
    }
}

/// Calculate stride (bytes per vector) aligned to 32 bytes for SIMD
#[inline]
pub const fn vector_stride(dimension: usize) -> usize {
    // 16 bytes metadata (id + norm + padding) + dimension * 4 bytes
    let raw_size = 16 + dimension * 4;
    // Round up to 32-byte boundary
    (raw_size + 31) & !31
}

/// Calculate total file size for storage
#[inline]
pub const fn file_size(dimension: usize, capacity: usize) -> usize {
    64 + vector_stride(dimension) * capacity
}

/// Stored vector metadata (16 bytes before vector data)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VectorMeta {
    pub id: u64,
    pub norm: f32,
    pub _padding: u32,
}

impl VectorMeta {
    pub fn new(id: u64, norm: f32) -> Self {
        Self {
            id,
            norm,
            _padding: 0,
        }
    }

    pub fn to_bytes(self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&self.id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.norm.to_le_bytes());
        buf
    }

    #[allow(dead_code)] // Part of the file format API
    pub fn from_bytes(buf: &[u8; 16]) -> Self {
        Self {
            id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            norm: f32::from_le_bytes(buf[8..12].try_into().unwrap()),
            _padding: 0,
        }
    }
}
