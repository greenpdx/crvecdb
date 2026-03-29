/// Unique identifier for a vector (user-provided)
pub type VectorId = u64;

/// Internal index within storage (max ~4B vectors)
pub type InternalId = u32;

/// Distance metric selection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum DistanceMetric {
    /// Cosine distance (1 - cosine_similarity)
    Cosine = 0,
    /// Squared Euclidean distance (L2²)
    #[default]
    Euclidean = 1,
    /// Negative dot product (higher dot = lower distance)
    DotProduct = 2,
}

impl DistanceMetric {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Cosine),
            1 => Some(Self::Euclidean),
            2 => Some(Self::DotProduct),
            _ => None,
        }
    }
}

/// HNSW algorithm configuration
#[derive(Clone, Copy, Debug)]
pub struct HnswConfig {
    /// Max connections per node (layers > 0)
    pub m: usize,
    /// Max connections at layer 0 (typically 2*M)
    pub m0: usize,
    /// Search width during construction
    pub ef_construction: usize,
    /// Search width at query time
    pub ef_search: usize,
    /// Layer probability factor: 1/ln(M)
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Create config with custom M value
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

/// Complete index configuration
#[derive(Clone, Debug)]
pub struct IndexConfig {
    /// Vector dimensionality
    pub dimension: usize,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// HNSW parameters
    pub hnsw: HnswConfig,
    /// Initial capacity (for pre-allocation)
    pub capacity: usize,
}

impl IndexConfig {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            metric: DistanceMetric::default(),
            hnsw: HnswConfig::default(),
            capacity: 10_000,
        }
    }
}
