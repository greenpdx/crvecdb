use std::fmt;
use std::io;

/// Result type alias for crvecdb operations
pub type Result<T> = std::result::Result<T, CrvecError>;

/// Error types for crvecdb operations
#[derive(Debug)]
pub enum CrvecError {
    /// Vector dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
    /// Index is empty (no entry point)
    EmptyIndex,
    /// Invalid file format
    InvalidFormat(String),
    /// IO error
    Io(io::Error),
    /// Index capacity exceeded
    CapacityExceeded { capacity: usize },
    /// Vector ID not found
    NotFound(u64),
    /// Invalid parameter value
    InvalidParameter(String),
}

impl fmt::Display for CrvecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyIndex => write!(f, "index is empty"),
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::CapacityExceeded { capacity } => {
                write!(f, "capacity exceeded: max {capacity}")
            }
            Self::NotFound(id) => write!(f, "vector not found: {id}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for CrvecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for CrvecError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
