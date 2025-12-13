//! # crvecdb
//!
//! A fast vector database library with HNSW indexing for ARM64 and x86-64.
//!
//! ## Features
//!
//! - HNSW (Hierarchical Navigable Small World) indexing for fast approximate nearest neighbor search
//! - Multiple distance metrics: Cosine, Euclidean (L2), Dot Product
//! - Memory-mapped file storage for persistence
//! - Cross-platform SIMD acceleration (ARM NEON, x86 SSE/AVX2)
//! - Scales to millions of vectors
//!
//! ## Example
//!
//! ```rust
//! use crvecdb::{Index, DistanceMetric};
//!
//! // Create an in-memory index
//! let mut index = Index::builder(128)
//!     .metric(DistanceMetric::Cosine)
//!     .m(16)
//!     .ef_construction(200)
//!     .capacity(10_000)
//!     .build()
//!     .unwrap();
//!
//! // Insert vectors
//! index.insert(1, &vec![0.1; 128]).unwrap();
//! index.insert(2, &vec![0.2; 128]).unwrap();
//!
//! // Search for nearest neighbors
//! let results = index.search(&vec![0.15; 128], 10).unwrap();
//! for result in results {
//!     println!("ID: {}, Distance: {:.4}", result.id, result.distance);
//! }
//! ```
//!
//! ## Memory-Mapped Storage
//!
//! ```rust,no_run
//! use crvecdb::{Index, DistanceMetric};
//!
//! // Create a persistent index
//! let mut index = Index::builder(768)
//!     .metric(DistanceMetric::DotProduct)
//!     .capacity(1_000_000)
//!     .build_mmap("/path/to/index.db")
//!     .unwrap();
//!
//! // Data is automatically persisted via mmap
//! index.insert(1, &vec![0.1; 768]).unwrap();
//! index.flush().unwrap(); // Ensure durability
//! ```

mod config;
mod distance;
mod error;
mod hnsw;
mod index;
mod storage;

pub use config::{DistanceMetric, HnswConfig, IndexConfig, InternalId, VectorId};
pub use error::{CrvecError, Result};
pub use hnsw::SearchResult;
pub use index::{Index, IndexBuilder};
