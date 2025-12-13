mod scalar;
#[cfg(feature = "simd")]
mod simd;

use crate::config::DistanceMetric;

/// Trait for distance/similarity metrics
#[allow(dead_code)]
pub trait Distance: Send + Sync {
    /// Compute distance between two vectors (lower = more similar)
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Compute distance using precomputed norm of `a` (optimization for cosine)
    fn distance_with_norm(&self, a: &[f32], b: &[f32], norm_a: f32) -> f32 {
        let _ = norm_a;
        self.distance(a, b)
    }

    /// Whether this metric benefits from precomputed norms
    fn uses_norm(&self) -> bool {
        false
    }
}

/// Cosine distance: 1 - (a · b) / (|a| * |b|)
pub struct CosineDistance;

/// Squared Euclidean distance: Σ(a - b)²
pub struct EuclideanDistance;

/// Negative dot product: -(a · b)
pub struct DotProductDistance;

impl Distance for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(feature = "simd")]
        {
            simd::cosine_distance(a, b)
        }
        #[cfg(not(feature = "simd"))]
        {
            scalar::cosine_distance(a, b)
        }
    }

    fn distance_with_norm(&self, a: &[f32], b: &[f32], norm_a: f32) -> f32 {
        #[cfg(feature = "simd")]
        {
            simd::cosine_distance_with_norm(a, b, norm_a)
        }
        #[cfg(not(feature = "simd"))]
        {
            scalar::cosine_distance_with_norm(a, b, norm_a)
        }
    }

    fn uses_norm(&self) -> bool {
        true
    }
}

impl Distance for EuclideanDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(feature = "simd")]
        {
            simd::squared_euclidean(a, b)
        }
        #[cfg(not(feature = "simd"))]
        {
            scalar::squared_euclidean(a, b)
        }
    }
}

impl Distance for DotProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(feature = "simd")]
        {
            -simd::dot_product(a, b)
        }
        #[cfg(not(feature = "simd"))]
        {
            -scalar::dot_product(a, b)
        }
    }
}

/// Create a boxed distance metric from enum
pub fn create_distance(metric: DistanceMetric) -> Box<dyn Distance> {
    match metric {
        DistanceMetric::Cosine => Box::new(CosineDistance),
        DistanceMetric::Euclidean => Box::new(EuclideanDistance),
        DistanceMetric::DotProduct => Box::new(DotProductDistance),
    }
}

/// Compute L2 norm of a vector
pub fn l2_norm(v: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        simd::l2_norm(v)
    }
    #[cfg(not(feature = "simd"))]
    {
        scalar::l2_norm(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = EuclideanDistance.distance(&a, &b);
        assert!((dist - 27.0).abs() < 1e-5); // (3² + 3² + 3²)
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = DotProductDistance.distance(&a, &b);
        assert!((dist - (-32.0)).abs() < 1e-5); // -(1*4 + 2*5 + 3*6)
    }

    #[test]
    fn test_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = CosineDistance.distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-5); // orthogonal = distance 1

        let c = vec![1.0, 0.0];
        let d = vec![1.0, 0.0];
        let dist2 = CosineDistance.distance(&c, &d);
        assert!(dist2.abs() < 1e-5); // identical = distance 0
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-5);
    }
}
