use simdeez::prelude::*;

simd_runtime_generate!(
    /// SIMD-accelerated dot product
    pub fn dot_product_impl(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let len = a.len();
        let mut sum = S::Vf32::zeroes();

        // Process full SIMD widths
        let simd_width = S::Vf32::WIDTH;
        let simd_len = len - (len % simd_width);

        let mut i = 0;
        while i < simd_len {
            let va = S::Vf32::load_from_slice(&a[i..]);
            let vb = S::Vf32::load_from_slice(&b[i..]);
            sum += va * vb;
            i += simd_width;
        }

        let mut result = sum.horizontal_add();

        // Scalar tail
        for j in simd_len..len {
            result += a[j] * b[j];
        }

        result
    }
);

simd_runtime_generate!(
    /// SIMD-accelerated squared Euclidean distance
    pub fn squared_euclidean_impl(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let len = a.len();
        let mut sum = S::Vf32::zeroes();

        let simd_width = S::Vf32::WIDTH;
        let simd_len = len - (len % simd_width);

        let mut i = 0;
        while i < simd_len {
            let va = S::Vf32::load_from_slice(&a[i..]);
            let vb = S::Vf32::load_from_slice(&b[i..]);
            let diff = va - vb;
            sum += diff * diff;
            i += simd_width;
        }

        let mut result = sum.horizontal_add();

        for j in simd_len..len {
            let d = a[j] - b[j];
            result += d * d;
        }

        result
    }
);

simd_runtime_generate!(
    /// SIMD-accelerated L2 norm squared
    pub fn l2_norm_squared_impl(v: &[f32]) -> f32 {
        let len = v.len();
        let mut sum = S::Vf32::zeroes();

        let simd_width = S::Vf32::WIDTH;
        let simd_len = len - (len % simd_width);

        let mut i = 0;
        while i < simd_len {
            let va = S::Vf32::load_from_slice(&v[i..]);
            sum += va * va;
            i += simd_width;
        }

        let mut result = sum.horizontal_add();

        for &val in &v[simd_len..len] {
            result += val * val;
        }

        result
    }
);

/// Public SIMD dot product with runtime dispatch
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    dot_product_impl(a, b)
}

/// Public SIMD squared Euclidean with runtime dispatch
#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean_impl(a, b)
}

/// Public SIMD L2 norm with runtime dispatch
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    l2_norm_squared_impl(v).sqrt()
}

/// SIMD cosine distance
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// SIMD cosine distance with precomputed norm_a
#[inline]
pub fn cosine_distance_with_norm(a: &[f32], b: &[f32], norm_a: f32) -> f32 {
    let dot = dot_product(a, b);
    let norm_b = l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}
