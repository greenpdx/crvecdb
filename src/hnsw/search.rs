use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ordered_float::OrderedFloat;

use crate::config::{InternalId, VectorId};
use crate::distance::Distance;
use crate::hnsw::HnswGraph;
use crate::storage::VectorStorage;

/// Search result with ID and distance
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

impl HnswGraph {
    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        storage: &dyn VectorStorage,
        distance: &dyn Distance,
    ) -> Vec<SearchResult> {
        // Atomic read of the (entry_point, max_level) pair. Reading them
        // separately lets a concurrent insert CAS pair an old entry point
        // with a new max_level, which then indexes past
        // `entry_point.neighbors` and panics.
        let (entry_point, max_level) = match self.get_entry_state() {
            (Some(ep), ml) => (ep, ml),
            (None, _) => return Vec::new(),
        };

        let ef = ef_search.max(k);
        let mut current = entry_point;

        // Phase 1: Traverse top layers with greedy search
        for level in (1..=max_level).rev() {
            current = self.search_layer_greedy(query, current, level, storage, distance);
        }

        // Phase 2: Search layer 0 with ef candidates
        let candidates = self.search_layer_ef(query, current, ef, storage, distance);

        // Return top k results
        candidates
            .into_iter()
            .take(k)
            .map(|(dist, internal_id)| SearchResult {
                id: storage.get_id(internal_id),
                distance: dist,
            })
            .collect()
    }

    /// Greedy search for single nearest at a layer
    fn search_layer_greedy(
        &self,
        query: &[f32],
        entry: InternalId,
        level: usize,
        storage: &dyn VectorStorage,
        distance: &dyn Distance,
    ) -> InternalId {
        let mut current = entry;
        let mut current_dist = distance.distance(query, storage.get_vector(entry));

        loop {
            let neighbor_ids: Vec<InternalId> = {
                let nodes = self.nodes.read();
                nodes[current as usize].neighbors[level].read().clone()
            };
            let mut improved = false;
            let mut best = current;
            let mut best_dist = current_dist;

            for neighbor_id in neighbor_ids {
                let neighbor_dist = distance.distance(query, storage.get_vector(neighbor_id));
                if neighbor_dist < best_dist {
                    best = neighbor_id;
                    best_dist = neighbor_dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
            current = best;
            current_dist = best_dist;
        }

        current
    }

    /// Search layer 0 with ef candidates
    fn search_layer_ef(
        &self,
        query: &[f32],
        entry: InternalId,
        ef: usize,
        storage: &dyn VectorStorage,
        distance: &dyn Distance,
    ) -> Vec<(f32, InternalId)> {
        let num_nodes = self.nodes.read().len();
        let num_words = (num_nodes + 63) / 64;
        let mut visited = vec![0u64; num_words];
        visited[entry as usize / 64] |= 1u64 << (entry as usize % 64);

        let entry_dist = distance.distance(query, storage.get_vector(entry));

        // Min-heap for candidates
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, InternalId)>> = BinaryHeap::new();
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));

        // Max-heap for results
        let mut results: BinaryHeap<(OrderedFloat<f32>, InternalId)> = BinaryHeap::new();
        results.push((OrderedFloat(entry_dist), entry));

        while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            let worst_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            let neighbor_ids: Vec<InternalId> = {
                let nodes = self.nodes.read();
                nodes[c_id as usize].neighbors[0].read().clone()
            };

            for neighbor_id in neighbor_ids {
                let idx = neighbor_id as usize;
                // Grow visited bitset if needed
                if idx / 64 >= visited.len() {
                    visited.resize(idx / 64 + 1, 0);
                }
                let word = idx / 64;
                let bit = 1u64 << (idx % 64);
                if visited[word] & bit != 0 {
                    continue;
                }
                visited[word] |= bit;

                let neighbor_dist = distance.distance(query, storage.get_vector(neighbor_id));
                let worst_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

                if neighbor_dist < worst_dist || results.len() < ef {
                    candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor_id)));
                    results.push((OrderedFloat(neighbor_dist), neighbor_id));

                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<_> = results.into_iter().map(|(d, id)| (d.0, id)).collect();
        result_vec.sort_by(|a, b| a.0.total_cmp(&b.0));
        result_vec
    }
}
