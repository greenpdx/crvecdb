use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::Rng;

use crate::config::{HnswConfig, InternalId};
use crate::distance::Distance;
use crate::storage::VectorStorage;

/// Maximum number of layers
pub const MAX_LAYERS: usize = 16;

/// Sentinel value for no entry point
const NO_ENTRY_POINT: u32 = u32::MAX;

/// A node in the HNSW graph
#[allow(dead_code)]
pub struct HnswNode {
    /// Maximum layer this node exists on
    pub max_layer: usize,
    /// Neighbors at each layer: neighbors[layer] = Vec<InternalId>
    pub neighbors: Vec<RwLock<Vec<InternalId>>>,
}

impl HnswNode {
    pub fn new(max_layer: usize, config: &HnswConfig) -> Self {
        let mut neighbors = Vec::with_capacity(max_layer + 1);
        for layer in 0..=max_layer {
            let capacity = if layer == 0 { config.m0 } else { config.m };
            neighbors.push(RwLock::new(Vec::with_capacity(capacity)));
        }
        Self {
            max_layer,
            neighbors,
        }
    }
}

/// The HNSW graph structure (thread-safe)
pub struct HnswGraph {
    /// Nodes indexed by InternalId
    pub nodes: RwLock<Vec<HnswNode>>,
    /// Entry point (highest layer node), NO_ENTRY_POINT if none
    entry_point: AtomicU32,
    /// Current maximum layer
    max_level: AtomicUsize,
    /// Configuration
    pub config: HnswConfig,
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
            entry_point: AtomicU32::new(NO_ENTRY_POINT),
            max_level: AtomicUsize::new(0),
            config,
        }
    }

    /// Get current entry point
    pub fn get_entry_point(&self) -> Option<InternalId> {
        let ep = self.entry_point.load(Ordering::Acquire);
        if ep == NO_ENTRY_POINT {
            None
        } else {
            Some(ep)
        }
    }

    /// Get current max level
    pub fn get_max_level(&self) -> usize {
        self.max_level.load(Ordering::Acquire)
    }

    /// Get number of nodes
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Generate random level for new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform: f64 = rng.r#gen();
        let level = (-uniform.ln() * self.config.ml).floor() as usize;
        level.min(MAX_LAYERS - 1)
    }

    /// Thread-safe insert - can be called from multiple threads
    pub fn insert(
        &self,
        internal_id: InternalId,
        storage: &dyn VectorStorage,
        distance: &dyn Distance,
    ) {
        let insert_level = self.random_level();
        let node = HnswNode::new(insert_level, &self.config);

        // Add node to the graph first
        {
            let mut nodes = self.nodes.write();
            // Ensure we have space up to this ID
            while nodes.len() <= internal_id as usize {
                // This shouldn't happen in normal use, but handle it
                nodes.push(HnswNode::new(0, &self.config));
            }
            nodes[internal_id as usize] = node;
        }

        // Try to become the first entry point
        if self
            .entry_point
            .compare_exchange(
                NO_ENTRY_POINT,
                internal_id,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.max_level.store(insert_level, Ordering::Release);
            return;
        }

        let query = storage.get_vector(internal_id);
        let mut current = self.entry_point.load(Ordering::Acquire);
        let current_max_level = self.max_level.load(Ordering::Acquire);

        // Phase 1: Traverse from top to insert_level + 1 with greedy search
        for level in (insert_level + 1..=current_max_level).rev() {
            current = self.search_layer_single(query, current, level, storage, distance);
        }

        // Phase 2: Insert at levels insert_level down to 0
        let top_level = insert_level.min(current_max_level);
        for level in (0..=top_level).rev() {
            let candidates =
                self.search_layer(query, current, self.config.ef_construction, level, storage, distance);

            let max_neighbors = if level == 0 {
                self.config.m0
            } else {
                self.config.m
            };

            let selected = self.select_neighbors_simple(&candidates, max_neighbors);

            // Add connections from new node to selected neighbors
            {
                let nodes = self.nodes.read();
                let mut new_node_neighbors = nodes[internal_id as usize].neighbors[level].write();
                *new_node_neighbors = selected.clone();
            }

            // Add reverse connections
            for &neighbor_id in &selected {
                let nodes = self.nodes.read();
                let mut neighbor_neighbors = nodes[neighbor_id as usize].neighbors[level].write();

                if neighbor_neighbors.len() < max_neighbors {
                    neighbor_neighbors.push(internal_id);
                } else {
                    // Shrink connections if over capacity
                    neighbor_neighbors.push(internal_id);
                    self.shrink_connections_locked(
                        neighbor_id,
                        &mut neighbor_neighbors,
                        max_neighbors,
                        storage,
                        distance,
                    );
                }
            }

            // Update entry point for next level
            if let Some(&nearest) = selected.first() {
                current = nearest;
            }
        }

        // Update entry point if new node has higher level (CAS loop)
        loop {
            let old_max = self.max_level.load(Ordering::Acquire);
            if insert_level <= old_max {
                break;
            }
            if self
                .max_level
                .compare_exchange(old_max, insert_level, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.entry_point.store(internal_id, Ordering::Release);
                break;
            }
        }
    }

    /// Greedy search for single nearest neighbor at a layer
    fn search_layer_single(
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
            let nodes = self.nodes.read();
            let neighbors = nodes[current as usize].neighbors[level].read();
            let mut improved = false;
            let mut best = current;
            let mut best_dist = current_dist;

            for &neighbor_id in neighbors.iter() {
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

    /// Search a layer for ef nearest candidates
    fn search_layer(
        &self,
        query: &[f32],
        entry: InternalId,
        ef: usize,
        level: usize,
        storage: &dyn VectorStorage,
        distance: &dyn Distance,
    ) -> Vec<(f32, InternalId)> {
        let num_nodes = self.nodes.read().len();
        let mut visited = vec![false; num_nodes.max(entry as usize + 1)];
        visited[entry as usize] = true;

        let entry_dist = distance.distance(query, storage.get_vector(entry));

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, InternalId)>> = BinaryHeap::new();
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));

        // Max-heap of results (worst at top)
        let mut results: BinaryHeap<(OrderedFloat<f32>, InternalId)> = BinaryHeap::new();
        results.push((OrderedFloat(entry_dist), entry));

        while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            // Get worst result distance
            let worst_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);

            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            // Get neighbors for this node
            let neighbor_ids: Vec<InternalId> = {
                let nodes = self.nodes.read();
                nodes[c_id as usize].neighbors[level].read().clone()
            };

            for neighbor_id in neighbor_ids {
                if neighbor_id as usize >= visited.len() {
                    visited.resize(neighbor_id as usize + 1, false);
                }
                if visited[neighbor_id as usize] {
                    continue;
                }
                visited[neighbor_id as usize] = true;

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

        // Convert to sorted vec (best first)
        let mut result_vec: Vec<_> = results.into_iter().map(|(d, id)| (d.0, id)).collect();
        result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result_vec
    }

    /// Simple neighbor selection (take closest)
    fn select_neighbors_simple(
        &self,
        candidates: &[(f32, InternalId)],
        max_neighbors: usize,
    ) -> Vec<InternalId> {
        candidates
            .iter()
            .take(max_neighbors)
            .map(|(_, id)| *id)
            .collect()
    }

    /// Shrink connections with already-locked neighbors
    fn shrink_connections_locked(
        &self,
        node_id: InternalId,
        neighbors: &mut Vec<InternalId>,
        max_neighbors: usize,
        storage: &dyn VectorStorage,
        distance: &dyn Distance,
    ) {
        if neighbors.len() <= max_neighbors {
            return;
        }

        let node_vec = storage.get_vector(node_id);

        // Calculate distances and sort
        let mut with_dist: Vec<_> = neighbors
            .iter()
            .map(|&id| {
                let dist = distance.distance(node_vec, storage.get_vector(id));
                (dist, id)
            })
            .collect();

        with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        neighbors.clear();
        neighbors.extend(with_dist.iter().take(max_neighbors).map(|(_, id)| *id));
    }
}
