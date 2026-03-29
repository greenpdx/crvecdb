use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::Rng;

use crate::config::{HnswConfig, InternalId};
use crate::distance::Distance;
use crate::error::{CrvecError, Result};
use crate::storage::VectorStorage;

/// Maximum number of layers
pub const MAX_LAYERS: usize = 16;

/// Sentinel value for no entry point
const NO_ENTRY_POINT: u32 = u32::MAX;

/// Pack entry_point (low 32 bits) and max_level (high 32 bits) into a u64
fn pack_entry_state(entry_point: u32, max_level: u32) -> u64 {
    (entry_point as u64) | ((max_level as u64) << 32)
}

/// Unpack entry_point and max_level from a u64
fn unpack_entry_state(state: u64) -> (u32, u32) {
    let entry_point = state as u32;
    let max_level = (state >> 32) as u32;
    (entry_point, max_level)
}

/// A node in the HNSW graph
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
    /// Packed entry_point (low 32) + max_level (high 32), atomically updated
    entry_state: AtomicU64,
    /// Configuration
    pub config: HnswConfig,
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
            entry_state: AtomicU64::new(pack_entry_state(NO_ENTRY_POINT, 0)),
            config,
        }
    }

    /// Get current entry point
    pub fn get_entry_point(&self) -> Option<InternalId> {
        let (ep, _) = unpack_entry_state(self.entry_state.load(Ordering::Acquire));
        if ep == NO_ENTRY_POINT {
            None
        } else {
            Some(ep)
        }
    }

    /// Get current max level
    pub fn get_max_level(&self) -> usize {
        let (_, ml) = unpack_entry_state(self.entry_state.load(Ordering::Acquire));
        ml as usize
    }

    /// Get entry point and max level atomically
    fn get_entry_state(&self) -> (Option<InternalId>, usize) {
        let (ep, ml) = unpack_entry_state(self.entry_state.load(Ordering::Acquire));
        let entry = if ep == NO_ENTRY_POINT { None } else { Some(ep) };
        (entry, ml as usize)
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
                nodes.push(HnswNode::new(0, &self.config));
            }
            nodes[internal_id as usize] = node;
        }

        // Try to become the first entry point (atomic CAS on packed state)
        let initial_state = pack_entry_state(NO_ENTRY_POINT, 0);
        let new_state = pack_entry_state(internal_id, insert_level as u32);
        if self
            .entry_state
            .compare_exchange(initial_state, new_state, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return;
        }

        let query = storage.get_vector(internal_id);
        let (_, current_max_level) = self.get_entry_state();
        let mut current = self.entry_state.load(Ordering::Acquire);
        let (mut current_ep, _) = unpack_entry_state(current);

        // Phase 1: Traverse from top to insert_level + 1 with greedy search
        for level in (insert_level + 1..=current_max_level).rev() {
            current_ep = self.search_layer_single(query, current_ep, level, storage, distance);
        }

        // Phase 2: Insert at levels insert_level down to 0
        let top_level = insert_level.min(current_max_level);
        for level in (0..=top_level).rev() {
            let candidates =
                self.search_layer(query, current_ep, self.config.ef_construction, level, storage, distance);

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
                current_ep = nearest;
            }
        }

        // Update entry point if new node has higher level (CAS loop on packed state)
        loop {
            current = self.entry_state.load(Ordering::Acquire);
            let (_, old_max) = unpack_entry_state(current);
            if insert_level <= old_max as usize {
                break;
            }
            let new_state = pack_entry_state(internal_id, insert_level as u32);
            if self
                .entry_state
                .compare_exchange(current, new_state, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
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
        let num_words = (num_nodes.max(entry as usize + 1) + 63) / 64;
        let mut visited = vec![0u64; num_words];
        visited[entry as usize / 64] |= 1u64 << (entry as usize % 64);

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

        // Convert to sorted vec (best first)
        let mut result_vec: Vec<_> = results.into_iter().map(|(d, id)| (d.0, id)).collect();
        result_vec.sort_by(|a, b| a.0.total_cmp(&b.0));
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

        with_dist.sort_by(|a, b| a.0.total_cmp(&b.0));

        neighbors.clear();
        neighbors.extend(with_dist.iter().take(max_neighbors).map(|(_, id)| *id));
    }

    /// Save graph to file
    ///
    /// File format:
    /// - Magic: 4 bytes "HNSW"
    /// - Version: 4 bytes (1)
    /// - Entry point: 4 bytes (u32, NO_ENTRY_POINT if none)
    /// - Max level: 4 bytes (u32)
    /// - M: 4 bytes (u32)
    /// - M0: 4 bytes (u32)
    /// - Num nodes: 4 bytes (u32)
    /// - For each node:
    ///   - max_layer: 1 byte
    ///   - For each layer 0..=max_layer:
    ///     - num_neighbors: 2 bytes (u16)
    ///     - neighbors: num_neighbors * 4 bytes (u32 each)
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let (entry_point, max_level) = unpack_entry_state(self.entry_state.load(Ordering::Acquire));

        // Magic
        writer.write_all(b"HNSW")?;
        // Version
        writer.write_all(&1u32.to_le_bytes())?;
        // Entry point
        writer.write_all(&entry_point.to_le_bytes())?;
        // Max level
        writer.write_all(&max_level.to_le_bytes())?;
        // M and M0
        writer.write_all(&(self.config.m as u32).to_le_bytes())?;
        writer.write_all(&(self.config.m0 as u32).to_le_bytes())?;

        let nodes = self.nodes.read();
        // Num nodes
        writer.write_all(&(nodes.len() as u32).to_le_bytes())?;

        // Write each node
        for node in nodes.iter() {
            // max_layer
            writer.write_all(&[node.max_layer as u8])?;

            // Each layer's neighbors
            for layer in 0..=node.max_layer {
                let neighbors = node.neighbors[layer].read();
                writer.write_all(&(neighbors.len() as u16).to_le_bytes())?;
                for &neighbor in neighbors.iter() {
                    writer.write_all(&neighbor.to_le_bytes())?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load graph from file
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"HNSW" {
            return Err(CrvecError::InvalidFormat("invalid graph magic".into()));
        }

        // Version
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(CrvecError::InvalidFormat(format!("unsupported graph version: {}", version)));
        }

        // Entry point
        reader.read_exact(&mut buf4)?;
        let entry_point = u32::from_le_bytes(buf4);

        // Max level
        reader.read_exact(&mut buf4)?;
        let max_level = u32::from_le_bytes(buf4);

        // M and M0
        reader.read_exact(&mut buf4)?;
        let m = u32::from_le_bytes(buf4) as usize;
        reader.read_exact(&mut buf4)?;
        let m0 = u32::from_le_bytes(buf4) as usize;

        // Num nodes
        reader.read_exact(&mut buf4)?;
        let num_nodes = u32::from_le_bytes(buf4) as usize;

        let config = HnswConfig {
            m,
            m0,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
        };

        let mut nodes = Vec::with_capacity(num_nodes);

        // Read each node
        for _ in 0..num_nodes {
            // max_layer
            let mut buf1 = [0u8; 1];
            reader.read_exact(&mut buf1)?;
            let max_layer = buf1[0] as usize;

            let mut neighbors = Vec::with_capacity(max_layer + 1);
            for _ in 0..=max_layer {
                // num_neighbors
                let mut buf2 = [0u8; 2];
                reader.read_exact(&mut buf2)?;
                let num_neighbors = u16::from_le_bytes(buf2) as usize;

                let mut layer_neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    reader.read_exact(&mut buf4)?;
                    layer_neighbors.push(u32::from_le_bytes(buf4));
                }
                neighbors.push(RwLock::new(layer_neighbors));
            }

            nodes.push(HnswNode { max_layer, neighbors });
        }

        Ok(Self {
            nodes: RwLock::new(nodes),
            entry_state: AtomicU64::new(pack_entry_state(entry_point, max_level)),
            config,
        })
    }
}
