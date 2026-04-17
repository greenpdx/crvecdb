use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::OnceLock;
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

/// Graph file format version.
const GRAPH_FORMAT_VERSION: u32 = 2;

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
///
/// `nodes` is pre-allocated to the index's capacity and never grows. Each slot
/// is a `OnceLock<HnswNode>` that is populated exactly once during `insert()`.
/// This removes the need for an outer `RwLock` on the `nodes` vector: readers
/// and concurrent inserters never contend on it, and a slot can never be
/// replaced while a reader is looking at it.
pub struct HnswGraph {
    /// Pre-allocated slots, one per possible `InternalId`. Each slot is set
    /// exactly once when the corresponding node is inserted.
    pub nodes: Vec<OnceLock<HnswNode>>,
    /// Packed entry_point (low 32) + max_level (high 32), atomically updated
    entry_state: AtomicU64,
    /// Configuration
    pub config: HnswConfig,
}

impl HnswGraph {
    pub fn new(config: HnswConfig, capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            nodes.push(OnceLock::new());
        }
        Self {
            nodes,
            entry_state: AtomicU64::new(pack_entry_state(NO_ENTRY_POINT, 0)),
            config,
        }
    }

    /// Borrow the node at `id`. Panics if the slot has not been inserted yet;
    /// by construction every id we traverse is reachable from a completed
    /// insert, so a missing slot is a logic bug.
    #[inline]
    fn node(&self, id: InternalId) -> &HnswNode {
        self.nodes[id as usize]
            .get()
            .expect("HNSW internal: accessed an uninitialized node slot")
    }

    /// Get entry point and max level atomically
    pub(crate) fn get_entry_state(&self) -> (Option<InternalId>, usize) {
        let (ep, ml) = unpack_entry_state(self.entry_state.load(Ordering::Acquire));
        let entry = if ep == NO_ENTRY_POINT { None } else { Some(ep) };
        (entry, ml as usize)
    }

    /// Get number of populated nodes.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|s| s.get().is_some()).count()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.nodes.iter().all(|s| s.get().is_none())
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

        // Populate our slot exactly once. If it's already set, that's a caller
        // bug (duplicate internal_id) — panic loudly rather than corrupt state.
        self.nodes[internal_id as usize]
            .set(node)
            .map_err(|_| ())
            .expect("HNSW internal: insert called twice for the same internal_id");

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
        // Atomic read of the (entry_point, max_level) pair — reading these
        // separately lets a concurrent CAS pair an old entry point with a new
        // max_level, which then indexes past `entry_point.neighbors`.
        let (ep_opt, current_max_level) = self.get_entry_state();
        let mut current_ep = match ep_opt {
            Some(ep) => ep,
            None => return,
        };

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

            // Add connections from new node to selected neighbors.
            // Our own slot was set above, so .get() returns Some here.
            {
                let mut new_node_neighbors = self.node(internal_id).neighbors[level].write();
                *new_node_neighbors = selected.clone();
            }

            // Add reverse connections. Each neighbor_id came from a completed
            // traversal at this layer, so its slot is populated and its
            // neighbors vec has at least `level + 1` layers.
            for &neighbor_id in &selected {
                let mut neighbor_neighbors = self.node(neighbor_id).neighbors[level].write();

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
            let current = self.entry_state.load(Ordering::Acquire);
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
            let neighbors = self.node(current).neighbors[level].read();
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

            drop(neighbors);

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
        let num_words = (self.nodes.len().max(entry as usize + 1) + 63) / 64;
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

            // Snapshot this node's neighbors at this level
            let neighbor_ids: Vec<InternalId> = self.node(c_id).neighbors[level].read().clone();

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
    /// File format (version 2):
    /// - Magic: 4 bytes "HNSW"
    /// - Version: 4 bytes (2)
    /// - Entry point: 4 bytes (u32, NO_ENTRY_POINT if none)
    /// - Max level: 4 bytes (u32)
    /// - M: 4 bytes (u32)
    /// - M0: 4 bytes (u32)
    /// - Capacity: 4 bytes (u32)
    /// - Num populated: 4 bytes (u32)
    /// - For each populated node:
    ///   - id: 4 bytes (u32)
    ///   - max_layer: 1 byte
    ///   - For each layer 0..=max_layer:
    ///     - num_neighbors: 2 bytes (u16)
    ///     - neighbors: num_neighbors * 4 bytes (u32 each)
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let (entry_point, max_level) = unpack_entry_state(self.entry_state.load(Ordering::Acquire));

        writer.write_all(b"HNSW")?;
        writer.write_all(&GRAPH_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&entry_point.to_le_bytes())?;
        writer.write_all(&max_level.to_le_bytes())?;
        writer.write_all(&(self.config.m as u32).to_le_bytes())?;
        writer.write_all(&(self.config.m0 as u32).to_le_bytes())?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;

        // Collect populated slots (no lock needed — OnceLock handles it).
        let populated: Vec<(u32, &HnswNode)> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.get().map(|n| (i as u32, n)))
            .collect();

        writer.write_all(&(populated.len() as u32).to_le_bytes())?;

        for (id, node) in populated {
            writer.write_all(&id.to_le_bytes())?;
            writer.write_all(&[node.max_layer as u8])?;

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

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"HNSW" {
            return Err(CrvecError::InvalidFormat("invalid graph magic".into()));
        }

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != GRAPH_FORMAT_VERSION {
            return Err(CrvecError::InvalidFormat(format!(
                "unsupported graph version: {} (expected {})",
                version, GRAPH_FORMAT_VERSION
            )));
        }

        reader.read_exact(&mut buf4)?;
        let entry_point = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let max_level = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let m = u32::from_le_bytes(buf4) as usize;
        reader.read_exact(&mut buf4)?;
        let m0 = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let capacity = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let num_populated = u32::from_le_bytes(buf4) as usize;

        let config = HnswConfig {
            m,
            m0,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
        };

        let mut nodes: Vec<OnceLock<HnswNode>> = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            nodes.push(OnceLock::new());
        }

        for _ in 0..num_populated {
            reader.read_exact(&mut buf4)?;
            let id = u32::from_le_bytes(buf4) as usize;
            if id >= capacity {
                return Err(CrvecError::InvalidFormat(format!(
                    "node id {} exceeds capacity {}",
                    id, capacity
                )));
            }

            let mut buf1 = [0u8; 1];
            reader.read_exact(&mut buf1)?;
            let max_layer = buf1[0] as usize;

            let mut neighbors = Vec::with_capacity(max_layer + 1);
            for _ in 0..=max_layer {
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

            nodes[id]
                .set(HnswNode { max_layer, neighbors })
                .map_err(|_| ())
                .expect("duplicate node id in graph file");
        }

        Ok(Self {
            nodes,
            entry_state: AtomicU64::new(pack_entry_state(entry_point, max_level)),
            config,
        })
    }
}
