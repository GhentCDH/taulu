//! Benchmark comparing HashMap vs BinaryHeap for edge candidate selection.
//!
//! This benchmarks the core pattern used in `TableGrower`:
//! - Repeatedly find the maximum-confidence edge candidate
//! - Remove it from the collection
//! - Add new candidates (simulating neighbor discovery)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Grid coordinate (row, col)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coord(usize, usize);

/// Pixel position
#[derive(Debug, Clone, Copy)]
struct Point(i32, i32);

// =============================================================================
// Current implementation: HashMap with O(n) max lookup
// =============================================================================

struct HashMapEdge {
    edge: HashMap<Coord, (Point, f64)>,
}

impl HashMapEdge {
    fn new() -> Self {
        Self {
            edge: HashMap::new(),
        }
    }

    fn insert(&mut self, coord: Coord, point: Point, confidence: f64) {
        self.edge
            .entry(coord)
            .and_modify(|entry| {
                if confidence > entry.1 {
                    entry.1 = confidence;
                    entry.0 = point;
                }
            })
            .or_insert((point, confidence));
    }

    fn pop_max(&mut self) -> Option<(Coord, Point, f64)> {
        let (&coord, &(point, confidence)) = self
            .edge
            .iter()
            .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(Ordering::Equal))?;
        self.edge.remove(&coord);
        Some((coord, point, confidence))
    }

    fn len(&self) -> usize {
        self.edge.len()
    }
}

// =============================================================================
// Proposed implementation: BinaryHeap with lazy deletion
// =============================================================================

#[derive(Clone)]
struct EdgeCandidate {
    coord: Coord,
    point: Point,
    confidence: f64,
}

impl PartialEq for EdgeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.confidence == other.confidence
    }
}

impl Eq for EdgeCandidate {}

impl PartialOrd for EdgeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by confidence
        self.confidence
            .partial_cmp(&other.confidence)
            .unwrap_or(Ordering::Equal)
    }
}

struct BinaryHeapEdge {
    heap: BinaryHeap<EdgeCandidate>,
    /// Maps coord -> current best confidence (for staleness check)
    index: HashMap<Coord, f64>,
}

impl BinaryHeapEdge {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            index: HashMap::new(),
        }
    }

    fn insert(&mut self, coord: Coord, point: Point, confidence: f64) {
        // Only insert if this is a new best for this coord
        let dominated = self.index.get(&coord).is_some_and(|&c| c >= confidence);
        if dominated {
            return;
        }
        self.index.insert(coord, confidence);
        self.heap.push(EdgeCandidate {
            coord,
            point,
            confidence,
        });
    }

    fn pop_max(&mut self) -> Option<(Coord, Point, f64)> {
        // Lazy deletion: skip stale entries
        while let Some(candidate) = self.heap.pop() {
            // Check if this entry is still current
            if let Some(&current_confidence) = self.index.get(&candidate.coord) {
                if (current_confidence - candidate.confidence).abs() < f64::EPSILON {
                    // This is the current best entry for this coord
                    self.index.remove(&candidate.coord);
                    return Some((candidate.coord, candidate.point, candidate.confidence));
                }
            }
            // Entry is stale (superseded or already removed), skip it
        }
        None
    }

    fn len(&self) -> usize {
        self.index.len()
    }
}

// =============================================================================
// Benchmark setup
// =============================================================================

/// Simulates the table growing pattern:
/// 1. Start with `initial_edges` candidates
/// 2. Repeatedly: pop max, add `new_edges_per_pop` new candidates
/// 3. Repeat for `iterations` cycles
fn simulate_hashmap(initial_edges: usize, new_edges_per_pop: usize, iterations: usize) {
    let mut edge = HashMapEdge::new();

    // Seed initial edges
    for i in 0..initial_edges {
        let coord = Coord(i / 10, i % 10);
        let point = Point(i as i32 * 50, i as i32 * 30);
        let confidence = (i as f64) / (initial_edges as f64);
        edge.insert(coord, point, confidence);
    }

    let mut next_row = initial_edges / 10 + 1;

    for iter in 0..iterations {
        // Pop the max
        if let Some((coord, _point, _confidence)) = edge.pop_max() {
            // Add new edges (simulating neighbor discovery)
            for j in 0..new_edges_per_pop {
                let new_coord = Coord(next_row, j);
                let new_point = Point(next_row as i32 * 50, j as i32 * 30);
                // Confidence varies to simulate real data
                let new_confidence = 0.3 + 0.5 * ((iter + j) as f64 / iterations as f64);
                edge.insert(new_coord, new_point, new_confidence);
            }
            next_row += 1;

            // Sometimes update existing edges with higher confidence
            if iter % 3 == 0 && edge.len() > 0 {
                let update_coord = Coord(coord.0.saturating_sub(1), coord.1);
                edge.insert(update_coord, Point(0, 0), 0.95);
            }
        }
    }

    black_box(edge.len());
}

fn simulate_binaryheap(initial_edges: usize, new_edges_per_pop: usize, iterations: usize) {
    let mut edge = BinaryHeapEdge::new();

    // Seed initial edges
    for i in 0..initial_edges {
        let coord = Coord(i / 10, i % 10);
        let point = Point(i as i32 * 50, i as i32 * 30);
        let confidence = (i as f64) / (initial_edges as f64);
        edge.insert(coord, point, confidence);
    }

    let mut next_row = initial_edges / 10 + 1;

    for iter in 0..iterations {
        // Pop the max
        if let Some((coord, _point, _confidence)) = edge.pop_max() {
            // Add new edges (simulating neighbor discovery)
            for j in 0..new_edges_per_pop {
                let new_coord = Coord(next_row, j);
                let new_point = Point(next_row as i32 * 50, j as i32 * 30);
                // Confidence varies to simulate real data
                let new_confidence = 0.3 + 0.5 * ((iter + j) as f64 / iterations as f64);
                edge.insert(new_coord, new_point, new_confidence);
            }
            next_row += 1;

            // Sometimes update existing edges with higher confidence
            if iter % 3 == 0 && edge.len() > 0 {
                let update_coord = Coord(coord.0.saturating_sub(1), coord.1);
                edge.insert(update_coord, Point(0, 0), 0.95);
            }
        }
    }

    black_box(edge.len());
}

fn bench_edge_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_selection");

    // Test different table sizes
    // (initial_edges, new_edges_per_pop, iterations) - simulating different table sizes
    let scenarios = [
        ("small_table_10x5", 50, 2, 50),     // 10 cols, ~5 rows
        ("medium_table_10x20", 50, 3, 200),  // 10 cols, ~20 rows
        ("large_table_15x50", 75, 3, 750),   // 15 cols, ~50 rows
        ("huge_table_20x100", 100, 4, 2000), // 20 cols, ~100 rows
    ];

    for (name, initial, new_per_pop, iters) in scenarios {
        group.bench_with_input(
            BenchmarkId::new("hashmap", name),
            &(initial, new_per_pop, iters),
            |b, &(initial, new_per_pop, iters)| {
                b.iter(|| simulate_hashmap(initial, new_per_pop, iters));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binaryheap", name),
            &(initial, new_per_pop, iters),
            |b, &(initial, new_per_pop, iters)| {
                b.iter(|| simulate_binaryheap(initial, new_per_pop, iters));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_edge_selection);
criterion_main!(benches);
