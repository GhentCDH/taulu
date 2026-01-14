//! Priority queue for edge candidates using lazy deletion pattern.
//!
//! This module provides an efficient priority queue for managing edge candidates
//! in the table growing algorithm. It uses a `BinaryHeap` for O(1) max lookups
//! combined with a `HashMap` for O(1) staleness checks, achieving better asymptotic
//! performance than a pure `HashMap` approach for repeated max-extraction.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::{Coord, Point};

/// A candidate edge point with its confidence score.
#[derive(Clone, Debug)]
struct EdgeCandidate {
    coord: Coord,
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

/// A priority queue for edge candidates that supports efficient max-extraction.
///
/// Uses the "lazy deletion" pattern: entries may become stale in the heap when
/// updated, but are filtered out during extraction by checking against the index.
///
/// # Complexity
///
/// | Operation | Time |
/// |-----------|------|
/// | `insert` | O(log n) |
/// | `pop_max` | O(log n) amortized |
/// | `remove` | O(1) (lazy) |
/// | `len` | O(1) |
#[derive(Debug)]
pub struct EdgeQueue {
    /// Max-heap of candidates (may contain stale entries)
    heap: BinaryHeap<EdgeCandidate>,
    /// Maps coord -> (point, current best confidence) for staleness checks
    index: HashMap<Coord, (Point, f64)>,
}

impl EdgeQueue {
    /// Creates an empty edge queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            index: HashMap::new(),
        }
    }

    /// Inserts or updates an edge candidate.
    ///
    /// If a candidate already exists at this coordinate with lower confidence,
    /// it will be superseded (the old entry becomes stale in the heap).
    pub fn insert(&mut self, coord: Coord, point: Point, confidence: f64) {
        // Only insert if this is a new best for this coord
        let dominated = self
            .index
            .get(&coord)
            .is_some_and(|(_, c)| *c >= confidence);
        if dominated {
            return;
        }
        self.index.insert(coord, (point, confidence));
        self.heap.push(EdgeCandidate { coord, confidence });
    }

    /// Removes and returns the highest-confidence candidate.
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop_max(&mut self) -> Option<(Coord, Point, f64)> {
        // Lazy deletion: skip stale entries
        while let Some(candidate) = self.heap.pop() {
            // Check if this entry is still current
            if let Some(&(point, current_confidence)) = self.index.get(&candidate.coord)
                && (current_confidence - candidate.confidence).abs() < f64::EPSILON
            {
                // This is the current best entry for this coord
                self.index.remove(&candidate.coord);
                return Some((candidate.coord, point, candidate.confidence));
            }
            // Entry is stale (superseded or already removed), skip it
        }
        None
    }

    /// Peeks at the highest-confidence candidate without removing it.
    #[must_use]
    pub fn peek_max_confidence(&self) -> Option<f64> {
        // Find the first non-stale entry
        for candidate in &self.heap {
            if let Some(&(_, current_confidence)) = self.index.get(&candidate.coord)
                && (current_confidence - candidate.confidence).abs() < f64::EPSILON
            {
                return Some(candidate.confidence);
            }
        }
        None
    }

    /// Removes a candidate at the given coordinate (lazy deletion).
    ///
    /// The entry remains in the heap but will be skipped during `pop_max`.
    pub fn remove(&mut self, coord: &Coord) {
        self.index.remove(coord);
    }

    /// Returns the number of active candidates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if there are no active candidates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Returns an iterator over all active (coord, point, confidence) entries.
    ///
    /// This iterates over the index, not the heap, so it only returns current entries.
    pub fn iter(&self) -> impl Iterator<Item = (Coord, Point, f64)> + '_ {
        self.index
            .iter()
            .map(|(&coord, &(point, confidence))| (coord, point, confidence))
    }
}

impl Default for EdgeQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_pop() {
        let mut queue = EdgeQueue::new();
        queue.insert(Coord::new(0, 0), Point(10, 20), 0.5);
        queue.insert(Coord::new(1, 0), Point(10, 50), 0.8);
        queue.insert(Coord::new(0, 1), Point(60, 20), 0.3);

        assert_eq!(queue.len(), 3);

        let (coord, point, conf) = queue.pop_max().unwrap();
        assert_eq!(coord, Coord::new(1, 0));
        assert_eq!(point, Point(10, 50));
        assert!((conf - 0.8).abs() < f64::EPSILON);

        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_update_higher_confidence() {
        let mut queue = EdgeQueue::new();
        queue.insert(Coord::new(0, 0), Point(10, 20), 0.5);
        queue.insert(Coord::new(0, 0), Point(15, 25), 0.9); // Same coord, higher confidence

        assert_eq!(queue.len(), 1);

        let (coord, point, conf) = queue.pop_max().unwrap();
        assert_eq!(coord, Coord::new(0, 0));
        assert_eq!(point, Point(15, 25)); // Should be the updated point
        assert!((conf - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_lower_confidence_ignored() {
        let mut queue = EdgeQueue::new();
        queue.insert(Coord::new(0, 0), Point(10, 20), 0.9);
        queue.insert(Coord::new(0, 0), Point(15, 25), 0.5); // Same coord, lower confidence

        assert_eq!(queue.len(), 1);

        let (_, point, conf) = queue.pop_max().unwrap();
        assert_eq!(point, Point(10, 20)); // Original point preserved
        assert!((conf - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove() {
        let mut queue = EdgeQueue::new();
        queue.insert(Coord::new(0, 0), Point(10, 20), 0.9);
        queue.insert(Coord::new(1, 0), Point(10, 50), 0.5);

        queue.remove(&Coord::new(0, 0));
        assert_eq!(queue.len(), 1);

        let (coord, _, _) = queue.pop_max().unwrap();
        assert_eq!(coord, Coord::new(1, 0));
    }

    #[test]
    fn test_empty() {
        let mut queue = EdgeQueue::new();
        assert!(queue.is_empty());
        assert!(queue.pop_max().is_none());

        queue.insert(Coord::new(0, 0), Point(10, 20), 0.5);
        assert!(!queue.is_empty());

        queue.pop_max();
        assert!(queue.is_empty());
    }
}
