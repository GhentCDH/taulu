//! Core Rust implementations for taulu's table segmentation algorithms.
//!
//! This module provides high-performance implementations of:
//! - A* pathfinding for rule following in table images
//! - Table grid growing algorithms
//! - Geometric utilities for line fitting and intersection detection

use std::{convert::Into, f64::consts::PI};

use numpy::{
    PyReadonlyArray2,
    ndarray::{ArrayBase, Dim, ViewRepr},
};
use pathfinding::prelude::astar as astar_rust;
use pyo3::prelude::*;

mod coord;
mod direction;
mod geom_util;
mod invert;
mod point;
mod step;
mod table_grower;
mod traits;

pub use coord::Coord;
pub use direction::Direction;
pub use point::Point;
pub use step::Step;
pub use table_grower::TableGrower;

type Image<'a> = ArrayBase<ViewRepr<&'a u8>, Dim<[usize; 2]>>;

/// Finds the shortest path between a start point and one of multiple goal points
/// using the A* algorithm, optimized for following table rules in binary images.
///
/// # Arguments
///
/// * `img` - Binary image where darker pixels indicate table rules
/// * `start` - Starting point (x, y) coordinates
/// * `goals` - List of possible goal points to reach
/// * `direction` - Search direction: "right", "down", "left", "up", "any", "straight", or "diagonal"
///
/// # Returns
///
/// `Some(Vec<(i32, i32)>)` containing the path points if found, `None` otherwise
///
/// # Example
///
/// ```python
/// from taulu._core import astar
/// import numpy as np
///
/// img = np.array([[255, 0, 255], [255, 0, 255]], dtype=np.uint8)
/// path = astar(img, (0, 0), [(2, 1)], "right")
/// ```
#[pyfunction]
fn astar(
    img: PyReadonlyArray2<'_, u8>,
    start: Point,
    goals: Vec<Point>,
    direction: &str,
) -> PyResult<Option<Vec<(i32, i32)>>> {
    let direction: Direction = direction.try_into()?;

    Ok(astar_rust(
        &start,
        |p| {
            p.successors(&direction, &img.as_array())
                .unwrap_or_default()
        },
        |p| p.min_distance(&goals),
        |p| p.at_goal(&goals),
    )
    .map(|r| r.0.into_iter().map(Into::into).collect()))
}

/// Return the circular median of angles in radians.
fn circular_median_angle(angles: Vec<f64>) -> PyResult<f64> {
    if angles.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute median of empty list",
        ));
    }

    // Helper function to calculate circular distance between two angles
    let circular_distance = |a: f64, b: f64| {
        let diff = (a - b).abs() % (2.0 * PI);
        diff.min(2.0 * PI - diff)
    };

    // Normalize all angles to [0, 2π)
    let angles: Vec<f64> = angles.into_iter().map(|angle| angle % (2.0 * PI)).collect();
    let n = angles.len();

    let mut best_median: Option<f64> = None;
    let mut min_total_distance = f64::INFINITY;

    // Try each angle as a potential "cut point" for linearization
    for &cut_point in &angles {
        // Reorder angles relative to this cut point
        let mut reordered = angles.clone();
        reordered.sort_by(|&x, &y| {
            let x_relative = (x - cut_point) % (2.0 * PI);
            let y_relative = (y - cut_point) % (2.0 * PI);
            x_relative
                .partial_cmp(&y_relative)
                .expect("Should be able to compare floats")
        });

        // Find median in this ordering
        let candidate = if n % 2 == 1 {
            reordered[n / 2]
        } else {
            let a1 = reordered[n / 2 - 1];
            let a2 = reordered[n / 2];

            // Take circular average of the two middle angles
            let mut diff = (a2 - a1) % (2.0 * PI);
            if diff > PI {
                diff -= 2.0 * PI;
            }
            (a1 + diff / 2.0) % (2.0 * PI)
        };

        // Calculate total circular distance to all points
        let total_distance: f64 = angles
            .iter()
            .map(|&angle| circular_distance(candidate, angle))
            .sum();

        if total_distance < min_total_distance {
            min_total_distance = total_distance;
            best_median = Some(candidate);
        }
    }

    Ok(best_median.expect("input shouldn't be empty"))
}

/// Calculate the median slope from a list of line segments.
/// Each line segment is represented as ((x1, y1), (x2, y2)).
#[pyfunction]
fn median_slope(lines: Vec<((f64, f64), (f64, f64))>) -> PyResult<f64> {
    if lines.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute median slope of empty list",
        ));
    }

    let mut angles = Vec::new();

    for ((x1, y1), (x2, y2)) in lines {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let angle = dy.atan2(dx);
        angles.push(angle);
    }

    let median_angle = circular_median_angle(angles)?;

    // Convert back to slope
    let cos_median = median_angle.cos();
    if cos_median.abs() < 1e-9 {
        Ok(f64::INFINITY) // Vertical line
    } else {
        Ok(median_angle.tan())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableGrower>()?;
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    m.add_function(wrap_pyfunction!(median_slope, m)?)?;
    Ok(())
}
