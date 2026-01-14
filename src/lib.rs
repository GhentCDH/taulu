//! Core Rust implementations for taulu's table segmentation algorithms.
//!
//! This module provides high-performance implementations of:
//! - A* pathfinding for rule following in table images
//! - Table grid growing algorithms
//! - Geometric utilities for line fitting and intersection detection

use std::convert::Into;

use numpy::{
    PyReadonlyArray2,
    ndarray::{ArrayBase, Dim, ViewRepr},
};
use pathfinding::prelude::astar as astar_rust;
use pyo3::prelude::*;

mod coord;
mod direction;
mod edge_queue;
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

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TableGrower>()?;
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    Ok(())
}
