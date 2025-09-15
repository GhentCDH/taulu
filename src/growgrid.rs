use std::collections::HashMap;
use std::ops::{Add, Index};

use numpy::PyReadonlyArray2;
use pathfinding::prelude::astar;
use pyo3::prelude::*;
use pyo3::{exceptions::PyException, FromPyObject, PyErr, PyResult};

use crate::Direction;

// A coordinate of the grid (row, col)
#[derive(FromPyObject, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Coord(usize, usize);

#[derive(FromPyObject, IntoPyObject, Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Corner(pub f64, pub f64);

/// A selection of rows of points that make up the corners of a table.
#[pyclass]
pub struct TableGrower {
    /// The points in the grid, indexed by (row, col)
    corners: Vec<Vec<Option<Corner>>>,
    /// The number of columns in the grid, being columns of the table + 1
    #[pyo3(get)]
    columns: usize,
    /// Edge of the table grid, where new points can be grown from
    edge: HashMap<Coord, (Corner, f64)>,
    /// The size of the search region to use when finding the best corner match
    search_region: usize,
    /// The distance penalty to use when finding the best corner match
    distance_penalty: f64,
    column_widths: Vec<f64>,
    row_heights: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Right,
    Down,
    Left,
    Up,
}

impl Add<Step> for Coord {
    type Output = Coord;

    fn add(self, rhs: Step) -> Self::Output {
        match rhs {
            Step::Right => Coord(self.0, self.1 + 1),
            Step::Down => Coord(self.0 + 1, self.1),
            Step::Left => Coord(self.0, self.1 - 1),
            Step::Up => Coord(self.0 - 1, self.1),
        }
    }
}

impl From<Corner> for crate::Point {
    fn from(val: Corner) -> Self {
        crate::Point(val.1 as i32, val.0 as i32)
    }
}

#[pymethods]
impl TableGrower {
    #[new]
    fn new(
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
        column_widths: Vec<f64>,
        row_heights: Vec<f64>,
        start_point: Corner,
        search_region: usize,
        distance_penalty: f64,
    ) -> PyResult<Self> {
        let corners = Vec::new();
        let mut table_grower = Self {
            edge: HashMap::new(),
            corners,
            columns: column_widths.len() + 1,
            column_widths,
            row_heights,
            search_region,
            distance_penalty,
        };

        let start_corner = find_best_corner_match(
            &cross_correlation,
            &start_point,
            search_region,
            distance_penalty,
        )
        .ok_or(PyErr::new::<PyException, _>(
            "Couldn't find corner match in search region",
        ))?;

        table_grower.add_corner(table_image, cross_correlation, start_corner.0, Coord(0, 0));

        Ok(table_grower)
    }

    fn get_corner(&self, coord: Coord) -> Option<Corner> {
        if coord.0 >= self.corners.len() || coord.1 >= self.corners[coord.0].len() {
            return None;
        }

        self.corners[coord.0][coord.1]
    }

    fn all_rows_complete(&self) -> bool {
        self.corners.iter().all(|row| row.len() == self.columns)
    }

    fn get_all_corners(&self) -> Vec<Vec<Option<Corner>>> {
        self.corners.clone()
    }

    /// Grow a grid of points starting from start and growing according to the given
    /// column widths and row heights. The table_image is used to guide the growth
    /// using the cross_correlation image to find the best positions for the grid points.
    fn grow_point(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<()> {
        // find the edge point with the highest confidence
        // without emptying the edge
        let (&coord, &(corner, confidence)) = self
            .edge
            .iter()
            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
            .unwrap();

        // print to python
        println!(
            "Growing point at coord {:?} with confidence {}",
            coord, confidence
        );

        self.add_corner(table_image, cross_correlation, corner, coord);

        Ok(())
    }
}

impl TableGrower {
    fn add_corner(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
        corner: Corner,
        coord: Coord,
    ) {
        assert!(coord.1 < self.columns);

        while coord.0 >= self.corners.len() {
            self.corners.push(Vec::with_capacity(self.columns));
        }
        let row = &mut self.corners[coord.0];

        while row.len() < coord.1 {
            row.push(None); // Placeholder points
        }

        if row.len() == coord.1 {
            row.push(Some(corner));
        } else {
            row[coord.1] = Some(corner);
        }

        // Update edge
        // Remove current point from edge
        self.edge.remove(&coord);

        // Add new edge points
        if (coord + Step::Right).1 < self.columns && self[coord + Step::Right].is_none() {
            if let Some((corner, confidence)) =
                self.step_from_coord(&table_image, &cross_correlation, coord, Step::Right)
            {
                self.update_edge(coord + Step::Right, corner, confidence);
            }
        }

        // FIX: prevent unlimited growth downwards by some kind of end condition
        if self[coord + Step::Down].is_none() {
            if let Some((corner, confidence)) =
                self.step_from_coord(&table_image, &cross_correlation, coord, Step::Down)
            {
                self.update_edge(coord + Step::Down, corner, confidence);
            }
        }

        if coord.1 > 0 && self[coord + Step::Left].is_none() {
            if let Some((corner, confidence)) =
                self.step_from_coord(&table_image, &cross_correlation, coord, Step::Left)
            {
                self.update_edge(coord + Step::Left, corner, confidence);
            }
        }

        if coord.0 > 0 && self[coord + Step::Up].is_none() {
            if let Some((corner, confidence)) =
                self.step_from_coord(&table_image, &cross_correlation, coord, Step::Up)
            {
                self.update_edge(coord + Step::Up, corner, confidence);
            }
        }
    }

    fn step_from_coord(
        &self,
        table_image: &PyReadonlyArray2<'_, u8>,
        cross_correlation: &PyReadonlyArray2<'_, u8>,
        coord: Coord,
        step: Step,
    ) -> Option<(Corner, f64)> {
        // construct the goals based on the step direction,
        // known column widths and row heights, and existing points
        let current_corner = self.get_corner(coord)?;

        let (direction, goals) = match step {
            Step::Right => {
                if coord.1 + 1 >= self.columns {
                    return None;
                }

                let next_x = current_corner.0 + self.column_widths[coord.1];
                let goals = (0..self.search_region)
                    .map(|i| {
                        Corner(
                            next_x,
                            current_corner.1 + (i as f64) - (self.search_region as f64) / 2.0,
                        )
                        .into()
                    })
                    .collect::<Vec<_>>();

                (Direction::RightStrict, goals)
            }
            Step::Down => {
                // extend row heights with last value if necessary
                let h = if coord.0 + 1 >= self.row_heights.len() {
                    *self.row_heights.last().unwrap()
                } else {
                    self.row_heights[coord.0]
                };

                let next_y = current_corner.1 + h;
                let goals = (0..self.search_region)
                    .map(|i| {
                        Corner(
                            current_corner.0 + (i as f64) - (self.search_region as f64) / 2.0,
                            next_y,
                        )
                        .into()
                    })
                    .collect::<Vec<_>>();

                (Direction::DownStrict, goals)
            }
            Step::Left => {
                if coord.1 == 0 {
                    return None;
                }

                let next_x = current_corner.0 - self.column_widths[coord.1 - 1];
                let goals = (0..self.search_region)
                    .map(|i| {
                        Corner(
                            next_x,
                            current_corner.1 + (i as f64) - (self.search_region as f64) / 2.0,
                        )
                        .into()
                    })
                    .collect::<Vec<_>>();

                (Direction::LeftStrict, goals)
            }
            Step::Up => {
                if coord.0 == 0 {
                    return None;
                }

                // extend row heights with last value if necessary
                let h = if coord.0 > self.row_heights.len() {
                    *self.row_heights.last().unwrap()
                } else {
                    self.row_heights[coord.0]
                };

                let next_y = current_corner.1 - h;
                let goals = (0..self.search_region)
                    .map(|i| {
                        Corner(
                            current_corner.0 + (i as f64) - (self.search_region as f64) / 2.0,
                            next_y,
                        )
                        .into()
                    })
                    .collect::<Vec<_>>();

                (Direction::UpStrict, goals)
            }
        };

        let path: Vec<(i32, i32)> = astar::<crate::Point, u32, _, _, _, _>(
            &current_corner.into(),
            |p| p.successors(&direction, table_image).unwrap_or_default(),
            |p| p.min_distance(&goals),
            |p| p.at_goal(&goals),
        )
        .map(|r| r.0.into_iter().map(|p| p.into()).collect())?;

        let approx = {
            let last = path.last().unwrap();
            Corner(last.0 as f64, last.1 as f64)
        };

        find_best_corner_match(
            cross_correlation,
            &approx,
            self.search_region,
            self.distance_penalty,
        )
    }

    fn update_edge(&mut self, coord: Coord, corner: Corner, confidence: f64) {
        self.edge
            .entry(coord)
            .and_modify(|entry| {
                if confidence > entry.1 {
                    entry.1 = confidence;
                    entry.0 = corner;
                }
            })
            .or_insert((corner, confidence));
    }
}

impl Index<Coord> for TableGrower {
    type Output = Option<Corner>;

    fn index(&self, index: Coord) -> &Self::Output {
        if index.0 >= self.corners.len() || index.1 >= self.corners[index.0].len() {
            return &None;
        }

        &self.corners[index.0][index.1]
    }
}

fn create_gaussian_weights(region_size: usize, distance_penalty: f64) -> Vec<Vec<f32>> {
    // If no distance penalty, return uniform weights
    if distance_penalty == 0.0 {
        return vec![vec![1.0; region_size]; region_size];
    }

    // Create normalized coordinate system from -1 to 1
    let mut weights = vec![vec![0.0; region_size]; region_size];

    // Calculate sigma based on distance_penalty
    let sigma = if distance_penalty >= 0.999 {
        0.1 // Small sigma for very sharp peak
    } else {
        (-1.0 / (2.0 * (1.0 - distance_penalty).ln())).sqrt()
    };

    (0..region_size).for_each(|i| {
        for j in 0..region_size {
            // Map indices to [-1, 1] range
            let y = -1.0 + 2.0 * (i as f64) / (region_size - 1) as f64;
            let x = -1.0 + 2.0 * (j as f64) / (region_size - 1) as f64;

            let dist_squared = x * x + y * y;
            weights[i][j] = (-dist_squared / (2.0 * sigma * sigma)).exp() as f32;
        }
    });

    weights
}

fn find_best_corner_match(
    cross_correlation: &PyReadonlyArray2<'_, u8>,
    approx: &Corner,
    search_region: usize,
    distance_penalty: f64, // This parameter isn't used in the Python version
) -> Option<(Corner, f64)> {
    let filtered = cross_correlation.as_array();

    // Check if image is empty
    if filtered.is_empty() {
        return None;
    }

    let (height, width) = filtered.dim();
    let x = approx.0.round() as i32;
    let y = approx.1.round() as i32;

    // Calculate crop boundaries (equivalent to Python logic)
    let crop_x = std::cmp::max(0, x - (search_region as i32) / 2) as usize;
    let crop_y = std::cmp::max(0, y - (search_region as i32) / 2) as usize;
    let crop_width = std::cmp::min(search_region, width.saturating_sub(crop_x));
    let crop_height = std::cmp::min(search_region, height.saturating_sub(crop_y));

    // Handle edge cases
    if crop_width == 0 || crop_height == 0 {
        return Some((*approx, 0.0)); // Return original point with 0 confidence
    }

    // Extract cropped region
    let mut cropped = vec![vec![0u8; crop_width]; crop_height];
    for i in 0..crop_height {
        for j in 0..crop_width {
            cropped[i][j] = filtered[[crop_y + i, crop_x + j]];
        }
    }

    if cropped.is_empty() || cropped[0].is_empty() {
        return Some((*approx, 0.0));
    }

    // Apply Gaussian weighting
    let weighted = if crop_height == search_region && crop_width == search_region {
        // Perfect size - apply weights directly
        let weights = create_gaussian_weights(search_region, distance_penalty);
        let mut result = vec![vec![0.0f32; crop_width]; crop_height];
        for i in 0..crop_height {
            for j in 0..crop_width {
                result[i][j] = cropped[i][j] as f32 * weights[i][j];
            }
        }
        result
    } else {
        // Extend crop to match search_region, apply weights, then restore
        let mut extended = vec![vec![0u8; search_region]; search_region];

        // Calculate offset to center the cropped region in extended array
        let offset_y = (search_region - crop_height) / 2;
        let offset_x = (search_region - crop_width) / 2;

        // Place cropped region in center of extended array
        for i in 0..crop_height {
            for j in 0..crop_width {
                extended[offset_y + i][offset_x + j] = cropped[i][j];
            }
        }

        // Apply Gaussian weights to extended array
        let weights = create_gaussian_weights(search_region, distance_penalty);
        let mut weighted_extended = vec![vec![0.0f32; search_region]; search_region];
        for i in 0..search_region {
            for j in 0..search_region {
                weighted_extended[i][j] = extended[i][j] as f32 * weights[i][j];
            }
        }

        // Extract the original region back out
        let mut result = vec![vec![0.0f32; crop_width]; crop_height];
        for i in 0..crop_height {
            for j in 0..crop_width {
                result[i][j] = weighted_extended[offset_y + i][offset_x + j];
            }
        }
        result
    };

    // Find the maximum value and its position
    let mut best_value = 0.0f32;
    let mut best_x = 0;
    let mut best_y = 0;

    for (i, row) in weighted.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_x = j;
                best_y = i;
            }
        }
    }

    // Convert back to global coordinates
    let result_x = (crop_x + best_x) as f64;
    let result_y = (crop_y + best_y) as f64;

    Some((Corner(result_x, result_y), best_value as f64))
}
