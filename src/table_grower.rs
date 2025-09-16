use std::collections::HashMap;
use std::ops::{Add, Index};

use numpy::PyReadonlyArray2;
use pathfinding::prelude::astar;
use pyo3::prelude::*;
use pyo3::{FromPyObject, PyResult};
use rayon::prelude::*;

use crate::traits::Xy;
use crate::{Direction, Image, Point};

// A coordinate of the grid (row, col)
// This struct is used to make clear that the order is (row, col) not (x, y)
#[derive(FromPyObject, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Coord(usize, usize);

impl Xy<usize> for Coord {
    #[inline]
    fn x(&self) -> usize {
        self.1
    }

    #[inline]
    fn y(&self) -> usize {
        self.0
    }
}

impl Coord {
    pub fn new(x: usize, y: usize) -> Self {
        Self(y, x)
    }
}

impl From<Coord> for Point {
    fn from(val: Coord) -> Self {
        Point(val.x() as i32, val.y() as i32)
    }
}

impl From<Point> for Coord {
    fn from(val: Point) -> Self {
        Coord(val.1 as usize, val.0 as usize)
    }
}

/// A selection of rows of points that make up the corners of a table.
#[pyclass]
#[derive(Debug)]
pub struct TableGrower {
    /// The points in the grid, indexed by (row, col)
    corners: Vec<Vec<Option<Point>>>,
    /// The number of columns in the grid, being columns of the table + 1
    #[pyo3(get)]
    columns: usize,
    /// Edge of the table grid, where new points can be grown from
    edge: HashMap<Coord, (Point, f64)>,
    /// The size of the search region to use when finding the best corner match
    search_region: usize,
    /// The distance penalty to use when finding the best corner match
    distance_penalty: f64,
    column_widths: Vec<i32>,
    row_heights: Vec<i32>,
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

#[pymethods]
impl TableGrower {
    #[new]
    #[allow(clippy::too_many_arguments)]
    /// Notice that the start_point is given as (x, y), both being integers
    fn new(
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
        column_widths: Vec<i32>,
        row_heights: Vec<i32>,
        start_point: Point,
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

        table_grower.add_corner(
            &table_image.as_array(),
            &cross_correlation.as_array(),
            start_point,
            Coord(0, 0),
        );

        Ok(table_grower)
    }

    fn get_corner(&self, coord: Coord) -> Option<Point> {
        if coord.0 >= self.corners.len() || coord.1 >= self.corners[coord.0].len() {
            return None;
        }

        self.corners[coord.0][coord.1]
    }

    fn all_rows_complete(&self) -> bool {
        self.corners.iter().all(|row| row.len() == self.columns)
    }

    fn get_all_corners(&self) -> Vec<Vec<Option<Point>>> {
        self.corners.clone()
    }

    fn get_edge_points(&self) -> Vec<(Point, f64)> {
        self.edge.values().cloned().collect()
    }

    /// Grow a grid of points starting from start and growing according to the given
    /// column widths and row heights. The table_image is used to guide the growth
    /// using the cross_correlation image to find the best positions for the grid points.
    fn grow_point(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<Option<f64>> {
        self.grow_point_internal(&table_image.as_array(), &cross_correlation.as_array())
    }

    fn grow_points(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<()> {
        loop {
            if self
                .grow_point_internal(&table_image.as_array(), &cross_correlation.as_array())?
                .is_none()
            {
                break Ok(());
            }
        }
    }
}

impl TableGrower {
    /// Grow a grid of points starting from start and growing according to the given
    /// column widths and row heights. The table_image is used to guide the growth
    /// using the cross_correlation image to find the best positions for the grid points.
    fn grow_point_internal(
        &mut self,
        table_image: &Image,
        cross_correlation: &Image,
    ) -> PyResult<Option<f64>> {
        // find the edge point with the highest confidence
        // without emptying the edge
        let Some((&coord, &(corner, confidence))) = self
            .edge
            .iter()
            .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
        else {
            return Ok(None);
        };

        let _ = self.add_corner(table_image, cross_correlation, corner, coord);

        Ok(Some(confidence))
    }

    fn add_corner(
        &mut self,
        table_image: &Image,
        cross_correlation: &Image,
        corner_point: Point,
        coord: Coord,
    ) -> bool {
        assert!(coord.x() < self.columns);

        while self.corners.len() <= coord.y() {
            self.corners.push(Vec::with_capacity(self.columns));
        }
        let row = &mut self.corners[coord.0];

        while row.len() < coord.x() {
            row.push(None);
        }

        if row.len() == coord.x() {
            row.push(Some(corner_point));
        } else {
            row[coord.x()] = Some(corner_point);
        }

        // Update edge: Remove current point from edge
        self.edge.remove(&coord);

        let directions = [
            (
                Step::Right,
                coord + Step::Right,
                (coord + Step::Right).x() < self.columns,
            ),
            (Step::Down, coord + Step::Down, true),
            (Step::Left, coord + Step::Left, coord.x() > 0),
            (Step::Up, coord + Step::Up, coord.y() > 0),
        ];

        let step_results: Vec<Option<(Coord, Point, f32)>> = directions
            .par_iter()
            .map(|(step, new_coord, condition)| {
                if *condition && self[*new_coord].is_none() {
                    if let Some((corner, confidence)) =
                        self.step_from_coord(&table_image, &cross_correlation, coord, *step)
                    {
                        Some((*new_coord, corner, confidence as f32))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        let mut edge_added = false;

        for (new_coord, corner, confidence) in step_results.iter().flatten().copied() {
            self.update_edge(new_coord, corner, confidence as f64);
            edge_added = true;
        }

        edge_added
    }

    fn step_from_coord(
        &self,
        table_image: &Image,
        cross_correlation: &Image,
        coord: Coord,
        step: Step,
    ) -> Option<(Point, f64)> {
        // construct the goals based on the step direction,
        // known column widths and row heights, and existing points
        let current_point = self.get_corner(coord)?;

        let image_size = table_image.shape();
        let height = image_size[0];
        let width = image_size[1];

        let (direction, goals) = match step {
            Step::Right => {
                if coord.x() + 1 >= self.columns {
                    return None;
                }

                let next_x = current_point.x() + self.column_widths[coord.x()];
                let goals: Vec<crate::Point> = (0..(self.search_region as i32))
                    .map(|i| {
                        Point(
                            next_x,
                            current_point.y() + i - self.search_region as i32 / 2,
                        )
                    })
                    .filter(|p| p.within((0, 0, width as i32, height as i32)))
                    .collect();

                (Direction::RightStrict, goals)
            }
            Step::Down => {
                // extend row heights with last value if necessary
                let h = if coord.y() >= self.row_heights.len() {
                    *self.row_heights.last().unwrap()
                } else {
                    self.row_heights[coord.y()]
                };

                let next_y = current_point.y() + h;
                let goals = (0..(self.search_region as i32))
                    .map(|i| {
                        Point(
                            current_point.x() + i - self.search_region as i32 / 2,
                            next_y,
                        )
                    })
                    .filter(|p| p.within((0, 0, width as i32, height as i32)))
                    .collect::<Vec<_>>();

                (Direction::DownStrict, goals)
            }
            Step::Left => {
                if coord.x() == 0 {
                    return None;
                }

                let next_x = current_point.x() - self.column_widths[coord.x() - 1];
                let goals = (0..(self.search_region as i32))
                    .map(|i| {
                        Point(
                            next_x,
                            current_point.y() + i - self.search_region as i32 / 2,
                        )
                    })
                    .filter(|p| p.within((0, 0, width as i32, height as i32)))
                    .collect::<Vec<_>>();

                (Direction::LeftStrict, goals)
            }
            Step::Up => {
                if coord.y() == 0 {
                    return None;
                }

                // extend row heights with last value if necessary
                let h = if coord.y() > self.row_heights.len() {
                    *self.row_heights.last().unwrap()
                } else {
                    self.row_heights[coord.y() - 1]
                };

                let next_y = current_point.1 - h;
                let goals = (0..(self.search_region as i32))
                    .map(|i| {
                        Point(
                            current_point.x() + i - self.search_region as i32 / 2,
                            next_y,
                        )
                    })
                    .filter(|p| p.within((0, 0, width as i32, height as i32)))
                    .collect::<Vec<_>>();

                (Direction::UpStrict, goals)
            }
        };

        if goals.is_empty() {
            return None;
        }

        let path: Vec<(i32, i32)> = astar::<crate::Point, u32, _, _, _, _>(
            &current_point,
            |p| p.successors(&direction, table_image).unwrap_or_default(),
            |p| p.min_distance(&goals),
            |p| p.at_goal(&goals),
        )
        .map(|r| r.0.into_iter().map(|p| p.into()).collect())?;

        let approx = {
            let last = path.last().unwrap();
            Point(last.0, last.1)
        };

        find_best_corner_match(
            cross_correlation,
            &approx,
            self.search_region,
            self.distance_penalty,
        )
    }

    fn update_edge(&mut self, coord: Coord, corner: Point, confidence: f64) {
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
    type Output = Option<Point>;

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
    cross_correlation: &Image,
    approx: &Point,
    search_region: usize,
    distance_penalty: f64, // This parameter isn't used in the Python version
) -> Option<(Point, f64)> {
    // Check if image is empty
    if cross_correlation.is_empty() {
        return None;
    }

    let (height, width) = cross_correlation.dim();
    let x = approx.x();
    let y = approx.y();

    // Calculate crop boundaries
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
            cropped[i][j] = cross_correlation[[crop_y + i, crop_x + j]];
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
    let result_x = crop_x + best_x;
    let result_y = crop_y + best_y;

    let best_value_normalized = best_value / 255.0;

    Some((
        Point(result_x as i32, result_y as i32),
        best_value_normalized as f64,
    ))
}
