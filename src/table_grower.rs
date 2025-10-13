use std::collections::HashMap;
use std::convert::Into;
use std::ops::{Index, Neg};

use numpy::PyReadonlyArray2;
use pathfinding::prelude::astar;
use pyo3::PyResult;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::geom_util::{
    evaluate_polynomial, gaussian_1d, linear_polynomial_least_squares, normalize, region_aware_fit,
};
use crate::traits::Xy;
use crate::{Coord, Image, Point, Step};

#[cfg(feature = "debug-tools")]
const RERUN_EXPECT: &str = "Should be able to log values to rerun server";

/// A table grid builder that grows from a seed point by detecting corner intersections.
///
/// This class implements an iterative algorithm that:
/// 1. Starts from a known corner point (typically from header alignment)
/// 2. Searches for adjacent corners using template matching on cross-correlation
/// 3. Grows the grid both horizontally and vertically
/// 4. Extrapolates missing corners using polynomial regression when detection fails
///
/// The algorithm adapts its confidence threshold dynamically to handle varying
/// image quality across the document.
///
/// # Algorithm Details
///
/// The growing process uses:
/// - **Cross-correlation scores** to identify high-confidence corners
/// - **Weighted search regions** with Gaussian falloff for robustness
/// - **Polynomial regression** (linear or quadratic) for extrapolation
/// - **Region-aware fitting** that considers neighboring rows/columns for consistency
///
/// # Parameters
///
/// * `table_image` - Grayscale image of the table (used for pathfinding)
/// * `cross_correlation` - Preprocessed image with high values at corner intersections
/// * `column_widths` - Expected widths of each column from the header template
/// * `row_heights` - Expected heights of rows (extends last value if needed)
/// * `start_point` - Initial corner point (x, y) to begin growing from
/// * `search_region` - Size of the square region to search for next corner (pixels)
/// * `distance_penalty` - Weight factor [0, 1] penalizing corners far from expected position
/// * `look_distance` - Number of adjacent rows/columns to consider for extrapolation
/// * `grow_threshold` - Minimum confidence [0, 1] to accept a detected corner
/// * `min_row_count` - Minimum rows required before considering table complete
#[pyclass]
#[derive(Debug)]
pub struct TableGrower {
    /// The points in the grid, indexed by (row, col)
    pub corners: Vec<Vec<Option<Point>>>,
    /// The number of columns in the grid, being columns of the table + 1
    #[pyo3(get)]
    pub columns: usize,
    /// Edge of the table grid, where new points can be grown from
    pub edge: HashMap<Coord, (Point, f64)>,
    /// The size of the search region to use when finding the best corner match
    pub search_region: usize,
    /// The distance penalty to use when finding the best corner match
    pub distance_penalty: f64,
    pub column_widths: Vec<i32>,
    pub row_heights: Vec<i32>,
    pub look_distance: usize,
    pub grow_threshold: f64,
    pub min_row_count: usize,
    #[cfg(feature = "debug-tools")]
    rec: rerun::RecordingStream,
}

#[cfg(feature = "debug-tools")]
fn start_rerun() -> rerun::RecordingStream {
    rerun::RecordingStreamBuilder::new("taulu")
        .connect_grpc()
        .expect("rerun recorder should spawn")
}

#[pymethods]
impl TableGrower {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        table_image,
        cross_correlation,
        column_widths,
        row_heights,
        start_point,
        search_region,
        distance_penalty = 0.5,
        look_distance = 3,
        grow_threshold = 0.5,
        min_row_count = 5,
    ))]
    /// Notice that the `start_point` is given as (x, y), both being integers
    fn new(
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
        column_widths: Vec<i32>,
        row_heights: Vec<i32>,
        start_point: Point,
        search_region: usize,
        distance_penalty: f64,
        look_distance: usize,
        grow_threshold: f64,
        min_row_count: usize,
    ) -> Self {
        let corners = Vec::new();
        let mut table_grower = Self {
            edge: HashMap::new(),
            corners,
            columns: column_widths.len() + 1,
            column_widths,
            row_heights,
            search_region,
            distance_penalty,
            look_distance,
            grow_threshold,
            min_row_count,
            #[cfg(feature = "debug-tools")]
            rec: start_rerun(),
        };

        table_grower.add_corner(
            &table_image.as_array(),
            &cross_correlation.as_array(),
            start_point,
            Coord::new(0, 0),
        );

        table_grower
    }

    fn get_corner(&self, coord: Coord) -> Option<Point> {
        if coord.y() >= self.corners.len() || coord.x() >= self.corners[coord.y()].len() {
            return None;
        }

        self.corners[coord.y()][coord.x()]
    }

    fn all_rows_complete(&self) -> bool {
        self.corners
            .iter()
            .all(|row| row.len() == self.columns && row.iter().all(std::option::Option::is_some))
    }

    fn get_all_corners(&self) -> Vec<Vec<Option<Point>>> {
        self.corners.clone()
    }

    fn get_edge_points(&self) -> Vec<(Point, f64)> {
        self.edge.values().copied().collect()
    }

    /// Grow a grid of points starting from start and growing according to the given
    /// column widths and row heights. The `table_image` is used to guide the growth
    /// using the `cross_correlation` image to find the best positions for the grid points.
    fn grow_point(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
    ) -> Option<f64> {
        Some(
            self.grow_point_internal(
                &table_image.as_array(),
                &cross_correlation.as_array(),
                self.grow_threshold,
            )?
            .1,
        )
    }

    fn grow_points(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<()> {
        loop {
            if self
                .grow_point_internal(
                    &table_image.as_array(),
                    &cross_correlation.as_array(),
                    self.grow_threshold,
                )
                .is_none()
            {
                break Ok(());
            }
        }
    }

    /// Returns None when no corner was added
    pub fn extrapolate_one(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
    ) -> Option<Point> {
        let (selected_location, point) = self.extrapolate_one_internal()?;
        self.add_corner(
            &table_image.as_array(),
            &cross_correlation.as_array(),
            point,
            selected_location,
        );

        Some(point)
    }

    fn is_table_complete(&self) -> bool {
        self.all_rows_complete() && self.corners.len() >= self.min_row_count
    }

    fn set_threshold(&mut self, value: f64) {
        self.grow_threshold = value;
    }

    #[allow(clippy::too_many_lines)]
    fn grow_table(
        &mut self,
        table_image: PyReadonlyArray2<'_, u8>,
        cross_correlation: PyReadonlyArray2<'_, u8>,
        py: Python,
    ) -> PyResult<()> {
        #[cfg(feature = "debug-tools")]
        {
            self.rec
                .log(
                    "table_image",
                    &rerun::Image::from_color_model_and_tensor(
                        rerun::ColorModel::L,
                        table_image.as_array().to_owned(),
                    )
                    .expect("should be able to create rerun image"),
                )
                .expect(RERUN_EXPECT);
            self.rec
                .log(
                    "cross_correlation",
                    &rerun::Image::from_color_model_and_tensor(
                        rerun::ColorModel::L,
                        cross_correlation.as_array().to_owned(),
                    )
                    .expect("should be able to create rerun image"),
                )
                .expect(RERUN_EXPECT);
        }

        let mut threshold = self.grow_threshold;

        assert!(threshold <= 1.0, "threshold should be <= 1.0");
        assert!(threshold >= 0.0, "threshold should be >= 0.0");

        let original_threshold = threshold;
        let table = table_image.as_array();
        let cross = cross_correlation.as_array();

        // first grow all points with the initial threshold until
        // there are no good candidates left
        #[cfg(feature = "debug-tools")]
        while let Some((point, _)) = self.grow_point_internal(&table, &cross, threshold) {
            #[allow(clippy::cast_precision_loss)]
            self.rec
                .log(
                    format!("points/grown/{:04}", self.len()),
                    &rerun::Points2D::new([(point.x() as f32, point.y() as f32)])
                        .with_colors([rerun::Color::from_rgb(255, 0, 0)])
                        .with_radii([3.0]),
                )
                .expect(RERUN_EXPECT);

            py.check_signals()?;
        }

        #[cfg(not(feature = "debug-tools"))]
        while self
            .grow_point_internal(&table, &cross, threshold)
            .is_some()
        {
            py.check_signals()?;
        }

        let mut loops_without_change = 0;

        // if the table hasn't been completed this way, extrapolate corners
        while !self.is_table_complete() {
            loops_without_change += 1;

            if loops_without_change > 50 {
                break;
            }

            py.check_signals()?;

            if let Some((coord, point)) = self.extrapolate_one_internal() {
                self.add_corner(&table, &cross, point, coord);

                #[cfg(feature = "debug-tools")]
                #[allow(clippy::cast_precision_loss)]
                self.rec
                    .log(
                        format!("points/extrapolated/{:04}", self.len()),
                        &rerun::Points2D::new([(point.x() as f32, point.y() as f32)])
                            .with_radii([3.0])
                            .with_colors([rerun::Color::from_rgb(0, 0, 255)]),
                    )
                    .expect(RERUN_EXPECT);

                loops_without_change = 0;
                let mut grown = false;

                #[allow(unused_variables)]
                while let Some((p, _)) = self.grow_point_internal(&table, &cross, threshold) {
                    #[cfg(feature = "debug-tools")]
                    #[allow(clippy::cast_precision_loss)]
                    self.rec
                        .log(
                            format!("points/grown/{:04}", self.len()),
                            &rerun::Points2D::new([(p.x() as f32, p.y() as f32)])
                                .with_radii([3.0])
                                .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                        )
                        .expect(RERUN_EXPECT);

                    grown = true;
                    // increase the threshold
                    threshold = (0.1 + 0.9 * threshold).min(original_threshold);
                }

                if !grown {
                    threshold *= 0.9;
                }
            } else {
                // couldn't extrapolate a corner, grow a new corner with a lowered threshold
                threshold *= 0.9;

                #[allow(unused_variables)]
                if let Some((p, _)) = self.grow_point_internal(&table, &cross, threshold) {
                    #[cfg(feature = "debug-tools")]
                    #[allow(clippy::cast_precision_loss)]
                    self.rec
                        .log(
                            format!("points/grown/{:04}", self.len()),
                            &rerun::Points2D::new([(p.x() as f32, p.y() as f32)])
                                .with_radii([3.0])
                                .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                        )
                        .expect(RERUN_EXPECT);

                    loops_without_change = 0;
                }
            }
        }

        Ok(())
    }

    #[pyo3(signature= (degree = 1, amount = 1.0))]
    fn smooth_grid(&mut self, degree: usize, amount: f32) {
        let degree = degree.clamp(1, 2);
        let amount = amount.clamp(0.0, 1.0);

        let mut new_corners = Vec::with_capacity(self.corners.len());

        for (y, row) in self.corners.iter().enumerate() {
            let mut new_row = Vec::with_capacity(row.len());
            for (x, cell) in row.iter().enumerate() {
                if let Some(current) = cell
                    && let Some(extrapolated) = self.extrapolate_coord(Coord::new(x, y), degree)
                {
                    let extrapolated = current * (1.0 - amount) + extrapolated * amount;
                    new_row.push(Some(extrapolated));
                } else if cell.is_some() {
                    new_row.push(*cell);
                } else {
                    new_row.push(None);
                }
            }
            new_corners.push(new_row);
        }

        self.corners = new_corners;
    }
}

impl TableGrower {
    /// Grow a grid of points starting from start and growing according to the given
    /// column widths and row heights. The `table_image` is used to guide the growth
    /// using the `cross_correlation` image to find the best positions for the grid points.
    fn grow_point_internal(
        &mut self,
        table_image: &Image,
        cross_correlation: &Image,
        threshold: f64,
    ) -> Option<(Point, f64)> {
        // find the edge point with the highest confidence
        // without emptying the edge
        let (&coord, &(corner, confidence)) = self.edge.iter().max_by(|a, b| {
            a.1.1
                .partial_cmp(&b.1.1)
                .expect("should be able to compare f64s")
        })?;

        if confidence < threshold {
            return None;
        }

        let _ = self.add_corner(table_image, cross_correlation, corner, coord);

        Some((corner, confidence))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.corners
            .iter()
            .map(|row| row.iter().filter(|c| c.is_some()).count())
            .sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
            self.corners.push(vec![None; self.columns]);
        }
        let row = &mut self.corners[coord.y()];

        row[coord.x()] = Some(corner_point);

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
                        self.step_from_coord(table_image, cross_correlation, coord, *step)
                    {
                        #[allow(clippy::cast_possible_truncation)]
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
            self.update_edge(new_coord, corner, f64::from(confidence));
            edge_added = true;
        }

        edge_added
    }

    /// Find the step size to take as an estimation based on the column width and row height of the
    /// table header
    fn header_based_step_from_coord(&self, coord: Coord, step: Step) -> Option<Point> {
        match step {
            Step::Right => {
                if coord.x() + 1 >= self.columns {
                    return None;
                }
                Some(Point(self.column_widths[coord.x()], 0))
            }
            Step::Left => {
                if coord.x() == 0 {
                    return None;
                }
                Some(Point(-self.column_widths[coord.x() - 1], 0))
            }
            Step::Down => {
                // extend row heights with last value if necessary
                let h = if coord.y() >= self.row_heights.len() {
                    *self
                        .row_heights
                        .last()
                        .expect("There should be at least one row already at this point")
                } else {
                    self.row_heights[coord.y()]
                };
                Some(Point(0, h))
            }
            Step::Up => {
                if coord.y() == 0 {
                    return None;
                }

                // extend row heights with last value if necessary
                let h = if coord.y() > self.row_heights.len() {
                    *self
                        .row_heights
                        .last()
                        .expect("There should be at least one row already at this point")
                } else {
                    self.row_heights[coord.y() - 1]
                };

                Some(Point(0, -h))
            }
        }
    }

    /// Find the best corner match when taking a step in given direction from the given coord
    fn step_from_coord(
        &self,
        table_image: &Image,
        cross_correlation: &Image,
        coord: Coord,
        step: Step,
    ) -> Option<(Point, f64)> {
        // construct the goals based on the step direction,
        // known column widths and row heights, and existing points
        let current_point = TableGrower::get_corner(self, coord)?;

        let image_size = table_image.shape();
        let height = image_size[0];
        let width = image_size[1];

        // let estimated_new_point = self.header_based_step_from_coord(coord, step)?;
        let estimated_new_point = self.approximate_best_step(coord, step)? + current_point;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let goals = match step {
            Step::Right => (0..(self.search_region * 2) as i32)
                .map(|i| estimated_new_point + Point(0, i - self.search_region as i32))
                .filter(|p| p.within((0, 0, width as i32, height as i32)))
                .collect(),
            Step::Down => (0..(self.search_region * 2) as i32)
                .map(|i| estimated_new_point + Point(i - self.search_region as i32, 0))
                .filter(|p| p.within((0, 0, width as i32, height as i32)))
                .collect::<Vec<_>>(),
            Step::Left => (0..(self.search_region * 2) as i32)
                .map(|i| estimated_new_point + Point(0, i - self.search_region as i32))
                .filter(|p| p.within((0, 0, width as i32, height as i32)))
                .collect::<Vec<_>>(),
            Step::Up => (0..(self.search_region * 2) as i32)
                .map(|i| estimated_new_point + Point(i - self.search_region as i32, 0))
                .filter(|p| p.within((0, 0, width as i32, height as i32)))
                .collect::<Vec<_>>(),
        };

        let direction = step.into();

        if goals.is_empty() {
            return None;
        }

        #[cfg(feature = "debug-tools")]
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_precision_loss
        )]
        self.rec
            .log(
                format!(
                    "astar/goals/{1}_{0}",
                    estimated_new_point.x()
                        + estimated_new_point.y()
                        + coord.x() as i32
                        + coord.y() as i32,
                    self.len()
                ),
                &rerun::LineStrips2D::new([goals.iter().map(|p| (p.x() as f32, p.y() as f32))])
                    .with_colors([rerun::Color::from_rgb(255, 255, 255)])
                    .with_radii([3.0]),
            )
            .expect("Should be able to send to rerun");

        let path: Vec<(i32, i32)> = astar::<crate::Point, u32, _, _, _, _>(
            &current_point,
            |p| p.successors(&direction, table_image).unwrap_or_default(),
            |p| p.min_distance(&goals),
            |p| p.at_goal(&goals),
        )
        .map(|r| r.0.into_iter().map(Into::into).collect())?;

        #[cfg(feature = "debug-tools")]
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            clippy::cast_precision_loss
        )]
        self.rec
            .log(
                format!("astar/paths/{1}_{0}", path.len(), self.len()),
                &rerun::LineStrips2D::new([path.iter().map(|(x, y)| (*x as f32, *y as f32))])
                    .with_colors([rerun::Color::from_rgb(0, 255, 0)])
                    .with_radii([3.0]),
            )
            .expect("Should be able to send to rerun");

        let approx = {
            let last = path.last().expect("path should have at least one entry");
            Point(last.0, last.1)
        };

        find_best_corner_match(
            cross_correlation,
            approx,
            self.search_region,
            self.distance_penalty,
        )
    }

    /// Update the edge with a new corner point and its confidence for the given coord
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

    #[must_use]
    pub fn width(&self) -> usize {
        self.columns
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.corners.len()
    }

    /// Returns the next-best empty corner to be used for extrapolation
    #[must_use]
    pub fn extendable_corner_and_neighbours(&self) -> Option<(Coord, Vec<Point>, Vec<Point>)> {
        self.corners
            .par_iter()
            .enumerate()
            .map(|(y, row)| {
                // TODO: try optimizing by only outputting number of neighbours here and
                // recalculate later
                (
                    y,
                    row.iter()
                        .enumerate()
                        .filter(|(_x, cell)| cell.is_none())
                        .map(|(x, _cell)| {
                            (
                                x,
                                self.neighbour_points_x(Coord::new(x, y)),
                                self.neighbour_points_y(Coord::new(x, y)),
                            )
                        })
                        .max_by_key(|(_x, x_nbs, y_nbs)| x_nbs.len().min(y_nbs.len())),
                )
            })
            .filter_map(|(y, opt_max)| opt_max.map(|(x, x_nbs, y_nbs)| (x, y, x_nbs, y_nbs)))
            .max_by_key(|(_y, _x, x_nbs, y_nbs)| x_nbs.len().min(y_nbs.len()))
            .and_then(|(x, y, x_nbs, y_nbs)| {
                if x_nbs.len().min(y_nbs.len()) > 0 {
                    Some((Coord::new(x, y), x_nbs, y_nbs))
                } else {
                    None
                }
            })
    }

    #[inline]
    #[must_use]
    pub fn in_bounds(&self, loc: Coord) -> bool {
        loc.x() < self.width() && loc.y() < self.height()
    }

    #[must_use]
    pub fn neighbour_points_x(&self, loc: Coord) -> Vec<Point> {
        let mut points: Vec<Point> = Vec::new();
        for dx in 1..=self.look_distance {
            let right = Coord::new(loc.x() + dx, loc.y());
            if self.in_bounds(right)
                && self.corners[right.y()][right.x()].is_some()
                && let Some(point) = self[right]
            {
                points.push(point);
            }

            let left = Coord::new(loc.x().saturating_sub(dx), loc.y());
            if self.in_bounds(left) && self.corners[left.y()][left.x()].is_some() {
                if let Some(last) = points.last()
                    && let Some(left) = self[left]
                    && *last == left
                {
                    continue;
                }
                if let Some(point) = self[left] {
                    points.push(point);
                }
            }
        }
        points
    }

    #[must_use]
    pub fn neighbour_points_y(&self, loc: Coord) -> Vec<Point> {
        let mut points = Vec::new();
        for dy in 1..=self.look_distance {
            let down = Coord::new(loc.x(), loc.y() + dy);
            if self.in_bounds(down)
                && self.corners[down.y()][down.x()].is_some()
                && let Some(point) = self[down]
            {
                points.push(point);
            }

            let up = Coord::new(loc.x(), loc.y().saturating_sub(dy));
            if self.in_bounds(up) && self.corners[up.y()][up.x()].is_some() {
                if let Some(last) = points.last()
                    && let Some(up) = self[up]
                    && *last == up
                {
                    continue;
                }
                if let Some(point) = self[up] {
                    points.push(point);
                }
            }
        }
        points
    }

    fn extrapolate_coord(&self, coord: Coord, degree: usize) -> Option<Point> {
        let neighbours_x = self.neighbour_points_x(coord);
        let neighbours_y = self.neighbour_points_y(coord);

        if neighbours_y.len() < degree + 1 || neighbours_x.len() < degree + 1 {
            return None;
        }

        let region = self.get_region(coord);
        self.intersect_regressions_region_aware(&neighbours_x, &neighbours_y, degree, &region)
    }

    fn extrapolate_one_internal(&self) -> Option<(Coord, Point)> {
        // Get the missing corner with the maxmin neighbours in each direction
        let (selected_location, neighbours_x, neighbours_y) =
            self.extendable_corner_and_neighbours()?;

        let degree = 1;

        let region = self.get_region(selected_location);

        let intersection =
            self.intersect_regressions_region_aware(&neighbours_x, &neighbours_y, degree, &region)?;

        Some((selected_location, intersection))
    }

    /// Get a square region of size `look_distance` * 2 around the given coordinate
    fn get_region(&self, coord: Coord) -> Vec<Vec<Option<Point>>> {
        let mut region = vec![vec![None; self.look_distance * 2 + 1]; self.look_distance * 2 + 1];

        #[allow(clippy::cast_possible_wrap)]
        for (dy, row) in region.iter_mut().enumerate() {
            for (dx, cell) in row.iter_mut().enumerate() {
                let Ok(x) =
                    usize::try_from(coord.x() as isize + dx as isize - self.look_distance as isize)
                else {
                    continue;
                };
                let Ok(y) =
                    usize::try_from(coord.y() as isize + dy as isize - self.look_distance as isize)
                else {
                    continue;
                };

                let nb_coord = Coord::new(x, y);
                if self.in_bounds(nb_coord) {
                    *cell = self[nb_coord];
                }
            }
        }

        region
    }

    /// Approximates the best step size to take from `current` in `step` direction
    /// this approximation is based on the steps that have been taken before, by neighbouring
    /// points
    fn approximate_best_step(&self, current: Coord, step: Step) -> Option<Point> {
        let mut gaussian_weights = gaussian_1d(self.look_distance * 2, None);
        let similar_steps = self.similar_neighbouring_steps(current, step);

        let mut count = 0;
        for (i, step) in similar_steps.iter().enumerate() {
            if step.is_none() {
                gaussian_weights[i] = 0.0;
            } else {
                count += 1;
            }
        }

        normalize(&mut gaussian_weights);

        let result = similar_steps
            .iter()
            .zip(gaussian_weights.iter())
            .filter(|(step, _weight)| step.is_some())
            .map(|(step, weight)| step.expect("filter") * *weight)
            .fold(Point(0, 0), |acc, val| acc + val);

        #[allow(clippy::cast_precision_loss)]
        if count <= 3 {
            let header_step = self.header_based_step_from_coord(current, step)?;
            Some(
                header_step * (1.0 / (1.0 + count as f32))
                    + result * (count as f32 / (1.0 + count as f32)),
            )
        } else {
            Some(result)
        }
    }

    /// Finds the step sizes that relevant neighbours took in the same direction
    /// to reach another corner point.
    ///
    /// The result is a vector of length `look_distance * 2`.
    /// For neighbours which don't exist or haven't taken the relevant step yet,
    /// use the default guestimated value
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    fn similar_neighbouring_steps(&self, current: Coord, step: Step) -> Vec<Option<Point>> {
        let mut steps = vec![None; self.look_distance * 2];

        let neighbours_one = (1..=self.look_distance).map(|offset| {
            current
                .take_amount_of_steps(offset, step.rotate_ninety())
                .filter(|&c| self.in_bounds(c))
        });
        let neighbours_two = (1..=self.look_distance).map(|offset| {
            current
                .take_amount_of_steps(offset, step.rotate_ninety().neg())
                .filter(|&c| self.in_bounds(c))
        });

        // get the step size they took in the step direction
        neighbours_one
            .chain(neighbours_two)
            .enumerate()
            .filter(|(_i, nb)| nb.is_some())
            .for_each(|(i, nb)| {
                let nb = nb.expect("filter");
                if let Some(current_point) = self[nb]
                    && let Some(step_coord) = nb.take_amount_of_steps(1, step)
                    && let Some(next_point) = self[step_coord]
                {
                    steps[i] = Some(&next_point - &current_point);
                }
            });

        steps
    }

    /// Given a set of horizontal and vertical points, fit polynomial regressions
    /// of the given degree to each set, and find their intersection point.
    #[allow(clippy::pedantic)]
    fn intersect_regressions_region_aware(
        &self,
        horizontal_points: &[Point],
        vertical_points: &[Point],
        degree: usize,
        region: &[Vec<Option<Point>>],
    ) -> Option<Point> {
        if horizontal_points.len() < degree + 1 && vertical_points.len() < degree + 1 {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let horizontal_xs = horizontal_points
            .iter()
            .map(|p| p.x() as f32)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
        let horizontal_ys = horizontal_points
            .iter()
            .map(|p| p.y() as f32)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
        let vertical_xs = vertical_points
            .iter()
            .map(|p| p.x() as f32)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
        let vertical_ys = vertical_points
            .iter()
            .map(|p| p.y() as f32)
            .collect::<Vec<_>>();

        let mut region_horizontal_xs = Vec::new();
        let mut region_horizontal_ys = Vec::new();

        for (y, row) in region.iter().enumerate() {
            let mut row_xs = Vec::new();
            let mut row_ys = Vec::new();
            for cell in row.iter() {
                // don't do the current row
                if y == region.len() / 2 {
                    continue;
                }
                if let Some(p) = cell {
                    row_xs.push(p.x() as f32);
                    row_ys.push(p.y() as f32);
                }
            }
            if row_xs.len() >= 2 && row_ys.len() >= 2 {
                #[cfg(feature = "debug-tools")]
                {
                    self.rec
                        .log(
                            format!("regression/region_horizontal/{}", y),
                            &rerun::Points2D::new(
                                row_xs.iter().zip(row_ys.iter()).map(|(x, y)| (*x, *y)),
                            )
                            .with_colors([rerun::Color::from_rgb(255, 255, 0)])
                            .with_radii([2.0]),
                        )
                        .expect(RERUN_EXPECT);
                }

                region_horizontal_xs.push(row_xs);
                region_horizontal_ys.push(row_ys);
            }
        }

        let mut region_vertical_xs = Vec::new();
        let mut region_vertical_ys = Vec::new();

        for x in 0..region[0].len() {
            let mut column_xs = Vec::new();
            let mut column_ys = Vec::new();

            for y in 0..region.len() {
                // don't do the current column
                if x == region.len() / 2 {
                    continue;
                }

                if let Some(p) = region[y][x] {
                    column_xs.push(p.x() as f32);
                    column_ys.push(p.y() as f32);
                }
            }

            if column_xs.len() >= 2 && column_ys.len() >= 2 {
                #[cfg(feature = "debug-tools")]
                {
                    self.rec
                        .log(
                            format!("regression/region_vertical/{}", x),
                            &rerun::Points2D::new(
                                column_xs
                                    .iter()
                                    .zip(column_ys.iter())
                                    .map(|(x, y)| (*x, *y)),
                            )
                            .with_colors([rerun::Color::from_rgb(0, 255, 255)])
                            .with_radii([2.0]),
                        )
                        .expect(RERUN_EXPECT);
                }

                region_vertical_xs.push(column_xs);
                region_vertical_ys.push(column_ys);
            }
        }

        let lambda = 0.2;
        let horizontal_coeffs = region_aware_fit(
            &horizontal_xs,
            &horizontal_ys,
            &region_horizontal_xs,
            &region_horizontal_ys,
            lambda,
        )
        .ok()?;

        let vertical_coeffs = region_aware_fit(
            &vertical_ys,
            &vertical_xs,
            &region_vertical_ys,
            &region_vertical_xs,
            lambda,
        )
        .ok()?;

        // iteratively solve for the intersection of both
        let (mut x, mut y) = {
            let point = horizontal_points.first()?;
            #[allow(clippy::cast_precision_loss)]
            (point.x() as f32, point.y() as f32)
        };

        let ho = &horizontal_coeffs;
        let ve = &vertical_coeffs;

        // Newton's method
        let (h, h_derivative) = match degree {
            1 => {
                let h = vec![ho[0] + ho[1] * ve[0], ho[1] * ve[1] - 1.0];
                let h_derivative = vec![ho[1] * ve[1] - 1.0];
                (h, h_derivative)
            }
            2 => {
                let h = vec![
                    ho[0] + ho[1] * ve[0] + ho[2] * ve[0] * ve[0],
                    ho[1] * ve[1] + 2.0 * ho[2] * ve[0] * ve[1] - 1.0,
                    ho[1] * ve[2] + 2.0 * ho[2] * ve[0] * ve[2] + ho[2] * ve[1] * ve[1],
                    2.0 * ho[2] * ve[1] * ve[2],
                    ho[2] * ve[2] * ve[2],
                ];
                let h_derivative = vec![
                    ho[1] * ve[1] + 2.0 * ho[2] * ve[0] * ve[1] - 1.0,
                    2.0 * ho[1] * ve[2] + 4.0 * ho[2] * ve[0] * ve[2] + 2.0 * ho[2] * ve[1] * ve[1],
                    6.0 * ho[2] * ve[1] * ve[2],
                    4.0 * ho[2] * ve[2] * ve[2],
                ];
                (h, h_derivative)
            }
            _ => {
                return None;
            }
        };

        let mut done = false;
        for _ in 0..20 {
            let h_y = evaluate_polynomial(&h, y);
            let h_der_y = evaluate_polynomial(&h_derivative, y);
            let new_y = y - h_y / h_der_y;

            let dist = (y - new_y).abs();

            if dist < 1e-6 {
                x = evaluate_polynomial(&vertical_coeffs, new_y);
                y = new_y;
                done = true;
                break;
            }

            if dist.is_nan() {
                return None;
            }

            y = new_y;
        }

        if !done {
            return None;
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let result = Point(x.round() as i32, y.round() as i32);

        #[cfg(feature = "debug-tools")]
        {
            // get two points on the horizontal regression line
            let h_start = Point(
                horizontal_points.first()?.x(),
                evaluate_polynomial(&horizontal_coeffs, horizontal_points.first()?.x() as f32)
                    .round() as i32,
            );
            let h_end = Point(
                horizontal_points.last()?.x(),
                evaluate_polynomial(&horizontal_coeffs, horizontal_points.last()?.x() as f32)
                    .round() as i32,
            );

            self.rec
                .log(
                    "regression/horizontal",
                    &rerun::LineStrips2D::new([[
                        (h_start.x() as f32, h_start.y() as f32),
                        (h_end.x() as f32, h_end.y() as f32),
                    ]])
                    .with_colors([rerun::Color::from_rgb(255, 0, 255)])
                    .with_radii([2.0]),
                )
                .expect(RERUN_EXPECT);

            // get two points on the vertical regression line
            let v_start = Point(
                evaluate_polynomial(&vertical_coeffs, vertical_points.first()?.y() as f32).round()
                    as i32,
                vertical_points.first()?.y(),
            );
            let v_end = Point(
                evaluate_polynomial(&vertical_coeffs, vertical_points.last()?.y() as f32).round()
                    as i32,
                vertical_points.last()?.y(),
            );

            self.rec
                .log(
                    "regression/vertical",
                    &rerun::LineStrips2D::new([[
                        (v_start.x() as f32, v_start.y() as f32),
                        (v_end.x() as f32, v_end.y() as f32),
                    ]])
                    .with_colors([rerun::Color::from_rgb(255, 0, 255)])
                    .with_radii([2.0]),
                )
                .expect(RERUN_EXPECT);
        }

        Some(result)
    }
}

impl Index<Coord> for TableGrower {
    type Output = Option<Point>;

    fn index(&self, index: Coord) -> &Self::Output {
        if index.y() >= self.corners.len() || index.x() >= self.corners[index.y()].len() {
            return &None;
        }

        &self.corners[index.y()][index.x()]
    }
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
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

/// Given a set of horizontal and vertical points, fit polynomial regressions
/// of the given degree to each set, and find their intersection point.
#[allow(clippy::similar_names, dead_code)]
fn intersect_regressions(
    horizontal_points: &[Point],
    vertical_points: &[Point],
    degree: usize,
) -> Option<Point> {
    if horizontal_points.len() < degree + 1 && vertical_points.len() < degree + 1 {
        return None;
    }

    #[allow(clippy::cast_precision_loss)]
    let horizontal_xs = horizontal_points
        .iter()
        .map(|p| p.x() as f32)
        .collect::<Vec<_>>();
    #[allow(clippy::cast_precision_loss)]
    let horizontal_ys = horizontal_points
        .iter()
        .map(|p| p.y() as f32)
        .collect::<Vec<_>>();
    #[allow(clippy::cast_precision_loss)]
    let vertical_xs = vertical_points
        .iter()
        .map(|p| p.x() as f32)
        .collect::<Vec<_>>();
    #[allow(clippy::cast_precision_loss)]
    let vertical_ys = vertical_points
        .iter()
        .map(|p| p.y() as f32)
        .collect::<Vec<_>>();

    let horizontal_coeffs =
        linear_polynomial_least_squares(degree, &horizontal_xs, &horizontal_ys).ok()?;
    let vertical_coeffs =
        linear_polynomial_least_squares(degree, &vertical_ys, &vertical_xs).ok()?;

    // iteratively solve for the intersection of both
    let (mut x, mut y) = {
        let point = horizontal_points.first()?;
        #[allow(clippy::cast_precision_loss)]
        (point.x() as f32, point.y() as f32)
    };

    let ho = &horizontal_coeffs;
    let ve = &vertical_coeffs;

    // Newton's method
    let (h, h_derivative) = match degree {
        1 => {
            let h = vec![ho[0] + ho[1] * ve[0], ho[1] * ve[1] - 1.0];
            let h_derivative = vec![ho[1] * ve[1] - 1.0];
            (h, h_derivative)
        }
        2 => {
            let h = vec![
                ho[0] + ho[1] * ve[0] + ho[2] * ve[0] * ve[0],
                ho[1] * ve[1] + 2.0 * ho[2] * ve[0] * ve[1] - 1.0,
                ho[1] * ve[2] + 2.0 * ho[2] * ve[0] * ve[2] + ho[2] * ve[1] * ve[1],
                2.0 * ho[2] * ve[1] * ve[2],
                ho[2] * ve[2] * ve[2],
            ];
            let h_derivative = vec![
                ho[1] * ve[1] + 2.0 * ho[2] * ve[0] * ve[1] - 1.0,
                2.0 * ho[1] * ve[2] + 4.0 * ho[2] * ve[0] * ve[2] + 2.0 * ho[2] * ve[1] * ve[1],
                6.0 * ho[2] * ve[1] * ve[2],
                4.0 * ho[2] * ve[2] * ve[2],
            ];
            (h, h_derivative)
        }
        _ => {
            return None;
        }
    };

    let mut done = false;
    for _ in 0..20 {
        let h_y = evaluate_polynomial(&h, y);
        let h_der_y = evaluate_polynomial(&h_derivative, y);
        let new_y = y - h_y / h_der_y;

        let dist = (y - new_y).abs();

        if dist < 1e-6 {
            x = evaluate_polynomial(&vertical_coeffs, new_y);
            y = new_y;
            done = true;
            break;
        }

        if dist.is_nan() {
            return None;
        }

        y = new_y;
    }

    if !done {
        return None;
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(Point(x.round() as i32, y.round() as i32))
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn find_best_corner_match(
    cross_correlation: &Image,
    approx: Point,
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
        return Some((approx, 0.0)); // Return original point with 0 confidence
    }

    // Extract cropped region
    let mut cropped = vec![vec![0u8; crop_width]; crop_height];
    for i in 0..crop_height {
        for j in 0..crop_width {
            cropped[i][j] = cross_correlation[[crop_y + i, crop_x + j]];
        }
    }

    if cropped.is_empty() || cropped[0].is_empty() {
        return Some((approx, 0.0));
    }

    // Apply Gaussian weighting
    let weighted = if crop_height == search_region && crop_width == search_region {
        // Perfect size - apply weights directly
        let weights = create_gaussian_weights(search_region, distance_penalty);
        let mut result = vec![vec![0.0f32; crop_width]; crop_height];
        for i in 0..crop_height {
            for j in 0..crop_width {
                result[i][j] = f32::from(cropped[i][j]) * weights[i][j];
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
                weighted_extended[i][j] = f32::from(extended[i][j]) * weights[i][j];
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
        f64::from(best_value_normalized),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn create_test_table_grower() -> TableGrower {
        let mut corners = vec![vec![None; 4]; 3];

        // x . x .
        // x . . .
        // x . . .

        // Add some points to create a pattern
        corners[0][0] = Some(Point(10, 10));
        corners[0][2] = Some(Point(30, 12));
        corners[1][0] = Some(Point(8, 25));
        corners[2][0] = Some(Point(6, 40));

        TableGrower {
            corners,
            columns: 4,
            edge: HashMap::new(),
            search_region: 10,
            distance_penalty: 0.5,
            column_widths: vec![4; 3],
            row_heights: vec![4; 2],
            look_distance: 3,
            grow_threshold: 1.0,
            min_row_count: 2,
            #[cfg(feature = "debug-tools")]
            rec: start_rerun(),
        }
    }

    fn create_empty_table_grower() -> TableGrower {
        let corners = vec![vec![None; 4]; 3];

        TableGrower {
            corners,
            columns: 4,
            edge: HashMap::new(),
            search_region: 10,
            distance_penalty: 0.5,
            column_widths: vec![4; 3],
            row_heights: vec![4; 2],
            look_distance: 3,
            grow_threshold: 1.0,
            min_row_count: 2,
            #[cfg(feature = "debug-tools")]
            rec: start_rerun(),
        }
    }

    #[test]
    fn test_table_completer_creation() {
        let grower = create_test_table_grower();

        assert_eq!(grower.width(), 4);
        assert_eq!(grower.height(), 3);
        assert_eq!(grower.corners.len(), 3);
        assert_eq!(grower.corners[0].len(), 4);
    }

    #[test]
    fn test_in_bounds() {
        let grower = create_test_table_grower();

        assert!(grower.in_bounds(Coord::new(0, 0)));
        assert!(grower.in_bounds(Coord::new(3, 2)));
        assert!(!grower.in_bounds(Coord::new(4, 0)));
        assert!(!grower.in_bounds(Coord::new(0, 3)));
        assert!(!grower.in_bounds(Coord::new(4, 3)));
    }

    #[test]
    fn test_neighbour_points_x() {
        let grower = create_test_table_grower();

        let neighbors = grower.neighbour_points_x(Coord::new(1, 0));

        // Should find the point at (0,0) and (2,0)
        assert!(!neighbors.is_empty());
        assert!(neighbors.contains(&Point(10, 10)));
        assert!(neighbors.contains(&Point(30, 12)));
    }

    #[test]
    fn test_neighbour_points_y() {
        let grower = create_test_table_grower();

        let neighbors = grower.neighbour_points_y(Coord::new(0, 1));

        // Should find points at (0,0) and (0,2)
        assert!(!neighbors.is_empty());
    }

    #[test]
    fn test_add_missing_corners_integration() {
        let grower = create_test_table_grower();
        let result = grower.extrapolate_one_internal();

        assert!(result.is_none());
    }

    #[test]
    fn test_table_completer_with_no_existing_corners() {
        let grower = create_empty_table_grower();
        let result = grower.extrapolate_one_internal();
        // Should return None since there are no existing corners to extrapolate from
        assert!(result.is_none());
    }

    #[test]
    fn test_table_completer_bounds_checking() {
        let grower = create_test_table_grower();

        // Test various boundary conditions
        assert!(grower.in_bounds(Coord::new(0, 0)));
        assert!(grower.in_bounds(Coord::new(3, 2))); // width-1, height-1
        assert!(!grower.in_bounds(Coord::new(4, 0))); // width
        assert!(!grower.in_bounds(Coord::new(0, 3))); // height
    }
}
