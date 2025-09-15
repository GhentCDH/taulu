use std::f64::consts::PI;
use std::ops::Add;

use numpy::PyReadonlyArray2;
use pathfinding::prelude::astar as astar_rust;
use pyo3::prelude::*;

#[derive(FromPyObject, PartialEq, PartialOrd, Eq, Hash, Clone)]
struct Point(i32, i32);

enum Direction {
    Right,
    RightStrict,
    Down,
    DownStrict,
    Left,
    LeftStrict,
    Up,
    UpStrict,
    Any,
    Straight,
    Diagonal,
}

impl Direction {
    fn offsets(&self) -> &[Point] {
        match self {
            Direction::Right => &[
                Point(1, -1),
                Point(1, 0),
                Point(1, 1),
                Point(0, -1),
                Point(0, 1),
            ],
            Direction::RightStrict => &[Point(1, -1), Point(1, 0), Point(1, 1)],
            Direction::Down => &[
                Point(-1, 1),
                Point(0, 1),
                Point(1, 1),
                Point(-1, 0),
                Point(1, 0),
            ],
            Direction::DownStrict => &[Point(-1, 1), Point(0, 1), Point(1, 1)],
            Direction::Left => &[
                Point(-1, -1),
                Point(-1, 0),
                Point(-1, 1),
                Point(0, -1),
                Point(0, 1),
            ],
            Direction::LeftStrict => &[Point(-1, -1), Point(-1, 0), Point(-1, 1)],
            Direction::Up => &[
                Point(-1, -1),
                Point(0, -1),
                Point(1, -1),
                Point(-1, 0),
                Point(1, 0),
            ],
            Direction::UpStrict => &[Point(-1, -1), Point(0, -1), Point(1, -1)],
            Direction::Any => &[
                Point(-1, -1),
                Point(0, -1),
                Point(1, -1),
                Point(-1, 0),
                Point(1, 0),
                Point(-1, 1),
                Point(0, 1),
                Point(1, 1),
            ],
            Direction::Straight => &[Point(0, -1), Point(-1, 0), Point(1, 0), Point(0, 1)],
            Direction::Diagonal => &[Point(-1, -1), Point(1, -1), Point(-1, 1), Point(1, 1)],
        }
    }
}

impl Point {
    fn distance(&self, other: &Point) -> u32 {
        ((self.0 - other.0).abs() + (self.1 - other.1).abs()) as u32
    }

    fn min_distance(&self, others: &[Point]) -> u32 {
        others.iter().map(|o| self.distance(o)).min().unwrap()
    }

    fn successors(
        &self,
        dir: &Direction,
        img: &PyReadonlyArray2<'_, u8>,
    ) -> Option<Vec<(Self, u32)>> {
        let &Self(x, y) = self;

        fn image_cost(img: &PyReadonlyArray2<'_, u8>, p: &Point) -> Option<u32> {
            Some(*img.get((p.1 as usize, p.0 as usize))? as u32 / 25)
        }

        fn step_cost(x: i32, y: i32, nx: i32, ny: i32, dir: &Direction) -> u32 {
            let dx = (x - nx).abs();
            let dy = (y - ny).abs();
            if (dx != 0 && dy != 0) || dir.perpendicular(dx, dy) {
                2
            } else {
                1
            }
        }

        dir.offsets()
            .iter()
            .map(|offset| {
                let n = self + offset;
                image_cost(img, &n).map(|icost| {
                    let cost = icost + 15 * step_cost(x, y, n.0, n.1, dir);
                    (n, cost)
                })
            })
            .collect()
    }

    fn at_goal(&self, goals: &[Point]) -> bool {
        goals.contains(self)
    }
}

impl<'a> Add<&'a Point> for &'_ Point {
    type Output = Point;

    fn add(self, rhs: &'a Point) -> Self::Output {
        Point(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl From<Point> for (i32, i32) {
    fn from(value: Point) -> Self {
        (value.0, value.1)
    }
}

impl Direction {
    fn perpendicular(&self, dx: i32, dy: i32) -> bool {
        match self {
            Direction::Right | Direction::RightStrict | Direction::Left | Direction::LeftStrict => {
                dx == 0 && dy != 0
            }
            Direction::Down | Direction::DownStrict | Direction::Up | Direction::UpStrict => {
                dy == 0 && dx != 0
            }
            Direction::Any | Direction::Diagonal | Direction::Straight => false,
        }
    }
}

impl TryFrom<&str> for Direction {
    type Error = PyErr;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "right" => Ok(Self::Right),
            "down" => Ok(Self::Down),
            "any" => Ok(Self::Any),
            "straight" => Ok(Self::Straight),
            "diagonal" => Ok(Self::Diagonal),
            "right_strict" => Ok(Self::RightStrict),
            "left_strict" => Ok(Self::LeftStrict),
            "left" => Ok(Self::Left),
            "up" => Ok(Self::Up),
            "up_strict" => Ok(Self::UpStrict),
            "down_strict" => Ok(Self::DownStrict),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Direction must be 'right', 'down', 'right_strict', 'left_strict', 'up', 'up_strict', 'diagonal', 'straight' or 'any'",
            )),
        }
    }
}

#[pyfunction]
fn astar(
    img: PyReadonlyArray2<'_, u8>, // NumPy 2D uint8 image
    start: Point,                  // start point
    goals: Vec<Point>,             // list of goal points
    direction: &str,               // "right" or "down"
) -> PyResult<Option<Vec<(i32, i32)>>> {
    let direction: Direction = direction.try_into()?;

    Ok(astar_rust(
        &start,
        |p| p.successors(&direction, &img).unwrap_or_default(),
        |p| p.min_distance(&goals),
        |p| p.at_goal(&goals),
    )
    .map(|r| r.0.into_iter().map(|p| p.into()).collect()))
}

/// Return the circular median of angles in radians.
#[pyfunction]
fn circular_median_angle(angles: Vec<f64>) -> PyResult<f64> {
    if angles.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot compute median of empty list",
        ));
    }

    // Helper function to calculate circular distance between two angles
    fn circular_distance(a: f64, b: f64) -> f64 {
        let diff = (a - b).abs() % (2.0 * PI);
        diff.min(2.0 * PI - diff)
    }

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
            x_relative.partial_cmp(&y_relative).unwrap()
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

    Ok(best_median.unwrap())
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
fn _core<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    m.add_function(wrap_pyfunction!(median_slope, m)?)?;
    Ok(())
}
