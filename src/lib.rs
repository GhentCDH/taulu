use numpy::PyReadonlyArray2;
use pathfinding::prelude::astar as astar_rust;
use pyo3::prelude::*;

#[derive(FromPyObject, PartialEq, PartialOrd, Eq, Hash, Clone)]
struct Point(i32, i32);

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

        match dir {
            Direction::Right => vec![
                Self(x + 1, y),
                Self(x + 1, y - 1),
                Self(x + 1, y + 1),
                Self(x, y - 1),
                Self(x, y + 1),
            ],
            Direction::Down => vec![
                Self(x - 1, y + 1),
                Self(x, y + 1),
                Self(x + 1, y + 1),
                Self(x + 1, y),
                Self(x - 1, y),
            ],
            Direction::Any => vec![
                Self(x + 1, y),
                Self(x + 1, y + 1),
                Self(x, y + 1),
                Self(x - 1, y + 1),
                Self(x - 1, y),
                Self(x - 1, y - 1),
                Self(x, y - 1),
                Self(x + 1, y - 1),
            ],
        }
        .into_iter()
        .map(|n| {
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

impl From<Point> for (i32, i32) {
    fn from(value: Point) -> Self {
        (value.0, value.1)
    }
}

enum Direction {
    Right,
    Down,
    Any,
}

impl Direction {
    fn perpendicular(&self, dx: i32, dy: i32) -> bool {
        match self {
            Direction::Right => dx == 0 && dy != 0,
            Direction::Down => dy == 0 && dx != 0,
            Direction::Any => false,
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
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Direction must be 'right', 'down' or 'any'",
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

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(astar, m)?)?;
    Ok(())
}
