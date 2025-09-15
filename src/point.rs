use std::ops::Add;

use numpy::PyReadonlyArray2;
use pyo3::FromPyObject;

use crate::Direction;

#[derive(FromPyObject, PartialEq, PartialOrd, Eq, Hash, Clone)]
pub struct Point(pub i32, pub i32);

impl Point {
    fn distance(&self, other: &Point) -> u32 {
        ((self.0 - other.0).abs() + (self.1 - other.1).abs()) as u32
    }

    pub fn min_distance(&self, others: &[Point]) -> u32 {
        others.iter().map(|o| self.distance(o)).min().unwrap()
    }

    pub fn successors(
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

    pub fn at_goal(&self, goals: &[Point]) -> bool {
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
