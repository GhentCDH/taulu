use std::ops::{Add, Mul, Sub};

use pyo3::{FromPyObject, IntoPyObject};

use crate::{
    Direction, Image, Step,
    traits::{self, Xy as _},
};

/// x, y
#[derive(FromPyObject, IntoPyObject, PartialEq, PartialOrd, Eq, Hash, Clone, Copy, Debug)]
pub struct Point(pub i32, pub i32);

fn image_cost(img: &Image, p: Point) -> Option<u32> {
    Some(
        u32::from(
            img.get((usize::try_from(p.y()).ok()?, usize::try_from(p.x()).ok()?))
                .copied()?,
        ) / 25,
    )
}

fn step_cost(x: i32, y: i32, nx: i32, ny: i32, dir: &Direction) -> u32 {
    let dx = (x - nx).abs();
    let dy = (y - ny).abs();
    if (dx != 0 && dy != 0) || dir.perpendicular(dx, dy) {
        14
    } else {
        10
    }
}

impl Point {
    fn distance(self, other: Point) -> u32 {
        u32::try_from((self.0 - other.0).abs() + (self.1 - other.1).abs()).expect("conversion")
    }

    // Rect given in (x1, y1, x2, y2) format
    #[must_use]
    pub fn within(self, rect: (i32, i32, i32, i32)) -> bool {
        let (x1, y1, x2, y2) = rect;
        self.x() >= x1 && self.x() < x2 && self.y() >= y1 && self.y() < y2
    }

    #[must_use]
    pub fn min_distance(self, others: &[Point]) -> u32 {
        others
            .iter()
            .map(|o| self.distance(*o))
            .min()
            .expect("minimum distance")
    }

    #[must_use]
    pub fn successors(self, dir: &Direction, img: &Image) -> Option<Vec<(Self, u32)>> {
        let Self(x, y) = self;

        dir.offsets()
            .iter()
            .map(|offset| {
                let n = &self + offset;
                image_cost(img, n).map(|icost| {
                    let cost = icost + 4 * step_cost(x, y, n.x(), n.y(), dir);
                    (n, cost)
                })
            })
            .collect()
    }

    #[must_use]
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

impl Add<Point> for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl<'a> Sub<&'a Point> for &'_ Point {
    type Output = Point;

    fn sub(self, rhs: &'a Point) -> Self::Output {
        Point(self.0 - rhs.0, self.1 - rhs.1)
    }
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
impl Mul<f32> for &'_ Point {
    type Output = Point;

    fn mul(self, rhs: f32) -> Self::Output {
        Point(
            (self.0 as f32 * rhs).round() as i32,
            (self.1 as f32 * rhs).round() as i32,
        )
    }
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
impl Mul<f32> for Point {
    type Output = Point;

    fn mul(self, rhs: f32) -> Self::Output {
        Point(
            (self.0 as f32 * rhs).round() as i32,
            (self.1 as f32 * rhs).round() as i32,
        )
    }
}

impl From<Point> for (i32, i32) {
    fn from(value: Point) -> Self {
        (value.0, value.1)
    }
}

impl From<Step> for Point {
    fn from(value: Step) -> Self {
        match value {
            Step::Right => Point(1, 0),
            Step::Down => Point(0, 1),
            Step::Left => Point(-1, 0),
            Step::Up => Point(0, -1),
        }
    }
}

impl traits::Xy<i32> for Point {
    fn x(&self) -> i32 {
        self.0
    }
    fn y(&self) -> i32 {
        self.1
    }
}

impl From<crate::Coord> for Point {
    fn from(val: crate::Coord) -> Self {
        Point(
            i32::try_from(val.x()).expect("conversion"),
            i32::try_from(val.y()).expect("conversion"),
        )
    }
}
