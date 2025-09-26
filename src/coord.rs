use std::ops::Add;

use pyo3::FromPyObject;

use crate::traits::Xy;
use crate::{Point, Step};

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
    #[must_use]
    pub fn new(x: usize, y: usize) -> Self {
        Self(y, x)
    }

    #[must_use]
    pub fn take_amount_of_steps(&self, amount: usize, step: Step) -> Option<Coord> {
        if match step {
            Step::Right | Step::Down => false,
            Step::Left => self.1 < amount,
            Step::Up => self.0 < amount,
        } {
            return None;
        }

        Some(match step {
            Step::Right => Coord(self.0, self.1 + amount),
            Step::Down => Coord(self.0 + amount, self.1),
            Step::Left => Coord(self.0, self.1 - amount),
            Step::Up => Coord(self.0 - amount, self.1),
        })
    }
}

impl Add<Step> for Coord {
    type Output = Coord;

    fn add(self, rhs: Step) -> Self::Output {
        match rhs {
            Step::Right => Coord(self.0, self.1 + 1),
            Step::Down => Coord(self.0 + 1, self.1),
            Step::Left => Coord(self.0, self.1.saturating_sub(1)),
            Step::Up => Coord(self.0.saturating_sub(1), self.1),
        }
    }
}

impl From<Point> for Coord {
    fn from(val: Point) -> Self {
        Coord(
            usize::try_from(val.1).expect("conversion"),
            usize::try_from(val.0).expect("convertion"),
        )
    }
}
