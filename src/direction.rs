use crate::{Step, point::Point};

#[derive(Debug)]
pub enum Direction {
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
    #[must_use]
    pub fn offsets(&self) -> &[Point] {
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

    #[must_use]
    pub fn perpendicular(&self, dx: i32, dy: i32) -> bool {
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
    type Error = pyo3::PyErr;

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
            _ => Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Direction must be 'right', 'down', 'right_strict', 'left_strict', 'up', 'up_strict', 'diagonal', 'straight' or 'any'",
            )),
        }
    }
}

impl From<Step> for Direction {
    fn from(value: Step) -> Self {
        match value {
            Step::Right => Self::RightStrict,
            Step::Down => Self::DownStrict,
            Step::Left => Self::LeftStrict,
            Step::Up => Self::UpStrict,
        }
    }
}
