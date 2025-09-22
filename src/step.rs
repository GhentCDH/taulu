use std::ops::Neg;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    Right,
    Down,
    Left,
    Up,
}

impl Neg for Step {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Step::Right => Self::Left,
            Step::Down => Self::Up,
            Step::Left => Self::Right,
            Step::Up => Self::Down,
        }
    }
}

impl Step {
    #[must_use]
    pub fn rotate_ninety(self) -> Self {
        match self {
            Step::Right => Step::Up,
            Step::Up => Step::Left,
            Step::Left => Step::Down,
            Step::Down => Step::Right,
        }
    }
}
