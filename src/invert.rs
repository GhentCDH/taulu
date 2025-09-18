use ndarray::{Array2, arr2};

#[derive(Debug)]
pub enum MatrixError {
    NotSquare,
    UnsupportedSize,
    Singular, // determinant is zero
}

impl std::error::Error for MatrixError {}
impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::NotSquare => write!(f, "Matrix is not square"),
            MatrixError::UnsupportedSize => write!(f, "Unsupported matrix size"),
            MatrixError::Singular => write!(f, "Matrix is singular (determinant is zero)"),
        }
    }
}

pub fn invert_matrix(matrix: &Array2<f32>) -> Result<Array2<f32>, MatrixError> {
    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(MatrixError::NotSquare);
    }

    let n = shape[0];
    match n {
        2 => invert_2x2(matrix),
        3 => invert_3x3(matrix),
        4 => invert_4x4(matrix),
        _ => Err(MatrixError::UnsupportedSize),
    }
}

fn invert_2x2(matrix: &Array2<f32>) -> Result<Array2<f32>, MatrixError> {
    let a = matrix[[0, 0]];
    let b = matrix[[0, 1]];
    let c = matrix[[1, 0]];
    let d = matrix[[1, 1]];

    let det = a * d - b * c;
    if det.abs() < 1e-12 {
        return Err(MatrixError::Singular);
    }

    let inv_det = 1.0 / det;
    Ok(arr2(&[
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det],
    ]))
}

fn invert_3x3(matrix: &Array2<f32>) -> Result<Array2<f32>, MatrixError> {
    let m = matrix;

    // Calculate determinant
    let det = m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]);

    if det.abs() < 1e-12 {
        return Err(MatrixError::Singular);
    }

    let inv_det = 1.0 / det;

    // Calculate adjugate matrix (cofactor matrix transposed)
    let adj = arr2(&[
        [
            (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]]) * inv_det,
            (m[[0, 2]] * m[[2, 1]] - m[[0, 1]] * m[[2, 2]]) * inv_det,
            (m[[0, 1]] * m[[1, 2]] - m[[0, 2]] * m[[1, 1]]) * inv_det,
        ],
        [
            (m[[1, 2]] * m[[2, 0]] - m[[1, 0]] * m[[2, 2]]) * inv_det,
            (m[[0, 0]] * m[[2, 2]] - m[[0, 2]] * m[[2, 0]]) * inv_det,
            (m[[0, 2]] * m[[1, 0]] - m[[0, 0]] * m[[1, 2]]) * inv_det,
        ],
        [
            (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]) * inv_det,
            (m[[0, 1]] * m[[2, 0]] - m[[0, 0]] * m[[2, 1]]) * inv_det,
            (m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]) * inv_det,
        ],
    ]);

    Ok(adj)
}

fn invert_4x4(matrix: &Array2<f32>) -> Result<Array2<f32>, MatrixError> {
    let m = matrix;

    // Calculate 2x2 determinants for cofactors
    let a2323 = m[[2, 2]] * m[[3, 3]] - m[[2, 3]] * m[[3, 2]];
    let a1323 = m[[2, 1]] * m[[3, 3]] - m[[2, 3]] * m[[3, 1]];
    let a1223 = m[[2, 1]] * m[[3, 2]] - m[[2, 2]] * m[[3, 1]];
    let a0323 = m[[2, 0]] * m[[3, 3]] - m[[2, 3]] * m[[3, 0]];
    let a0223 = m[[2, 0]] * m[[3, 2]] - m[[2, 2]] * m[[3, 0]];
    let a0123 = m[[2, 0]] * m[[3, 1]] - m[[2, 1]] * m[[3, 0]];
    let a2313 = m[[1, 2]] * m[[3, 3]] - m[[1, 3]] * m[[3, 2]];
    let a1313 = m[[1, 1]] * m[[3, 3]] - m[[1, 3]] * m[[3, 1]];
    let a1213 = m[[1, 1]] * m[[3, 2]] - m[[1, 2]] * m[[3, 1]];
    let a2312 = m[[1, 2]] * m[[2, 3]] - m[[1, 3]] * m[[2, 2]];
    let a1312 = m[[1, 1]] * m[[2, 3]] - m[[1, 3]] * m[[2, 1]];
    let a1212 = m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]];
    let a0313 = m[[1, 0]] * m[[3, 3]] - m[[1, 3]] * m[[3, 0]];
    let a0213 = m[[1, 0]] * m[[3, 2]] - m[[1, 2]] * m[[3, 0]];
    let a0312 = m[[1, 0]] * m[[2, 3]] - m[[1, 3]] * m[[2, 0]];
    let a0212 = m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]];
    let a0113 = m[[1, 0]] * m[[3, 1]] - m[[1, 1]] * m[[3, 0]];
    let a0112 = m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]];

    // Calculate determinant
    let det = m[[0, 0]] * (m[[1, 1]] * a2323 - m[[1, 2]] * a1323 + m[[1, 3]] * a1223)
        - m[[0, 1]] * (m[[1, 0]] * a2323 - m[[1, 2]] * a0323 + m[[1, 3]] * a0223)
        + m[[0, 2]] * (m[[1, 0]] * a1323 - m[[1, 1]] * a0323 + m[[1, 3]] * a0123)
        - m[[0, 3]] * (m[[1, 0]] * a1223 - m[[1, 1]] * a0223 + m[[1, 2]] * a0123);

    if det.abs() < 1e-12 {
        return Err(MatrixError::Singular);
    }

    let inv_det = 1.0 / det;

    // Calculate adjugate matrix
    let adj = arr2(&[
        [
            (m[[1, 1]] * a2323 - m[[1, 2]] * a1323 + m[[1, 3]] * a1223) * inv_det,
            (m[[0, 2]] * a1323 - m[[0, 1]] * a2323 - m[[0, 3]] * a1223) * inv_det,
            (m[[0, 1]] * a2313 - m[[0, 2]] * a1313 + m[[0, 3]] * a1213) * inv_det,
            (m[[0, 2]] * a1312 - m[[0, 1]] * a2312 - m[[0, 3]] * a1212) * inv_det,
        ],
        [
            (m[[1, 2]] * a0323 - m[[1, 0]] * a2323 - m[[1, 3]] * a0223) * inv_det,
            (m[[0, 0]] * a2323 - m[[0, 2]] * a0323 + m[[0, 3]] * a0223) * inv_det,
            (m[[0, 2]] * a0313 - m[[0, 0]] * a2313 - m[[0, 3]] * a0213) * inv_det,
            (m[[0, 0]] * a2312 - m[[0, 2]] * a0312 + m[[0, 3]] * a0212) * inv_det,
        ],
        [
            (m[[1, 0]] * a1323 - m[[1, 1]] * a0323 + m[[1, 3]] * a0123) * inv_det,
            (m[[0, 1]] * a0323 - m[[0, 0]] * a1323 - m[[0, 3]] * a0123) * inv_det,
            (m[[0, 0]] * a1313 - m[[0, 1]] * a0313 + m[[0, 3]] * a0113) * inv_det,
            (m[[0, 1]] * a0312 - m[[0, 0]] * a1312 - m[[0, 3]] * a0112) * inv_det,
        ],
        [
            (m[[1, 1]] * a0223 - m[[1, 0]] * a1223 - m[[1, 2]] * a0123) * inv_det,
            (m[[0, 0]] * a1223 - m[[0, 1]] * a0223 + m[[0, 2]] * a0123) * inv_det,
            (m[[0, 1]] * a0213 - m[[0, 0]] * a1213 - m[[0, 2]] * a0113) * inv_det,
            (m[[0, 0]] * a1212 - m[[0, 1]] * a0212 + m[[0, 2]] * a0112) * inv_det,
        ],
    ]);

    Ok(adj)
}
