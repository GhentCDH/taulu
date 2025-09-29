use ndarray::prelude::*;

use crate::invert::invert_matrix;

#[derive(Debug)]
pub enum Error {
    NotEnoughPoints,
    Matrix,
    Conversion,
}

pub fn gaussian_1d(width: usize, sigma: Option<f32>) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    let sigma = sigma.unwrap_or(width as f32 * 0.15 + 0.35);
    let mut kernel = Vec::with_capacity(width);
    #[allow(clippy::cast_precision_loss)]
    let mean = (width as f32 - 1.0) / 2.0;
    let coeff = 1.0 / (sigma * (2.0 * std::f32::consts::PI).sqrt());
    let denom = 2.0 * sigma * sigma;

    for x in 0..width {
        #[allow(clippy::cast_precision_loss)]
        let exponent = -((x as f32 - mean).powi(2)) / denom;
        kernel.push(coeff * exponent.exp());
    }

    // Normalize the kernel
    normalize(&mut kernel);

    kernel
}

pub fn normalize(kernel: &mut [f32]) {
    let sum: f32 = kernel.iter().sum();
    if sum != 0.0 {
        for k in kernel {
            *k /= sum;
        }
    }
}

/// Solves the least squares problem for fitting a polynomial
/// of given degree to the set of points
pub fn linear_polynomial_least_squares(
    degree: usize,
    sample_input: &[f32],
    sample_output: &[f32],
) -> Result<Vec<f32>, Error> {
    assert_eq!(
        sample_input.len(),
        sample_output.len(),
        "input and output should have equal length"
    );
    let n = sample_input.len();

    if n < degree + 1 {
        return Err(Error::NotEnoughPoints);
    }

    // linear least square fit x and y separately on linear curve
    let mut features: Array2<f32> = Array2::zeros((n, degree + 1));
    for ((sample, power), element) in features.indexed_iter_mut() {
        *element =
            (sample_input[sample]).powi(i32::try_from(power).map_err(|_| Error::Conversion)?);
    }

    let mut samples: Array2<f32> = Array2::zeros((n, 1));
    for ((sample, _), element) in samples.indexed_iter_mut() {
        *element = sample_output[sample];
    }

    // X^T X
    let xt_x = features.t().dot(&features);

    // (X^T X)^(-1)
    let xt_x_inv = invert_matrix(&xt_x).map_err(|_| Error::Matrix)?;

    // X^T W Y
    let xt_y = features.t().dot(&samples);

    let coeffs = xt_x_inv.dot(&xt_y);

    assert_eq!(coeffs.shape(), &[degree + 1, 1]);

    Ok(coeffs.into_raw_vec_and_offset().0)
}

/// Given a primary set of points which you want to fit a line to,
/// and a selection of secondary sets of points which should be also approximately colinear,
/// fit a line to the primary set such that this line is mostly parallel to the secondary lines.
#[allow(clippy::similar_names)]
pub fn region_aware_fit(
    sample_input: &[f32],
    sample_output: &[f32],
    other_inputs: &[Vec<f32>],
    other_outputs: &[Vec<f32>],
    lambda: f32,
) -> Result<Vec<f32>, Error> {
    assert_eq!(
        sample_input.len(),
        sample_output.len(),
        "input and output should have equal length"
    );
    assert_eq!(
        other_inputs.len(),
        other_outputs.len(),
        "outher inputs and outputs should have equal length"
    );

    let n = sample_input.len();

    if n < 2 {
        return Err(Error::NotEnoughPoints);
    }

    // get the second coefficients of the other lines
    //
    // fit the other lines
    let mut avg_slope = 0.0;
    let mut found_slope = false;
    #[allow(clippy::cast_precision_loss)]
    for (i, (other_input, other_output)) in
        other_inputs.iter().zip(other_outputs.iter()).enumerate()
    {
        if other_input.len() < 2 || other_output.len() < 2 {
            continue;
        }
        let coeffs = linear_polynomial_least_squares(1, other_input, other_output)?;
        assert_eq!(coeffs.len(), 2);
        avg_slope = (coeffs[1] + avg_slope * (i as f32)) / ((i + 1) as f32);
        found_slope = true;
    }

    if !found_slope {
        return Err(Error::NotEnoughPoints);
    }

    // fit the primary line

    // linear least square fit x and y separately on linear curve
    let mut features: Array2<f32> = Array2::zeros((n, 2));
    for ((sample, power), element) in features.indexed_iter_mut() {
        *element =
            (sample_input[sample]).powi(i32::try_from(power).map_err(|_| Error::Conversion)?);
    }

    let mut samples: Array2<f32> = Array2::zeros((n, 1));
    for ((sample, _), element) in samples.indexed_iter_mut() {
        *element = sample_output[sample];
    }

    // selection matrix that selects the slope coefficient
    let c: Array2<f32> = array![[0.0, 0.0], [0.0, 1.0]];
    let desired_slope = array![[0.0], [avg_slope]];

    // X^T X
    let xt_x = features.t().dot(&features);

    let lambda = lambda.max(0.0) * max_eigenvalue_2d(&xt_x).ok_or(Error::Matrix)?;
    let xt_x_reg = xt_x + lambda * c;

    // (X^T X)^(-1)
    let xt_x_inv = invert_matrix(&xt_x_reg).map_err(|_| Error::Matrix)?;

    // X^T W Y
    let xt_y = features.t().dot(&samples);

    let xt_y_reg = xt_y + lambda * desired_slope;

    let coeffs = xt_x_inv.dot(&xt_y_reg);

    assert_eq!(coeffs.shape(), &[2, 1]);

    Ok(coeffs.into_raw_vec_and_offset().0)
}

pub fn evaluate_polynomial(coeffs: &[f32], input: f32) -> f32 {
    coeffs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            c * input.powi(
                i32::try_from(i)
                    .expect("coefficients index should convert to i32 without overflow"),
            )
        })
        .sum()
}

pub fn max_eigenvalue_2d(matrix: &Array2<f32>) -> Option<f32> {
    if matrix.shape() != [2, 2] {
        return None;
    }

    let trace = matrix[[0, 0]] + matrix[[1, 1]];
    let determinant = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];

    let discriminant = trace * trace - 4.0 * determinant;
    if discriminant < 0.0 {
        return None;
    }

    Some(f32::midpoint(trace, discriminant.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_evaluate_polynomial() {
        // Test linear polynomial: y = 2x + 3
        let coeffs = vec![3.0, 2.0]; // constant term first, then x term

        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 0.0), 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 1.0), 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 2.0), 7.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, -1.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_evaluate_polynomial_quadratic() {
        // Test quadratic polynomial: y = x^2 + 2x + 1 = (x+1)^2
        let coeffs = vec![1.0, 2.0, 1.0]; // constant, x, x^2

        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 0.0), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 1.0), 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, -1.0), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 2.0), 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_evaluate_polynomial_constant() {
        let coeffs = vec![5.0]; // constant polynomial

        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 0.0), 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, 100.0), 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(evaluate_polynomial(&coeffs, -50.0), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_polynomial_least_squares_two_points() {
        let x_values = vec![2.0, 3.0];
        let y_values = vec![5.0, 10.0];

        let result = linear_polynomial_least_squares(1, &x_values, &y_values);
        assert!(result.is_ok());

        let coeffs = result.expect("result should be ok because this was just asserted");
        assert_eq!(coeffs.len(), 2);

        assert_abs_diff_eq!(coeffs[0], -5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(coeffs[1], 5.0, epsilon = 1e-6);

        let eval = evaluate_polynomial(&coeffs, 4.0);
        assert_abs_diff_eq!(eval, 15.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_linear_polynomial_least_squares_mismatched_lengths() {
        let x_values = vec![0.0, 1.0, 2.0];
        let y_values = vec![0.0, 1.0]; // Different length

        let _ = linear_polynomial_least_squares(2, &x_values, &y_values);
    }

    #[test]
    fn test_linear_polynomial_least_squares_empty() {
        let x_values: Vec<f32> = vec![];
        let y_values: Vec<f32> = vec![];

        let result = linear_polynomial_least_squares(1, &x_values, &y_values);
        assert!(result.is_err());
    }
}
