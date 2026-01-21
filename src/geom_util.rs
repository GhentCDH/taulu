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

/// Finds the intersection of two polynomial curves using Newton's method.
///
/// Given coefficients for a horizontal curve `y = h(x)` and a vertical curve `x = v(y)`,
/// finds the point `(x, y)` where both equations are satisfied.
///
/// # Arguments
///
/// * `horizontal_coeffs` - Polynomial coefficients for `y = h(x)` (constant term first)
/// * `vertical_coeffs` - Polynomial coefficients for `x = v(y)` (constant term first)
/// * `degree` - Polynomial degree (1 for linear, 2 for quadratic)
/// * `initial_y` - Starting y value for Newton iteration
///
/// # Returns
///
/// `Some((x, y))` if convergence is achieved, `None` otherwise.
///
/// # Algorithm
///
/// Substitutes `x = v(y)` into `y = h(x)` to get `y = h(v(y))`, then solves
/// `y - h(v(y)) = 0` using Newton's method.
///
/// # Examples
///
/// ```ignore
/// // Two linear polynomials: y = 2 (horizontal line) and x = 3 (vertical line)
/// // h(x) = 2, so coefficients are [2.0, 0.0]
/// // v(y) = 3, so coefficients are [3.0, 0.0]
/// let h_coeffs = [2.0, 0.0];
/// let v_coeffs = [3.0, 0.0];
/// let result = find_polynomial_intersection(&h_coeffs, &v_coeffs, 1, 0.0);
/// assert!(result.is_some());
/// let (x, y) = result.unwrap();
/// assert!((x - 3.0).abs() < 1e-5);
/// assert!((y - 2.0).abs() < 1e-5);
/// ```
pub fn find_polynomial_intersection(
    horizontal_coeffs: &[f32],
    vertical_coeffs: &[f32],
    degree: usize,
    initial_y: f32,
) -> Option<(f32, f32)> {
    let ho = horizontal_coeffs;
    let ve = vertical_coeffs;

    // Build the composed polynomial h(v(y)) - y and its derivative
    // For degree 1: h(x) = ho[0] + ho[1]*x, v(y) = ve[0] + ve[1]*y
    //   h(v(y)) = ho[0] + ho[1]*(ve[0] + ve[1]*y) = (ho[0] + ho[1]*ve[0]) + ho[1]*ve[1]*y
    //   h(v(y)) - y = (ho[0] + ho[1]*ve[0]) + (ho[1]*ve[1] - 1)*y
    // For degree 2: more complex composition
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

    let mut y = initial_y;

    // Newton's method iteration
    for _ in 0..20 {
        let h_y = evaluate_polynomial(&h, y);
        let h_der_y = evaluate_polynomial(&h_derivative, y);
        let new_y = y - h_y / h_der_y;

        let dist = (y - new_y).abs();

        if dist < 1e-6 {
            let x = evaluate_polynomial(vertical_coeffs, new_y);
            return Some((x, new_y));
        }

        if dist.is_nan() {
            return None;
        }

        y = new_y;
    }

    // Did not converge
    None
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

    #[test]
    fn test_find_polynomial_intersection_linear() {
        // Two perpendicular lines:
        // Horizontal: y = 10 (constant, so y = 10 for all x)
        // Vertical: x = 20 (constant, so x = 20 for all y)
        // Intersection should be at (20, 10)

        // y = h(x) = 10 -> coeffs [10, 0]
        let horizontal_coeffs = vec![10.0, 0.0];
        // x = v(y) = 20 -> coeffs [20, 0]
        let vertical_coeffs = vec![20.0, 0.0];

        let result = find_polynomial_intersection(&horizontal_coeffs, &vertical_coeffs, 1, 0.0);
        assert!(result.is_some());

        let (x, y) = result.unwrap();
        assert_abs_diff_eq!(x, 20.0, epsilon = 1e-4);
        assert_abs_diff_eq!(y, 10.0, epsilon = 1e-4);
    }

    #[test]
    fn test_find_polynomial_intersection_sloped_lines() {
        // Horizontal line: y = 0.5 * x + 5 (slope 0.5, intercept 5)
        // At x = 10, y = 10
        let horizontal_coeffs = vec![5.0, 0.5];

        // Vertical line: x = -0.5 * y + 15 (slope -0.5, intercept 15)
        // At y = 10, x = 10
        let vertical_coeffs = vec![15.0, -0.5];

        // They should intersect at (10, 10)
        let result = find_polynomial_intersection(&horizontal_coeffs, &vertical_coeffs, 1, 5.0);
        assert!(result.is_some());

        let (x, y) = result.unwrap();
        assert_abs_diff_eq!(x, 10.0, epsilon = 1e-4);
        assert_abs_diff_eq!(y, 10.0, epsilon = 1e-4);
    }

    #[test]
    fn test_find_polynomial_intersection_unsupported_degree() {
        let horizontal_coeffs = vec![1.0, 2.0, 3.0, 4.0]; // degree 3
        let vertical_coeffs = vec![1.0, 2.0, 3.0, 4.0];

        let result = find_polynomial_intersection(&horizontal_coeffs, &vertical_coeffs, 3, 0.0);
        assert!(result.is_none());
    }
}
