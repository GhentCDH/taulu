//! Auto-detection of horizontal-rule offsets along vertical rules.
//!
//! For each vertical-rule top point we run a 3-connected A* downward on a
//! (typically scaled-down) grayscale image. The cost function strongly
//! prefers straight downward motion, with only a slight bias toward dark
//! pixels — the assumption being that vertical rules are nearly straight
//! but may tilt slightly. The resulting path is rescaled to full
//! resolution and treated as a piecewise-linear curve along which we
//! sample the cross-correlation map. Peaks of that 1D profile are
//! clustered across columns to recover horizontal-rule offsets.

use numpy::PyReadonlyArray2;
use pathfinding::prelude::astar;
use pyo3::prelude::*;

use crate::Image;

#[cfg(feature = "debug-tools")]
const RERUN_EXPECT: &str = "Should be able to log values to rerun server";

#[cfg(feature = "debug-tools")]
fn start_rerun() -> rerun::RecordingStream {
    rerun::RecordingStreamBuilder::new("taulu")
        .connect_grpc()
        .expect("rerun recorder should spawn")
}

/// Convert arc-length along a piecewise-linear path to (x, y).
#[cfg(feature = "debug-tools")]
fn arc_length_to_point(path: &[(f32, f32)], target: f32) -> Option<(f32, f32)> {
    let mut acc: f32 = 0.0;
    for w in path.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        let dx = x1 - x0;
        let dy = y1 - y0;
        let seg = (dx * dx + dy * dy).sqrt();
        if seg <= f32::EPSILON {
            continue;
        }
        if target <= acc + seg {
            let t = (target - acc) / seg;
            return Some((x0 + t * dx, y0 + t * dy));
        }
        acc += seg;
    }
    path.last().copied()
}

/// Bilinear sample of an 8-bit single-channel image at fractional (x, y).
/// Returns 0 outside bounds.
fn bilinear_sample(img: &Image, x: f32, y: f32) -> f32 {
    let h = i32::try_from(img.shape()[0]).unwrap_or_default();
    let w = i32::try_from(img.shape()[1]).unwrap_or_default();

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    if x0 < 0 || y0 < 0 || x1 >= w || y1 >= h {
        return 0.0;
    }

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let v00 = f32::from(img[(y0 as usize, x0 as usize)]);
    let v10 = f32::from(img[(y0 as usize, x1 as usize)]);
    let v01 = f32::from(img[(y1 as usize, x0 as usize)]);
    let v11 = f32::from(img[(y1 as usize, x1 as usize)]);

    let v0 = v00 * (1.0 - dx) + v10 * dx;
    let v1 = v01 * (1.0 - dx) + v11 * dx;
    v0 * (1.0 - dy) + v1 * dy
}

/// 3-connected A* down a grayscale image.
///
/// * `start`, `goal` are integer pixel coords on `gray`.
/// * `straight_cost` is the cost of an aligned step (down/up).
/// * `perpendicular_cost` is the cost of a left/right step. Make this
///   large to penalize lateral deviation.
/// * `darkness_divisor` controls how strongly dark pixels reduce cost:
///   step cost adds `pixel / darkness_divisor`. Larger = less line bias.
///
/// Returns `None` if no path is found.
fn astar_vertical(
    gray: &Image,
    start: (i32, i32),
    goal: (i32, i32),
    straight_cost: u32,
    perpendicular_cost: u32,
    darkness_divisor: u32,
) -> Option<Vec<(i32, i32)>> {
    let h = gray.shape()[0] as i32;
    let w = gray.shape()[1] as i32;

    if start.0 < 0 || start.0 >= w || start.1 < 0 || start.1 >= h {
        return None;
    }
    if goal.0 < 0 || goal.0 >= w || goal.1 < 0 || goal.1 >= h {
        return None;
    }

    let divisor = darkness_divisor.max(1);
    let min_step = straight_cost.min(perpendicular_cost);

    let result = astar(
        &start,
        |&(x, y)| {
            let mut succ = Vec::with_capacity(3);
            // (dx, dy, base_cost)
            let neighbours = [
                (0_i32, 1_i32, straight_cost),
                (1, 0, perpendicular_cost),
                (-1, 0, perpendicular_cost),
            ];
            for (dx, dy, c) in neighbours {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let pixel = u32::from(gray[(ny as usize, nx as usize)]);
                let img_cost = pixel / divisor;
                succ.push(((nx, ny), c + img_cost));
            }
            succ
        },
        |&(x, y)| ((x - goal.0).abs() + (y - goal.1).abs()) as u32 * min_step,
        |&p| p == goal,
    );

    result.map(|(path, _)| path)
}

/// Sample `img` along a piecewise-linear path, stepping 1 px of arc-length.
fn sample_along_path(img: &Image, path: &[(f32, f32)]) -> Vec<f32> {
    let mut profile = Vec::new();
    if path.len() < 2 {
        return profile;
    }

    let mut acc: f32 = 0.0;
    let mut next_target: f32 = 0.0;

    for window in path.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];
        let dx = x1 - x0;
        let dy = y1 - y0;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len <= f32::EPSILON {
            continue;
        }
        while next_target <= acc + seg_len {
            let t = (next_target - acc) / seg_len;
            let x = x0 + t * dx;
            let y = y0 + t * dy;
            profile.push(bilinear_sample(img, x, y));
            next_target += 1.0;
        }
        acc += seg_len;
    }
    profile
}

/// Find local maxima with a minimum-distance constraint and a prominence
fn find_peaks(profile: &[f32], min_distance: i32, prominence: f32) -> Vec<i32> {
    if profile.len() < 3 {
        return Vec::new();
    }

    let mut candidates: Vec<(i32, f32)> = Vec::new();
    for i in 1..profile.len() - 1 {
        let v = profile[i];
        if v >= profile[i - 1] && v > profile[i + 1] && v >= prominence {
            candidates.push((i as i32, v));
        }
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<i32> = Vec::new();
    for (idx, _) in candidates {
        if idx < min_distance {
            continue;
        }
        let too_close = kept.iter().any(|&k| (k - idx).abs() < min_distance);
        if !too_close {
            kept.push(idx);
        }
    }

    kept.sort_unstable();
    kept
}

fn median(values: &mut [i32]) -> i32 {
    values.sort_unstable();
    let n = values.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        i32::midpoint(values[n / 2 - 1], values[n / 2])
    }
}

/// Cross-column clustering. For each peak in the densest column, collect
/// the nearest peak from every other column within `tolerance`. Keep the
/// cluster only if a sufficient fraction of columns contributed; emit the
/// median offset.
fn cluster_peaks(per_column: &[Vec<i32>], tolerance: i32, min_fraction: f32) -> Vec<i32> {
    if per_column.is_empty() {
        // no columns
        return Vec::new();
    }

    // column with the highest number of peaks
    let ref_idx = per_column
        .iter()
        .enumerate()
        .max_by_key(|(_, p)| p.len())
        .map_or(0, |(i, _)| i);

    let num_cols = per_column.len();
    let min_cols = ((num_cols as f32) * min_fraction).ceil().max(1.0) as usize;

    let mut clusters: Vec<i32> = Vec::new();

    for &ref_peak in &per_column[ref_idx] {
        let mut matches: Vec<i32> = vec![ref_peak];
        for (col_idx, peaks) in per_column.iter().enumerate() {
            if col_idx == ref_idx {
                continue;
            }
            let mut best: Option<i32> = None;
            let mut best_d = i32::MAX;
            for &p in peaks {
                let d = (p - ref_peak).abs();
                if d <= tolerance && d < best_d {
                    best_d = d;
                    best = Some(p);
                }
            }
            if let Some(p) = best {
                matches.push(p);
            }
        }

        if matches.len() >= min_cols {
            clusters.push(median(&mut matches));
        }
    }

    clusters.sort_unstable();
    let mut deduped: Vec<i32> = Vec::new();
    for c in clusters {
        if let Some(&last) = deduped.last()
            && (c - last).abs() < tolerance.max(1)
        {
            continue;
        }
        deduped.push(c);
    }

    deduped
}

/// Enforce min/max gap between consecutive offsets:
/// - drop offsets that produce gap < min,
/// - split offsets where gap > max via linear interpolation.
fn enforce_range(offsets: Vec<i32>, min_distance: i32, max_distance: i32) -> Vec<i32> {
    if offsets.is_empty() {
        return offsets;
    }

    let mut filtered: Vec<i32> = Vec::with_capacity(offsets.len());
    for o in offsets {
        if let Some(&last) = filtered.last()
            && o - last < min_distance
        {
            continue;
        }
        filtered.push(o);
    }

    let mut result: Vec<i32> = Vec::with_capacity(filtered.len());
    if filtered.is_empty() {
        return result;
    }
    result.push(filtered[0]);
    for cur in filtered.iter().skip(1) {
        let prev = result[result.len() - 1];
        let gap = cur - prev;
        if gap > max_distance {
            let n = (gap + max_distance - 1) / max_distance; // ceil
            let step = gap / n;
            for k in 1..n {
                result.push(prev + step * k);
            }
        }
        result.push(*cur);
    }

    result
}

/// Detect arc-length offsets of horizontal rules from each vertical-rule top point.
///
/// # Arguments
///
/// * `cross_correlation` - 2D `u8` cross-correlation map (full-resolution).
/// * `scaled_gray` - 2D `u8` grayscale image used for A* path-following
///   (typically a downscaled copy of the table image for speed).
/// * `top_points` - One (x, y) per vertical rule, in full-resolution image
///   coordinates.
/// * `scale` - Factor that was used to produce `scaled_gray` from the
///   full-resolution image (e.g. 0.25). Use 1.0 to skip downscaling.
/// * `min_distance`, `max_distance` - Min/max allowed row height in
///   pixels (full-resolution).
/// * `prominence` - Minimum peak value [0, 255]. Default ≈ 0.15 × 255.
/// * `cluster_tolerance` - Cross-column matching tolerance in pixels.
///   `-1` (default) selects `min_distance / 2`.
/// * `min_columns_for_rule` - Fraction of columns that must agree on a
///   rule. Default 0.4.
/// * `straight_cost` - A* cost per straight (down/up) step. Default 10.
/// * `perpendicular_cost` - A* cost per lateral step. Default 30 (strong
///   straight-line bias).
/// * `darkness_divisor` - A* image cost is `pixel / darkness_divisor`.
///   Default 100 (light line bias).
///
/// # Returns
///
/// Ascending arc-length offsets from the start (in full-resolution pixels).
/// Empty if detection failed.
#[pyfunction]
#[pyo3(signature = (
    cross_correlation,
    scaled_gray,
    top_points,
    scale,
    min_distance,
    max_distance,
    prominence = 38.0,
    cluster_tolerance = -1,
    min_columns_for_rule = 0.4,
    straight_cost = 10,
    perpendicular_cost = 30,
    darkness_divisor = 100,
))]
#[allow(clippy::too_many_arguments)]
pub fn detect_row_offsets(
    cross_correlation: PyReadonlyArray2<'_, u8>,
    scaled_gray: PyReadonlyArray2<'_, u8>,
    top_points: Vec<(f32, f32)>,
    scale: f32,
    min_distance: i32,
    max_distance: i32,
    prominence: f32,
    cluster_tolerance: i32,
    min_columns_for_rule: f32,
    straight_cost: u32,
    perpendicular_cost: u32,
    darkness_divisor: u32,
) -> PyResult<Vec<i32>> {
    if top_points.is_empty() {
        return Ok(Vec::new());
    }
    if min_distance <= 0 || max_distance < min_distance {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "require 0 < min_distance <= max_distance",
        ));
    }
    if scale <= 0.0 || scale > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scale must be in (0, 1]",
        ));
    }

    let cc = cross_correlation.as_array();
    let gray = scaled_gray.as_array();
    let scaled_h = gray.shape()[0] as i32;
    let scaled_w = gray.shape()[1] as i32;
    let inv_scale = 1.0 / scale;
    let tolerance = if cluster_tolerance <= 0 {
        (min_distance / 2).max(1)
    } else {
        cluster_tolerance
    };

    #[cfg(feature = "debug-tools")]
    let rec = start_rerun();
    #[cfg(feature = "debug-tools")]
    {
        rec.log(
            "row_detector/cross_correlation",
            &rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, cc.to_owned())
                .expect("should be able to create rerun image"),
        )
        .expect(RERUN_EXPECT);
        rec.log(
            "row_detector/scaled_gray",
            &rerun::Image::from_color_model_and_tensor(rerun::ColorModel::L, gray.to_owned())
                .expect("should be able to create rerun image"),
        )
        .expect(RERUN_EXPECT);
    }

    #[cfg(feature = "debug-tools")]
    let mut full_paths: Vec<Vec<(f32, f32)>> = Vec::with_capacity(top_points.len());

    let mut per_column: Vec<Vec<i32>> = Vec::with_capacity(top_points.len());
    for (col_idx, &(tx, ty)) in top_points.iter().enumerate() {
        let _ = col_idx;
        let sx = (tx * scale).round() as i32;
        let sy = (ty * scale).round() as i32;
        let sx = sx.clamp(0, scaled_w - 1);
        let sy = sy.clamp(0, scaled_h - 1);
        let goal = (sx, scaled_h - 1);

        let Some(path_scaled) = astar_vertical(
            &gray,
            (sx, sy),
            goal,
            straight_cost,
            perpendicular_cost,
            darkness_divisor,
        ) else {
            per_column.push(Vec::new());
            #[cfg(feature = "debug-tools")]
            full_paths.push(Vec::new());
            continue;
        };

        let path_full: Vec<(f32, f32)> = path_scaled
            .into_iter()
            .map(|(x, y)| (x as f32 * inv_scale, y as f32 * inv_scale))
            .collect();

        let profile = sample_along_path(&cc, &path_full);
        let peaks = find_peaks(&profile, min_distance, prominence);

        #[cfg(feature = "debug-tools")]
        {
            rec.log(
                format!("row_detector/paths/{col_idx}"),
                &rerun::LineStrips2D::new([path_full.clone()])
                    .with_colors([rerun::Color::from_rgb(0, 200, 255)])
                    .with_radii([1.0]),
            )
            .expect(RERUN_EXPECT);

            let prof_f64: Vec<f64> = profile.iter().map(|&v| v as f64).collect();
            rec.log(
                format!("row_detector/profile/{col_idx}"),
                &rerun::BarChart::new(prof_f64),
            )
            .expect(RERUN_EXPECT);

            let peak_points: Vec<(f32, f32)> = peaks
                .iter()
                .filter_map(|&p| arc_length_to_point(&path_full, p as f32))
                .collect();
            if !peak_points.is_empty() {
                rec.log(
                    format!("row_detector/peaks/{col_idx}"),
                    &rerun::Points2D::new(peak_points)
                        .with_colors([rerun::Color::from_rgb(255, 0, 0)])
                        .with_radii([3.0]),
                )
                .expect(RERUN_EXPECT);
            }
        }

        per_column.push(peaks);
        #[cfg(feature = "debug-tools")]
        full_paths.push(path_full);
    }

    if per_column.iter().all(Vec::is_empty) {
        return Ok(Vec::new());
    }

    let clustered = cluster_peaks(&per_column, tolerance, min_columns_for_rule);
    let final_offsets = enforce_range(clustered, min_distance, max_distance);

    #[cfg(feature = "debug-tools")]
    {
        let mut final_points: Vec<(f32, f32)> = Vec::new();
        for path in &full_paths {
            if path.is_empty() {
                continue;
            }
            for &off in &final_offsets {
                if let Some(pt) = arc_length_to_point(path, off as f32) {
                    final_points.push(pt);
                }
            }
        }
        if !final_points.is_empty() {
            rec.log(
                "row_detector/final_offsets",
                &rerun::Points2D::new(final_points)
                    .with_colors([rerun::Color::from_rgb(0, 255, 0)])
                    .with_radii([4.0]),
            )
            .expect(RERUN_EXPECT);
        }

        let offsets_f64: Vec<f64> = final_offsets.iter().map(|&v| v as f64).collect();
        if !offsets_f64.is_empty() {
            rec.log(
                "row_detector/final_offsets_chart",
                &rerun::BarChart::new(offsets_f64),
            )
            .expect(RERUN_EXPECT);
        }
    }

    Ok(final_offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_profile_with_peaks(len: usize, peaks: &[(usize, f32)]) -> Vec<f32> {
        let mut p = vec![0.0; len];
        for &(i, v) in peaks {
            p[i] = v;
        }
        p
    }

    #[test]
    fn peak_detection_min_distance() {
        let prof = flat_profile_with_peaks(100, &[(10, 200.0), (15, 180.0), (50, 220.0)]);
        let peaks = find_peaks(&prof, 20, 50.0);
        assert_eq!(peaks, vec![10, 50]);
    }

    #[test]
    fn peak_detection_skip_initial() {
        let prof = flat_profile_with_peaks(100, &[(2, 200.0), (40, 200.0)]);
        let peaks = find_peaks(&prof, 10, 50.0);
        assert_eq!(peaks, vec![40]);
    }

    #[test]
    fn cluster_basic() {
        let cols = vec![vec![10, 50, 90], vec![11, 49, 91], vec![10, 51]];
        let out = cluster_peaks(&cols, 3, 0.4);
        assert_eq!(out, vec![10, 50, 90]);
    }

    #[test]
    fn enforce_range_splits_large_gap() {
        let out = enforce_range(vec![0, 100], 10, 40);
        assert_eq!(out, vec![0, 33, 66, 100]);
    }

    #[test]
    fn enforce_range_drops_close() {
        let out = enforce_range(vec![0, 5, 30], 10, 100);
        assert_eq!(out, vec![0, 30]);
    }
}
