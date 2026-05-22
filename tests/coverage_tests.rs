/*
Copyright 2021 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//! Comprehensive coverage tests targeting uncovered lines across all modules.

#[macro_use]
extern crate approx;

use lcso::quadratics::bq::BoundedQuadratic;
use lcso::quadratics::envelope::envelope;
use lcso::quadratics::pwq::{PiecewiseQuadratic, SyncWorkspace};
use len_trait::{Empty, Len};
use num::Zero;
use std::f64;

// ============== BoundedQuadratic Tests ==============

mod bq_tests {
    use super::*;

    // --- restrict_domain_in_place (lines 170-178) ---
    #[test]
    fn test_restrict_domain_in_place() {
        let mut bq = BoundedQuadratic::new(-2., 2., 1., 0., 0.);
        bq.restrict_domain_in_place(-1., 1.);
        assert_eq!(bq.lower, -1.);
        assert_eq!(bq.upper, 1.);
    }

    #[test]
    fn test_restrict_domain_in_place_wider() {
        let mut bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        bq.restrict_domain_in_place(-5., 5.);
        assert_eq!(bq.lower, -1.);
        assert_eq!(bq.upper, 1.);
    }

    // --- extend_domain_in_place (lines 216-218) ---
    #[test]
    fn test_extend_domain_in_place() {
        let mut bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        bq.extend_domain_in_place();
        assert_eq!(bq.lower, f64::NEG_INFINITY);
        assert_eq!(bq.upper, f64::INFINITY);
    }

    // --- scale (line 232-239) ---
    #[test]
    fn test_scale() {
        let bq = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
        let scaled = bq.scale(2.);
        assert_relative_eq!(scaled.a, 2., epsilon = f64::EPSILON);
        assert_relative_eq!(scaled.b, 2., epsilon = f64::EPSILON);
        assert_relative_eq!(scaled.c, 2., epsilon = f64::EPSILON);
    }

    // --- scale_in_place (lines 242-245) ---
    #[test]
    fn test_scale_in_place() {
        let mut bq = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
        bq.scale_in_place(3.);
        assert_relative_eq!(bq.a, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.b, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.c, 3., epsilon = f64::EPSILON);
    }

    // --- new_scaled_interval with negative scale (lines 249-258) ---
    #[test]
    fn test_scale_arg_negative() {
        let bq = BoundedQuadratic::new(-2., 4., 1., 0., 0.);
        let scaled = bq.scale_arg(-1.);
        // with negative scale, bounds swap
        assert!(scaled.lower <= scaled.upper);
    }

    // --- perspective_in_place (lines 329-336) ---
    #[test]
    fn test_perspective_in_place() {
        let mut bq = BoundedQuadratic::new(-1., 1., 2., 3., 4.);
        bq.perspective_in_place(2.);
        assert_relative_eq!(bq.a, 1., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.b, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.c, 8., epsilon = f64::EPSILON);
    }

    // --- shift_in_place (lines 362-368) ---
    #[test]
    fn test_shift_in_place() {
        let mut bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        bq.shift_in_place(2.);
        assert_relative_eq!(bq.lower, 1., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.upper, 3., epsilon = f64::EPSILON);
    }

    // --- _minimize edge cases (lines 542, 559, 568) ---
    #[test]
    fn test_minimize_upward_sloping_infinite_lower() {
        // b > 0, lower is -inf => NaN x, -inf value
        let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 1., 0.);
        let (x, val) = bq.minimize();
        assert!(x.is_nan());
        assert_eq!(val, f64::NEG_INFINITY);
    }

    #[test]
    fn test_minimize_downward_sloping_infinite_upper() {
        // b < 0, upper is inf => NaN x, -inf value
        let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., -1., 0.);
        let (x, val) = bq.minimize();
        assert!(x.is_nan());
        assert_eq!(val, f64::NEG_INFINITY);
    }

    #[test]
    fn test_minimize_constant_both_infinite() {
        // constant function with both bounds infinite => returns (0, c)
        let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 0., 5.);
        let (x, val) = bq.minimize();
        assert_relative_eq!(x, 0., epsilon = f64::EPSILON);
        assert_relative_eq!(val, 5., epsilon = f64::EPSILON);
    }

    #[test]
    fn test_minimize_constant_finite_lower() {
        let bq = BoundedQuadratic::new(1., f64::INFINITY, 0., 0., 5.);
        let (x, val) = bq.minimize();
        assert_relative_eq!(x, 1., epsilon = f64::EPSILON);
        assert_relative_eq!(val, 5., epsilon = f64::EPSILON);
    }

    #[test]
    fn test_minimize_constant_only_upper_finite() {
        let bq = BoundedQuadratic::new(f64::NEG_INFINITY, 3., 0., 0., 7.);
        let (x, val) = bq.minimize();
        assert_relative_eq!(x, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(val, 7., epsilon = f64::EPSILON);
    }

    // --- Display trait (lines 634-692) ---
    #[test]
    fn test_display_full_quadratic() {
        let bq = BoundedQuadratic::new(-1., 1., 2., 3., 4.);
        let s = format!("{}", bq);
        assert!(s.contains("BoundedQuadratic"));
        assert!(s.contains("2")); // coefficient
    }

    #[test]
    fn test_display_negative_coefficients() {
        let bq = BoundedQuadratic::new(-1., 1., -2., -3., -4.);
        let s = format!("{}", bq);
        assert!(s.contains("-"));
    }

    #[test]
    fn test_display_unit_coefficients() {
        let bq = BoundedQuadratic::new(-1., 1., 1., 1., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("x"));
    }

    #[test]
    fn test_display_zero_function() {
        let bq = BoundedQuadratic::new(-1., 1., 0., 0., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("0"));
    }

    #[test]
    fn test_display_linear_only() {
        let bq = BoundedQuadratic::new(-1., 1., 0., 2., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("x"));
    }

    #[test]
    fn test_display_constant_only() {
        let bq = BoundedQuadratic::new(-1., 1., 0., 0., 5.);
        let s = format!("{}", bq);
        assert!(s.contains("5"));
    }

    #[test]
    fn test_display_infinite_bounds() {
        let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 1., 0., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("(")); // open bracket for infinite
        assert!(s.contains(")")); // close bracket for infinite
    }

    #[test]
    fn test_display_finite_bounds() {
        let bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("[")); // closed bracket for finite
        assert!(s.contains("]")); // closed bracket for finite
    }

    #[test]
    fn test_display_unit_neg_linear() {
        let bq = BoundedQuadratic::new(-1., 1., 0., -1., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("-"));
    }

    #[test]
    fn test_display_neg_unit_quadratic() {
        let bq = BoundedQuadratic::new(-1., 1., -1., 0., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("-"));
    }

    #[test]
    fn test_display_positive_constant_with_quadratic() {
        let bq = BoundedQuadratic::new(-1., 1., 1., 0., 5.);
        let s = format!("{}", bq);
        assert!(s.contains("+"));
    }

    #[test]
    fn test_display_positive_linear_with_quadratic() {
        let bq = BoundedQuadratic::new(-1., 1., 1., 2., 0.);
        let s = format!("{}", bq);
        assert!(s.contains("+"));
    }

    // --- derivative (line 510-511) ---
    #[test]
    fn test_derivative() {
        let bq = BoundedQuadratic::new(-1., 1., 3., 2., 1.);
        let d = bq.derivative();
        assert_relative_eq!(d.a, 0., epsilon = f64::EPSILON);
        assert_relative_eq!(d.b, 6., epsilon = f64::EPSILON); // 2*a = 6
        assert_relative_eq!(d.c, 2., epsilon = f64::EPSILON); // b
    }

    // --- find_roots edge cases ---
    #[test]
    fn test_find_roots_linear() {
        let bq = BoundedQuadratic::new(-10., 10., 0., 1., -3.);
        let (x1, x2) = bq.find_roots();
        assert_relative_eq!(x1, 3., epsilon = 1e-10);
    }

    #[test]
    fn test_find_roots_negative_discriminant() {
        let bq = BoundedQuadratic::new(-10., 10., 1., 0., 1.);
        let (x1, x2) = bq.find_roots();
        assert_eq!(x1, f64::INFINITY);
        assert_eq!(x2, f64::INFINITY);
    }

    #[test]
    fn test_find_roots_out_of_domain() {
        let bq = BoundedQuadratic::new(5., 10., 1., -1., -6.);
        let (x1, x2) = bq.find_roots();
        // roots are -2 and 3, both out of [5, 10]
        assert_eq!(x1, f64::INFINITY);
    }

    #[test]
    fn test_find_roots_negative_b() {
        let bq = BoundedQuadratic::new(-10., 10., 1., -5., 6.);
        let (x1, x2) = bq.find_roots();
        assert!(x1.is_finite());
        assert!(x2.is_finite());
    }

    // --- PartialEq for points ---
    #[test]
    fn test_partial_eq_points() {
        let p1 = BoundedQuadratic::new_point(1., 2.);
        let p2 = BoundedQuadratic::new_point(1., 2.);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_partial_eq_different_points() {
        let p1 = BoundedQuadratic::new_point(1., 2.);
        let p2 = BoundedQuadratic::new_point(1., 3.);
        assert_ne!(p1, p2);
    }

    // --- is_affine ---
    #[test]
    fn test_is_affine() {
        let line = BoundedQuadratic::new(-1., 1., 0., 1., 0.);
        assert!(line.is_affine());
        let quad = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        assert!(!quad.is_affine());
    }

    // --- is_empty ---
    #[test]
    fn test_is_empty() {
        let empty = BoundedQuadratic::new(2., 1., 0., 0., 0.); // lower > upper
        assert!(empty.is_empty());
        let nonempty = BoundedQuadratic::new(1., 2., 0., 0., 0.);
        assert!(!nonempty.is_empty());
    }

    // --- eval out of domain ---
    #[test]
    fn test_eval_out_of_domain() {
        let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        assert_eq!(bq.eval(5.), f64::INFINITY);
    }

    // --- eval_derivative out of domain ---
    #[test]
    fn test_eval_derivative_out_of_domain() {
        let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        assert_eq!(bq.eval_derivative(5.), f64::INFINITY);
    }

    // --- reflect_over_y_in_place ---
    #[test]
    fn test_reflect_over_y_in_place() {
        let mut bq = BoundedQuadratic::new(-2., 1., 1., 3., 0.);
        bq.reflect_over_y_in_place();
        assert_relative_eq!(bq.lower, -1., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.upper, 2., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.b, -3., epsilon = f64::EPSILON);
    }

    // --- scale_arg_in_place ---
    #[test]
    fn test_scale_arg_in_place() {
        let mut bq = BoundedQuadratic::new(-1., 1., 1., 2., 3.);
        bq.scale_arg_in_place(2.);
        assert_relative_eq!(bq.a, 4., epsilon = f64::EPSILON);
        assert_relative_eq!(bq.b, 4., epsilon = f64::EPSILON);
    }

    // --- sum_bq ---
    #[test]
    fn test_sum_bq() {
        let f = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let sum = BoundedQuadratic::sum_bq(&[f, g]).unwrap();
        assert_relative_eq!(sum.a, 1., epsilon = f64::EPSILON);
        assert_relative_eq!(sum.b, 1., epsilon = f64::EPSILON);
    }

    #[test]
    fn test_sum_bq_no_common_domain() {
        let f = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        let g = BoundedQuadratic::new(2., 3., 0., 1., 0.);
        assert!(BoundedQuadratic::sum_bq(&[f, g]).is_none());
    }

    // --- approx ---
    #[test]
    fn test_approx() {
        let f = BoundedQuadratic::new(-1., 1., 1., 2., 3.);
        let g = BoundedQuadratic::new(-1., 1., 1., 2., 3.);
        assert!(f.approx(&g));
    }

    // --- new_line_from_points ---
    #[test]
    fn test_new_line_from_points() {
        let line = BoundedQuadratic::new_line_from_points((0., 0.), (1., 2.));
        assert_relative_eq!(line.b, 2., epsilon = 1e-10);
        assert_relative_eq!(line.c, 0., epsilon = 1e-10);
    }

    // --- minimize with point domain ---
    #[test]
    fn test_minimize_point() {
        let bq = BoundedQuadratic::new_point(3., 7.);
        let (x, val) = bq.minimize();
        assert_relative_eq!(x, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(val, 7., epsilon = f64::EPSILON);
    }
}

// ============== PiecewiseQuadratic Tests ==============

mod pwq_tests {
    use super::*;

    // --- minimize (lines 203-213) ---
    #[test]
    fn test_minimize() {
        let f = BoundedQuadratic::new(-2., 0., 1., 0., 0.);
        let g = BoundedQuadratic::new(0., 2., 1., 0., 1.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let (x, val, idx) = pwq.minimize();
        assert_relative_eq!(x, 0., epsilon = 1e-10);
        assert_relative_eq!(val, 0., epsilon = 1e-10);
        assert_eq!(idx, 0);
    }

    // --- scale_in_place (lines 227-231) ---
    #[test]
    fn test_scale_in_place() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.scale_in_place(3.);
        assert_relative_eq!(pwq[0].b, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(pwq[1].b, 6., epsilon = f64::EPSILON);
    }

    // --- scale_arg_in_place (lines 248-256) ---
    #[test]
    fn test_scale_arg_in_place() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.scale_arg_in_place(2.);
        assert_eq!(pwq.len(), 2);
    }

    #[test]
    fn test_scale_arg_in_place_negative() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.scale_arg_in_place(-1.);
        assert_eq!(pwq.len(), 2);
    }

    // --- perspective_in_place (lines 273-281) ---
    #[test]
    fn test_perspective_in_place() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.perspective_in_place(2.);
        assert_eq!(pwq.len(), 2);
    }

    #[test]
    fn test_perspective_in_place_negative() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.perspective_in_place(-2.);
        assert_eq!(pwq.len(), 2);
    }

    // --- shift_in_place (lines 288-292) ---
    #[test]
    fn test_shift_in_place() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.shift_in_place(5.);
        assert_relative_eq!(pwq[0].lower, 4., epsilon = f64::EPSILON);
    }

    // --- reflect_over_y_in_place (lines 303-308) ---
    #[test]
    fn test_reflect_over_y_in_place() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let mut pwq = PiecewiseQuadratic::new(vec![f, g]);
        pwq.reflect_over_y_in_place();
        assert_eq!(pwq.len(), 2);
        // order should be reversed
        assert_relative_eq!(pwq[0].lower, -1., epsilon = f64::EPSILON);
    }

    // --- Display trait (lines 650-660) ---
    #[test]
    fn test_pwq_display() {
        let f = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let s = format!("{}", pwq);
        assert!(s.contains("PiecewiseQuadratic"));
    }

    // --- Zero and is_zero (lines 673-685) ---
    #[test]
    fn test_zero() {
        let z = PiecewiseQuadratic::zero();
        assert!(z.is_zero());
    }

    #[test]
    fn test_not_zero() {
        let f = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        assert!(!pwq.is_zero());
    }

    #[test]
    fn test_is_zero_multi_piece() {
        let f = BoundedQuadratic::new(-1., 0., 0., 0., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        assert!(!pwq.is_zero()); // multi-piece so not zero
    }

    // --- is_empty ---
    #[test]
    fn test_is_empty_pwq() {
        let pwq = PiecewiseQuadratic::new(vec![]);
        assert!(pwq.is_empty());
    }

    // --- PiecewiseQuadratic::new with validation (line 97) ---
    #[test]
    fn test_new_single_piece() {
        let f = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        assert_eq!(pwq.len(), 1);
    }

    // --- is_convex (lines 162-180) ---
    #[test]
    fn test_is_convex_single() {
        let f = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        assert!(pwq.is_convex());
    }

    #[test]
    fn test_is_convex_multi() {
        let f = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
        let g = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        assert!(pwq.is_convex());
    }

    #[test]
    fn test_not_convex() {
        let f = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., f64::INFINITY, 0., -1., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        assert!(!pwq.is_convex());
    }

    // --- eval edge cases ---
    #[test]
    fn test_eval_at_boundary() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 1.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        // At boundary x=0, both f(0)=1 and g(0)=0; eval should return min(1,0) = 0
        assert_eq!(pwq.eval(0.), 0.);
    }

    // --- sum_pwq with syncd = true ---
    #[test]
    fn test_sum_pwq_syncd() {
        let f = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let p1 = PiecewiseQuadratic::new(vec![f]);
        let p2 = PiecewiseQuadratic::new(vec![g]);
        let mut work = SyncWorkspace::new(2);
        let sum = PiecewiseQuadratic::sum_pwq(&mut work, &[&p1, &p2], true);
        assert_eq!(sum.len(), 1);
        assert_relative_eq!(sum[0].a, 1., epsilon = f64::EPSILON);
        assert_relative_eq!(sum[0].b, 1., epsilon = f64::EPSILON);
    }

    // --- conjugate_maximizer with quadratic pieces ---
    #[test]
    fn test_conjugate_maximizer_quadratic() {
        let f = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        let cm = pwq.conjugate_maximizer();
        assert!(cm.len() >= 1);
    }

    #[test]
    fn test_conjugate_maximizer_with_gap() {
        // convex function with affine left piece extending to -inf
        let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
        let right = BoundedQuadratic::new(0., f64::INFINITY, 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![left, right]);
        let cm = pwq.conjugate_maximizer();
        assert!(cm.len() >= 1);
    }

    #[test]
    fn test_conjugate_maximizer_finite_upper() {
        // convex function with finite upper bound on the last piece
        let f = BoundedQuadratic::new(0., 2., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        let cm = pwq.conjugate_maximizer();
        // Should have piece extending to infinity
        assert!(cm.len() >= 1);
    }

    // --- simplify edge case: empty piece in the middle ---
    #[test]
    fn test_simplify_with_empty_piece() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let empty = BoundedQuadratic::new(1., 0., 0., 0., 0.); // empty
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, empty, g]);
        let s = pwq.simplify();
        assert!(s.len() <= 3);
    }

    // --- scale (non-in-place, lines 218-224) ---
    #[test]
    fn test_scale() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let scaled = pwq.scale(3.);
        assert_eq!(scaled.len(), 2);
        assert_relative_eq!(scaled[0].b, 3., epsilon = f64::EPSILON);
        assert_relative_eq!(scaled[1].b, 6., epsilon = f64::EPSILON);
    }

    // --- shift (non-in-place, lines 284-285) ---
    #[test]
    fn test_shift() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let shifted = pwq.shift(5.);
        assert_eq!(shifted.len(), 2);
        assert_relative_eq!(shifted[0].lower, 4., epsilon = f64::EPSILON);
    }

    // --- reflect_over_y (non-in-place, lines 296-300) ---
    #[test]
    fn test_reflect_over_y() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let reflected = pwq.reflect_over_y();
        assert_eq!(reflected.len(), 2);
    }

    // --- simplify with empty first piece (line 473) ---
    #[test]
    fn test_simplify_empty_first_piece() {
        let empty = BoundedQuadratic::new(5., 0., 0., 0., 0.); // lower > upper = empty
        let f = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        let g = BoundedQuadratic::new(2., 3., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![empty, f, g]);
        let s = pwq.simplify();
        assert!(s.len() >= 1);
    }

    // --- new with 3+ pieces (line 97 - the loop body) ---
    #[test]
    fn test_new_three_pieces() {
        let f = BoundedQuadratic::new(-2., -1., 1., 0., 0.);
        let g = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let h = BoundedQuadratic::new(0., 1., 0., 0., 1.);
        let pwq = PiecewiseQuadratic::new(vec![f, g, h]);
        assert_eq!(pwq.len(), 3);
    }

    // --- Add trait (should panic) ---
    #[test]
    #[should_panic]
    fn test_add_trait_panics() {
        let f = BoundedQuadratic::new(-1., 1., 0., 0., 0.);
        let p1 = PiecewiseQuadratic::new(vec![f]);
        let p2 = PiecewiseQuadratic::new(vec![f]);
        let _ = p1 + p2;
    }
}

// ============== Envelope Tests ==============

mod envelope_tests {
    use super::*;

    #[test]
    fn test_envelope_single_convex() {
        let f = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        let env = envelope(&pwq);
        assert_eq!(env.len(), 1);
    }

    #[test]
    fn test_envelope_non_convex() {
        // Non-convex function: f(x) = -x^2
        let f = BoundedQuadratic::new(-1., 1., -1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_v_shape() {
        // Already convex |x|
        let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
        let right = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
        let pwq = PiecewiseQuadratic::new(vec![left, right]);
        let env = envelope(&pwq);
        assert_eq!(env.len(), 2);
    }

    #[test]
    fn test_envelope_with_gap() {
        // Pieces with a gap
        let f = BoundedQuadratic::new(-2., -1., 1., 0., 0.);
        let g = BoundedQuadratic::new(1., 2., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_point_and_curve() {
        let pt = BoundedQuadratic::new_point(0., 0.);
        let curve = BoundedQuadratic::new(0., 2., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![pt, curve]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_constant_pieces() {
        let f = BoundedQuadratic::new(-1., 0., 0., 0., 1.);
        let g = BoundedQuadratic::new(0., 1., 0., 0., 2.);
        let pwq = PiecewiseQuadratic::new(vec![f, g]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_two_points_adjacent() {
        // Two points at the same x coordinate
        let p1 = BoundedQuadratic::new_point(0., 1.);
        let p2 = BoundedQuadratic::new_point(0., 2.);
        let pwq = PiecewiseQuadratic::new(vec![p1, p2]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_point_before_curve() {
        // Point with value >= curve's starting value
        let pt = BoundedQuadratic::new_point(-1., 5.);
        let curve = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![pt, curve]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_curve_before_point() {
        // Curve ending at a point where point value >= curve value
        let curve = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let pt = BoundedQuadratic::new_point(0., 5.);
        let pwq = PiecewiseQuadratic::new(vec![curve, pt]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_three_pieces_non_convex() {
        // W-shaped: two quadratic humps
        let f = BoundedQuadratic::new(-2., -1., -1., 0., 0.);
        let g = BoundedQuadratic::new(-1., 1., 1., 0., -1.);
        let h = BoundedQuadratic::new(1., 2., -1., 0., 0.);
        let pwq = PiecewiseQuadratic::new(vec![f, g, h]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }

    #[test]
    fn test_envelope_concave_bounded() {
        let f = BoundedQuadratic::new(-2., 2., -1., 0., 4.);
        let pwq = PiecewiseQuadratic::new(vec![f]);
        let env = envelope(&pwq);
        assert!(env.len() >= 1);
    }
}

// ============== Opto/Structs Tests ==============

mod structs_tests {
    use super::*;
    use lcso::opto::structs::{ProblemData, Settings, Variables};
    use ndarray::{array, Array2};
    use sprs::CsMat;

    // --- Variables::Display (lines 280-286) ---
    #[test]
    fn test_variables_display() {
        let vars = Variables::default(1, 2);
        let s = format!("{}", vars);
        assert!(s.contains("x:"));
        assert!(s.contains("xt:"));
    }

    // --- Stats.update with compute_stats = true (lines 62-63) ---
    #[test]
    fn test_settings_defaults() {
        let s = Settings::defaults(4, 2);
        assert_eq!(s.max_iter, 10000);
        assert_relative_eq!(s.alpha, 1.5, epsilon = f64::EPSILON);
    }

    // --- validate (lines 289-296) ---
    #[test]
    fn test_validate_matching() {
        let p = CsMat::csc_from_dense(array![[1., 0.], [0., 1.]].view(), f64::EPSILON);
        let q = array![0., 0.];
        let a = CsMat::csc_from_dense(array![[1., 1.]].view(), f64::EPSILON);
        let b = array![0.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];
        let pd = ProblemData::new(p, q, a, b, g);
        let settings = Settings::new(1.0, array![1.], array![1., 1.], 100, 10, true);
        let vars = Variables::from_problem_data(&pd);
        lcso::opto::structs::validate(&vars, &settings, &pd);
    }

    // --- Variables::new ---
    #[test]
    fn test_variables_new() {
        use ndarray::Array1;
        let vars = Variables::new(
            Array1::zeros(3),
            Array1::zeros(2),
            Array1::zeros(3),
            Array1::zeros(2),
            Array1::zeros(3),
            Array1::zeros(2),
        );
        assert_eq!(vars.x.len(), 3);
        assert_eq!(vars.z.len(), 2);
    }
}

// ============== Opto/ADMM Tests ==============

mod admm_tests {
    use super::*;
    use lcso::opto::admm::{optimize, optimize_structs};
    use lcso::opto::structs::{ProblemData, Settings, Variables};
    use lcso::opto::term::{Objective, Residual};
    use ndarray::{array, Array1, Array2};
    use ndarray::linalg::Dot;
    use sprs::CsMat;

    #[derive(Clone)]
    struct TestObjRes {}

    impl Objective for TestObjRes {
        fn objective(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
            let f_obj = vars.x.t().dot(&problem_data.p.dot(&vars.x))
                + problem_data.q.t().dot(&vars.x);
            let g_obj: f64 = (0..problem_data.q.len())
                .map(|i| problem_data.g[i].eval(vars.x[i]))
                .sum();
            f_obj + g_obj
        }
    }

    impl Residual for TestObjRes {
        fn residual(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
            (&problem_data.a.dot(&vars.x) - &problem_data.b)
                .mapv(|x| x.abs())
                .sum()
        }
    }

    // --- optimize (top-level function, lines 266-279) ---
    #[test]
    fn test_optimize_simple() {
        let a = CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON);
        let b = array![0.];
        let p = CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON);
        let q = array![1., 1.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];
        let (sol, stats) = optimize(a, b, p, q, g, &TestObjRes {}, false);
        assert!(sol.x[0] < 0.);
        assert!(sol.x[1] < 0.);
    }

    // --- optimize with convexify=true (lines 209-238) ---
    #[test]
    fn test_optimize_with_convexify() {
        let a = CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON);
        let b = array![0.];
        let p = CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON);
        let q = array![1., 1.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];
        let (sol, stats) = optimize(a, b, p, q, g, &TestObjRes {}, true);
        assert!(sol.x[0] < 0.);
    }

    // --- optimize_structs with compute_stats = true (triggers Stats.update lines 62-63) ---
    #[test]
    fn test_optimize_with_stats() {
        let settings = Settings::new(1., array![1.], array![1., 1.], 50, 2, true);
        let mut pd = ProblemData::new(
            CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON),
            array![1., 1.],
            CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON),
            array![0.],
            array![
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );
        let mut vars = Variables::from_problem_data(&pd);
        let (sol, stats) = optimize_structs(&settings, &mut pd, &mut vars, &TestObjRes {}, false);
        assert!(!stats.objective.is_empty());
        assert!(!stats.residual.is_empty());
    }

    // --- optimize with constraint ---
    #[test]
    fn test_optimize_with_constraint() {
        // x1 + x2 = 1
        let a = CsMat::csc_from_dense(array![[1., 1.]].view(), f64::EPSILON);
        let b = array![1.];
        let p = CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON);
        let q = array![0., 0.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];
        let (sol, _) = optimize(a, b, p, q, g, &TestObjRes {}, false);
        assert_relative_eq!(sol.x[0] + sol.x[1], 1., epsilon = 1e-3);
    }

    // --- optimize with box constraint separable term ---
    #[test]
    fn test_optimize_with_box_constraint() {
        let a = CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON);
        let b = array![0.];
        let p = CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON);
        let q = array![-10., -10.];
        // box constraint: x in [0, 1]
        let g = array![
            PiecewiseQuadratic::indicator(0., 1.),
            PiecewiseQuadratic::indicator(0., 1.),
        ];
        let (sol, _) = optimize(a, b, p, q, g, &TestObjRes {}, false);
        assert!(sol.x[0] > -0.1);
        assert!(sol.x[1] > -0.1);
    }
}

// ============== Utils Tests ==============

mod utils_tests {
    use lcso::quadratics::bq::BoundedQuadratic;

    // --- clamp (line 37 - the middle branch) ---
    #[test]
    fn test_clamp_in_range() {
        let bq = BoundedQuadratic::new(0., 10., 1., -10., 0.);
        let (x, _) = bq.minimize();
        // should clamp to [0, 10]
        assert!(x >= 0. && x <= 10.);
    }
}
