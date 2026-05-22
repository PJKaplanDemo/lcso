use lcso::quadratics::bq::BoundedQuadratic;
use lcso::quadratics::pwq::{PiecewiseQuadratic, SyncWorkspace};
use lcso::opto::structs::{
    HasConstraints, HasVariables, ProblemData, Settings, Stats, Variables,
};
use lcso::opto::term::{Objective, Residual};
use lcso::opto::admm;
use len_trait::Len;
use ndarray::{array, linalg::Dot, Array2};
use num::traits::Zero;
use sprs::CsMat;
use std::f64;

// ─── BoundedQuadratic in-place methods ───

#[test]
fn test_restrict_domain_in_place() {
    let mut bq = BoundedQuadratic::new(0., 10., 1., 0., 0.);
    bq.restrict_domain_in_place(2., 8.);
    assert_eq!(bq.lower, 2.);
    assert_eq!(bq.upper, 8.);
}

#[test]
fn test_extend_domain_in_place() {
    let mut bq = BoundedQuadratic::new(0., 10., 1., 0., 0.);
    bq.extend_domain_in_place();
    assert_eq!(bq.lower, f64::NEG_INFINITY);
    assert_eq!(bq.upper, f64::INFINITY);
}

#[test]
fn test_scale_method() {
    let bq = BoundedQuadratic::new(-1., 1., 1., 2., 3.);
    let scaled = bq.scale(2.);
    assert_eq!(scaled.a, 2.);
    assert_eq!(scaled.b, 4.);
    assert_eq!(scaled.c, 6.);
    assert_eq!(scaled.lower, -1.);
    assert_eq!(scaled.upper, 1.);
}

#[test]
fn test_scale_in_place() {
    let mut bq = BoundedQuadratic::new(-1., 1., 1., 2., 3.);
    bq.scale_in_place(2.);
    assert_eq!(bq.a, 2.);
    assert_eq!(bq.b, 4.);
    assert_eq!(bq.c, 6.);
}

#[test]
fn test_perspective_in_place() {
    let mut bq = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
    bq.perspective_in_place(2.);
    assert_eq!(bq.a, 0.5);
    assert_eq!(bq.c, 12.);
}

#[test]
fn test_shift_in_place() {
    let mut bq = BoundedQuadratic::new(0., 2., 1., 1., 1.);
    let orig_a = bq.a;
    let orig_b = bq.b;
    let orig_c = bq.c;
    bq.shift_in_place(3.);
    assert_eq!(bq.lower, 3.);
    assert_eq!(bq.upper, 5.);
    assert_eq!(bq.a, orig_a);
    assert_eq!(bq.b, orig_b - 2. * orig_a * 3.);
    assert_eq!(bq.c, orig_a * 9. - orig_b * 3. + orig_c);
}

#[test]
fn test_scale_arg_in_place() {
    let mut bq = BoundedQuadratic::new(2., 4., 1., -5., 6.);
    bq.scale_arg_in_place(2.);
    assert_eq!(bq.a, 4.);
    assert_eq!(bq.b, -10.);
}

#[test]
fn test_reflect_over_y_in_place() {
    let mut bq = BoundedQuadratic::new(1., 3., 1., 2., 3.);
    bq.reflect_over_y_in_place();
    assert_eq!(bq.lower, -3.);
    assert_eq!(bq.upper, -1.);
    assert_eq!(bq.b, -2.);
}

#[test]
fn test_derivative() {
    let bq = BoundedQuadratic::new(0., 1., 3., 2., 1.);
    let d = bq.derivative();
    assert_eq!(d.a, 0.);
    assert_eq!(d.b, 6.);
    assert_eq!(d.c, 2.);
}

// ─── BoundedQuadratic minimize edge cases ───

#[test]
fn test_minimize_constant_both_infinite() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 0., 5.);
    let (x, val) = bq.minimize();
    assert_eq!(x, 0.);
    assert_eq!(val, 5.);
}

#[test]
fn test_minimize_downward_slope_finite_upper() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, 10., 0., -1., 0.);
    let (x, val) = bq.minimize();
    assert_eq!(x, 10.);
    assert_eq!(val, -10.);
}

#[test]
fn test_minimize_downward_slope_infinite_upper() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., -1., 0.);
    let (x, _val) = bq.minimize();
    assert!(x.is_nan());
}

#[test]
fn test_minimize_constant_lower_finite() {
    let bq = BoundedQuadratic::new(5., f64::INFINITY, 0., 0., 3.);
    let (x, val) = bq.minimize();
    assert_eq!(x, 5.);
    assert_eq!(val, 3.);
}

#[test]
fn test_minimize_constant_upper_finite() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, 7., 0., 0., 3.);
    let (x, val) = bq.minimize();
    assert_eq!(x, 7.);
    assert_eq!(val, 3.);
}

// ─── BoundedQuadratic Display ───

#[test]
fn test_display_zero_function() {
    let bq = BoundedQuadratic::new(0., 1., 0., 0., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("0"));
}

#[test]
fn test_display_full_quadratic() {
    let bq = BoundedQuadratic::new(0., 1., 2., 3., 4.);
    let s = format!("{}", bq);
    assert!(s.contains("x²"));
    assert!(s.contains("x "));
}

#[test]
fn test_display_unit_coefficients() {
    let bq = BoundedQuadratic::new(0., 1., 1., 1., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("x²"));
}

#[test]
fn test_display_negative_coefficients() {
    let bq = BoundedQuadratic::new(0., 1., -1., -1., -1.);
    let s = format!("{}", bq);
    assert!(s.contains("-"));
}

#[test]
fn test_display_infinite_bounds() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 1., 0., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("("));
    assert!(s.contains(")"));
}

#[test]
fn test_display_finite_bounds() {
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("["));
    assert!(s.contains("]"));
}

#[test]
fn test_display_const_only() {
    let bq = BoundedQuadratic::new(0., 1., 0., 0., 5.);
    let s = format!("{}", bq);
    assert!(s.contains("5"));
}

#[test]
fn test_display_linear_only() {
    let bq = BoundedQuadratic::new(0., 1., 0., 2., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("2x"));
}

// ─── PiecewiseQuadratic methods ───

#[test]
fn test_pwq_minimize() {
    let left = BoundedQuadratic::new(-10., 0., 0., -1., 0.);
    let right = BoundedQuadratic::new(0., 10., 0., 1., 0.);
    let abs = PiecewiseQuadratic::new(vec![left, right]);
    let (x, val, _idx) = abs.minimize();
    assert!((x - 0.).abs() < 1e-10);
    assert!((val - 0.).abs() < 1e-10);
}

#[test]
fn test_pwq_scale() {
    let bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let scaled = pwq.scale(3.);
    assert_eq!(scaled[0].a, 3.);
}

#[test]
fn test_pwq_scale_in_place() {
    let bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![bq]);
    pwq.scale_in_place(3.);
    assert_eq!(pwq[0].a, 3.);
}

#[test]
fn test_pwq_scale_arg_negative() {
    let left = BoundedQuadratic::new(-2., 0., 0., -1., 0.);
    let right = BoundedQuadratic::new(0., 2., 0., 1., 0.);
    let pwq = PiecewiseQuadratic::new(vec![left, right]);
    let scaled = pwq.scale_arg(-1.);
    assert_eq!(scaled.len(), 2);
}

#[test]
fn test_pwq_scale_arg_in_place() {
    let bq = BoundedQuadratic::new(1., 2., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![bq]);
    pwq.scale_arg_in_place(2.);
    assert_eq!(pwq[0].a, 4.);
}

#[test]
fn test_pwq_perspective() {
    let bq = BoundedQuadratic::new(1., 2., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let persp = pwq.perspective(2.);
    assert_eq!(persp.len(), 1);
}

#[test]
fn test_pwq_perspective_in_place() {
    let bq = BoundedQuadratic::new(1., 2., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![bq]);
    pwq.perspective_in_place(2.);
    assert_eq!(pwq.len(), 1);
}

#[test]
fn test_pwq_shift() {
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let shifted = pwq.shift(5.);
    assert_eq!(shifted[0].lower, 5.);
    assert_eq!(shifted[0].upper, 6.);
}

#[test]
fn test_pwq_shift_in_place() {
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![bq]);
    pwq.shift_in_place(5.);
    assert_eq!(pwq[0].lower, 5.);
}

#[test]
fn test_pwq_reflect_over_y() {
    let bq = BoundedQuadratic::new(1., 3., 1., 2., 3.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let reflected = pwq.reflect_over_y();
    assert_eq!(reflected[0].lower, -3.);
    assert_eq!(reflected[0].upper, -1.);
}

#[test]
fn test_pwq_reflect_over_y_in_place() {
    let bq = BoundedQuadratic::new(1., 3., 1., 2., 3.);
    let mut pwq = PiecewiseQuadratic::new(vec![bq]);
    pwq.reflect_over_y_in_place();
    assert_eq!(pwq[0].lower, -3.);
}

// ─── PiecewiseQuadratic Zero trait ───

#[test]
fn test_pwq_zero() {
    let z = PiecewiseQuadratic::zero();
    assert!(z.is_zero());
    assert_eq!(z.len(), 1);
}

#[test]
fn test_pwq_is_zero_false() {
    let bq = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    assert!(!pwq.is_zero());
}

// ─── PiecewiseQuadratic conjugate_maximizer ───

#[test]
fn test_conjugate_maximizer_quadratic() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let cm = pwq.conjugate_maximizer();
    assert!(cm.len() >= 1);
}

// ─── PiecewiseQuadratic sum_pwq ───

#[test]
fn test_sum_pwq_syncd() {
    let bq1 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let bq2 = BoundedQuadratic::new(-1., 1., 0., 1., 0.);
    let p1 = PiecewiseQuadratic::new(vec![bq1]);
    let p2 = PiecewiseQuadratic::new(vec![bq2]);
    let mut ws = SyncWorkspace::new(2);
    let result = PiecewiseQuadratic::sum_pwq(&mut ws, &[&p1, &p2], true);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].a, 1.);
    assert_eq!(result[0].b, 1.);
}

#[test]
fn test_sum_pwq_not_syncd() {
    let bq1 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let bq2 = BoundedQuadratic::new(-2., 2., 0., 1., 0.);
    let p1 = PiecewiseQuadratic::new(vec![bq1]);
    let p2 = PiecewiseQuadratic::new(vec![bq2]);
    let mut ws = SyncWorkspace::new(2);
    let result = PiecewiseQuadratic::sum_pwq(&mut ws, &[&p1, &p2], false);
    assert!(result.len() >= 1);
}

// ─── PiecewiseQuadratic new with validation ───

#[test]
fn test_pwq_new_nonempty_ordered() {
    let a = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
    let b = BoundedQuadratic::new(0., 1., 0., 1., 0.);
    let c = BoundedQuadratic::new(1., 2., 0., 1., 1.);
    let pwq = PiecewiseQuadratic::new(vec![a, b, c]);
    assert_eq!(pwq.len(), 3);
}

// ─── opto structs coverage ───

#[derive(Clone)]
struct TestObjRes;

impl Objective for TestObjRes {
    fn objective(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
        use ndarray::linalg::Dot;
        let f_obj =
            vars.x.t().dot(&problem_data.p.dot(&vars.x)) + problem_data.q.t().dot(&vars.x);
        let g_obj: f64 = (0..problem_data.n_vars())
            .map(|i| problem_data.g[i].eval(vars.x[i]))
            .sum();
        f_obj + g_obj
    }
}

impl Residual for TestObjRes {
    fn residual(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
        (&problem_data.a.dot(&vars.x) - &problem_data.b).sum().abs()
    }
}

#[test]
fn test_settings_defaults() {
    let s = Settings::defaults(3, 2);
    assert_eq!(s.n_vars(), 3);
    assert_eq!(s.n_constrs(), 2);
}

#[test]
fn test_stats_update_with_compute_stats() {
    let settings = Settings::new(1., array![1.], array![1., 1.], 50, 2, true);
    let problem_data = ProblemData::new(
        CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON),
        array![0., 0.],
        CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON),
        array![0.],
        array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ],
    );
    let vars = Variables::from_problem_data(&problem_data);
    let mut stats = Stats::new(50);
    stats.update(&vars, &problem_data, &TestObjRes, settings.compute_stats);
    assert_eq!(stats.objective.len(), 1);
    assert_eq!(stats.residual.len(), 1);
}

#[test]
fn test_variables_display() {
    let vars = Variables::default(1, 2);
    let s = format!("{}", vars);
    assert!(s.contains("x:"));
    assert!(s.contains("z:"));
}

// ─── admm::optimize (the top-level function) ───

#[test]
fn test_optimize_top_level() {
    let a = CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON);
    let b = array![0.];
    let p = CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON);
    let q = array![1., 1.];
    let g = array![
        PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
    ];
    let (sol, stats) = admm::optimize(a, b, p, q, g, &TestObjRes, false);
    assert!(sol.x[0] < 0.);
    assert!(sol.x[1] < 0.);
    assert!(stats.iters > 0);
}

// ─── utils clamp upper branch ───

#[test]
fn test_clamp_above_max() {
    use lcso::quadratics::bq::BoundedQuadratic;
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    // This calls clamp indirectly through minimize with a>0
    // Minimizer of x^2 is at x=0, which is clamped to [0,1]
    let (x, _) = bq.minimize();
    assert_eq!(x, 0.);

    // Force the upper clamp path: upward-opening quad with min at x=5, domain [0,3]
    let bq2 = BoundedQuadratic::new(0., 3., 1., -10., 0.);
    let (x2, _) = bq2.minimize();
    assert_eq!(x2, 3.);
}

// ─── BoundedQuadratic PartialEq for points ───

#[test]
fn test_partial_eq_points() {
    let p1 = BoundedQuadratic::new(1., 1., 1., 0., 0.);
    let p2 = BoundedQuadratic::new(1., 1., 0., 0., 1.);
    assert_eq!(p1, p2);
}

// ─── PiecewiseQuadratic eval edge cases ───

#[test]
fn test_pwq_eval_at_boundary() {
    let left = BoundedQuadratic::new(0., 1., 0., 0., 1.);
    let right = BoundedQuadratic::new(1., 2., 0., 0., 2.);
    let pwq = PiecewiseQuadratic::new(vec![left, right]);
    let val = pwq.eval(1.);
    assert_eq!(val, 1.);
}

#[test]
fn test_pwq_eval_out_of_domain() {
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let val = pwq.eval(5.);
    assert_eq!(val, f64::INFINITY);
}

#[test]
fn test_pwq_extends_left_right() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    assert!(pwq.extends_left());
    assert!(pwq.extends_right());
}

// ─── PiecewiseQuadratic is_convex ───

#[test]
fn test_pwq_is_convex_single_piece() {
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    assert!(pwq.is_convex());
}

#[test]
fn test_pwq_is_convex_multi_piece() {
    let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
    let right = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
    let abs = PiecewiseQuadratic::new(vec![left, right]);
    assert!(abs.is_convex());
}

// ─── PiecewiseQuadratic simplify edge cases ───

#[test]
fn test_pwq_simplify_single() {
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![bq]);
    let s = pwq.simplify();
    assert_eq!(s.len(), 1);
}
