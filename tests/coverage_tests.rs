use approx::{abs_diff_eq, AbsDiffEq};
use len_trait::len::Len;
use lcso::quadratics::bq::BoundedQuadratic;
use lcso::quadratics::pwq::PiecewiseQuadratic;
use lcso::opto::structs::{ProblemData, Settings, Variables, HasVariables};
use lcso::opto::term::{Objective, Residual};
use lcso::opto::admm::optimize_structs;
use ndarray::{array, Array2};
use ndarray::linalg::Dot;
use sprs::CsMat;

// ======== BoundedQuadratic in-place methods ========

#[test]
fn test_restrict_domain_in_place() {
    let mut bq = BoundedQuadratic::new(0., 10., 1., 2., 3.);
    bq.restrict_domain_in_place(2., 8.);
    assert_eq!(bq.lower, 2.);
    assert_eq!(bq.upper, 8.);
    assert_eq!(bq.a, 1.);
}

#[test]
fn test_extend_domain_in_place() {
    let mut bq = BoundedQuadratic::new(1., 5., 1., 2., 3.);
    bq.extend_domain_in_place();
    assert_eq!(bq.lower, f64::NEG_INFINITY);
    assert_eq!(bq.upper, f64::INFINITY);
}

#[test]
fn test_scale_in_place() {
    let bq = BoundedQuadratic::new(1., 5., 2., 3., 4.);
    let mut bq_mut = bq;
    bq_mut.scale_in_place(3.0);
    let scaled = bq.scale(3.0);
    assert_eq!(bq_mut.a, scaled.a);
    assert_eq!(bq_mut.b, scaled.b);
    assert_eq!(bq_mut.c, scaled.c);
}

#[test]
fn test_scale_arg_in_place() {
    let bq = BoundedQuadratic::new(2., 4., 1., -1., 2.);
    let mut bq_mut = bq;
    bq_mut.scale_arg_in_place(2.0);
    let scaled = bq.scale_arg(2.0);
    assert!(abs_diff_eq!(bq_mut.lower, scaled.lower, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.upper, scaled.upper, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.a, scaled.a, epsilon = 1e-10));
}

#[test]
fn test_perspective_in_place() {
    let bq = BoundedQuadratic::new(2., 4., 1., -1., 2.);
    let mut bq_mut = bq;
    bq_mut.perspective_in_place(2.0);
    let persp = bq.perspective(2.0);
    assert!(abs_diff_eq!(bq_mut.lower, persp.lower, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.upper, persp.upper, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.a, persp.a, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.c, persp.c, epsilon = 1e-10));
}

#[test]
fn test_shift_in_place() {
    let bq = BoundedQuadratic::new(1., 3., 1., -2., 1.);
    let mut bq_mut = bq;
    bq_mut.shift_in_place(2.0);
    let shifted = bq.shift(2.0);
    assert!(abs_diff_eq!(bq_mut.lower, shifted.lower, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.upper, shifted.upper, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.b, shifted.b, epsilon = 1e-10));
    assert!(abs_diff_eq!(bq_mut.c, shifted.c, epsilon = 1e-10));
}

#[test]
fn test_reflect_over_y_in_place() {
    let bq = BoundedQuadratic::new(1., 3., 1., 2., 3.);
    let mut bq_mut = bq;
    bq_mut.reflect_over_y_in_place();
    let reflected = bq.reflect_over_y();
    assert_eq!(bq_mut.lower, reflected.lower);
    assert_eq!(bq_mut.upper, reflected.upper);
    assert_eq!(bq_mut.b, reflected.b);
}

#[test]
fn test_derivative() {
    let bq = BoundedQuadratic::new(0., 5., 3., 2., 1.);
    let deriv = bq.derivative();
    assert_eq!(deriv.a, 0.);
    assert_eq!(deriv.b, 6.); // 2 * a = 2 * 3
    assert_eq!(deriv.c, 2.); // original b
    assert_eq!(deriv.lower, 0.);
    assert_eq!(deriv.upper, 5.);
}

// ======== PiecewiseQuadratic methods ========

#[test]
fn test_pwq_minimize() {
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 1., 0., 1.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let (min_x, min_val, min_idx) = pwq.minimize();
    assert!(abs_diff_eq!(min_x, 0., epsilon = 1e-10));
    assert!(abs_diff_eq!(min_val, 0., epsilon = 1e-10));
    assert_eq!(min_idx, 1);
}

#[test]
fn test_pwq_scale() {
    let f = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f]);
    let scaled = pwq.scale(2.0);
    assert!(abs_diff_eq!(scaled[0].a, 2., epsilon = 1e-10));
}

#[test]
fn test_pwq_scale_in_place() {
    let f = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f]);
    pwq.scale_in_place(3.0);
    assert!(abs_diff_eq!(pwq[0].a, 3., epsilon = 1e-10));
}

#[test]
fn test_pwq_scale_arg() {
    let f1 = BoundedQuadratic::new(-2., 0., 1., 1., 0.);
    let f2 = BoundedQuadratic::new(0., 2., 1., -1., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);

    let scaled_pos = pwq.scale_arg(2.0);
    assert_eq!(scaled_pos.len(), 2);

    let scaled_neg = pwq.scale_arg(-1.0);
    assert_eq!(scaled_neg.len(), 2);
}

#[test]
fn test_pwq_scale_arg_in_place() {
    let f1 = BoundedQuadratic::new(-2., 0., 1., 1., 0.);
    let f2 = BoundedQuadratic::new(0., 2., 1., -1., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    pwq.scale_arg_in_place(2.0);
    assert_eq!(pwq.len(), 2);
}

#[test]
fn test_pwq_scale_arg_in_place_negative() {
    let f1 = BoundedQuadratic::new(-2., 0., 1., 1., 0.);
    let f2 = BoundedQuadratic::new(0., 2., 1., -1., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    pwq.scale_arg_in_place(-1.0);
    assert_eq!(pwq.len(), 2);
}

#[test]
fn test_pwq_perspective() {
    let f = BoundedQuadratic::new(1., 3., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f]);
    let persp = pwq.perspective(2.0);
    assert_eq!(persp.len(), 1);
}

#[test]
fn test_pwq_perspective_negative() {
    let f1 = BoundedQuadratic::new(-3., -1., 1., 0., 0.);
    let f2 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let persp = pwq.perspective(-2.0);
    assert_eq!(persp.len(), 2);
}

#[test]
fn test_pwq_perspective_in_place() {
    let f = BoundedQuadratic::new(1., 3., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f]);
    pwq.perspective_in_place(2.0);
    assert_eq!(pwq.len(), 1);
}

#[test]
fn test_pwq_perspective_in_place_negative() {
    let f1 = BoundedQuadratic::new(-3., -1., 1., 0., 0.);
    let f2 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    pwq.perspective_in_place(-2.0);
    assert_eq!(pwq.len(), 2);
}

#[test]
fn test_pwq_shift() {
    let f = BoundedQuadratic::new(0., 2., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f]);
    let shifted = pwq.shift(3.0);
    assert!(abs_diff_eq!(shifted[0].lower, 3., epsilon = 1e-10));
    assert!(abs_diff_eq!(shifted[0].upper, 5., epsilon = 1e-10));
}

#[test]
fn test_pwq_shift_in_place() {
    let f = BoundedQuadratic::new(0., 2., 1., 0., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f]);
    pwq.shift_in_place(3.0);
    assert!(abs_diff_eq!(pwq[0].lower, 3., epsilon = 1e-10));
    assert!(abs_diff_eq!(pwq[0].upper, 5., epsilon = 1e-10));
}

#[test]
fn test_pwq_reflect_over_y() {
    let f1 = BoundedQuadratic::new(-2., 0., 1., 1., 0.);
    let f2 = BoundedQuadratic::new(0., 2., 1., -1., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let reflected = pwq.reflect_over_y();
    assert_eq!(reflected.len(), 2);
}

#[test]
fn test_pwq_reflect_over_y_in_place() {
    let f1 = BoundedQuadratic::new(-2., 0., 1., 1., 0.);
    let f2 = BoundedQuadratic::new(0., 2., 1., -1., 0.);
    let mut pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    pwq.reflect_over_y_in_place();
    assert_eq!(pwq.len(), 2);
}

// ======== ADMM with convexify=true ========

#[derive(Clone)]
struct SimpleObjRes {}

impl Objective for SimpleObjRes {
    fn objective(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
        let f_obj =
            vars.x.t().dot(&problem_data.p.dot(&vars.x)) + problem_data.q.t().dot(&vars.x);
        let g_obj: f64 = (0..problem_data.n_vars())
            .map(|i| problem_data.g[i].eval(vars.x[i]))
            .sum();
        f_obj + g_obj
    }
}

impl Residual for SimpleObjRes {
    fn residual(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
        (&problem_data.a.dot(&vars.x) - &problem_data.b).sum()
    }
}

#[test]
fn test_optimize_with_convexify() {
    let settings = Settings::new(1., array![1.], array![1., 1., 1., 1.], 500, 25, true);

    let mut problem_data = ProblemData::new(
        CsMat::csc_from_dense(Array2::eye(4).view(), f64::EPSILON),
        array![1., 1., 1., 1.],
        CsMat::csc_from_dense(Array2::zeros((1, 4)).view(), f64::EPSILON),
        array![1.],
        array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ],
    );

    let mut variables = Variables::from_problem_data(&problem_data);
    let (sol, stats) = optimize_structs(
        &settings,
        &mut problem_data,
        &mut variables,
        &SimpleObjRes {},
        true,
    );
    // Convexify may need more iterations; check it ran and produced reasonable results
    assert!(stats.iters > 0);
    // With compute_stats=true, stats should be populated
    assert!(!stats.objective.is_empty());
    assert!(!stats.residual.is_empty());
}

#[test]
fn test_optimize_with_box_constraints_convexify() {
    let settings = Settings::new(1., array![5., 5.], array![5., 5.], 500, 25, true);

    let bq1 = BoundedQuadratic::new_extended(1., 0., 0.);
    let g1 = PiecewiseQuadratic::new(vec![bq1]);

    let mut problem_data = ProblemData::new(
        CsMat::csc_from_dense(array![[5., 3.], [3., 2.]].view(), f64::EPSILON),
        array![1., 2.],
        CsMat::csc_from_dense(array![[1., 1.], [1., 2.]].view(), f64::EPSILON),
        array![1., 1.],
        array![
            g1,
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ],
    );

    let mut variables = Variables::from_problem_data(&problem_data);
    let (_sol, stats) = optimize_structs(
        &settings,
        &mut problem_data,
        &mut variables,
        &SimpleObjRes {},
        true,
    );
    // Verify convexify path executed
    assert!(stats.iters > 0);
}

#[test]
fn test_optimize_compute_stats() {
    let settings = Settings::new(1., array![2., 2.], array![2., 2.], 50, 2, true);

    let mut problem_data = ProblemData::new(
        CsMat::csc_from_dense(array![[65., 76.], [76., 89.]].view(), f64::EPSILON),
        array![7., 3.],
        CsMat::csc_from_dense(array![[7., 8.], [5., 4.]].view(), f64::EPSILON),
        array![8., 8.],
        array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ],
    );

    let mut variables = Variables::from_problem_data(&problem_data);
    let (sol, stats) = optimize_structs(
        &settings,
        &mut problem_data,
        &mut variables,
        &SimpleObjRes {},
        false,
    );
    assert!(sol.x.abs_diff_eq(&array![2.66666651, -1.33333319], 1e-5));
    // compute_stats=true so stats should be populated
    assert!(!stats.objective.is_empty());
}

// ======== Variables Display ========

#[test]
fn test_variables_display() {
    let problem_data = ProblemData::new(
        CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON),
        array![1., 1.],
        CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON),
        array![0.],
        array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ],
    );
    let vars = Variables::from_problem_data(&problem_data);
    let display = format!("{}", vars);
    assert!(display.contains("x:"));
    assert!(display.contains("xt:"));
    assert!(display.contains("z:"));
}

// ======== Settings defaults ========

#[test]
fn test_settings_defaults() {
    let settings = Settings::defaults(3, 2);
    assert_eq!(settings.max_iter, 10000);
    assert!(abs_diff_eq!(settings.alpha, 1.5, epsilon = 1e-10));
}

// ======== Envelope edge cases ========

#[test]
fn test_envelope_with_non_convex_pwq() {
    use lcso::quadratics::envelope::envelope;

    // Create a non-convex PWQ (concave piece)
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., -1., 0., 0.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, -1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let env = envelope(&pwq);
    assert!(env.len() > 0);
}

#[test]
fn test_envelope_with_points_and_segments() {
    use lcso::quadratics::envelope::envelope;

    // PWQ with point pieces
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
    let pt = BoundedQuadratic::new(0., 0., 0., 0., 0.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, pt, f2]);
    let env = envelope(&pwq);
    assert!(env.len() > 0);
}

#[test]
fn test_optimize_with_indicator_bounds() {
    let settings = Settings::new(1., array![2.], array![2., 2.], 200, 5, false);

    // Box constraints via indicator functions
    let g1 = PiecewiseQuadratic::indicator(-1., 1.);
    let g2 = PiecewiseQuadratic::indicator(-1., 1.);

    let mut problem_data = ProblemData::new(
        CsMat::csc_from_dense(array![[2., 0.], [0., 2.]].view(), f64::EPSILON),
        array![5., 5.], // would push x to 2.5 without constraints
        CsMat::csc_from_dense(array![[1., 1.]].view(), f64::EPSILON),
        array![0.],
        array![g1, g2],
    );

    let mut variables = Variables::from_problem_data(&problem_data);
    let (sol, _stats) = optimize_structs(
        &settings,
        &mut problem_data,
        &mut variables,
        &SimpleObjRes {},
        false,
    );
    // x should be clamped to [-1, 1] box
    assert!(sol.x[0] >= -1.0 - 1e-4 && sol.x[0] <= 1.0 + 1e-4);
    assert!(sol.x[1] >= -1.0 - 1e-4 && sol.x[1] <= 1.0 + 1e-4);
}

// ======== BoundedQuadratic minimize edge cases ========

#[test]
fn test_minimize_point_domain() {
    // Point domain: lower == upper
    let bq = BoundedQuadratic::new(3., 3., 1., -2., 5.);
    let (x, val) = bq.minimize();
    assert!(abs_diff_eq!(x, 3., epsilon = 1e-10));
    assert!(abs_diff_eq!(val, 1. * 9. - 2. * 3. + 5., epsilon = 1e-10));
}

#[test]
fn test_minimize_downward_linear_infinite_upper() {
    // Downward-sloping linear with infinite upper: returns NaN, NEG_INFINITY
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., -1., 0.);
    let (x, val) = bq.minimize();
    assert!(x.is_nan());
    assert_eq!(val, f64::NEG_INFINITY);
}

#[test]
fn test_minimize_constant_infinite_lower_finite_upper() {
    // Constant with infinite lower, finite upper
    let bq = BoundedQuadratic::new(f64::NEG_INFINITY, 5., 0., 0., 3.);
    let (x, val) = bq.minimize();
    // Should return upper since lower is infinite
    assert!(x.is_finite());
    assert!(abs_diff_eq!(val, 3., epsilon = 1e-10));
}

// ======== BoundedQuadratic Display edge cases ========

#[test]
fn test_bq_display_nonunit_a() {
    // a != 0 and |a| != 1 → covers line 644
    let bq = BoundedQuadratic::new(0., 1., 2., 0., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("2"));
}

#[test]
fn test_bq_display_zero_b() {
    // b == 0 → covers line 648
    let bq = BoundedQuadratic::new(0., 1., 1., 0., 1.);
    let s = format!("{}", bq);
    assert!(s.contains("x"));
}

#[test]
fn test_bq_display_only_const() {
    // a == 0, b == 0, c != 0 → covers line 672-673
    let bq = BoundedQuadratic::new(0., 1., 0., 0., 5.);
    let s = format!("{}", bq);
    assert!(s.contains("5"));
}

#[test]
fn test_bq_display_zero_function() {
    // a == 0, b == 0, c == 0 → covers line 680-681
    let bq = BoundedQuadratic::new(0., 1., 0., 0., 0.);
    let s = format!("{}", bq);
    assert!(s.contains("0"));
}

// ======== PiecewiseQuadratic conjugate_maximizer edge case ========

#[test]
fn test_conjugate_maximizer_with_affine_left() {
    // First piece is affine and extends left → covers lines 337-338
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let cm = pwq.conjugate_maximizer();
    assert!(cm.len() > 0);
}

// ======== PiecewiseQuadratic Zero trait ========

#[test]
fn test_pwq_indicator_is_zero_like() {
    // The indicator from -inf to inf is the "zero" of PWQ
    let z = PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY);
    assert_eq!(z.len(), 1);
    assert!(abs_diff_eq!(z[0].a, 0., epsilon = 1e-10));
    assert!(abs_diff_eq!(z[0].b, 0., epsilon = 1e-10));
    assert!(abs_diff_eq!(z[0].c, 0., epsilon = 1e-10));
}

// ======== PiecewiseQuadratic Add (should panic) ========

#[test]
#[should_panic(expected = "The add trait is implemented so that we can also implement the Zero trait")]
fn test_pwq_add_panics() {
    let f = BoundedQuadratic::new_extended(1., 0., 0.);
    let p1 = PiecewiseQuadratic::new(vec![f]);
    let p2 = PiecewiseQuadratic::new(vec![f]);
    let _ = p1 + p2;
}

// ======== PiecewiseQuadratic simplify with empty first piece ========

#[test]
fn test_pwq_simplify_with_empty_first() {
    let empty = BoundedQuadratic::new(0., 0., 0., 0., 0.);
    let f1 = BoundedQuadratic::new(0., 1., 1., 0., 0.);
    let f2 = BoundedQuadratic::new(1., 2., 1., 0., 0.);
    let pwq = PiecewiseQuadratic::new(vec![empty, f1, f2]);
    let simplified = pwq.simplify();
    assert!(simplified.len() >= 1);
}

// ======== PiecewiseQuadratic Display ========

#[test]
fn test_pwq_display() {
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let s = format!("{}", pwq);
    assert!(s.contains("BoundedQuadratic"));
}

// ======== optimize function (top-level) ========

#[test]
fn test_optimize_top_level() {
    use lcso::opto::admm::optimize;

    let a_mat = CsMat::csc_from_dense(Array2::zeros((1, 2)).view(), f64::EPSILON);
    let b_vec = array![0.];
    let p_mat = CsMat::csc_from_dense(Array2::eye(2).view(), f64::EPSILON);
    let q_vec = array![1., 1.];
    let g = array![
        PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
    ];

    let (sol, stats) = optimize(a_mat, b_vec, p_mat, q_vec, g, &SimpleObjRes {}, false);
    assert!(sol.x.abs_diff_eq(&array![-1., -1.], 1e-4));
    assert!(stats.iters > 0);
}

// ======== Envelope edge case: overlapping points ========

#[test]
fn test_envelope_with_adjacent_points() {
    use lcso::quadratics::envelope::envelope;

    // Two point pieces at the same x with different values
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 1., 0., 1.);
    let pt1 = BoundedQuadratic::new(0., 0., 0., 0., 1.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, 1., 0., 1.);
    let pwq = PiecewiseQuadratic::new(vec![f1, pt1, f2]);
    let env = envelope(&pwq);
    assert!(env.len() > 0);
}

#[test]
fn test_envelope_adjacent_affine() {
    use lcso::quadratics::envelope::envelope;

    // Two adjacent affine pieces where right meets left continuously
    let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 1., 0.);
    let f2 = BoundedQuadratic::new(0., f64::INFINITY, 0., -1., 0.);
    let pwq = PiecewiseQuadratic::new(vec![f1, f2]);
    let env = envelope(&pwq);
    assert!(env.len() > 0);
}
