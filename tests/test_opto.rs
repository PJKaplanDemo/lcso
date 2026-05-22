use lcso::opto::structs::{HasConstraints, HasVariables, ProblemData, Settings, Variables};
use lcso::quadratics::pwq::PiecewiseQuadratic;
use ndarray::{array, Array1};
use sprs::CsMat;
use std::f64;

#[test]
fn test_settings_defaults() {
    let settings = Settings::defaults(3, 2);
    assert_eq!(settings.n_vars(), 3);
    assert_eq!(settings.n_constrs(), 2);
    assert_eq!(settings.max_iter, 10000);
    assert!((settings.alpha - 1.5).abs() < 1e-10);
}

#[test]
fn test_settings_new() {
    let alpha = 1.0;
    let rho = Array1::ones(2);
    let sigma = Array1::ones(3);
    let settings = Settings::new(alpha, rho, sigma, 500, 10, true);
    assert_eq!(settings.max_iter, 500);
    assert!(settings.compute_stats);
}

#[test]
fn test_variables_default() {
    let vars = Variables::default(3, 2);
    assert_eq!(vars.x.len(), 2);
    assert_eq!(vars.z.len(), 3);
    assert_eq!(vars.w.len(), 2);
    assert_eq!(vars.y.len(), 3);
}

#[test]
fn test_variables_new() {
    let x = array![1.0, 2.0];
    let xt = array![1.0, 2.0];
    let w = array![0.0, 0.0];
    let z = array![3.0];
    let zt = array![3.0];
    let y = array![0.0];
    let vars = Variables::new(xt, zt, x, z, w, y);
    assert_eq!(vars.x.len(), 2);
    assert_eq!(vars.z.len(), 1);
}

#[test]
fn test_problem_data_new() {
    let p = CsMat::csc_from_dense(array![[1.0, 0.0], [0.0, 2.0]].view(), f64::EPSILON);
    let q = array![1.0, 2.0];
    let a = CsMat::csc_from_dense(array![[1.0, 1.0]].view(), f64::EPSILON);
    let b = array![1.0];
    let g = array![
        PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
    ];
    let data = ProblemData::new(p, q, a, b, g);
    assert_eq!(data.n_vars(), 2);
    assert_eq!(data.n_constrs(), 1);
}

#[test]
fn test_stats_new() {
    use lcso::opto::structs::Stats;
    let stats = Stats::new(100);
    assert_eq!(stats.iters, 0);
    assert_eq!(stats.prox_iters, 0);
    assert_eq!(stats.solve_time_ms, 0);
}
