use lcso::quadratics::bq::BoundedQuadratic;
use lcso::quadratics::pwq::PiecewiseQuadratic;
use std::f64;

#[test]
fn test_bq_new_creates_valid_quadratic() {
    let bq = BoundedQuadratic::new(-1.0, 1.0, 1.0, 2.0, 3.0);
    assert_eq!(bq.lower, -1.0);
    assert_eq!(bq.upper, 1.0);
    assert_eq!(bq.a, 1.0);
    assert_eq!(bq.b, 2.0);
    assert_eq!(bq.c, 3.0);
}

#[test]
fn test_bq_new_extended() {
    let bq = BoundedQuadratic::new_extended(1.0, 0.0, 0.0);
    assert_eq!(bq.lower, f64::NEG_INFINITY);
    assert_eq!(bq.upper, f64::INFINITY);
    assert_eq!(bq.a, 1.0);
}

#[test]
fn test_bq_new_line() {
    let line = BoundedQuadratic::new_line(-5.0, 5.0, 2.0, 1.0);
    assert_eq!(line.a, 0.0);
    assert_eq!(line.b, 2.0);
    assert_eq!(line.c, 1.0);
}

#[test]
fn test_bq_new_point() {
    let pt = BoundedQuadratic::new_point(3.0, 7.0);
    assert_eq!(pt.lower, 3.0);
    assert_eq!(pt.upper, 3.0);
    assert_eq!(pt.c, 7.0);
}

#[test]
fn test_bq_is_convex() {
    let convex = BoundedQuadratic::new(-1.0, 1.0, 1.0, 0.0, 0.0);
    assert!(convex.is_convex());

    let concave = BoundedQuadratic::new(-1.0, 1.0, -1.0, 0.0, 0.0);
    assert!(!concave.is_convex());
}

#[test]
fn test_bq_is_affine() {
    let affine = BoundedQuadratic::new(-1.0, 1.0, 0.0, 1.0, 2.0);
    assert!(affine.is_affine());

    let non_affine = BoundedQuadratic::new(-1.0, 1.0, 1.0, 1.0, 2.0);
    assert!(!non_affine.is_affine());
}

#[test]
fn test_bq_is_empty() {
    let empty = BoundedQuadratic::new(2.0, 1.0, 0.0, 0.0, 0.0);
    assert!(empty.is_empty());

    let non_empty = BoundedQuadratic::new(-1.0, 1.0, 0.0, 0.0, 0.0);
    assert!(!non_empty.is_empty());
}

#[test]
fn test_pwq_abs_value() {
    let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0.0, 0.0, -1.0, 0.0);
    let right = BoundedQuadratic::new(0.0, f64::INFINITY, 0.0, 1.0, 0.0);
    let abs = PiecewiseQuadratic::new(vec![left, right]);
    assert_eq!(abs.functions.len(), 2);
}

#[test]
fn test_pwq_indicator() {
    let indicator = PiecewiseQuadratic::indicator(0.0, 1.0);
    assert!(!indicator.functions.is_empty());
}

#[test]
fn test_pwq_empty_with_capacity() {
    let mut pwq = PiecewiseQuadratic::new_empty_with_capacity(5);
    assert_eq!(pwq.functions.len(), 0);
    let bq = BoundedQuadratic::new(0.0, 1.0, 1.0, 0.0, 0.0);
    pwq.add_piece_at_right(bq);
    assert_eq!(pwq.functions.len(), 1);
}

#[test]
fn test_bq_eval() {
    let bq = BoundedQuadratic::new(-10.0, 10.0, 1.0, 0.0, 0.0);
    let val = bq.eval(2.0);
    assert!((val - 4.0).abs() < 1e-10);
}

#[test]
fn test_bq_line_from_points() {
    let line = BoundedQuadratic::new_line_from_points((0.0, 0.0), (1.0, 1.0));
    assert!((line.b - 1.0).abs() < 1e-10);
    assert!((line.c - 0.0).abs() < 1e-10);
}

#[test]
fn test_pwq_new_empty() {
    let pwq = PiecewiseQuadratic::new(vec![]);
    assert_eq!(pwq.functions.len(), 0);
}
