use burn::Tensor;
use catalyst_lib::math::*;

mod utils;
use utils::*;

#[test]
fn test_cholesky() {
    // Test with a simple 3x3 symmetric positive-definite matrix
    let a = Tensor::<B, 2>::from_floats(
        [
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ],
        &Default::default(),
    );

    let l = cholesky(a.clone());

    // Verify A = L·L^T
    let reconstructed = l.clone().matmul(l.clone().transpose());

    let tolerance = Tensor::<B, 2>::from_floats(
        [[1e-5, 1e-5, 1e-5], [1e-5, 1e-5, 1e-5], [1e-5, 1e-5, 1e-5]],
        &Default::default(),
    );
    let equal = (reconstructed.clone() - a.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;
    assert!(
        equal,
        "Reconstruction L·L^T differs from A beyond tolerance.\nA: {}\nL·L^T: {}",
        a.to_data(),
        reconstructed.to_data()
    );
}

#[test]
fn test_cholesky_solve() {
    let a = Tensor::<B, 2>::from_floats(
        [
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0],
        ],
        &Default::default(),
    );

    let b = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &Default::default());

    let l = cholesky(a.clone());
    let x = cholesky_solve(l, b.clone());

    println!("Solution x:\n{}", x);

    // Verify Ax = b
    let result = a.matmul(x.clone().reshape([3, 1])).reshape([3]);
    let tolerance = Tensor::<B, 1>::from_floats([1e-4, 1e-4, 1e-4], &Default::default());
    let equal = (result.clone() - b.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;
    assert!(
        equal,
        "Solution Ax differs from b beyond tolerance.\nAx: {}\nb: {}",
        result.to_data(),
        b.to_data()
    );
}

#[test]
fn test_cholesky_simple() {
    // Simple 2x2 case
    let a = Tensor::<B, 2>::from_floats([[4.0, 2.0], [2.0, 3.0]], &Default::default());

    let l = cholesky(a.clone());
    let reconstructed = l.clone().matmul(l.clone().transpose());

    let tolerance = Tensor::<B, 2>::from_floats([[1e-5, 1e-5], [1e-5, 1e-5]], &Default::default());
    let equal = (reconstructed.clone() - a.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;
    assert!(
        equal,
        "Reconstruction L·L^T differs from A beyond tolerance.\nA: {}\nL·L^T: {}",
        a.to_data(),
        reconstructed.to_data()
    );
}
