use burn::{Tensor, backend::ndarray::NdArrayDevice};
use catalyst::*;

mod utils;
use utils::*;

#[test]
fn test_rbf_kernel_at_zero() {
    let backend = instantiate_backend();
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &backend);
    let kernel = Kernel::<B>::new(KernelKind::Rbf, vec![1.0, 1.0], 2.5, &backend);
    let k = kernel.execute(x.clone(), x);

    // At distance 0, RBF kernel should equal variance
    let expected = Tensor::<B, 2>::from_floats([[2.5]], &backend);
    let diff = (k.clone() - expected.clone()).abs().into_scalar();
    assert!(
        diff < 1e-5,
        "RBF at zero: Expected {}, got {}",
        2.5,
        k.into_scalar()
    );
}

#[test]
fn test_rbf_kernel_basic() {
    let backend = instantiate_backend();
    let x1 = Tensor::<B, 2>::from_floats([[0.0, 0.0], [1.0, 0.0]], &backend);
    let x2 = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &backend);
    let kernel = Kernel::<B>::new(KernelKind::Rbf, vec![1.0, 1.0], 1.0, &backend);
    let k = kernel.execute(x1, x2);

    // Distance from [0,0] to [0,0]: d² = 0, k = exp(0) = 1.0
    // Distance from [1,0] to [0,0]: d² = 1, k = exp(-0.5 × 1) ≈ 0.6065
    let expected = Tensor::<B, 2>::from_floats([[1.0], [0.6065]], &backend);
    let tolerance = Tensor::<B, 2>::from_floats([[1e-3], [1e-3]], &backend);

    let equal = (k.clone() - expected.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;

    assert!(
        equal,
        "RBF kernel: Computed {} and expected {} differ",
        k.to_data(),
        expected.to_data(),
    );
}

#[test]
fn test_matern_nu1_2_at_zero() {
    let backend = instantiate_backend();
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &backend);
    let kernel = Kernel::<B>::new(
        KernelKind::Matern(NuKind::Nu1_2),
        vec![1.0, 1.0],
        2.5,
        &backend,
    );
    let k = kernel.execute(x.clone(), x);

    let expected = Tensor::<B, 2>::from_floats([[2.5]], &backend);
    let diff = (k.clone() - expected.clone()).abs().into_scalar();
    assert!(
        diff < 1e-5,
        "Matérn 1/2 at zero: Expected {}, got {}",
        2.5,
        k.into_scalar()
    );
}

#[test]
fn test_matern_nu1_2_basic() {
    let backend = instantiate_backend();
    let x1 = Tensor::<B, 2>::from_floats([[0.0, 0.0], [1.0, 0.0]], &backend);
    let x2 = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &backend);
    let kernel = Kernel::<B>::new(
        KernelKind::Matern(NuKind::Nu1_2),
        vec![1.0, 1.0],
        1.0,
        &backend,
    );
    let k = kernel.execute(x1, x2);

    // Matérn 1/2: k(d) = exp(-d)
    // Distance from [0,0] to [0,0]: d = 0, k = exp(0) = 1.0
    // Distance from [1,0] to [0,0]: d = 1, k = exp(-1) ≈ 0.3679
    let expected = Tensor::<B, 2>::from_floats([[1.0], [0.3679]], &backend);
    let tolerance = Tensor::<B, 2>::from_floats([[1e-2], [1e-2]], &backend);

    let equal = (k.clone() - expected.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;

    assert!(
        equal,
        "Matérn 1/2: Computed {} and expected {} differ",
        k.to_data(),
        expected.to_data(),
    );
}

#[test]
fn test_matern_nu3_2_at_zero() {
    let backend = instantiate_backend();
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &backend);
    let kernel = Kernel::<B>::new(
        KernelKind::Matern(NuKind::Nu3_2),
        vec![1.0, 1.0],
        2.5,
        &backend,
    );
    let k = kernel.execute(x.clone(), x);

    let expected = Tensor::<B, 2>::from_floats([[2.5]], &backend);
    let diff = (k.clone() - expected.clone()).abs().into_scalar();
    assert!(
        diff < 1e-5,
        "Matérn 3/2 at zero: Expected {}, got {}",
        2.5,
        k.into_scalar()
    );
}

#[test]
fn test_matern_nu3_2_basic() {
    let backend = instantiate_backend();
    let x1 = Tensor::<B, 2>::from_floats([[0.0, 0.0], [1.0, 0.0]], &backend);
    let x2 = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &backend);
    let kernel = Kernel::<B>::new(
        KernelKind::Matern(NuKind::Nu3_2),
        vec![1.0, 1.0],
        1.0,
        &backend,
    );
    let k = kernel.execute(x1, x2);

    // Matérn 3/2: k(d) = (1 + √3×d) × exp(-√3×d)
    // Distance from [0,0] to [0,0]: d = 0, k = 1.0
    // Distance from [1,0] to [0,0]: d = 1
    // r = √3 ≈ 1.732, k = (1 + 1.732) × exp(-1.732) ≈ 2.732 × 0.1769 ≈ 0.4832
    let expected = Tensor::<B, 2>::from_floats([[1.0], [0.4832]], &backend);
    let tolerance = Tensor::<B, 2>::from_floats([[1e-2], [1e-2]], &backend);

    let equal = (k.clone() - expected.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;

    assert!(
        equal,
        "Matérn 3/2: Computed {} and expected {} differ",
        k.to_data(),
        expected.to_data(),
    );
}

#[test]
fn test_matern_nu5_2_at_zero() {
    let backend = instantiate_backend();
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &backend);
    let kernel = Kernel::<B>::new(
        KernelKind::Matern(NuKind::Nu5_2),
        vec![1.0, 1.0],
        2.5,
        &backend,
    );
    let k = kernel.execute(x.clone(), x);

    let expected = Tensor::<B, 2>::from_floats([[2.5]], &backend);
    let diff = (k.clone() - expected.clone()).abs().into_scalar();
    assert!(
        diff < 1e-5,
        "Matérn 5/2 at zero: Expected {}, got {}",
        2.5,
        k.into_scalar()
    );
}

#[test]
fn test_matern_nu5_2_basic() {
    let backend = instantiate_backend();
    let x1 = Tensor::<B, 2>::from_floats([[0.0, 0.0], [1.0, 0.0]], &backend);
    let x2 = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &backend);
    let kernel = Kernel::<B>::new(
        KernelKind::Matern(NuKind::Nu5_2),
        vec![1.0, 1.0],
        1.0,
        &backend,
    );
    let k = kernel.execute(x1, x2);

    // Matérn 5/2: k(d) = (1 + √5×d + 5d²/3) × exp(-√5×d)
    // Distance from [0,0] to [0,0]: d = 0, k = 1.0
    // Distance from [1,0] to [0,0]: d = 1
    // r = √5 ≈ 2.236, k = (1 + 2.236 + 2.236²/3) × exp(-2.236)
    //                    = (1 + 2.236 + 1.667) × 0.1066 ≈ 4.903 × 0.1066 ≈ 0.5227
    let expected = Tensor::<B, 2>::from_floats([[1.0], [0.5227]], &backend);
    let tolerance = Tensor::<B, 2>::from_floats([[1e-2], [1e-2]], &backend);

    let equal = (k.clone() - expected.clone())
        .abs()
        .lower_equal(tolerance)
        .all()
        .into_scalar()
        != 0;

    assert!(
        equal,
        "Matérn 5/2: Computed {} and expected {} differ",
        k.to_data(),
        expected.to_data(),
    );
}

#[test]
fn test_ard_different_length_scales() {
    let backend = instantiate_backend();
    // Test that different length scales affect dimensions differently
    let x1 = Tensor::<B, 2>::from_floats([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &backend);
    let x2 = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &backend);

    // Length scale 1.0 for first dim, 10.0 for second dim
    // This means variations in second dimension matter less
    let kernel = Kernel::<B>::new(KernelKind::Rbf, vec![1.0, 10.0], 1.0, &backend);
    let k = kernel.execute(x1, x2);

    // Distance from [1,0]: After scaling: [1/1, 0/10] = [1, 0], d² = 1
    // Distance from [0,1]: After scaling: [0/1, 1/10] = [0, 0.1], d² = 0.01
    // So [0,1] should have higher kernel value than [1,0]
    let k_data = k.to_data();
    let k1 = k_data.as_slice::<f32>().unwrap()[1]; // k([1,0], [0,0])
    let k2 = k_data.as_slice::<f32>().unwrap()[2]; // k([0,1], [0,0])

    assert!(
        k2 > k1,
        "ARD: Movement in high-length-scale dimension should have higher kernel value. Got k1={}, k2={}",
        k1,
        k2
    );
}

#[test]
fn test_kernel_symmetry() {
    // NdArray for this one, since the WGPU driver has a bug
    type B = burn::backend::NdArray<f32>;
    let backend = NdArrayDevice::default();

    let x1 = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &backend);
    let x2 = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &backend);

    let kernel = Kernel::<B>::new(KernelKind::Rbf, vec![1.0, 1.0], 1.0, &backend);

    let k12 = kernel.execute(x1.clone(), x2.clone());
    let k21 = kernel.execute(x2, x1).transpose();

    let diff = (k12 - k21).abs().max().into_scalar();
    assert!(
        diff < 1e-5,
        "Kernel should be symmetric: max diff = {}",
        diff
    );
}
