use burn::{Tensor, backend::Autodiff};
use catalyst::{train::*, *};

mod utils;
use utils::*;

#[test]
fn test_gaussian_process_basic() {
    // Instantiate backend
    let backend = instantiate_backend();

    // Small training data: y = x^2
    let x_train = Tensor::<B, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &backend);
    let y_train = Tensor::<B, 1>::from_floats([1.0, 4.0, 9.0, 16.0], &backend);

    // Test points
    let x_test = Tensor::<B, 2>::from_floats([[1.5], [2.5], [3.5]], &backend);

    // RBF kernel with length scale 1.0 and variance 1.0
    let kernel = Kernel::<B>::new(KernelKind::Rbf, vec![1.0], 1.0, &backend);

    // Run Gaussian Process
    let (mean, var) = gaussian(x_train, y_train, x_test, kernel);

    // Assert shapes
    assert_eq!(mean.dims()[0], 3);
    assert_eq!(var.dims()[0], 3);

    // Assert that predictions are roughly increasing and reasonable
    let mean_data = mean.to_data();
    let data = mean_data.as_slice::<f32>().unwrap();
    assert!(data[0] < data[1]);
    assert!(data[1] < data[2]);

    // Variances should be positive
    for v in var.to_data().as_slice::<f32>().unwrap() {
        assert!(*v >= 0.0);
    }

    // Optional: print results
    println!("Predicted mean: {}", mean.to_data());
    println!("Predicted variance: {}", var.to_data());
}

#[test]
fn test_train_gp() {
    type AB = Autodiff<B>;
    // Small 1D training data: y = x^2
    let x_train = Tensor::<AB, 2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &Default::default());
    let y_train = Tensor::<AB, 1>::from_floats([1.0, 4.0, 9.0, 16.0], &Default::default());

    // Initialize kernel
    let device = x_train.device();
    let initial_kernel = Kernel::new(KernelKind::Rbf, vec![1.0], 1.0, &device);

    // Compute initial loss
    let initial_loss =
        negative_log_marginal_likelihood(x_train.clone(), y_train.clone(), &initial_kernel);
    let initial_loss_val: f32 = initial_loss.clone().into_scalar();

    // Train kernel for a few epochs
    let trained_kernel = train_gp(
        initial_kernel.clone(),
        x_train.clone(),
        y_train.clone(),
        100,
        0.01,
    );

    // Compute loss after training
    let final_loss =
        negative_log_marginal_likelihood(x_train.clone(), y_train.clone(), &trained_kernel);
    let final_loss_val: f32 = final_loss.clone().into_scalar();

    println!("Initial NLL: {}", initial_loss_val);
    println!("Final NLL: {}", final_loss_val);

    println!(
        "Initial kernel: {}, {}",
        initial_kernel.length_scale(),
        initial_kernel.variance()
    );
    println!(
        "Final kernel: {}, {}",
        trained_kernel.length_scale(),
        trained_kernel.variance()
    );

    // Assert that loss decreased
    assert!(
        final_loss_val < initial_loss_val,
        "Training did not decrease the negative log likelihood"
    );
}
