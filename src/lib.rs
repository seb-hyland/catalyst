use crate::math::{cholesky, cholesky_solve, cholesky_solve_batch};
use burn::{
    module::Param,
    prelude::*,
    tensor::linalg::{diag, l2_norm},
};
use std::marker::PhantomData;

pub mod math;
pub mod train;

const NOISE: f64 = 1e-6;

#[derive(Module, Debug)]
pub struct Kernel<B: Backend> {
    kind: KernelKind,

    // Hyperparameters
    log_length_scale: Param<Tensor<B, 1>>,
    log_variance: Param<Tensor<B, 1>>,

    _marker: PhantomData<B>,
}
#[derive(Debug, Clone, Copy, Module)]
pub enum KernelKind {
    Rbf,
    Matern(NuKind),
}
#[derive(Debug, Clone, Copy, Module)]
pub enum NuKind {
    Nu1_2,
    Nu3_2,
    Nu5_2,
}

impl<B: Backend> Kernel<B> {
    pub fn new(
        kind: KernelKind,
        length_scales: Vec<f32>,
        variance: f32,
        device: &B::Device,
    ) -> Self {
        let log_ls: Vec<_> = length_scales.into_iter().map(|l| l.ln()).collect();
        Self {
            kind,
            log_length_scale: Param::from_tensor(Tensor::from_floats(log_ls.as_slice(), device)),
            log_variance: Param::from_tensor(Tensor::from_floats([variance.ln()], device)),
            _marker: PhantomData,
        }
    }

    pub fn length_scale(&self) -> Tensor<B, 1> {
        self.log_length_scale.val().clone().exp()
    }

    pub fn variance(&self) -> Tensor<B, 1> {
        self.log_variance.val().clone().exp()
    }

    pub fn execute(&self, x1: Tensor<B, 2>, x2: Tensor<B, 2>) -> Tensor<B, 2> {
        match self.kind {
            KernelKind::Rbf => self.rbf_kernel(x1, x2),
            KernelKind::Matern(ref nu) => self.matern_kernel(x1, x2, nu),
        }
    }

    fn rbf_kernel(&self, x1: Tensor<B, 2>, x2: Tensor<B, 2>) -> Tensor<B, 2> {
        let length_scale = self.length_scale().unsqueeze();
        let x1_scaled = x1 / length_scale.clone();
        let x2_scaled = x2 / length_scale;

        // Unsqueeze then resequeeze for pairwise difference
        let diff: Tensor<B, 3> = x1_scaled.unsqueeze_dim(1) - x2_scaled.unsqueeze_dim(0);
        let dist_sq: Tensor<B, 2> = diff.square().sum_dims_squeeze(&[2]);
        let k = (-0.5_f32 * dist_sq).exp();
        self.variance().unsqueeze() * k
    }

    fn matern_kernel(&self, x1: Tensor<B, 2>, x2: Tensor<B, 2>, nu_kind: &NuKind) -> Tensor<B, 2> {
        let length_scale = self.length_scale().unsqueeze();
        let x1_scaled = x1 / length_scale.clone();
        let x2_scaled = x2 / length_scale;

        const SQRT_3: f32 = 1.7320508;
        const SQRT_5: f32 = 2.236068;

        let diff: Tensor<B, 3> = x1_scaled.unsqueeze_dim(1) - x2_scaled.unsqueeze_dim(0);
        let dist: Tensor<B, 2> = l2_norm(diff, 2).squeeze_dim(2);

        let k = match nu_kind {
            NuKind::Nu1_2 => (-dist).exp(),
            NuKind::Nu3_2 => {
                let scaled_dist = dist * SQRT_3;
                (1 + scaled_dist.clone()) * (-scaled_dist).exp()
            }
            NuKind::Nu5_2 => {
                let scaled_dist = dist * SQRT_5;
                (1 + scaled_dist.clone() + scaled_dist.clone().square() / 3.0)
                    * (-scaled_dist).exp()
            }
        };

        self.variance().unsqueeze() * k
    }
}

pub fn gaussian<B: Backend>(
    x_train: Tensor<B, 2>,
    y_train: Tensor<B, 1>,
    x_test: Tensor<B, 2>,
    kernel: Kernel<B>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    // Kernel matrices
    let k_tt = kernel.execute(x_train.clone(), x_train.clone());
    let k_ts = kernel.execute(x_train.clone(), x_test.clone());
    let k_ss = kernel.execute(x_test.clone(), x_test.clone());

    let n = k_tt.dims()[0];
    // Add noise
    let k_y = k_tt + Tensor::<B, 2>::eye(n, &x_train.device()) * NOISE;

    let l = cholesky(k_y);
    let alpha = cholesky_solve(l.clone(), y_train.clone());
    let mean = k_ts
        .clone()
        .transpose()
        .matmul(alpha.unsqueeze_dim(1))
        .squeeze();

    let v = cholesky_solve_batch(l, k_ts);
    let predictive_var = diag(k_ss) - v.powi_scalar(2).sum_dim(0).squeeze();

    (mean, predictive_var)
}
