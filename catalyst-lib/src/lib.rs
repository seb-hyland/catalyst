use crate::math::{cholesky, cholesky_solve, cholesky_solve_batch};
use burn::{
    module::Param,
    prelude::*,
    tensor::linalg::{diag, l2_norm},
};
use libm::erff;
use std::f32;

pub mod math;
pub mod train;

const NOISE: f64 = 1e-4;

#[derive(Module, Debug)]
pub struct Kernel<B: Backend> {
    kind: KernelKind,

    // Hyperparameters
    log_length_scale: Param<Tensor<B, 1>>,
    log_variance: Param<Tensor<B, 1>>,
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
        }
    }

    pub fn length_scale(&self) -> Tensor<B, 1> {
        self.log_length_scale.val().clone().exp().clamp_min(1e-4)
    }

    pub fn variance(&self) -> Tensor<B, 1> {
        self.log_variance.val().clone().exp().clamp_min(1e-6)
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
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    // Kernel matrices
    let k_tt = kernel.execute(x_train.clone(), x_train.clone());
    let k_ts = kernel.execute(x_train.clone(), x_test.clone());
    let k_ss = kernel.execute(x_test.clone(), x_test.clone());

    let [n, _] = k_tt.dims();
    // Add noise
    let k_y = k_tt + Tensor::<B, 2>::eye(n, &x_train.device()) * NOISE;

    let l = cholesky(k_y);
    let alpha = cholesky_solve(l.clone(), y_train.clone());
    let mean = k_ts
        .clone()
        .transpose()
        .matmul(alpha.unsqueeze_dim(1))
        .squeeze_dim(1);

    let v = cholesky_solve_batch(l, k_ts);
    let predictive_cov = k_ss - v.clone().transpose().matmul(v);

    (mean, predictive_cov)
}

pub fn select_batch<B: Backend>(
    mut x_train: Tensor<B, 2>,
    mut y_train: Tensor<B, 1>,
    model: &Kernel<B>,
    mut x_candidates: Tensor<B, 2>,
    batch_size: usize,
) -> Tensor<B, 2> {
    let mut selected_points: Vec<Tensor<B, 2>> = Vec::new();
    let xi_max = 1.0;

    for i in 0..batch_size {
        let expected_improvements = ei_all(
            x_train.clone(),
            y_train.clone(),
            x_candidates.clone(),
            model,
            xi_max * i as f32 / batch_size as f32,
        );
        let (best_candidate_idx, _) = expected_improvements
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let best_x = x_candidates.clone().slice(best_candidate_idx);
        selected_points.push(best_x.clone());

        // Predict its value using the surrogate
        let (estimated_mu, _cov) = gaussian(
            x_train.clone(),
            y_train.clone(),
            best_x.clone(),
            model.clone(),
        );
        let y_fantasy = estimated_mu.clone();
        x_train = Tensor::cat(vec![x_train, best_x.clone()], 0);
        y_train = Tensor::cat(vec![y_train, y_fantasy.clone()], 0);

        let [n_rows, n_cols] = x_candidates.dims();
        let device = x_train.device();

        let before = if best_candidate_idx != 0 {
            x_candidates.clone().slice(0..best_candidate_idx)
        } else {
            Tensor::empty([0, n_cols], &device)
        };
        let after = if best_candidate_idx != n_rows - 1 {
            x_candidates.clone().slice(best_candidate_idx + 1..)
        } else {
            Tensor::empty([0, n_cols], &device)
        };
        x_candidates = Tensor::cat(vec![before, after], 0);
    }

    Tensor::cat(selected_points, 0)
}

fn ei_all<B: Backend>(
    x_train: Tensor<B, 2>,
    y_train: Tensor<B, 1>,
    x_candidates: Tensor<B, 2>,
    model: &Kernel<B>,
    xi: f32,
) -> Vec<f32> {
    let f_best: f32 = y_train
        .clone()
        .into_data()
        .iter()
        .fold(f32::NEG_INFINITY, f32::max);

    let (mu, cov) = gaussian(
        x_train.clone(),
        y_train.clone(),
        x_candidates.clone(),
        model.clone(),
    );

    let mu_vec = mu.clone().into_data().to_vec().unwrap();
    let var_vec = diag::<B, _, 1, _>(cov).into_data().to_vec().unwrap();

    mu_vec
        .iter()
        .zip(var_vec.iter())
        .map(|(&m, &v)| ei_scalar(m, v, f_best, xi))
        .collect()
}

fn ei_scalar(mu: f32, var: f32, f_best: f32, xi: f32) -> f32 {
    if var <= 0.0 {
        return 0.0;
    }

    let improvement = mu - f_best - xi;
    let sigma = var.sqrt();

    let z = improvement / sigma;
    let phi = (1.0 / f32::sqrt(2.0 * f32::consts::PI)) * f32::exp(-0.5 * z.powi(2));
    let cdf = 0.5 * (1.0 + erff(z / f32::consts::SQRT_2));

    improvement * cdf + sigma * phi
}
