use crate::{
    Kernel, NOISE,
    math::{cholesky, cholesky_solve},
};
use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer as _},
    prelude::*,
    tensor::{backend::AutodiffBackend, linalg::diag},
};

pub fn train_gp<B: AutodiffBackend>(
    mut model: Kernel<B>,
    x_train: Tensor<B, 2>,
    y_train: Tensor<B, 1>,
    epochs: usize,
    lr: f64,
) -> Kernel<B> {
    let mut optim = AdamConfig::new().init();

    #[cfg(feature = "visualize")]
    let (mut losses, mut length_scales, mut variances) = (
        Vec::with_capacity(epochs),
        Vec::with_capacity(epochs),
        Vec::with_capacity(epochs),
    );

    for _ in 0..epochs {
        let loss = negative_log_marginal_likelihood(x_train.clone(), y_train.clone(), &model);

        #[cfg(feature = "visualize")]
        {
            let loss_val: f32 = loss.clone().into_scalar().to_f32();
            losses.push(loss_val);

            let ls = model
                .length_scale()
                .into_data()
                .to_vec()
                .expect("Length scales");
            length_scales.push(ls);

            let variance: f32 = model.variance().clone().into_scalar().to_f32();
            variances.push(variance);
        }

        let grads = loss.backward();
        model = optim.step(
            lr,
            model.clone(),
            GradientsParams::from_grads(grads, &model),
        );
    }

    // After training, plot results
    #[cfg(feature = "visualize")]
    plot_training(losses, length_scales, variances);

    model
}

#[cfg(feature = "visualize")]
fn plot_training(losses: Vec<f32>, length_scales: Vec<Vec<f32>>, variances: Vec<f32>) {
    use plotly::{Layout, Plot, Scatter, layout::LayoutGrid};

    let mut plot = Plot::new();
    let n_dims = length_scales[0].len();
    let layout = Layout::new().grid(LayoutGrid::new().rows(2 + n_dims).columns(1));
    plot.set_layout(layout);

    plot.add_trace(
        Scatter::new((0..losses.len()).collect(), losses)
            .name("Loss")
            .x_axis("x1")
            .y_axis("y1"),
    );
    plot.add_trace(
        Scatter::new((0..variances.len()).collect(), variances)
            .name("Variance")
            .x_axis("x2")
            .y_axis("y2"),
    );
    for dim in 0..n_dims {
        let ls_dim: Vec<f32> = length_scales.iter().map(|v| v[dim]).collect();
        plot.add_trace(
            Scatter::new((0..ls_dim.len()).collect(), ls_dim)
                .name(format!("Length scale dim {}", dim))
                .x_axis(format!("x{}", 3 + dim))
                .y_axis(format!("y{}", 3 + dim)),
        );
    }

    plot.write_html("training_loss.html");
}

pub fn negative_log_marginal_likelihood<B: Backend>(
    x_train: Tensor<B, 2>,
    y_train: Tensor<B, 1>,
    kernel: &Kernel<B>,
) -> Tensor<B, 1> {
    let [n, _m] = x_train.dims();
    let device = x_train.device();

    let k = kernel.execute(x_train.clone(), x_train.clone());
    let k_y = k + Tensor::<B, 2>::eye(n, &device) * NOISE;

    let l = cholesky(k_y);
    let alpha = cholesky_solve(l.clone(), y_train.clone());
    let term1 = 0.5
        * y_train
            .reshape([1, n])
            .matmul(alpha.reshape([n, 1]))
            .squeeze_dim(1);

    let diag_l: Tensor<B, 1> = diag(l);
    let log_det = diag_l.log().sum() * 2.0;
    let term2 = 0.5 * log_det;

    let term3 = Tensor::from_floats(
        [(n as f32 * (2.0 * core::f32::consts::PI).ln()) * 0.5],
        &device,
    );

    term1 + term2 + term3
}
