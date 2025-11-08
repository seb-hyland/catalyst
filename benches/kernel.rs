use burn::{
    Tensor,
    backend::{NdArray, Rocm, Wgpu, ndarray::NdArrayDevice, rocm::RocmDevice, wgpu::WgpuDevice},
    prelude::Backend,
    tensor::Device,
};
use catalyst::*;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};

fn rand_matrix<B: Backend>(rows: usize, cols: usize, backend: &Device<B>) -> Tensor<B, 2> {
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.random_range(-5.0..5.0))
        .collect();
    let tensor_1d = Tensor::<B, 1>::from_floats(data.as_slice(), backend);
    tensor_1d.reshape([rows, cols])
}

const ROWS: usize = 1000;
const COLS: usize = 100;
fn bench_rbf_ndarray(c: &mut Criterion) {
    let backend = NdArrayDevice::default();

    let x1 = rand_matrix::<NdArray>(ROWS, COLS, &backend);
    let x2 = rand_matrix::<NdArray>(ROWS, COLS, &backend);

    let kernel = Kernel::new(KernelKind::Rbf, 1.0, 1.0);
    c.bench_function("rbf_kernel_ndarray", |b| {
        b.iter(|| kernel.execute(x1.clone(), x2.clone()))
    });
}
fn bench_rbf_wgpu(c: &mut Criterion) {
    let backend = WgpuDevice::default();

    let x1 = rand_matrix::<Wgpu>(ROWS, COLS, &backend);
    let x2 = rand_matrix::<Wgpu>(ROWS, COLS, &backend);

    let kernel = Kernel::new(KernelKind::Rbf, 1.0, 1.0);
    c.bench_function("rbf_kernel_wgpu", |b| {
        b.iter(|| kernel.execute(x1.clone(), x2.clone()))
    });
}
fn bench_rbf_rocm(c: &mut Criterion) {
    let backend = RocmDevice::default();

    let x1 = rand_matrix::<Rocm>(ROWS, COLS, &backend);
    let x2 = rand_matrix::<Rocm>(ROWS, COLS, &backend);

    let kernel = Kernel::new(KernelKind::Rbf, 1.0, 1.0);
    c.bench_function("rbf_kernel_rocm", |b| {
        b.iter(|| kernel.execute(x1.clone(), x2.clone()))
    });
}

criterion_group!(benches, bench_rbf_ndarray, bench_rbf_wgpu, bench_rbf_rocm);
criterion_main!(benches);
