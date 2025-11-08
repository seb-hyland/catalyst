use burn::backend::{Wgpu, wgpu::WgpuDevice};

pub type B = Wgpu;

#[inline]
#[allow(dead_code)]
pub fn instantiate_backend() -> WgpuDevice {
    WgpuDevice::default()
}
