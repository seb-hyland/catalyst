use burn::prelude::*;
use std::{
    mem::MaybeUninit,
    ops::{Bound, RangeBounds},
};

/**
   A raw buffer that may contain chunks of uninitialized memory.
*/
pub struct RawBuffer<T>(Vec<MaybeUninit<T>>);

impl<T: Clone> RawBuffer<T> {
    /**
       Initialize a buffer of uninitialized memory
       that can hold `size` items of type T.
    */
    pub fn init(size: usize) -> Self {
        let mut inner = Vec::with_capacity(size);
        inner.resize_with(size, MaybeUninit::uninit);
        Self(inner)
    }

    /**
       Mutate the `index` item in the buffer,
       setting it to `element`.
    */
    pub fn set(&mut self, index: usize, element: T) {
        self.0[index] = MaybeUninit::new(element);
    }

    /**
       Extract a slice of the buffer bounded by `range`.
       All items will be cloned, and returned as a [`Vec`] of owned elements.

       # Safety
       All items in `range` must have been previously initialized.
    */
    pub unsafe fn slice_vec<R: RangeBounds<usize>>(&self, range: R) -> Vec<T> {
        let start = match range.start_bound() {
            Bound::Included(&s) => s,
            Bound::Excluded(&s) => s + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&s) => s + 1,
            Bound::Excluded(&s) => s,
            Bound::Unbounded => self.0.len(),
        };

        self.0[start..end]
            .iter()
            .map(|item| unsafe { item.assume_init_ref().clone() })
            .collect()
    }

    /**
        Consumes the buffer and returns it as a [`Vec`] of initialized elements.

        # Safety
        All items in the buffer must have been previously initialized.
    */
    pub unsafe fn into_inner(self) -> Vec<T> {
        self.0
            .into_iter()
            .map(|item| unsafe { item.assume_init() })
            .collect()
    }
}

/// Compute the Cholesky decomposition of a Hermitian, positive-definite matrix
pub fn cholesky<B: Backend>(a: Tensor<B, 2>) -> Tensor<B, 2> {
    // Check shape
    let [n, m] = a.shape().dims::<2>();
    assert_eq!(n, m, "Matrix must be square");

    let device = a.device();
    let mut l = Tensor::<B, 2>::zeros([n, n], &device);

    for i in 0..n {
        const CLAMP: f32 = 1e-12;
        let a_elem = a.clone().slice([i, i]);

        let diag = if i == 0 {
            Tensor::sqrt(a_elem.clamp_min(CLAMP))
        } else {
            let prev_cols = l.clone().slice(s![i, 0..i]);
            let diag_inner = a_elem
                - Tensor::dot(
                    prev_cols.clone().squeeze_dim(0),
                    prev_cols.clone().squeeze_dim(0),
                )
                .unsqueeze_dim(0);
            Tensor::sqrt(diag_inner.clamp_min(CLAMP))
        };

        l = l.slice_assign([i, i], diag.clone());

        // There are still rows below the current
        if i + 1 < n {
            let rest_cols_range = (i + 1)..;
            let a_elem = a.clone().slice(s![rest_cols_range.clone(), i]);
            let diag = diag.into_scalar();

            let l_rest_i = if i == 0 {
                a_elem / diag
            } else {
                let prev_cols = l.clone().slice(s![i, 0..i]);
                a_elem
                    - Tensor::matmul(
                        l.clone().slice(s![rest_cols_range.clone(), 0..i]),
                        prev_cols.transpose(),
                    ) / diag
            };

            l = l.slice_assign(s![rest_cols_range, i], l_rest_i);
        }
    }

    l
}

/// Solves the system $Ax = b$ using Cholesky decomposition
/// where $A = L \cdot L^T$ has already been decomposed
pub fn cholesky_solve<B: Backend>(l: Tensor<B, 2>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let [n, _] = l.dims();

    // Forward substitution: L y = b
    let mut y_rows = Vec::with_capacity(n);
    for i in 0..n {
        let b_elem = b.clone().slice(i);
        let diag = l.clone().slice([i, i]).into_scalar();

        let row = if i == 0 {
            b_elem.clone() / diag
        } else {
            let dot_product = Tensor::dot(
                l.clone().slice(s![i, 0..i]).squeeze_dim(0),
                Tensor::cat(y_rows[0..i].to_owned(), 0),
            );
            (b_elem - dot_product) / diag
        };

        y_rows.insert(i, row);
    }
    let y = Tensor::cat(y_rows, 0);

    // Backward substitution: L^T x = y
    let mut x_rows = RawBuffer::init(n);
    for i in (0..n).rev() {
        let y_elem = y.clone().slice(i);
        let diag = l.clone().slice([i, i]).into_scalar();

        let row = if i == n - 1 {
            y_elem / diag
        } else {
            let dot_product = Tensor::dot(
                l.clone().slice(s![i + 1.., i]).squeeze_dim(1),
                Tensor::cat(unsafe { x_rows.slice_vec(i + 1..) }, 0),
            );
            (y_elem - dot_product) / diag
        };

        x_rows.set(i, row);
    }

    Tensor::cat(unsafe { x_rows.into_inner() }, 0)
}

/// Solve $L \cdot L^T X = B$ where $B$ is $2D$ ($n \times m$). Returns $X$ ($n \times m$).
pub fn cholesky_solve_batch<B: Backend>(l: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    let [_n, m] = b.dims();
    let mut cols = Vec::with_capacity(m);

    for j in 0..m {
        let col = b.clone().slice(s![.., j]).squeeze_dim(1);
        let x_col = cholesky_solve(l.clone(), col);
        cols.push(x_col);
    }

    Tensor::stack(cols, 1)
}
