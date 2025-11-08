use burn::prelude::*;

pub fn cholesky<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    // Check shape
    let [n, m] = tensor.shape().dims::<2>();
    assert_eq!(n, m, "Matrix must be square");

    let device = tensor.device();
    let mut l = Tensor::<B, 2>::zeros([n, n], &device);

    for j in 0..n {
        // diagonal element
        let a_jj = tensor.clone().slice([j..j + 1, j..j + 1]); // [1,1]

        let mut sum_sq = Tensor::<B, 2>::zeros([1, 1], &device);
        if j > 0 {
            let l_row = l.clone().slice([j..j + 1, 0..j]); // [1, j]
            sum_sq = (l_row.clone() * l_row).sum().reshape([1, 1]);
        }

        let diag = a_jj - sum_sq;
        let diag_val = diag.clone().into_scalar().to_f64();
        assert!(
            diag_val > 0.0,
            "Matrix is not positive definite at position ({}, {})",
            j,
            j
        );

        let l_jj = diag.sqrt(); // [1,1]
        l = l.slice_assign([j..j + 1, j..j + 1], l_jj.clone());

        // column below diagonal
        if j < n - 1 {
            let mut col = tensor.clone().slice([(j + 1)..n, j..j + 1]); // [n-j-1, 1]

            if j > 0 {
                let rows_below = l.clone().slice([(j + 1)..n, 0..j]); // [n-j-1, j]
                let row_j = l.clone().slice([j..j + 1, 0..j]); // [1, j]
                let dots = rows_below.matmul(row_j.transpose()); // [n-j-1, 1]
                col = col - dots;
            }

            let col = col / l_jj.clone();
            l = l.slice_assign([(j + 1)..n, j..j + 1], col);
        }
    }

    l
}

/// Solves the system Ax = b using Cholesky decomposition
/// where A = L·L^T has already been decomposed
pub fn cholesky_solve<B: Backend>(l: Tensor<B, 2>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    let n = l.dims()[0];
    let device = l.device();

    // Forward: L y = b
    let mut y = Tensor::<B, 1>::zeros([n], &device);
    for i in 0..n {
        // val: [1]
        let mut val = b.clone().slice(i).reshape([1]);

        if i > 0 {
            // l_row: [1, i], y_prev: [i, 1] -> matmul -> [1,1]
            let l_row = l.clone().slice(s![i, 0..i]); // [1, i]
            let y_prev = y.clone().slice(0..i).reshape([i, 1]); // [i,1]
            let dot = l_row.matmul(y_prev).reshape([1]); // [1]
            val = val - dot;
        }

        let l_ii = l.clone().slice([i, i]).reshape([1]); // [1]
        let y_i = (val / l_ii).reshape([1]); // [1]
        y = y.slice_assign(i, y_i);
    }

    // Backward: L^T x = y
    let mut x = Tensor::<B, 1>::zeros([n], &device);
    for i in (0..n).rev() {
        let mut val = y.clone().slice(i).reshape([1]);

        if i < n - 1 {
            // l_col: [n-i-1, 1], x_rest: [n-i-1, 1]
            let l_col = l.clone().slice(s![(i + 1)..n, i]); // [n-i-1, 1]
            let x_rest = x.clone().slice((i + 1)..n).reshape([n - i - 1, 1]);
            // (L^T row) * x_rest = l_col.transpose() matmul x_rest -> [1,1]
            let dot = l_col.transpose().matmul(x_rest).reshape([1]);
            val = val - dot;
        }

        let l_ii = l.clone().slice([i, i]).reshape([1]); // [1]
        let x_i = (val / l_ii).reshape([1]);
        x = x.slice_assign(i, x_i);
    }

    x
}

/// Solve L·L^T X = B where B is 2-D (n x m). Returns X (n x m).
pub fn cholesky_solve_batch<B: Backend>(l: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    // Solve each column (this is straightforward and correct).
    let [n, m] = b.dims();
    let mut cols: Vec<Tensor<B, 1>> = Vec::with_capacity(m);
    for j in 0..m {
        let col = b.clone().slice(s![0..n, j]).reshape([n]);
        let x_col = cholesky_solve(l.clone(), col);
        cols.push(x_col);
    }
    Tensor::stack(cols, 1)
}
