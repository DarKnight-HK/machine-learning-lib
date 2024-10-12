use machine_learning::nn::Matrix;

struct Xor {
    a0: Matrix,
    w1: Matrix,
    w2: Matrix,
    b1: Matrix,
    b2: Matrix,
    a1: Matrix,
    a2: Matrix,
}

fn xor_cost(xor: &mut Xor, input: &Matrix, output: &Matrix) -> f64 {
    let mut result = 0.0;
    for i in 0..input.rows {
        xor.a0.data[0] = input.data[i].clone();
        let y = forward_xor(xor, input.data[i][0], input.data[i][1]);
        result += (y - output.data[i][0]).powi(2);
    }
    result / input.rows as f64
}

fn finite_diff(xor: &mut Xor, g: &mut Xor, eps: f64, input: &Matrix, output: &Matrix) {
    let mut saved = 0.0;
    let c = xor_cost(xor, input, output);

    for i in 0..xor.w1.rows {
        for j in 0..xor.w1.cols {
            saved = xor.w1.data[i][j];
            xor.w1.data[i][j] += eps;
            g.w1.data[i][j] = (xor_cost(xor, input, output) - c) / eps;
            xor.w1.data[i][j] = saved;
        }
    }
    for i in 0..xor.w2.rows {
        for j in 0..xor.w2.cols {
            saved = xor.w2.data[i][j];
            xor.w2.data[i][j] += eps;
            g.w2.data[i][j] = (xor_cost(xor, input, output) - c) / eps;
            xor.w2.data[i][j] = saved;
        }
    }

    for i in 0..xor.b1.rows {
        for j in 0..xor.b1.cols {
            saved = xor.b1.data[i][j];
            xor.b1.data[i][j] += eps;
            g.b1.data[i][j] = (xor_cost(xor, input, output) - c) / eps;
            xor.b1.data[i][j] = saved;
        }
    }

    for i in 0..xor.b2.rows {
        for j in 0..xor.b2.cols {
            saved = xor.b2.data[i][j];
            xor.b2.data[i][j] += eps;
            g.b2.data[i][j] = (xor_cost(xor, input, output) - c) / eps;
            xor.b2.data[i][j] = saved;
        }
    }
}

fn forward_xor(xor: &mut Xor, x1: f64, x2: f64) -> f64 {
    xor.a0.data[0] = vec![x1, x2];
    xor.a1 = (&(&xor.a0 * &xor.w1).unwrap() + &xor.b1).unwrap();
    xor.a1.sigmoid();
    xor.a2 = (&(&xor.a1 * &xor.w2).unwrap() + &xor.b2).unwrap();
    xor.a2.sigmoid();
    xor.a2.data[0][0]
}

fn alc() -> Xor {
    let a0: Matrix = Matrix::new(1, 2);
    let w1: Matrix = Matrix::new(2, 2);
    let w2: Matrix = Matrix::new(2, 1);
    let b1: Matrix = Matrix::new(1, 2);
    let b2: Matrix = Matrix::new(1, 1);
    let a1: Matrix = Matrix::new(1, 2);
    let a2: Matrix = Matrix::new(1, 1);

    let mut xor = Xor {
        a0,
        w1,
        w2,
        b1,
        b2,
        a1,
        a2,
    };
    xor.w1.fill_rand(0.0, 1.0);
    xor.w2.fill_rand(0.0, 1.0);
    xor.b1.fill_rand(0.0, 1.0);
    xor.b2.fill_rand(0.0, 1.0);
    xor
}

fn xor_learn(xor: &mut Xor, grad: &mut Xor, lr: f64) {
    for i in 0..xor.w1.rows {
        for j in 0..xor.w1.cols {
            xor.w1.data[i][j] -= lr * grad.w1.data[i][j];
        }
    }
    for i in 0..xor.w2.rows {
        for j in 0..xor.w2.cols {
            xor.w2.data[i][j] -= lr * grad.w2.data[i][j];
        }
    }

    for i in 0..xor.b1.rows {
        for j in 0..xor.b1.cols {
            xor.b1.data[i][j] -= lr * grad.b1.data[i][j];
        }
    }

    for i in 0..xor.b2.rows {
        for j in 0..xor.b2.cols {
            xor.b2.data[i][j] -= lr * grad.b2.data[i][j];
        }
    }
}

fn main() {
    let mut inp_mat = Matrix::new(4, 2);
    let mut out_mat = Matrix::new(4, 1);
    inp_mat.data[0] = vec![0.0, 0.0];
    inp_mat.data[1] = vec![0.0, 1.0];
    inp_mat.data[2] = vec![1.0, 0.0];
    inp_mat.data[3] = vec![1.0, 1.0];

    out_mat.data[0] = vec![0.0];
    out_mat.data[1] = vec![1.0];
    out_mat.data[2] = vec![1.0];
    out_mat.data[3] = vec![0.0];
    let mut xor = alc();
    let cost = xor_cost(&mut xor, &inp_mat, &out_mat);
    println!("Cost: {:.2}", cost);

    let mut grad = alc();
    for _ in 0..10 * 1000 {
        finite_diff(&mut xor, &mut grad, 1e-1, &inp_mat, &out_mat);
        xor_learn(&mut xor, &mut grad, 1e-1);
    }

    println!("cost = {}", xor_cost(&mut xor, &inp_mat, &out_mat));
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{} XOR {} = {}",
                i,
                j,
                forward_xor(&mut xor, i as f64, j as f64)
            );
        }
    }
}
