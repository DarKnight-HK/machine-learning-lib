use machine_learning::nn::Matrix;

struct Xor {
    a0: Matrix<f64>,
    w1: Matrix<f64>,
    w2: Matrix<f64>,
    b1: Matrix<f64>,
    b2: Matrix<f64>,
    a1: Matrix<f64>,
    a2: Matrix<f64>,
}

fn forward_xor(xor: &mut Xor, x1: f64, x2: f64) -> f64 {
    xor.a0.data[0] = vec![x1, x2];
    xor.a1 = ((xor.a0.clone() * xor.w1.clone()).unwrap() + xor.b1.clone()).unwrap();
    xor.a1.sigmoid();
    xor.a2 = ((xor.a1.clone() * xor.w2.clone()).unwrap() + xor.b2.clone()).unwrap();
    xor.a2.sigmoid();
    xor.a2.data[0][0]
}

fn main() {
    let a0: Matrix<f64> = Matrix::new(1, 2);
    let w1: Matrix<f64> = Matrix::new(2, 2);
    let w2: Matrix<f64> = Matrix::new(2, 1);
    let b1: Matrix<f64> = Matrix::new(1, 2);
    let b2: Matrix<f64> = Matrix::new(1, 1);
    let a1: Matrix<f64> = Matrix::new(1, 2);
    let a2: Matrix<f64> = Matrix::new(1, 1);

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
