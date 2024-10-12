use machine_learning::nn::Matrix;
fn main() {
    let mut w1: Matrix<f64> = Matrix::new(2, 2);
    let mut w2: Matrix<f64> = Matrix::new(2, 1);
    let mut b1: Matrix<f64> = Matrix::new(1, 2);
    let mut b2: Matrix<f64> = Matrix::new(1, 1);
    w1.fill_rand(0.0, 1.0);
    w2.fill_rand(0.0, 1.0);
    b1.fill_rand(0.0, 1.0);
    b2.fill_rand(0.0, 1.0);
}
