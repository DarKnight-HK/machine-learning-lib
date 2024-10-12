use machine_learning::nn::Matrix;
fn main() {
    let mut x: Matrix<f64> = Matrix::new(1, 2);
    x.data[0] = vec![0.0, 1.0];
    let mut w1: Matrix<f64> = Matrix::new(2, 2);
    let mut w2: Matrix<f64> = Matrix::new(2, 1);
    let mut b1: Matrix<f64> = Matrix::new(1, 2);
    let mut b2: Matrix<f64> = Matrix::new(1, 1);
    w1.fill_rand(0.0, 1.0);
    w2.fill_rand(0.0, 1.0);
    b1.fill_rand(0.0, 1.0);
    b2.fill_rand(0.0, 1.0);

    let mut a1 = (x * w1).unwrap() + b1;
    a1.as_ref().unwrap().print("a1");
    a1.as_mut().unwrap().sigmoid();
    a1.unwrap().print("a1");
}
