use machine_learning::nn::Matrix;
fn main() {
    let mut matrix1: Matrix<f32> = Matrix::new(4, 2);
    let mut matrix2: Matrix<f32> = Matrix::new(2, 4);
    matrix1.fill(1.0);
    matrix2.fill(2.01);
    let mat3 = matrix1 * matrix2;
    mat3.unwrap().print();
}
