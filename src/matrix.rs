use rand::Rng;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn print(&self, name: &str) {
        println!("{} = [", name);
        for row in self.data.iter() {
            for value in row.iter() {
                print!("    {:.5} ", value);
            }
            println!();
        }
        println!("]");
    }

    pub fn fill(&mut self, value: f64) {
        for row in self.data.iter_mut() {
            for mat_value in row.iter_mut() {
                *mat_value = value;
            }
        }
    }

    pub fn fill_rand(&mut self, mut low: f64, mut high: f64) {
        if low > high {
            std::mem::swap(&mut low, &mut high);
        }
        let mut rng = rand::thread_rng();
        for row in self.data.iter_mut() {
            for value in row.iter_mut() {
                *value = rng.gen_range(low..high);
            }
        }
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.data[i][j];
                let sigmoid_value = 1.0 / (1.0 + (-value).exp());
                self.data[i][j] = sigmoid_value;
            }
        }
    }
}

impl<'a> Add<&'a Matrix> for &'a Matrix {
    type Output = Result<Matrix, &'static str>;

    fn add(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimensions must be the same");
        }

        let mut new_mat = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_mat.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        Ok(new_mat)
    }
}

impl<'a> Mul<&'a Matrix> for &'a Matrix {
    type Output = Result<Matrix, &'static str>;

    fn mul(self, other: Self) -> Self::Output {
        if self.cols != other.rows {
            return Err("Inner dimensions must be same");
        }

        let mut new_mat = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                new_mat.data[i][j] = 0.0;
                for k in 0..self.cols {
                    new_mat.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Ok(new_mat)
    }
}
