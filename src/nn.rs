use rand::prelude::Distribution;
use rand::{distributions::Standard, Rng};
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul};

pub trait MatrixOps:
    Add<Output = Self> + Copy + Default + Display + PartialOrd + Mul<Output = Self> + AddAssign
{
}

#[derive(Debug)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<T>>,
}

impl<T> MatrixOps for T
where
    T: Add<Output = T>
        + AddAssign
        + Copy
        + Default
        + Display
        + rand::distributions::uniform::SampleUniform
        + PartialOrd
        + Mul<Output = T>,
    Standard: Distribution<T>,
{
}

impl<T> Matrix<T>
where
    T: MatrixOps + rand::distributions::uniform::SampleUniform,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![T::default(); cols]; rows],
        }
    }

    pub fn print(&self, name: &str) {
        println!("{} = [", name);
        for row in self.data.iter() {
            for value in row.iter() {
                print!("    {:.2} ", value);
            }
            println!();
        }
        println!("]");
    }

    pub fn fill(&mut self, value: T) {
        for row in self.data.iter_mut() {
            for mat_value in row.iter_mut() {
                *mat_value = value;
            }
        }
    }

    pub fn fill_rand(&mut self, mut low: T, mut high: T) {
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
}

impl<T> Add for Matrix<T>
where
    T: MatrixOps + rand::distributions::uniform::SampleUniform,
{
    type Output = Result<Matrix<T>, &'static str>;

    fn add(self, other: Matrix<T>) -> Self::Output {
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

impl<T> Mul for Matrix<T>
where
    T: MatrixOps + rand::distributions::uniform::SampleUniform,
{
    type Output = Result<Matrix<T>, &'static str>;

    fn mul(self, other: Matrix<T>) -> Self::Output {
        if self.cols != other.rows {
            return Err("Inner dimensions must be same");
        }

        let mut new_mat = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                new_mat.data[i][j] = T::default();
                for k in 0..self.cols {
                    new_mat.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Ok(new_mat)
    }
}
