use crate::matrix::Matrix;

#[derive(Debug)]
pub struct NN {
    //inner layers = count
    pub count: usize,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub activations: Vec<Matrix>,
}

impl NN {
    pub fn new(architecture: &[usize]) -> Self {
        let mut nn = NN {
            count: architecture.len() - 1,
            weights: vec![],
            biases: vec![],
            activations: vec![],
        };
        nn.activations.push(Matrix::new(1, architecture[0]));
        for i in 1..architecture.len() {
            nn.weights
                .push(Matrix::new(nn.activations[i - 1].cols, architecture[i]));

            nn.biases.push(Matrix::new(1, architecture[i]));

            nn.activations.push(Matrix::new(1, architecture[i]));
        }
        nn
    }
    pub fn print(&self) {
        for i in 0..self.count {
            self.activations[i].print(&format!("activation[{}]", i));
            self.weights[i].print(&format!("weight[{}]", i));
            self.biases[i].print(&format!("bias[{}]", i));
        }
    }

    pub fn randomize(&mut self, low: f64, high: f64) {
        for i in 0..self.count {
            self.weights[i].fill_rand(low, high);
            self.biases[i].fill_rand(low, high);
        }
    }

    pub fn forward(&mut self) {
        for i in 0..self.count {
            self.activations[i + 1] =
                (&(&self.activations[i] * &self.weights[i]).unwrap() + &self.biases[i]).unwrap();
            self.activations[i + 1].sigmoid();
        }
    }

    pub fn cost(&mut self, features: &Matrix, labels: &Matrix) -> Result<f64, &'static str> {
        if features.rows != labels.rows {
            return Err("Rows should be same");
        }
        let mut cost = 0.0;
        for i in 0..features.rows {
            self.activations[0].data[0] = features.data[i].clone();
            self.forward();
            for j in 0..labels.cols {
                let diff = self.activations[self.count].data[0][j] - labels.data[0][j];
                cost += diff * diff;
            }
        }

        Ok(cost / features.rows as f64)
    }
}
