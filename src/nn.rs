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

    pub fn forward(&mut self) {}
}
