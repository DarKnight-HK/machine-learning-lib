use machine_learning::nn::NN;

fn main() {
    let arch: [usize; 4] = [2, 2, 2, 1];
    let mut nn = NN::new(&arch);
    nn.randomize(0.0, 1.0);
    nn.print();
}
