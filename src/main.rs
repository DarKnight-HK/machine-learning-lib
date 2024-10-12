use machine_learning::nn::NN;

fn main() {
    let arch: [usize; 3] = [2, 2, 1];
    let nn = NN::new(&arch);
    print!("{:?}", nn)
}
