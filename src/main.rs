use machine_learning::{matrix::Matrix, nn::NN};

fn main() {
    //xor input data
    let mut xor_inputs = Matrix::new(4, 2);
    xor_inputs.data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let mut xor_outputs = Matrix::new(4, 1);
    xor_outputs.data = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let arch: [usize; 3] = [2, 2, 1];
    let mut nn = NN::new(&arch);
    nn.randomize(0.0, 1.0);
    let mut grad = NN::new(&arch);
    grad.randomize(0.0, 1.0);
    let cost = nn.cost(&xor_inputs, &xor_outputs).unwrap();
    println!("Cost = {cost}");
    for _ in 0..1000000 {
        nn.finite_diff(&mut grad, 1e-1, &xor_inputs, &xor_outputs);
        nn.learn(&mut grad, 1e-1);
    }
    let cost = nn.cost(&xor_inputs, &xor_outputs).unwrap();
    println!("Cost = {cost}");

    for i in 0..2 {
        for j in 0..2 {
            nn.parameters[0].data[0][0] = i as f64;
            nn.parameters[0].data[0][1] = j as f64;
            nn.forward();
            println!("Prediction: {:?}", nn.parameters[nn.count].data[0][0])
        }
    }
}
