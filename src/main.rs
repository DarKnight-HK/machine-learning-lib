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
    let cost = nn.cost(&xor_inputs, &xor_outputs).unwrap();
    println!("Cost = {cost}")

    // for i in 0..2 {
    //     for j in 0..2 {
    //         nn.activations[0].data[0] = xor_inputs[i].clone();
    //         nn.forward();
    //         print!("{i} ^ {j} ");
    //         nn.activations[nn.count].print("Output");
    //         println!()
    //     }
    // }
}
