use aicaramba::functions::*;
use aicaramba::neural_net::NeuralNet;

fn main() {
    let mut net = NeuralNet::new(vec![2, 3, 1], SIGMOID, MSE, 0.05);
    let epochs = 10_000;

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let expected = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    net.train_basic(inputs, expected, epochs);
}
