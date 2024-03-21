use aicaramba::functions::*;
use aicaramba::matrix::Mat;
use aicaramba::neural_net::NeuralNet;

fn main() {
    let mut net = NeuralNet::new(vec![2, 3, 1], RELU, MSE, 0.05);
    let epochs = 500;

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let expected = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    net.train_basic(inputs.clone(), expected, epochs);

    for input in inputs {
        let output = net.forward(Mat::from(input.clone()));
        let o = output.into_iter().collect::<Vec<_>>();
        println!("{} ^ {} = {:.20}", input[0], input[1], o[0]);
    }
}
