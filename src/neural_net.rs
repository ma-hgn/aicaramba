use crate::functions::*;
use crate::matrix::{Mat, MatElem};

/// Contains the following values:
/// - `architecture: Vec<usize>`: The node counts for each layer (eg. `vec![2, 3, 1]`)
/// - `weights: Vec<Mat>`: The weight matrices between two layers.
/// - `biases: Vec<Mat>`: The bias matrices of the layers.
/// - `learning_rate: f64`: The scalar learning rate.
/// - `activation: ActivationFn`: Struct containing activation function and derivative
/// - `loss: LossFn`: Struct containing loss function and derivative
/// - `data: Vec<Mat>`: A buffer for the activated values during forward- and backward pass.
pub struct NeuralNet<T>
where
    T: MatElem,
{
    architecture: Vec<usize>,
    weights: Vec<Mat<T>>,
    biases: Vec<Mat<T>>,

    learning_rate: T,

    activation: ActivationFn<T>,
    loss: LossFn<T>,

    data: Vec<Mat<T>>,
}

impl<T> NeuralNet<T>
where
    T: MatElem,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    pub fn new(
        layers: Vec<usize>,
        activation: ActivationFn<T>,
        loss: LossFn<T>,
        learning_rate: T,
    ) -> Self {
        let mut weights = vec![];

        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Mat::random(layers[i + 1], layers[i]));
            biases.push(Mat::random(layers[i + 1], 1));
        }

        NeuralNet {
            architecture: layers,
            weights,
            biases,
            data: vec![],
            activation,
            loss,
            learning_rate,
        }
    }
}

impl<T> NeuralNet<T>
where
    T: MatElem,
{
    pub fn forward(&mut self, inputs: Mat<T>) -> Mat<T> {
        if self.architecture[0] != inputs.rows {
            panic!("Input vector does not have correct number of rows.")
        }

        let mut current = inputs;
        self.data = vec![current.clone()];

        for i in 0..self.architecture.len() - 1 {
            current = self.weights[i]
                .dot(&current)
                .add(&self.biases[i])
                .map(self.activation.f);

            self.data.push(current.clone());
        }

        current
    }

    pub fn backprop(&mut self, prediction: Mat<T>, truth: Mat<T>) {
        let mut losses = prediction.elementwise(&truth, self.loss.f_prime);
        let mut gradients = prediction.clone().map(self.activation.f_prime);

        for i in (0..self.architecture.len() - 1).rev() {
            gradients = gradients
                .elementwise_mul(&losses)
                .map(|x| x * self.learning_rate.clone());

            self.weights[i] = self.weights[i].add(&gradients.dot(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            losses = self.weights[i].transpose().dot(&losses);
            gradients = self.data[i].map(self.activation.f_prime);
        }
    }

    // TODO: add batch-wise training
    // TODO: refactor to use matrices instead of 2d-vecs
    pub fn train_basic(&mut self, inputs: Vec<Vec<T>>, truth: Vec<Vec<T>>, epochs: u32)
    where
        T: std::fmt::Display,
    {
        let width = epochs.ilog10() as usize + 1;

        for i in 1..=epochs {
            let mut outputs: Mat<T>;
            for j in 0..inputs.len() {
                outputs = self.forward(Mat::from(inputs[j].clone()));
                self.backprop(outputs, Mat::from(truth[j].clone()));
            }

            if epochs < 20 || i % (epochs / 20) == 0 {
                let mut loss = T::default();
                for j in 0..inputs.len() {
                    outputs = self.forward(Mat::from(inputs[j].clone()));
                    loss = loss
                        + outputs
                            .into_iter()
                            .zip(&truth[j])
                            .fold(T::default(), |sum, (y_hat, y)| {
                                sum + (self.loss.f)(y_hat, y.clone())
                            });
                }
                println!(
                    "epoch: {i:0>width$} / {epochs:0>width$} ;\tloss: {:.5}",
                    loss
                );
            }
        }
    }
}
