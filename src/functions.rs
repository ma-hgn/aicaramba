use crate::matrix::MatElem;

#[derive(Clone, Copy, Debug)]
pub struct ActivationFn<T>
where
    T: MatElem,
{
    pub f: fn(T) -> T,
    pub f_prime: fn(T) -> T,
}

#[derive(Clone, Copy, Debug)]
pub struct LossFn<T>
where
    T: MatElem,
{
    pub f: fn(T, T) -> T,
    pub f_prime: fn(T, T) -> T,
}

pub const SIGMOID: ActivationFn<f64> = ActivationFn {
    f: |x| 1.0 / (1.0 + f64::exp(-x)),
    f_prime: |x| x * (1.0 - x),
};

pub const MSE: LossFn<f64> = LossFn {
    f: |y_hat, y| (y_hat - y).powi(2),
    f_prime: |y_hat, y| -2.0 * (y_hat - y),
};
