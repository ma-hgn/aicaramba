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

pub const RELU: ActivationFn<f64> = ActivationFn {
    f: |x| x.max(0.0),
    f_prime: |x| if x > 0.0 { 1.0 } else { 0.0 },
};

pub const MSE: LossFn<f64> = LossFn {
    f: |y_hat, y| (y - y_hat).powi(2),
    f_prime: |y_hat, y| -2.0 * (y - y_hat),
};
