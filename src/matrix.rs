use rand::Rng;
use std::ops::{Add, Mul, Sub};

// NOTE: might want to rethink design (to 2d-vec?) to enable `matrix[i][j]`
// indexing and make nice row-iterator implementation possible
#[derive(Debug, Clone)]
pub struct Mat<T>
where
    T: MatElem,
{
    pub rows: usize,
    pub cols: usize,
    data: Vec<T>,
}

/// Shorthand/alias trait for types that are valid as matrix elements.
pub trait MatElem:
    PartialEq + Clone + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
}

impl<T> MatElem for T where
    T: PartialEq + Clone + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>
{
}

impl<T> Mat<T>
where
    T: MatElem,
{
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Mat<T> {
        assert!(data.len() == rows * cols, "Invalid Size");
        Mat { rows, cols, data }
    }

    pub fn at(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.cols + col]
    }

    pub fn default_with_size(rows: usize, cols: usize) -> Mat<T> {
        Mat {
            rows,
            cols,
            data: vec![T::default(); cols * rows],
        }
    }

    pub fn add(&self, other: &Mat<T>) -> Mat<T> {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to add matrices with differing shapes.");
        }
        self.elementwise(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Mat<T>) -> Mat<T>
    where
        T: std::ops::Sub<Output = T>,
    {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to subtract matrices with differing shapes.");
        }
        self.elementwise(other, |a, b| a - b)
    }

    pub fn elementwise_mul(&self, other: &Mat<T>) -> Mat<T>
    where
        T: std::ops::Mul<Output = T>,
    {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to elementwise-multiply matrices of differing shapes.");
        }
        self.elementwise(other, |a, b| a * b)
    }

    pub fn elementwise(&self, other: &Mat<T>, f: fn(T, T) -> T) -> Mat<T> {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to apply element-wise operation to matrices with differing shapes.");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| f(a.clone(), b.clone()))
            .collect::<Vec<_>>();

        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn dot(&self, other: &Mat<T>) -> Mat<T> {
        if self.cols != other.rows {
            panic!(
                "Attempted to take dot product of incompatible matrix shapes. (A.cols != B.rows)"
            );
        }

        let mut data = vec![T::default(); self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum
                        + self.data[i * self.cols + k].clone()
                            * other.data[k * other.cols + j].clone();
                }
                data[i * other.cols + j] = sum;
            }
        }

        Mat {
            rows: self.rows,
            cols: other.cols,
            data,
        }
    }

    pub fn transpose(&self) -> Mat<T> {
        let mut buffer = vec![T::default(); self.cols * self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                buffer[j * self.rows + i] = self.data[i * self.cols + j].clone();
            }
        }

        Mat {
            rows: self.cols,
            cols: self.rows,
            data: buffer,
        }
    }

    pub fn map<F>(&self, f: F) -> Mat<T>
    where
        F: FnMut(T) -> T,
    {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone().into_iter().map(f).collect(),
        }
    }
}

pub trait Collect<T>
where
    T: MatElem,
{
    fn collect_mat(self, rows: usize, cols: usize) -> Mat<T>;
}

impl<T> Collect<T> for T
where
    T: MatElem + std::iter::IntoIterator<Item = T>,
{
    fn collect_mat(self, rows: usize, cols: usize) -> Mat<T> {
        let data = self.into_iter().collect::<Vec<T>>();
        if data.len() != rows * cols {
            panic!("Collecting iterator into matrix failed due to incompatible matrix shape.")
        }
        Mat { rows, cols, data }
    }
}

// the random function is only available if `rand` supports randomizing the element type
impl<T> Mat<T>
where
    T: MatElem,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    // TODO: depend on randomization feature
    pub fn random(rows: usize, cols: usize) -> Mat<T> {
        let mut data = Vec::with_capacity(rows * cols);

        for _ in 0..rows * cols {
            data.push(rand::thread_rng().gen());
        }

        Mat { rows, cols, data }
    }
}

// NOTE: might want to change this to two row- and col-iters in the future
//       then might implement something like `flat_iter` that mirrors
//       current behavior.
impl<T> IntoIterator for Mat<T>
where
    T: MatElem,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T> From<Vec<T>> for Mat<T>
where
    T: MatElem,
{
    fn from(value: Vec<T>) -> Self {
        let rows = value.len();
        let cols = 1;
        Mat {
            rows,
            cols,
            data: value,
        }
    }
}

impl<T> From<Vec<Vec<T>>> for Mat<T>
where
    T: MatElem,
{
    fn from(value: Vec<Vec<T>>) -> Self {
        let rows = value.len();
        let cols = value.first().map(Vec::len).unwrap_or(0);
        Mat {
            rows,
            cols,
            data: value.into_iter().flatten().collect(),
        }
    }
}

impl<T> PartialEq for Mat<T>
where
    T: MatElem,
{
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl<T> std::ops::Index<usize> for Mat<T>
where
    T: MatElem,
{
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.cols..(index + 1) * self.cols]
    }
}

impl<T> std::fmt::Display for Mat<T>
where
    T: MatElem + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{}", self.data[row * self.cols + col])?;
                if col < self.cols - 1 {
                    write!(f, "\t")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! matrix {
    ( $( $($val:expr),+ );* $(;)? ) => {
        {
            let mut data = Vec::<f64>::new();
            let mut rows = 0;
            let mut cols = 0;
            $(
                let row_data = vec![$($val),+];
                data.extend(row_data);
                rows += 1;
                let row_len = vec![$($val),+].len();
                if cols == 0 {
                    cols = row_len;
                } else if cols != row_len {
                    panic!("Inconsistent number of elements in the matrix rows");
                }
            )*

            Mat {
                rows,
                cols,
                data,
            }
        }
    };
}
