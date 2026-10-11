pub use crate::measurements::Measurement;

use std::{
    error, fmt,
    ops::{Deref, DerefMut},
};

#[derive(Debug, Clone)]
pub(crate) struct Wrapper<T>(T);
impl<T> Deref for Wrapper<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for Wrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
pub enum DataProcessingError {
    MismatchedSize(usize, usize),
    ParsingError(String),
    InvalidSymbol(String),
    EvaluationError(String),
    OutOfBounds,
}

pub type ErrorResult<T> = std::result::Result<T, DataProcessingError>;

impl error::Error for DataProcessingError {}

impl fmt::Display for DataProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataProcessingError::MismatchedSize(values, errors) => write!(
                f,
                "Sizes of vectors is incorrect ({} items and {} items)",
                values, errors
            ),
            DataProcessingError::OutOfBounds => {
                write!(f, "Index provided is out of bounds of the array.")
            }
            DataProcessingError::ParsingError(error_msg) => {
                write!(f, "Parsing failed with error: {}", error_msg)
            }
            DataProcessingError::InvalidSymbol(symbol) => {
                write!(f, "Encountered invalid symbol: {}", symbol)
            }
            DataProcessingError::EvaluationError(error_msg) => {
                write!(f, "Evaluation failed with error: {}", error_msg)
            }
        }
    }
}
