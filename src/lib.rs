use std::{fmt::Display, iter::Enumerate, vec};

use crate::MeasurementValueError::MismatchedSize;

#[derive(Debug, Clone, Copy)]
pub struct Measurement(f64, f64);
#[derive(Debug, Clone)]
pub struct MeasurementVec(Vec::<f64>,Vec::<f64>);

#[derive(Debug, Clone)]
pub enum MeasurementError<T> {
    Relative(T),
    Absolute(T),
}


#[derive(Debug, Clone)]
pub enum MeasurementValueError {
    MismatchedSize(usize, usize)
}
type ErrorResult<T> = std::result::Result<T, MeasurementValueError>;
impl Display for MeasurementValueError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self{
            MismatchedSize(values, errors) =>  write!(f,"Sizes of vectors is incorrect ({} items and {} items)", values, errors)
        }
       
    }
}

impl Measurement {
    pub fn value(&self) -> f64 {
        self.0
    }
    pub fn error(&self) -> f64 {
        self.1
    }
    pub fn rel_error(&self) -> f64 {
        self.1 / self.0
    }
    pub fn new(value: f64, error: MeasurementError<f64>) -> Self {
        let error_val = match error {
            MeasurementError::Absolute(val) => val,
            MeasurementError::Relative(rel) => rel * value,
        };
        Measurement(value, error_val)
    }
}

impl MeasurementVec {
    pub fn new(values:Vec::<f64>,errors:Vec<MeasurementError<f64>>) -> ErrorResult<Self> {
        let length = errors.len();
        if values.len() != length:
            return Err(MeasurementValueError::MismatchedSize(values.len(), length));
        let error_vec : Vec::<f64> = Vec::with_capacity(length);
        
        for (i, error) in errors.iter().enumerate(){
            match *error {
                MeasurementError::Relative(rel) => rel*values[i],
                MeasurementError::Absolute(err) => err
            }
        };
        MeasurementVec((), error_vec);
    }
}
