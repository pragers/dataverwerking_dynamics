use crate::prelude::*;

#[derive(Debug, Clone)]
pub enum MeasurementError<T> {
    Relative(T),
    Absolute(T),
}

impl MeasurementError<f64> {
    pub fn get_val(&self, value: f64) -> f64 {
        match self {
            MeasurementError::Absolute(val) => *val,
            MeasurementError::Relative(val) => value * val,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Measurement(pub f64, pub f64);

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

#[derive(Debug, Clone)]
pub struct MeasurementVec(Vec<f64>, Vec<f64>);

impl MeasurementVec {
    pub fn new(values: Vec<f64>, errors: Vec<MeasurementError<f64>>) -> ErrorResult<Self> {
        let length = errors.len();
        if values.len() != length {
            return Err(DataProcessingError::MismatchedSize(values.len(), length));
        }
        let error_vec = errors
            .iter()
            .zip(&values)
            .map(|(error, value)| -> f64 {
                match *error {
                    MeasurementError::Absolute(val) => val,
                    MeasurementError::Relative(rel) => rel * value,
                }
            })
            .collect();
        Ok(MeasurementVec(values, error_vec))
    }
    pub fn index(&self, idx: usize) -> ErrorResult<Measurement> {
        if idx >= self.0.len() {
            return Err(DataProcessingError::OutOfBounds);
        }
        Ok(Measurement(self.0[idx], self.1[idx]))
    }
}
