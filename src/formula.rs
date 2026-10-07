use crate::prelude::*;
use std::collections::HashMap;
use symplex::prelude::*;

#[derive(Debug, Clone)]
pub struct Formula {
    formula: String,
    expression: Ex,
    vars: Vec<MeasurementSymbol>,
    ctx: Context,
}

#[derive(Debug, Clone)]
pub struct MeasurementSymbol {
    name: String,
    symbol: Ex,
    error_symbol: Ex,
}

impl MeasurementSymbol {
    pub fn from_string(ctx: &Context, name: String) -> ErrorResult<MeasurementSymbol> {
        if name.contains("err") {
            return Err(DataProcessingError::InvalidSymbol(name));
        }

        if name.is_empty() {
            return Err(DataProcessingError::InvalidSymbol(name));
        }

        let symbol = ctx.symbol(&name);
        let error_symbol = ctx.symbol(&format!("{name}_err"));

        Ok(MeasurementSymbol {
            name,
            symbol,
            error_symbol,
        })
    }
}

impl Formula {
    pub fn new(formula: String, variable_names: Vec<String>) -> ErrorResult<Self> {
        let ctx = Context::new();

        // Validate the formula immediately.
        let expression = ctx
            .parse(&formula)
            .map_err(|e| DataProcessingError::ParsingError(e.to_string()))?;

        let vars = variable_names
            .into_iter()
            .map(|name| MeasurementSymbol::from_string(&ctx, name))
            .collect::<ErrorResult<Vec<MeasurementSymbol>>>()?;

        Ok(Self {
            formula,
            expression,
            vars,
            ctx,
        })
    }

    pub fn get_error_formula(&self) -> ErrorResult<Formula> {
        let expression = self
            .ctx
            .parse(&self.formula)
            .map_err(|e| DataProcessingError::ParsingError(e.to_string()))?;

        let mut result = self.ctx.int(0);

        for var in &self.vars {
            let derivative = expression.diff(&var.symbol);

            let term = (&derivative * &var.error_symbol).powi(2);

            result = result + term;
        }
        result = result.sqrt();
        Ok(Formula {
            formula: result.to_string(),
            expression: result,
            vars: self.vars.clone(),
            ctx: self.ctx.clone(),
        })
    }
    pub fn evaluate(&self, values: HashMap<String, Measurement>) -> ErrorResult<Measurement> {
        let final_value = {
            let substitutions_values: Vec<(&Ex, f64)> = self
                .vars
                .iter()
                .filter_map(|var| values.get(&var.name).map(|value| (&var.symbol, value.0)))
                .collect();
            self.expression
                .eval_f64_with(&substitutions_values)
                .map_err(|e| DataProcessingError::EvaluationError(e.to_string()))
        }?;
        let final_error = {
            let substitutions_error: Vec<(&Ex, f64)> = self
                .vars
                .iter()
                .filter_map(|var| values.get(&var.name).map(|value| (&var.symbol, value.0)))
                .collect();
            self.get_error_formula()?
                .expression
                .eval_f64_with(&substitutions_error)
                .map_err(|e| DataProcessingError::EvaluationError(e.to_string()))
        }?;
        return Ok(Measurement(final_value, final_error));
    }
}
