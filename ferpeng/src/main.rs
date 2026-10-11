#![cfg(feature = "build-binary")]

use std::{collections::HashMap, error::Error};

use ferpeng::{
    formula::{self, Formula},
    prelude::*,
};
use inquire::{Confirm, CustomType, Text};

pub fn main() -> Result<(), Box<dyn Error>> {
    let formula_string = get_str_val("Enter formula", None);
    let mut variable_string = get_str_val(
        format!("Enter (comma seperated) variables for {} ", formula_string).as_str(),
        None,
    );
    variable_string.retain(|char| !char.is_whitespace());
    let variables: Vec<String> = variable_string.split(",").map(|f| f.to_string()).collect();

    let formula = Formula::new(formula_string, variables)?;
    println!("{:#?}", formula);
    loop {
        let mut variable_values = HashMap::<String, Measurement>::with_capacity(formula.vars.len());
        for var in &formula.vars {
            let amount = CustomType::<f64>::new(format!("{}:", var.name).as_str())
                .with_formatter(&|i| format!("{:.2}", i))
                .with_error_message("Please type a valid number")
                .with_help_message("Use decimal point as seperator")
                .prompt()?;
            let error = CustomType::<f64>::new(format!("error {}:", var.name).as_str())
                .with_formatter(&|i| format!("{:.2}", i))
                .with_error_message("Please type a valid number")
                .with_help_message("Use decimal point as seperator")
                .prompt()?;
            variable_values.insert(var.name.clone(), Measurement(amount, error));
        }
        let result = formula.evaluate(variable_values)?;
        println!("result: {}", result);
        let ans = Confirm::new("Exit?").with_default(false).prompt()?;
        if ans {
            break;
        }
    }

    Ok(())
}

fn get_str_val(prompt: &str, max_retries: Option<u32>) -> String {
    let mut amount_retries = 0;
    let max_retries = max_retries.unwrap_or(10);
    loop {
        let input = Text::new(prompt).prompt();
        match input {
            Ok(var) => break var,
            Err(_) => {
                if amount_retries < max_retries {
                    amount_retries += 1;
                } else {
                    panic!("Failed to get proper formula.")
                }
            }
        }
    }
}
