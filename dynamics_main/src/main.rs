use std::fs::File;

use dynamics_main::load_data;
use polars::{prelude::*, series::ops::NullBehavior};
fn main() -> PolarsResult<()> {
    let (mut data_ball, _data_stick) = load_data("./output.csv".into(), "./output.csv".into())?;
    let mut dx = diff(
        &data_ball.column("x")?.as_materialized_series(),
        4,
        NullBehavior::Ignore,
    )?
    .into_column();
    dx.rename("dx".into());
    let mut vx = &dx / (4.0 / 60.0);
    vx.rename("vx".into());

    data_ball.with_column(vx)?;
    println!("{}", data_ball);
    let mut file = File::create("Test.csv").expect("Could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut data_ball)?;
    Ok(())
}
