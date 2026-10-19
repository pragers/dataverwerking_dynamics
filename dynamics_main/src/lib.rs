use std::path::PathBuf;

use polars::prelude::*;

pub fn load_data(path_data_ball: PathBuf, path_data_stick: PathBuf) -> PolarsResult<(DataFrame,DataFrame)> {
    let ball_data_frame = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path_data_ball))?
        .finish()?;
    let stick_data_frame = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path_data_stick))?
        .finish()?;

    Ok((ball_data_frame,stick_data_frame))
}

