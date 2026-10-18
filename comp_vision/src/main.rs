use std::path::Path;

use comp_vision::{tracker::TrackerType, video::Video};

fn main() -> opencv::Result<()> {
    #[cfg(debug_assertions)]
    env_logger::init();

    #[cfg(debug_assertions)]
    let path = inquire::Text::new("Enter path")
        .with_default(".")
        .prompt()
        .map_err(|_| opencv::Error::new(-2, "Failed to get path from user input"))?;

    let path = Path::new(&path);

    let vid = Video::new(path)?;

    let tracked_vid = vid
        .add_tracker(
            TrackerType::OSTrack,
            Path::new(r"B:\Projects\dataverwerking_dynamics\models"),
        )?
        .track()?;

    tracked_vid.save_to_csv(Path::new("./output.csv"))?;

    Ok(())
}
