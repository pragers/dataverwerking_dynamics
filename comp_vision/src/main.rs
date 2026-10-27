use std::{fs, path::Path};

use comp_vision::{tracker::TrackerType, video::Video};

fn main() -> opencv::Result<()> {
    let path = inquire::Text::new("Enter path")
        .with_default(".")
        .prompt()
        .map_err(|_| opencv::Error::new(-2, "Failed to get path from user input"))?;

    let path = Path::new(&path);
    for item in fs::read_dir(path).expect("Cant read dir") {
        let file = item.expect("Entry failed");
        let path = file.path();
        let name = path.file_stem().expect("File has no name");
        let Ok(vid) = Video::new(file.path()) else {
            println!("Video {:#?} failed to be initialized", file.path());
            continue;
        };

        let tracked_vid = match vid
            .add_tracker(
                TrackerType::OSTrack,
                Path::new(r"B:\Projects\dataverwerking_dynamics\models"),
            )?
            .track()
        {
            Ok(tracked_vid) => tracked_vid,
            Err(msg) => {
                println!("{msg:#?}");
                println!("Video {:#?} failed to be tracked", file.path());
                continue;
            }
        };

        let csv_name = format!("./output/data/data_{}.csv", name.to_string_lossy());
        tracked_vid.save_to_csv(Path::new(&csv_name))?;
    }

    Ok(())
}
