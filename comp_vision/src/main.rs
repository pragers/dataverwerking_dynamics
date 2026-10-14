use std::{ffi::OsStr, path::Path};

use comp_vision::{
    create_output_video, open_video,
    tracker::{Coordinates, TrackerType, create_tracker, get_roi_box},
};
use opencv::{
    core::MatTraitConst,
    highgui::{destroy_all_windows, imshow, wait_key},
    videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst, VideoWriterTrait},
};

fn main() -> opencv::Result<()> {
    #[cfg(debug_assertions)]
    env_logger::init();

    let (mut vid_cap, mut frame) =
        open_video(OsStr::new(r"B:\recordings\2025-03-15_16-30-30.mp4"))?;

    let mut tracker = create_tracker(
        TrackerType::VIT,
        Path::new("B:/Projects/dataverwerking_dynamics/models"),
    )?;
    println!("{:#?}", tracker);
    let mut box1 = get_roi_box(&frame)?;

    let frame_size = frame.size()?;
    let fps = vid_cap.get(videoio::CAP_PROP_FPS)?;

    let mut output_video = create_output_video("output.mp4", fps, frame_size)?;

    tracker.init(&frame, box1)?;

    println!("{:.0} fps", fps);
    let total_frame_count = vid_cap.get(videoio::CAP_PROP_FRAME_COUNT)?;
    println!("{:.2}s duration", total_frame_count / fps);
    let mut output_vec: Vec<Coordinates> = Vec::new();
    let mut failed_frames = 0;
    while vid_cap.read(&mut frame)? && !frame.empty() {
        if tracker.update(&frame, &mut box1)? {
            failed_frames = 0;
            opencv::imgproc::rectangle(
                &mut frame,
                box1,
                opencv::core::Scalar::new(255.0, 0.0, 0.0, 255.0),
                1,
                opencv::imgproc::LINE_8,
                0,
            )?;

            output_video.write(&frame)?;
            imshow("Frame", &frame)?;
        } else {
            failed_frames += 1;
            if failed_frames > 5 {
                println!("Tracking failed, object lost");
                break;
            }
        }
        output_vec.push(box1.into());
        let key = wait_key(1)?;
        if key == 113 {
            break;
        }
    }
    let Ok(mut csv_out) = csv::Writer::from_path("Output.csv") else {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "Failed to open output csv.",
        ));
    };
    output_vec
        .iter()
        .for_each(|val| match csv_out.serialize(val) {
            Ok(_) => {}
            Err(_) => {}
        });
    output_video.release()?;
    vid_cap.release()?;
    destroy_all_windows()?;
    Ok(())
}
