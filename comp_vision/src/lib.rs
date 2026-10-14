use std::{ffi::OsStr, marker::PhantomData};

use opencv::{
    core::{Mat, MatTraitConst, Size_},
    videoio::{
        self, VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, VideoWriter,
        VideoWriterTrait, VideoWriterTraitConst,
    },
};

use crate::tracker::{AnyTracker, Coordinates};

pub mod filter;
pub mod tracker;

pub struct Loaded;
pub struct WithTracker {
    tracker: Box<dyn AnyTracker>,
    frame: Mat,
}
pub struct Tracking_finished {
    output: Vec<Coordinates>,
}

pub struct Video<S> {
    vid_cap: VideoCapture,
    fps: f64,
    length: f64,
    state: S,
}

pub fn open_video(filename: &OsStr) -> opencv::Result<(VideoCapture, Mat)> {
    let mut vid_cap = videoio::VideoCapture::from_file(filename, opencv::videoio::CAP_ANY)?;

    if !vid_cap.is_opened()? {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "Failed to open video stream or file.",
        ));
    }

    let mut frame = opencv::core::Mat::default();
    if !vid_cap.read(&mut frame)? || frame.empty() {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "Could not read first frame.",
        ));
    }
    return Ok((vid_cap, frame));
}

pub fn create_output_video(
    filename: impl AsRef<OsStr>,
    fps: f64,
    frame_size: Size_<i32>,
) -> opencv::Result<VideoWriter> {
    let mut output_video = VideoWriter::default()?;
    output_video.open(
        filename,
        VideoWriter::fourcc('m', 'p', '4', 'v')?,
        fps,
        frame_size,
        true,
    )?;

    if !output_video.is_opened()? {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "Failed to open video.",
        ));
    }
    return Ok(output_video);
}
