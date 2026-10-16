use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use log::warn;
use opencv::{
    core::{Mat, MatTraitConst, Size_},
    highgui::imshow,
    videoio::{
        self, VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, VideoWriter,
        VideoWriterTrait, VideoWriterTraitConst,
    },
};

use crate::tracker::{AnyTracker, Coordinates, TrackerType, create_tracker, get_roi_box};

pub mod filter;
pub mod tracker;

pub struct Loaded;
pub struct WithTracker {
    tracker: Box<dyn AnyTracker>,
}
pub struct TrackingFinished {
    output: Vec<Coordinates>,
    video: VideoWriter,
}

pub struct Video<S> {
    path: PathBuf,
    vid_cap: VideoCapture,
    fps: f64,
    length: f64,
    frame: Mat,
    state: S,
}

impl<v> Video<v> {
    fn new(path: &Path) -> opencv::Result<Video<Loaded>> {
        todo!("Video struct is not yet implemented")
    }
    fn get_fps(&self) -> f64 {
        self.fps
    }
    fn get_len(&self) -> f64 {
        self.length
    }
    fn get_frame(&self) -> Mat {
        self.frame.to_owned()
    }
    fn get_path(&self) -> PathBuf {
        self.path.to_owned()
    }
    fn get_state(self) -> v {
        self.state
    }
}
impl Video<Loaded> {
    fn add_tracker(
        self,
        tracker_type: TrackerType,
        model_dir: &Path,
    ) -> opencv::Result<Video<WithTracker>> {
        let tracker = create_tracker(tracker_type, model_dir)?;
        Ok(Video {
            path: self.path,
            vid_cap: self.vid_cap,
            fps: self.fps,
            length: self.length,
            frame: self.frame,
            state: WithTracker { tracker },
        })
    }
}
impl Video<WithTracker> {
    fn track(mut self) -> opencv::Result<Video<TrackingFinished>> {
        let mut bbox = get_roi_box(&self.frame)?;
        let mut output_video = create_output_video("output.mp4", self.fps, self.frame.size()?)?;
        let mut output_vec: Vec<Coordinates> =
            Vec::with_capacity(self.vid_cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize);
        let mut failed_frames = 0;
        while self.vid_cap.read(&mut self.frame)? && !self.frame.empty() {
            if self.state.tracker.update(&self.frame, &mut bbox)? {
                failed_frames = 0;
                opencv::imgproc::rectangle(
                    &mut self.frame,
                    bbox,
                    opencv::core::Scalar::new(255.0, 0.0, 0.0, 255.0),
                    1,
                    opencv::imgproc::LINE_8,
                    0,
                )?;
                output_video.write(&self.frame)?;
                imshow("Frame", &self.frame)?;
            } else {
                failed_frames += 1;
                if failed_frames > 5 {
                    println!("Tracking failed, object lost");
                    break;
                }
            }
            output_vec.push(bbox.into());
        }
        Ok(Video {
            path: self.path,
            vid_cap: self.vid_cap,
            fps: self.fps,
            length: self.length,
            frame: self.frame,
            state: TrackingFinished {
                output: output_vec,
                video: output_video,
            },
        })
    }
}
impl Video<TrackingFinished> {
    fn save_to_csv(&self, path: &Path) -> opencv::Result<()> {
        if path.parent().is_none() || path.parent().is_some_and(|s| !s.exists()) {
            return Err(opencv::Error::new(
                opencv::core::StsError,
                "Path does not exist.",
            ));
        }
        let Ok(mut csv_out) = csv::Writer::from_path(path) else {
            return Err(opencv::Error::new(
                opencv::core::StsError,
                "Failed to open output csv.",
            ));
        };
        let mut amount_failed = 0;
        self.state
            .output
            .iter()
            .for_each(|val| match csv_out.serialize(val) {
                Ok(_) => {}
                Err(_) => amount_failed += 1,
            });
        #[cfg(debug_assertions)]
        warn!("failed {} writes to csv", amount_failed);
        Ok(())
    }
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
    Ok((vid_cap, frame))
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
    Ok(output_video)
}
