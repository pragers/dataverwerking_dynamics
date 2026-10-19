use std::{
    ffi::OsStr,
    fmt::Debug,
    path::{Path, PathBuf},
};

use opencv::{
    core::{Mat, MatTraitConst, Point_, Size_},
    highgui::{destroy_all_windows, imshow, select_roi_1, wait_key},
    videoio::{
        self, VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst, VideoWriter,
        VideoWriterTrait, VideoWriterTraitConst,
    },
};

use crate::{
    prelude::WithCenter,
    tracker::{AnyTracker, Coordinates, TrackerType, create_tracker},
};

pub struct Loaded;
pub struct WithTracker {
    tracker: Box<dyn AnyTracker>,
}
#[derive(Debug)]
pub struct TrackingFinished {
    output: Vec<Coordinates>,
    video: VideoWriter,
}

#[derive(Debug)]
pub struct Video<S> {
    path: PathBuf,
    vid_cap: VideoCapture,
    fps: f64,
    length: f64,
    frame: Mat,
    state: S,
}

impl<V> Video<V> {
    pub fn get_fps(&self) -> f64 {
        self.fps
    }
    pub fn get_len(&self) -> f64 {
        self.length
    }
    pub fn get_frame(&self) -> Mat {
        self.frame.to_owned()
    }
    pub fn get_path(&self) -> PathBuf {
        self.path.to_owned()
    }
    pub fn get_state_owned(self) -> V {
        self.state
    }
    pub fn get_state(&self) -> &V {
        &self.state
    }
}

impl Video<Loaded> {
    pub fn new(path: PathBuf) -> opencv::Result<Video<Loaded>> {
        let mut vid_cap = videoio::VideoCapture::from_file(&path, opencv::videoio::CAP_ANY)?;

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
        let frame_count = vid_cap.get(videoio::CAP_PROP_FRAME_COUNT)?;
        let fps = vid_cap.get(videoio::CAP_PROP_FPS)?;

        Ok(Video {
            path,
            vid_cap,
            fps,
            length: frame_count / fps,
            frame,
            state: Loaded,
        })
    }
    pub fn add_tracker(
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
    pub fn track(mut self) -> opencv::Result<Video<TrackingFinished>> {
        let mut bbox = get_roi_box(&self.frame)?;
        self.state.tracker.init(&self.frame, bbox)?;

        let mut output_video = create_output_video(
            format!(
                "./output_6/{}.mp4",
                self.path
                    .file_stem()
                    .expect("File has no name")
                    .to_string_lossy()
            ),
            self.fps,
            self.frame.size()?,
        )?;
        let mut output_vec: Vec<Coordinates> =
            Vec::with_capacity(self.vid_cap.get(videoio::CAP_PROP_FRAME_COUNT)? as usize);

        let mut display_frame: Mat;
        while self.vid_cap.read(&mut self.frame)? && !self.frame.empty() {
            self.state.tracker.update(&self.frame, &mut bbox)?;

            display_frame = self.frame.clone();

            opencv::imgproc::rectangle(
                &mut display_frame,
                bbox,
                opencv::core::Scalar::new(255.0, 0.0, 0.0, 255.0),
                1,
                opencv::imgproc::LINE_8,
                0,
            )?;

            let bbox_center = bbox.get_center();

            opencv::imgproc::circle(
                &mut display_frame,
                Point_::new(bbox_center.0 as i32, bbox_center.1 as i32),
                1,
                opencv::core::Scalar::new(255.0, 0.0, 0.0, 255.0),
                3,
                opencv::imgproc::LINE_8,
                0,
            )?;

            output_video.write(&display_frame)?;

            imshow("Frame", &display_frame)?;
            if wait_key(1)? == 27 {
                break; // ESC to exit
            }

            output_vec.push(bbox.into());
        }
        destroy_all_windows()?;
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
    pub fn save_to_csv(&self, path: &Path) -> opencv::Result<()> {
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

        Ok(())
    }
}

fn create_output_video(
    filename: impl AsRef<OsStr> + Debug,
    fps: f64,
    frame_size: Size_<i32>,
) -> opencv::Result<VideoWriter> {
    let mut output_video = VideoWriter::default()?;
    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    println!("{:#?}", filename);
    println!("Output path: {:?}", filename);
    println!("FPS: {}", fps);
    println!("Frame size: {:?}", frame_size);
    println!("FourCC: {}", fourcc);

    output_video.open(filename, fourcc, fps, frame_size, true)?;

    if !output_video.is_opened()? {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "Failed to open output video.",
        ));
    }

    Ok(output_video)
}

// Get the bounding box of the item at frame 1.
pub fn get_roi_box(frame: &Mat) -> opencv::Result<opencv::core::Rect> {
    let box1 = select_roi_1(&frame, true, false, false)?;

    if box1.area() == 0 {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "No ROI is selected, Exiting...",
        ));
    }

    destroy_all_windows()?;

    Ok(box1)
}
