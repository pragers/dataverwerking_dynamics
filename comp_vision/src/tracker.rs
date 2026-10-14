use std::path::Path;

use opencv::{
    core::{self, Mat, Rect_},
    dnn,
    highgui::{destroy_all_windows, select_roi_1},
    video::{
        self, TrackerDaSiamRPN, TrackerDaSiamRPN_Params, TrackerDaSiamRPN_ParamsTrait,
        TrackerNano_ParamsTrait, TrackerVit_ParamsTrait,
    },
};

#[derive(serde::Serialize)]
pub struct Coordinates {
    x: f64,
    y: f64,
}

impl Coordinates {
    pub fn new(x: f64, y: f64) -> Self {
        return Self { x, y };
    }
    pub fn x(&self) -> f64 {
        self.x
    }
    pub fn y(&self) -> f64 {
        self.y
    }
}

impl From<Rect_<i32>> for Coordinates {
    fn from(value: Rect_<i32>) -> Self {
        let x = (value.x as f64) + (value.width as f64 / 2.);
        let y = (value.y as f64) + (value.height as f64 / 2.);
        return Self { x, y };
    }
}

// trait that implements the functions for trackers such that then can be returned as a box<dyn> object.

pub trait AnyTracker: std::fmt::Debug {
    fn init(&mut self, frame: &core::Mat, bbox: core::Rect) -> opencv::Result<()>;
    fn update(&mut self, frame: &core::Mat, bbox: &mut core::Rect) -> opencv::Result<bool>;
}

impl<T> AnyTracker for T
where
    T: video::TrackerTrait + std::fmt::Debug,
{
    fn init(&mut self, frame: &core::Mat, bbox: core::Rect) -> opencv::Result<()> {
        video::TrackerTrait::init(self, frame, bbox)
    }
    fn update(&mut self, frame: &core::Mat, bbox: &mut core::Rect) -> opencv::Result<bool> {
        video::TrackerTrait::update(self, frame, bbox)
    }
}

// Tracker type that decides which tracker will be used
#[derive(Debug, Clone)]
pub enum TrackerType {
    DASIAMRPN,
    NANO,
    VIT,
    MIL,
}

// Create a tracker using the models in model_dir
pub fn create_tracker(
    variant: TrackerType,
    model_dir: &Path,
) -> opencv::Result<Box<dyn AnyTracker>> {
    match variant {
        TrackerType::DASIAMRPN => {
            let mut params = TrackerDaSiamRPN_Params::default()?;

            let model_str = model_dir
                .join("object_tracking_dasiamrpn_model_2021nov.onnx")
                .to_string_lossy()
                .into_owned();
            params.set_model(&model_str);

            let kernel_cls1_str = model_dir
                .join("object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx")
                .to_string_lossy()
                .into_owned();
            params.set_kernel_cls1(&kernel_cls1_str);

            let kernel_r1_str = model_dir
                .join("object_tracking_dasiamrpn_kernel_r1_2021nov.onnx")
                .to_string_lossy()
                .into_owned();
            params.set_kernel_r1(&kernel_r1_str);

            let (backend, target) = select_dnn_backend()?;

            params.set_backend(backend.into());
            params.set_target(target.into());

            let tracker = TrackerDaSiamRPN::create(&params)?;

            return Ok(Box::new(tracker));
        }
        TrackerType::NANO => {
            let mut params = video::TrackerNano_Params::default()?;

            let backbone_str = model_dir
                .join("nanotrack_backbone_sim.onnx")
                .to_string_lossy()
                .into_owned();

            params.set_backbone(&backbone_str);

            let neckhead_str = model_dir
                .join("nanotrack_head_sim.onnx")
                .to_string_lossy()
                .into_owned();

            params.set_neckhead(&neckhead_str);

            let (backend, target) = select_dnn_backend()?;

            params.set_backend(backend.into());
            params.set_target(target.into());

            let tracker = video::TrackerNano::create(&params)?;

            return Ok(Box::new(tracker));
        }
        TrackerType::VIT => {
            let mut params = video::TrackerVit_Params::default()?;

            let net_str = model_dir
                .join("object_tracking_vittrack_2023sep.onnx")
                .to_string_lossy()
                .into_owned();

            params.set_net(&net_str);

            let tracker = video::TrackerVit::create(&params)?;

            return Ok(Box::new(tracker));
        }
        TrackerType::MIL => {
            let params = video::TrackerMIL_Params::default()?;

            let tracker = video::TrackerMIL::create(params)?;
            return Ok(Box::new(tracker));
        }
    }
}

// Get the bounding box of the item at frame 1.
pub fn get_roi_box(frame: &Mat) -> opencv::Result<core::Rect> {
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

// Select the backend for deep neural networks to run on, based on the currently available backends.
fn select_dnn_backend() -> opencv::Result<(dnn::Backend, dnn::Target)> {
    let backends = dnn::get_available_backends()?;

    let preferred = [
        (dnn::Backend::DNN_BACKEND_CUDA, dnn::Target::DNN_TARGET_CUDA),
        (
            dnn::Backend::DNN_BACKEND_OPENCV,
            dnn::Target::DNN_TARGET_OPENCL,
        ),
        (
            dnn::Backend::DNN_BACKEND_OPENCV,
            dnn::Target::DNN_TARGET_CPU,
        ),
    ];

    for (backend, target) in preferred {
        if backends
            .iter()
            .any(|pair| pair.get_0() == backend && pair.get_1() == target)
        {
            #[cfg(debug_assertions)]
            log::debug!(
                "Backend: {:#?}, Target:{:#?}",
                backend.clone(),
                target.clone()
            );
            return Ok((backend, target));
        }
    }

    Err(opencv::Error::new(
        opencv::core::StsError,
        "No supported DNN backend found",
    ))
}
