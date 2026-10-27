use std::path::Path;

pub mod ostrack;

use opencv::{
    core::Rect_,
    dnn,
    video::{
        self, TrackerDaSiamRPN, TrackerDaSiamRPN_Params, TrackerDaSiamRPN_ParamsTrait,
        TrackerNano_ParamsTrait, TrackerVit_ParamsTrait,
    },
};

use crate::{prelude::WithCenter, tracker::ostrack::OSTrack};

#[derive(Debug, Clone, serde::Serialize)]
pub struct Coordinates {
    x: f32,
    y: f32,
    width: Option<f32>,
    height: Option<f32>,
}

impl Coordinates {
    pub fn new(x: f32, y: f32, width: Option<f32>, height: Option<f32>) -> Self {
        return Self {
            x,
            y,
            width,
            height,
        };
    }
    pub fn x(&self) -> f32 {
        self.x
    }
    pub fn y(&self) -> f32 {
        self.y
    }
}

impl From<Rect_<i32>> for Coordinates {
    fn from(value: Rect_<i32>) -> Self {
        let (x, y) = value.get_center();
        return Self {
            x,
            y,
            width: Some(value.width as f32),
            height: Some(value.height as f32),
        };
    }
}

// trait that implements the functions for trackers such that then can be returned as a box<dyn> object.

pub trait AnyTracker: std::fmt::Debug {
    fn init(&mut self, frame: &opencv::core::Mat, bbox: opencv::core::Rect) -> opencv::Result<()>;
    fn update(
        &mut self,
        frame: &opencv::core::Mat,
        bbox: &mut opencv::core::Rect,
    ) -> opencv::Result<bool>;
}

impl<T> AnyTracker for T
where
    T: video::TrackerTrait + std::fmt::Debug,
{
    fn init(&mut self, frame: &opencv::core::Mat, bbox: opencv::core::Rect) -> opencv::Result<()> {
        video::TrackerTrait::init(self, frame, bbox)
    }
    fn update(
        &mut self,
        frame: &opencv::core::Mat,
        bbox: &mut opencv::core::Rect,
    ) -> opencv::Result<bool> {
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
    OSTrack,
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
        TrackerType::OSTrack => {
            return Ok(Box::new(OSTrack::new(model_dir.to_path_buf())));
        }
    }
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
            return Ok((backend, target));
        }
    }

    Err(opencv::Error::new(
        opencv::core::StsError,
        "No supported DNN backend found",
    ))
}
