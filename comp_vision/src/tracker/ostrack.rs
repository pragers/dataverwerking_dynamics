use std::{
    f32::consts::PI,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, Rect, Size};
use ort::{
    session::{Session, builder::AutoDevicePolicy},
    value::Tensor,
};

use crate::{prelude::WithCenter, tracker::AnyTracker};

#[derive(Debug, Clone)]
struct Crop {
    image: Mat,
    resize_factor: f32,
}

#[derive(Debug, Clone, Copy)]
struct Prediction {
    center_x: f32,
    center_y: f32,
    width: f32,
    height: f32,
}
impl Prediction {
    pub fn scale(&mut self, factor: f32) {
        self.center_x *= factor;
        self.center_y *= factor;
        self.width *= factor;
        self.height *= factor;
    }
}

#[derive(Debug)]
pub struct OSTrack<'a> {
    session: &'a Mutex<Session>,
    template: Option<Vec<f32>>,
    state: Option<Rect>,

    template_size: i32,
    search_size: i32,
    template_factor: f32,
    search_factor: f32,
}

impl<'a> OSTrack<'a> {
    pub fn new(path: PathBuf) -> Self {
        let session = get_session(path.join("ostrack384.onnx"));
        Self {
            session,
            template: None,
            state: None,
            template_size: 192,
            search_size: 384,
            template_factor: 2.0,
            search_factor: 5.0,
        }
    }
    pub fn extract_template(&self, frame: &Mat, bbox: Rect) -> opencv::Result<Vec<f32>> {
        let crop = sample_target(frame, bbox, self.template_factor, self.template_size)?;
        preprocess(&crop.image)
    }
    fn extract_search(&self, frame: &Mat, bbox: Rect) -> opencv::Result<(Vec<f32>, f32)> {
        let crop = sample_target(frame, bbox, self.search_factor, self.search_size)?;

        let processed_img = preprocess(&crop.image)?;

        Ok((processed_img, crop.resize_factor))
    }
    fn run_inference(
        &mut self,
        search: &[f32],
    ) -> Result<Option<Prediction>, Box<dyn std::error::Error>> {
        let template = self.template.as_ref().ok_or("Tracker not initialized")?;

        let z = Tensor::from_array(([1, 3, 192, 192], template.clone()))?;

        let x = Tensor::from_array(([1, 3, 384, 384], search.to_vec()))?;
        let mut session = self.session.lock().expect("Failed to lock session");

        let outputs = session.run(ort::inputs![
            "z" => z,
            "x" => x,
        ])?;

        let (_, scores) = outputs["score_map"].try_extract_tensor::<f32>()?;

        let (_, sizes) = outputs["size_map"].try_extract_tensor::<f32>()?;

        let (_, offsets) = outputs["offset_map"].try_extract_tensor::<f32>()?;

        let prediction = decode_ostrack(scores, sizes, offsets)?;

        Ok(prediction)
    }
    fn decode_prediction(
        &self,
        mut pred: Prediction,
        previous: Rect,
        resize_factor: f32,
        frame_size: Size,
    ) -> opencv::Result<Option<Rect>> {
        // Check for valid input
        if resize_factor <= 0.0
            || !resize_factor.is_finite()
            || ![pred.center_x, pred.center_y, pred.width, pred.height]
                .iter()
                .all(|v| v.is_finite())
            || pred.width <= 0.0
            || pred.height <= 0.0
        {
            return Ok(None);
        }

        let previous_center_x = previous.x as f32 + previous.width as f32 / 2.0;
        let previous_center_y = previous.y as f32 + previous.height as f32 / 2.0;

        let crop_size = self.search_size as f32 / resize_factor;
        pred.scale(crop_size);

        let x = previous_center_x - crop_size / 2.0 + pred.center_x - pred.width / 2.0;
        let y = previous_center_y - crop_size / 2.0 + pred.center_y - pred.height / 2.0;

        let top_left_corner_x = x.clamp(0.0, frame_size.width as f32);
        let top_left_corner_y = y.clamp(0.0, frame_size.height as f32);

        let bottom_right_corner_x = (x + pred.width).clamp(0.0, frame_size.width as f32);
        let bottom_right_corner_y = (y + pred.height).clamp(0.0, frame_size.height as f32);

        // Completely outside image
        if bottom_right_corner_x <= top_left_corner_x || bottom_right_corner_y <= top_left_corner_y
        {
            return Ok(None);
        }

        let left = top_left_corner_x.floor() as i32;
        let top = top_left_corner_y.floor() as i32;
        let right = bottom_right_corner_x.ceil() as i32;
        let bottom = bottom_right_corner_y.ceil() as i32;
        let new = Rect::new(left, top, right - left, bottom - top);
        if previous.area() as f32 * 1.5 < new.area() as f32
        // || ((previous.x as f32 - new.x as f32).powf(2.)
        //     + (previous.y as f32 - new.y as f32).powf(2.))
        // .sqrt()
        //     > 150.
        {
            return Ok(None);
        }

        Ok(Some(Rect::new(left, top, right - left, bottom - top)))
    }
}

impl<'a> AnyTracker for OSTrack<'a> {
    fn init(&mut self, frame: &Mat, bbox: Rect) -> opencv::Result<()> {
        if bbox.width <= 0 || bbox.height <= 0 {
            return Err(opencv::Error::new(
                opencv::core::StsBadArg,
                "Invalid initial bounding box".to_string(),
            ));
        }

        self.template = Some(self.extract_template(frame, bbox)?);
        self.state = Some(bbox);

        Ok(())
    }

    fn update(&mut self, frame: &Mat, bbox: &mut Rect) -> opencv::Result<bool> {
        let Some(previous) = self.state else {
            return Ok(false);
        };

        let (search, resize_factor) = self.extract_search(frame, previous)?;

        let prediction = self
            .run_inference(&search)
            .map_err(|e| opencv::Error::new(opencv::core::StsError, e.to_string()))?;
        if let Some(pred) = prediction {
            let Some(new_bbox) =
                self.decode_prediction(pred, previous, resize_factor, frame.size()?)?
            else {
                *bbox = previous;
                return Ok(false);
            };
            self.state = Some(new_bbox);
            *bbox = new_bbox;

            return Ok(true);
        } else {
            *bbox = previous;
            return Ok(false);
        };
    }
}

fn sample_target(frame: &Mat, bbox: Rect, factor: f32, output_size: i32) -> opencv::Result<Crop> {
    if bbox.width <= 0
        || bbox.height <= 0
        || factor <= 0.
        || output_size <= 0
        || frame.cols() <= 0
        || frame.rows() <= 0
    {
        return Err(opencv::Error::new(
            opencv::core::StsBadArg,
            "Invalid crop parameters".to_string(),
        ));
    }

    let side_length =
        (((bbox.width as f32 * bbox.height as f32).sqrt() * factor).ceil() as i32).max(1);

    let (center_x, center_y) = bbox.get_center();

    let top_left_corner_x = (center_x - side_length as f32 / 2.0).round() as i32;
    let top_left_corner_y = (center_y - side_length as f32 / 2.0).round() as i32;

    let bottom_right_corner_x = top_left_corner_x + side_length;
    let bottom_right_corner_y = top_left_corner_y + side_length;

    let left = top_left_corner_x.clamp(0, frame.cols());
    let top = top_left_corner_y.clamp(0, frame.rows());
    let right = bottom_right_corner_x.clamp(0, frame.cols());
    let bottom = bottom_right_corner_y.clamp(0, frame.rows());

    if right <= left || bottom <= top {
        return Err(opencv::Error::new(
            opencv::core::StsBadArg,
            "Crop is outside image".to_string(),
        ));
    };
    let roi = Mat::roi(frame, Rect::new(left, top, right - left, bottom - top))?;

    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &roi,
        &mut padded,
        top - top_left_corner_y,
        bottom_right_corner_y - bottom,
        left - top_left_corner_x,
        bottom_right_corner_x - right,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::all(0.0),
    )?;
    let mut resized_image = Mat::default();
    opencv::imgproc::resize(
        &padded,
        &mut resized_image,
        opencv::core::Size::new(output_size, output_size),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    Ok(Crop {
        image: resized_image,
        resize_factor: output_size as f32 / side_length as f32,
    })
}

fn preprocess(image: &Mat) -> opencv::Result<Vec<f32>> {
    // Based on:
    // https://github.com/botaoye/OSTrack/blob/main/lib/test/tracker/data_utils.py
    let mut rgb = Mat::default();
    opencv::imgproc::cvt_color(
        image,
        &mut rgb,
        opencv::imgproc::COLOR_BGR2RGB,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut float_image = Mat::default();

    rgb.convert_to(&mut float_image, opencv::core::CV_32F, 1.0 / 255.0, 0.0)?;
    let height = float_image.rows() as usize;
    let width = float_image.cols() as usize;

    let pixels = float_image.data_typed::<opencv::core::Vec3f>()?;

    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];

    let mut tensor = vec![0f32; 3 * height * width];

    // Convert from HWC to CHW format
    for y in 0..height {
        for x in 0..width {
            let pixel = pixels[y * width + x];

            for channel in 0..3 {
                let new_index = channel * height * width + y * width + x;

                tensor[new_index] = (pixel[channel] - mean[channel]) / std[channel];
            }
        }
    }

    Ok(tensor)
}

const FEATURE_SIZE: usize = 24;
const MAP_AREA: usize = FEATURE_SIZE * FEATURE_SIZE;

fn decode_ostrack(
    scores: &[f32],
    sizes: &[f32],
    offsets: &[f32],
) -> Result<Option<Prediction>, String> {
    // Check if the ouput matches:
    // score_map: shape = [1,1,24,24] probability score for each position.
    // size_map: shape = [1,2,24,24] width and height of bounding box.
    // offset_map: shape = [1,2,24,24] x and y position offsets.
    if scores.len() != MAP_AREA || sizes.len() != 2 * MAP_AREA || offsets.len() != 2 * MAP_AREA {
        return Err("Unexpected OSTrack output sizes".into());
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_index = 0;
    // use hann window to discourage jumping around.
    for y in 0..FEATURE_SIZE {
        for x in 0..FEATURE_SIZE {
            let index = y * FEATURE_SIZE + x;

            // calculate the hann window value for x and y
            let window_x =
                0.5 * (1.0 - (2.0 * PI * (x + 1) as f32 / (FEATURE_SIZE + 1) as f32).cos());
            let window_y =
                0.5 * (1.0 - (2.0 * PI * (y + 1) as f32 / (FEATURE_SIZE + 1) as f32).cos());
            let penalty_strength = 1.1_f32;
            let score = scores[index] * (window_x * window_y).powf(penalty_strength);

            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
    }
    if best_score < 0.4 {
        return Ok(None);
    }
    // Use integer division and modulo to turn 1d index into a 2d coordinate.
    let x = best_index % FEATURE_SIZE;
    let y = best_index / FEATURE_SIZE;

    // associated width and height are stored with offset of MAP_AREA
    let width = sizes[best_index];
    let height = sizes[MAP_AREA + best_index];

    let offset_x = offsets[best_index];
    let offset_y = offsets[MAP_AREA + best_index];

    Ok(Some(Prediction {
        center_x: (x as f32 + offset_x) / FEATURE_SIZE as f32,
        center_y: (y as f32 + offset_y) / FEATURE_SIZE as f32,
        width,
        height,
    }))
}

static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();

pub fn get_session(path: PathBuf) -> &'static Mutex<Session> {
    SESSION.get_or_init(|| Mutex::new(load_model(path).expect("Failed to load OSTrack model")))
}

pub fn load_model(path: PathBuf) -> ort::Result<Session> {
    let env = ort::init().build()?;
    let policy = AutoDevicePolicy::MaxPerformance;
    let session = Session::builder(&env)?
        .with_auto_device(policy)?
        .commit_from_file(path)?;

    Ok(session)
}
