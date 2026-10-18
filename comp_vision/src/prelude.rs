use opencv::core::Rect;

pub trait WithCenter {
    fn get_center(&self) -> (f32, f32);
}
impl WithCenter for Rect {
    fn get_center(&self) -> (f32, f32) {
        (
            self.x as f32 + self.width as f32 / 2.0,
            self.y as f32 + self.height as f32 / 2.0,
        )
    }
}
