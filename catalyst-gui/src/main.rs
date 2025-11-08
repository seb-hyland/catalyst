use std::sync::Arc;

use eframe::{App, HardwareAcceleration, NativeOptions};
use egui::{
    CentralPanel, Color32, FontData, FontDefinitions, FontFamily, Style, ViewportBuilder, Window,
};

fn main() {
    eframe::run_native(
        "catalyst",
        NativeOptions {
            hardware_acceleration: HardwareAcceleration::Preferred,
            viewport: ViewportBuilder::default()
                .with_inner_size([600.0, 400.0])
                .with_decorations(false),
            ..Default::default()
        },
        Box::new(|_| Ok(Box::<Application>::default())),
    )
    .expect("Failed to spawn catalyst window");
}

#[derive(Default)]
struct Application {}

impl App for Application {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut fonts = FontDefinitions::default();
        fonts.font_data.insert(
            "JetBrainsMono-Regular".to_owned(),
            Arc::new(FontData::from_static(include_bytes!(
                "../JetBrainsMono/fonts/ttf/JetBrainsMono-Regular.ttf"
            ))),
        );
        fonts
            .families
            .get_mut(&FontFamily::Proportional)
            .expect("Font families")
            .insert(0, "JetBrainsMono-Regular".to_owned());
        ctx.set_fonts(fonts);

        CentralPanel::default().show(ctx, |ui| {
            let mut style = Style::default();
            style.visuals.override_text_color = Some(Color32::WHITE);

            ui.set_style(style);
            ui.heading("Hello, world!");
        });
    }
}
