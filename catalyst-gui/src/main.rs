use std::{path::PathBuf, sync::Arc};

use eframe::{App, HardwareAcceleration, NativeOptions};
use egui::{
    CentralPanel, Color32, Direction, FontData, FontDefinitions, FontFamily, Layout, RichText,
    Vec2, ViewportBuilder, vec2,
};
use rfd::FileDialog;

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
struct Application {
    picked_file: Option<PathBuf>,
}

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

        ctx.style_mut(|style| style.spacing.button_padding = vec2(12.0, 8.0));

        CentralPanel::default().show(ctx, |ui| {
            ui.centered_and_justified(|ui| {
                if ui
                    .button(
                        RichText::new("Upload file")
                            .background_color(Color32::LIGHT_BLUE)
                            .color(Color32::WHITE)
                            .size(30.0),
                    )
                    .clicked()
                {
                    self.picked_file = FileDialog::new().pick_file();
                };
            });

            ui.add_space(20.0);

            if let Some(ref file) = self.picked_file {
                ui.label(format!("File picked: {}", file.display()));
            }
        });
    }
}
