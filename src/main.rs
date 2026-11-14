mod audio;
mod gui;
mod physics;
mod render;
mod types;

use gui::app::ChladniApp;

#[cfg(target_os = "windows")]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(uPeriod: u32) -> u32;
    fn timeEndPeriod(uPeriod: u32) -> u32;
}

#[cfg(target_os = "windows")]
#[link(name = "user32")]
extern "system" {
    fn GetSystemMetrics(nIndex: i32) -> i32;
}

#[cfg(target_os = "windows")]
pub fn detect_primary_screen_resolution() -> (u32, u32) {
    unsafe {
        let w = GetSystemMetrics(0);
        let h = GetSystemMetrics(1);
        if w > 0 && h > 0 {
            (w as u32, h as u32)
        } else {
            (1920, 1080)
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn detect_primary_screen_resolution() -> (u32, u32) {
    (1920, 1080)
}

pub struct Win32HighResolutionTimerGuard;

impl Win32HighResolutionTimerGuard {
    pub fn new() -> Self {
        #[cfg(target_os = "windows")]
        unsafe {
            let res = timeBeginPeriod(1);
            if res != 0 {
                eprintln!("Warning: timeBeginPeriod(1) returned {}", res);
            }
        }
        Self
    }
}

impl Default for Win32HighResolutionTimerGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Win32HighResolutionTimerGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            let _ = timeEndPeriod(1);
        }
    }
}

fn main() -> eframe::Result<()> {
    let _timer_guard = Win32HighResolutionTimerGuard::new();

    let (screen_w, screen_h) = detect_primary_screen_resolution();
    let init_w = (screen_w as f32 * 0.82).clamp(960.0, 3840.0);
    let init_h = (screen_h as f32 * 0.82).clamp(600.0, 2160.0);

    #[cfg(target_os = "windows")]
    if std::env::var_os("WGPU_BACKEND").is_none() {
        std::env::set_var("WGPU_BACKEND", "dx12");
    }
    if std::env::var_os("WGPU_PRESENT_MODE").is_none() {
        std::env::set_var("WGPU_PRESENT_MODE", "mailbox");
    }

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Chladni Plate Audio Visualizer")
            .with_inner_size([init_w, init_h])
            .with_min_inner_size([960.0, 600.0])
            .with_active(true),
        wgpu_options: egui_wgpu::WgpuConfiguration::default(),
        ..Default::default()
    };

    println!(
        "Starting Chladni Plate Audio Visualizer [wgpu / {}x{}]...",
        screen_w, screen_h
    );

    eframe::run_native(
        "Chladni Plate Visualizer",
        native_options,
        Box::new(|cc| Ok(Box::new(ChladniApp::new(cc)))),
    )
}
