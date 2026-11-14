use crate::audio::dsp::AudioDspProcessor;
use crate::audio::engine::AudioEngine;
use crate::render::pipeline::{
    ChladniPaintCallback, ChladniRenderResources, ShaderUniforms, DEFAULT_PARTICLE_CAP,
    MAX_GPU_PARTICLES,
};
use crate::types::{
    AppPreset, AudioSettings, ColorMappingMode, EdgeBoundaryMode, FrequencyMappingCurve,
    FrequencyTrackingMode, InputSourceMode, ModalEquationSolver, PhysicsSettings, PlateGeometry,
    SynthSettings, VisualSettings,
};
use eframe::egui::{self, Color32, RichText, Slider};
use std::sync::atomic::Ordering;
use std::time::Instant;

pub struct ChladniApp {
    audio_engine: AudioEngine,
    dsp: AudioDspProcessor,
    audio_settings: AudioSettings,
    physics_settings: PhysicsSettings,
    visual_settings: VisualSettings,
    synth_settings: SynthSettings,
    presets: Vec<AppPreset>,
    sample_buffer: Vec<f32>,
    start_time: Instant,
    last_frame_time: Instant,
    fps: f32,
    frame_time_ms: f32,
    sidebar_expanded: bool,
    is_fullscreen: bool,
    prev_maximized: bool,
    prev_os_fullscreen: bool,
    detected_monitor_px: [u32; 2],
    detected_canvas_px: [u32; 2],
    unlocked_particle_cap: bool,
    preset_save_name: String,
    status_message: Option<(String, Instant)>,
}

impl ChladniApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut audio_engine = AudioEngine::new();
        let audio_settings = AudioSettings::default();
        let physics_settings = PhysicsSettings::default();
        let visual_settings = VisualSettings::default();
        let synth_settings = SynthSettings::default();

        let dsp = AudioDspProcessor::new(audio_settings.fft_size);
        let presets = AppPreset::load_from_disk();
        let (screen_w, screen_h) = crate::detect_primary_screen_resolution();

        audio_engine.check_and_update_stream(&audio_settings, &synth_settings);

        if let Some(wgpu_render_state) = &cc.wgpu_render_state {
            let resources = ChladniRenderResources::new(
                &wgpu_render_state.device,
                wgpu_render_state.target_format,
            );
            wgpu_render_state
                .renderer
                .write()
                .callback_resources
                .insert(resources);
        }

        let now = Instant::now();

        Self {
            audio_engine,
            dsp,
            audio_settings,
            physics_settings,
            visual_settings,
            synth_settings,
            presets,
            sample_buffer: Vec::with_capacity(8192),
            start_time: now,
            last_frame_time: now,
            fps: 60.0,
            frame_time_ms: 16.6,
            sidebar_expanded: true,
            is_fullscreen: false,
            prev_maximized: false,
            prev_os_fullscreen: false,
            detected_monitor_px: [screen_w, screen_h],
            detected_canvas_px: [1360, 850],
            unlocked_particle_cap: false,
            preset_save_name: "My Custom Preset".to_string(),
            status_message: Some((
                "Chladni GPU Audio Visualizer Initialized".to_string(),
                Instant::now(),
            )),
        }
    }

    fn apply_preset(&mut self, preset: &AppPreset) {
        let keep_screen_borders = self.physics_settings.use_screen_borders;
        self.audio_settings = preset.audio.clone();
        self.physics_settings = preset.physics.clone();
        self.physics_settings.use_screen_borders = keep_screen_borders;
        self.visual_settings = preset.visual.clone();
        self.unlocked_particle_cap = self.visual_settings.particle_count > DEFAULT_PARTICLE_CAP;
        self.synth_settings = preset.synth.clone();
        self.dsp.resize_fft(self.audio_settings.fft_size);
        self.dsp.beat_kick = self.dsp.beat_kick.max(1.4);
        self.set_status(&format!("Applied preset: {}", preset.name));
    }

    fn set_fullscreen(&mut self, ctx: &egui::Context, fullscreen: bool) {
        self.is_fullscreen = fullscreen;
        if fullscreen {
            self.sidebar_expanded = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(true));
        } else {
            self.sidebar_expanded = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
        }
    }

    fn set_status(&mut self, msg: &str) {
        self.status_message = Some((msg.to_string(), Instant::now()));
    }
}

impl eframe::App for ChladniApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_frame_time)
            .as_secs_f32()
            .min(0.05);
        self.last_frame_time = now;
        self.frame_time_ms = dt * 1000.0;
        self.fps = self.fps * 0.9 + (1.0 / dt.max(0.001)) * 0.1;

        // 0. Hotkeys, native maximize -> true fullscreen promotion, and auto monitor resolution
        let ppp = ui.ctx().pixels_per_point().max(1.0);
        let (f11_pressed, esc_pressed, vp_maximized, vp_fullscreen, vp_monitor_size) =
            ui.ctx().input(|i| {
                let vp = i.viewport();
                (
                    i.key_pressed(egui::Key::F11),
                    i.key_pressed(egui::Key::Escape),
                    vp.maximized.unwrap_or(false),
                    vp.fullscreen,
                    vp.monitor_size,
                )
            });

        if let Some(mon_size) = vp_monitor_size {
            if mon_size.x > 0.0 && mon_size.y > 0.0 {
                self.detected_monitor_px = [
                    (mon_size.x * ppp).round() as u32,
                    (mon_size.y * ppp).round() as u32,
                ];
            }
        }

        if let Some(os_fs) = vp_fullscreen {
            if os_fs != self.prev_os_fullscreen {
                self.prev_os_fullscreen = os_fs;
                self.is_fullscreen = os_fs;
            }
        }

        if f11_pressed {
            self.set_fullscreen(ui.ctx(), !self.is_fullscreen);
        } else if esc_pressed && self.is_fullscreen {
            self.set_fullscreen(ui.ctx(), false);
        } else if vp_maximized && !self.prev_maximized && !self.is_fullscreen {
            self.set_fullscreen(ui.ctx(), true);
        }
        self.prev_maximized = vp_maximized;

        // 1. Audio stream sync & sample fetch
        self.audio_engine
            .check_and_update_stream(&self.audio_settings, &self.synth_settings);

        self.sample_buffer.clear();
        let sample_rate = self.audio_engine.sample_rate.load(Ordering::Relaxed);

        if self.audio_settings.source_mode == InputSourceMode::InternalSynth {
            let step_samples = (sample_rate as f32 * dt).round().max(1.0) as usize;
            self.audio_engine.generate_synth_samples(
                &mut self.sample_buffer,
                step_samples.min(8192),
                &self.synth_settings,
                sample_rate,
            );
        } else {
            self.audio_engine
                .fetch_samples(&mut self.sample_buffer, 8192);
        }

        // 2. DSP (sliding-window FFT, perceptual tilt, tri-band modal extraction, beat kick)
        self.dsp.process_samples(
            &self.sample_buffer,
            sample_rate,
            &self.audio_settings,
            &mut self.physics_settings,
            dt,
        );

        // 4. Status bar (Bottom Panel - hidden in true fullscreen)
        if !self.is_fullscreen {
            egui::Panel::bottom("bottom_status_bar")
                .frame(egui::Frame::default().fill(Color32::from_black_alpha(200)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("⚡ GPU Pro-Audio")
                                .color(Color32::from_rgb(0, 220, 180))
                                .strong(),
                        );
                        ui.separator();
                        ui.label(format!(
                            "FPS: {:.0} ({:.1} ms)",
                            self.fps, self.frame_time_ms
                        ));
                        ui.separator();
                        ui.label(format!(
                            "Res: {}×{}",
                            self.detected_canvas_px[0], self.detected_canvas_px[1]
                        ));
                        ui.separator();
                        ui.label(format!(
                            "Particles: {}",
                            self.visual_settings.particle_count
                        ));
                        ui.separator();
                        ui.label(format!("Sample Rate: {} Hz", sample_rate));
                        ui.separator();
                        ui.label(format!("Pitch: {:.1} Hz", self.dsp.detected_frequency));
                        ui.separator();
                        ui.label(format!(
                            "Modes: m={:.1}, n={:.1}",
                            self.physics_settings.m, self.physics_settings.n
                        ));

                        if let Some((msg, time)) = &self.status_message {
                            if time.elapsed().as_secs() < 5 {
                                ui.separator();
                                ui.label(RichText::new(msg).color(Color32::YELLOW));
                            }
                        }
                    });
                });
        }

        // 5. Sidebar (Left Panel)
        if self.sidebar_expanded {
            egui::Panel::left("controls_panel")
                .default_size(360.0)
                .size_range(300.0..=500.0)
                .frame(
                    egui::Frame::default()
                        .fill(Color32::from_rgba_premultiplied(18, 22, 30, 230))
                        .inner_margin(egui::Margin::same(12)),
                )
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading(
                            RichText::new("🌊 Chladni Cymatics")
                                .color(Color32::WHITE)
                                .strong(),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.button("◀ Collapse").clicked() {
                                    self.sidebar_expanded = false;
                                }
                                let fs_label = if self.is_fullscreen {
                                    "⛶ Exit Full"
                                } else {
                                    "⛶ Fullscreen"
                                };
                                if ui
                                    .button(fs_label)
                                    .on_hover_text("Toggle True Fullscreen (F11)")
                                    .clicked()
                                {
                                    self.set_fullscreen(ui.ctx(), !self.is_fullscreen);
                                }
                            },
                        );
                    });

                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        // Presets
                        ui.collapsing(RichText::new("💾 Presets").strong(), |ui| {
                            let mut remove_idx: Option<usize> = None;
                            let mut clicked_preset: Option<AppPreset> = None;

                            ui.horizontal_wrapped(|ui| {
                                for (idx, preset) in self.presets.iter().enumerate() {
                                    ui.push_id(idx, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().item_spacing.x = 2.0;
                                            if ui.button(&preset.name).clicked() {
                                                clicked_preset = Some(preset.clone());
                                            }
                                            if ui
                                                .small_button(
                                                    RichText::new("✕")
                                                        .color(Color32::from_rgb(220, 100, 100)),
                                                )
                                                .on_hover_text(format!(
                                                    "Remove preset '{}'",
                                                    preset.name
                                                ))
                                                .clicked()
                                            {
                                                remove_idx = Some(idx);
                                            }
                                        });
                                    });
                                }
                            });

                            if let Some(preset) = clicked_preset {
                                self.apply_preset(&preset);
                            }
                            if let Some(idx) = remove_idx {
                                let removed = self.presets.remove(idx);
                                match AppPreset::save_to_disk(&self.presets) {
                                    Ok(()) => self.set_status(&format!(
                                        "Removed preset: {}",
                                        removed.name
                                    )),
                                    Err(e) => self.set_status(&e),
                                }
                            }

                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.text_edit_singleline(&mut self.preset_save_name);
                                if ui.button("Save").clicked() {
                                    let trimmed = self.preset_save_name.trim();
                                    let name = if trimmed.is_empty() {
                                        "Custom Preset".to_string()
                                    } else {
                                        trimmed.to_string()
                                    };
                                    let new_preset = AppPreset {
                                        name: name.clone(),
                                        audio: self.audio_settings.clone(),
                                        physics: self.physics_settings.clone(),
                                        visual: self.visual_settings.clone(),
                                        synth: self.synth_settings.clone(),
                                    };
                                    if let Some(existing) =
                                        self.presets.iter_mut().find(|p| p.name == name)
                                    {
                                        *existing = new_preset;
                                    } else {
                                        self.presets.push(new_preset);
                                    }
                                    match AppPreset::save_to_disk(&self.presets) {
                                        Ok(()) => self.set_status(&format!("Saved preset: {}", name)),
                                        Err(e) => self.set_status(&e),
                                    }
                                }
                                if ui
                                    .button("↺ Defaults")
                                    .on_hover_text("Restore built-in default presets")
                                    .clicked()
                                {
                                    for def in AppPreset::built_in_presets() {
                                        if !self.presets.iter().any(|p| p.name == def.name) {
                                            self.presets.push(def);
                                        }
                                    }
                                    match AppPreset::save_to_disk(&self.presets) {
                                        Ok(()) => self.set_status("Restored built-in presets"),
                                        Err(e) => self.set_status(&e),
                                    }
                                }
                            });
                        });

                        ui.separator();

                        // Audio & DSP
                        ui.collapsing(RichText::new("🎚 Audio & DSP Routing").strong(), |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Source Mode:");
                                egui::ComboBox::from_id_salt("source_mode")
                                    .selected_text(format!(
                                        "{:?}",
                                        self.audio_settings.source_mode
                                    ))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.audio_settings.source_mode,
                                            InputSourceMode::SystemLoopback,
                                            "System Loopback",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.source_mode,
                                            InputSourceMode::SingleAppProcess,
                                            "Single App Process",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.source_mode,
                                            InputSourceMode::Microphone,
                                            "Microphone",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.source_mode,
                                            InputSourceMode::InternalSynth,
                                            "Internal Synth",
                                        );
                                    });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Device:");
                                let cur_dev = self
                                    .audio_settings
                                    .selected_device_name
                                    .clone()
                                    .unwrap_or_else(|| "Default Device".to_string());
                                egui::ComboBox::from_id_salt("audio_device_combo")
                                    .selected_text(cur_dev)
                                    .show_ui(ui, |ui| {
                                        let devices =
                                            if self.audio_settings.source_mode
                                                == InputSourceMode::Microphone
                                            {
                                                &self.audio_engine.input_devices
                                            } else {
                                                &self.audio_engine.output_devices
                                            };

                                        if ui
                                            .selectable_label(
                                                self.audio_settings.selected_device_name.is_none(),
                                                "Default Device",
                                            )
                                            .clicked()
                                        {
                                            self.audio_settings.selected_device_name = None;
                                        }

                                        for dev in devices {
                                            let is_selected =
                                                self.audio_settings.selected_device_name.as_ref()
                                                    == Some(&dev.name);
                                            if ui
                                                .selectable_label(is_selected, &dev.name)
                                                .clicked()
                                            {
                                                self.audio_settings.selected_device_name =
                                                    Some(dev.name.clone());
                                            }
                                        }
                                    });
                                if ui
                                    .button("⟳")
                                    .on_hover_text("Refresh Devices")
                                    .clicked()
                                {
                                    self.audio_engine.refresh_devices_and_processes();
                                }
                            });

                            if self.audio_settings.source_mode
                                == InputSourceMode::SingleAppProcess
                            {
                                ui.horizontal(|ui| {
                                    ui.label("App Process:");
                                    let cur_proc = self
                                        .audio_settings
                                        .target_process_name
                                        .clone()
                                        .unwrap_or_else(|| "Choose .exe".to_string());
                                    egui::ComboBox::from_id_salt("app_proc_combo")
                                        .selected_text(cur_proc)
                                        .show_ui(ui, |ui| {
                                            for proc in &self.audio_engine.running_processes {
                                                let is_sel = self
                                                    .audio_settings
                                                    .target_process_pid
                                                    == Some(proc.pid);
                                                if ui
                                                    .selectable_label(
                                                        is_sel,
                                                        format!(
                                                            "{} (PID: {})",
                                                            proc.name, proc.pid
                                                        ),
                                                    )
                                                    .clicked()
                                                {
                                                    self.audio_settings.target_process_name =
                                                        Some(proc.name.clone());
                                                    self.audio_settings.target_process_pid =
                                                        Some(proc.pid);
                                                }
                                            }
                                        });
                                });
                            }

                            ui.add(
                                Slider::new(&mut self.audio_settings.input_gain, 0.1..=10.0)
                                    .text("Pre-Amp Gain"),
                            );
                            ui.add(
                                Slider::new(
                                    &mut self.audio_settings.noise_gate_threshold,
                                    0.0..=0.08,
                                )
                                .text("Noise Gate"),
                            );

                            ui.horizontal(|ui| {
                                ui.label("FFT Window Size:");
                                let cur_fft = self.audio_settings.fft_size;
                                egui::ComboBox::from_id_salt("fft_size")
                                    .selected_text(format!("{} samples", cur_fft))
                                    .show_ui(ui, |ui| {
                                        for &size in &[512, 1024, 2048, 4096] {
                                            if ui
                                                .selectable_label(
                                                    cur_fft == size,
                                                    format!("{}", size),
                                                )
                                                .clicked()
                                            {
                                                self.audio_settings.fft_size = size;
                                                self.dsp.resize_fft(size);
                                            }
                                        }
                                    });
                            });

                            ui.horizontal(|ui| {
                                ui.label("Tracking Mode:");
                                egui::ComboBox::from_id_salt("tracking_mode")
                                    .selected_text(format!(
                                        "{:?}",
                                        self.audio_settings.tracking_mode
                                    ))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.audio_settings.tracking_mode,
                                            FrequencyTrackingMode::TriBandSuperposition,
                                            "Tri-Band Superposition",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.tracking_mode,
                                            FrequencyTrackingMode::DominantPeak,
                                            "Dominant Peak",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.tracking_mode,
                                            FrequencyTrackingMode::SubBassFocus,
                                            "Sub-Bass Focus",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.tracking_mode,
                                            FrequencyTrackingMode::HarmonicChordBlend,
                                            "Harmonic Chord Blend",
                                        );
                                        ui.selectable_value(
                                            &mut self.audio_settings.tracking_mode,
                                            FrequencyTrackingMode::MultiBandWeighted,
                                            "Multi-Band Weighted",
                                        );
                                    });
                            });

                            ui.add(
                                Slider::new(
                                    &mut self.audio_settings.attack_speed,
                                    0.05..=1.0,
                                )
                                .text("Attack Smoothing"),
                            );
                            ui.add(
                                Slider::new(
                                    &mut self.audio_settings.release_speed,
                                    0.01..=0.5,
                                )
                                .text("Release Decay"),
                            );

                            let vu = f32::from_bits(
                                self.audio_engine.current_vu.load(Ordering::Relaxed),
                            );
                            ui.horizontal(|ui| {
                                ui.label("Input VU:");
                                let prog = (vu * 3.0).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(prog).desired_width(160.0),
                                );
                            });
                        });

                        ui.separator();

                        // Physics
                        ui.collapsing(
                            RichText::new("⚙ Chladni Physics & Plate").strong(),
                            |ui| {
                                let prev_borders = self.physics_settings.use_screen_borders;
                                ui.checkbox(
                                    &mut self.physics_settings.use_screen_borders,
                                    "Use Screen as Borders",
                                )
                                .on_hover_text(
                                    "Use the full available screen/viewport space as plate borders instead of a confined shape",
                                );
                                if self.physics_settings.use_screen_borders && !prev_borders {
                                    self.dsp.beat_kick = self.dsp.beat_kick.max(1.8);
                                }
                                ui.label(
                                    RichText::new(format!(
                                        "Auto Resolution: {}×{} px (Display: {}×{} px)",
                                        self.detected_canvas_px[0],
                                        self.detected_canvas_px[1],
                                        self.detected_monitor_px[0],
                                        self.detected_monitor_px[1]
                                    ))
                                    .small()
                                    .color(Color32::from_gray(170)),
                                );
                                ui.add_space(4.0);

                                ui.horizontal(|ui| {
                                    ui.label("Geometry:");
                                    egui::ComboBox::from_id_salt("geom_combo")
                                        .selected_text(format!(
                                            "{:?}",
                                            self.physics_settings.geometry
                                        ))
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.physics_settings.geometry,
                                                PlateGeometry::Square,
                                                "Square",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.geometry,
                                                PlateGeometry::Circular,
                                                "Circular",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.geometry,
                                                PlateGeometry::Hexagonal,
                                                "Hexagonal",
                                            );
                                        });
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Modal Solver:");
                                    egui::ComboBox::from_id_salt("solver_combo")
                                        .selected_text(format!(
                                            "{:?}",
                                            self.physics_settings.solver
                                        ))
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.physics_settings.solver,
                                                ModalEquationSolver::ClassicSymmetric,
                                                "Classic Symmetric",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.solver,
                                                ModalEquationSolver::AsymmetricMixed,
                                                "Asymmetric Mixed",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.solver,
                                                ModalEquationSolver::BesselCircular,
                                                "Bessel Circular Modes",
                                            );
                                        });
                                });

                                ui.horizontal(|ui| {
                                    ui.label("Freq-to-Mode Mapping:");
                                    egui::ComboBox::from_id_salt("mapping_combo")
                                        .selected_text(format!(
                                            "{:?}",
                                            self.physics_settings.mapping_curve
                                        ))
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.physics_settings.mapping_curve,
                                                FrequencyMappingCurve::Linear,
                                                "Linear",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.mapping_curve,
                                                FrequencyMappingCurve::LogarithmicOctaves,
                                                "Logarithmic Octaves",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.mapping_curve,
                                                FrequencyMappingCurve::QuantizedHarmonics,
                                                "Quantized Harmonics",
                                            );
                                        });
                                });

                                ui.checkbox(
                                    &mut self.physics_settings.manual_override,
                                    "Manual Modal Override",
                                );
                                if self.physics_settings.manual_override {
                                    ui.add(
                                        Slider::new(
                                            &mut self.physics_settings.m,
                                            1.0..=12.0,
                                        )
                                        .text("Modal m"),
                                    );
                                    ui.add(
                                        Slider::new(
                                            &mut self.physics_settings.n,
                                            1.0..=14.0,
                                        )
                                        .text("Modal n"),
                                    );
                                    ui.add(
                                        Slider::new(
                                            &mut self.physics_settings.a,
                                            0.0..=2.0,
                                        )
                                        .text("Mixing a"),
                                    );
                                    ui.add(
                                        Slider::new(
                                            &mut self.physics_settings.b,
                                            0.0..=2.0,
                                        )
                                        .text("Mixing b"),
                                    );
                                }

                                ui.add(
                                    Slider::new(
                                        &mut self.physics_settings.kick_force,
                                        0.001..=0.02,
                                    )
                                    .text("Kick Force"),
                                );
                                ui.add(
                                    Slider::new(
                                        &mut self.physics_settings.friction,
                                        0.01..=0.25,
                                    )
                                    .text("Surface Friction"),
                                );
                                ui.add(
                                    Slider::new(
                                        &mut self.physics_settings.brownian_jitter,
                                        0.0001..=0.005,
                                    )
                                    .text("Brownian Jitter"),
                                );

                                ui.horizontal(|ui| {
                                    ui.label("Edge Boundary:");
                                    egui::ComboBox::from_id_salt("boundary_combo")
                                        .selected_text(format!(
                                            "{:?}",
                                            self.physics_settings.boundary_mode
                                        ))
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.physics_settings.boundary_mode,
                                                EdgeBoundaryMode::Bounce,
                                                "Bounce",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.boundary_mode,
                                                EdgeBoundaryMode::Wrap,
                                                "Wrap",
                                            );
                                            ui.selectable_value(
                                                &mut self.physics_settings.boundary_mode,
                                                EdgeBoundaryMode::RespawnRandom,
                                                "Respawn Random",
                                            );
                                        });
                                });
                            },
                        );

                        ui.separator();

                        // Visuals
                        ui.collapsing(
                            RichText::new("🎨 Visuals & Shader Tweaks").strong(),
                            |ui| {
                                let max_particles = if self.unlocked_particle_cap {
                                    MAX_GPU_PARTICLES
                                } else {
                                    DEFAULT_PARTICLE_CAP
                                };
                                ui.add(
                                    Slider::new(
                                        &mut self.visual_settings.particle_count,
                                        10_000..=max_particles,
                                    )
                                    .step_by(5000.0)
                                    .text("Particle Count"),
                                );
                                if !self.unlocked_particle_cap {
                                    let unlock_btn = egui::Button::new(
                                        RichText::new("⚠ Unlock 1,000,000 Particle Cap")
                                            .color(Color32::from_rgb(255, 195, 65)),
                                    )
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        Color32::from_rgb(200, 130, 30),
                                    ));
                                    if ui.add(unlock_btn).clicked() {
                                        self.unlocked_particle_cap = true;
                                        self.set_status(
                                            "Unlocked 1,000,000 max particle cap (Warning: High GPU load)",
                                        );
                                    }
                                    ui.label(
                                        RichText::new(
                                            "Warning: Above 250k particles heavily increases GPU compute & fill-rate load and may reduce FPS.",
                                        )
                                        .small()
                                        .color(Color32::from_rgb(220, 165, 80)),
                                    );
                                } else {
                                    ui.label(
                                        RichText::new(
                                            "⚠ 1,000,000 Particle Cap Unlocked — High GPU / VRAM usage!",
                                        )
                                        .small()
                                        .color(Color32::from_rgb(255, 185, 65)),
                                    );
                                    if ui.button("🔒 Reset Cap to 250,000").clicked() {
                                        self.visual_settings.particle_count =
                                            self.visual_settings.particle_count.min(DEFAULT_PARTICLE_CAP);
                                        self.unlocked_particle_cap = false;
                                    }
                                }
                                ui.add_space(4.0);
                                ui.add(
                                    Slider::new(
                                        &mut self.visual_settings.particle_size,
                                        0.8..=5.0,
                                    )
                                    .text("Particle Size"),
                                );

                                ui.horizontal(|ui| {
                                    ui.label("Color Mode:");
                                    egui::ComboBox::from_id_salt("color_mode_combo")
                                        .selected_text(format!(
                                            "{:?}",
                                            self.visual_settings.color_mode
                                        ))
                                        .show_ui(ui, |ui| {
                                            ui.selectable_value(
                                                &mut self.visual_settings.color_mode,
                                                ColorMappingMode::VelocityHeatmap,
                                                "Velocity Heatmap",
                                            );
                                            ui.selectable_value(
                                                &mut self.visual_settings.color_mode,
                                                ColorMappingMode::NodalLineGlow,
                                                "Nodal Line Glow",
                                            );
                                            ui.selectable_value(
                                                &mut self.visual_settings.color_mode,
                                                ColorMappingMode::MetallicSand,
                                                "Metallic Sand",
                                            );
                                            ui.selectable_value(
                                                &mut self.visual_settings.color_mode,
                                                ColorMappingMode::CyberPhosphor,
                                                "Cyber Phosphor",
                                            );
                                            ui.selectable_value(
                                                &mut self.visual_settings.color_mode,
                                                ColorMappingMode::CustomGradient,
                                                "Custom Gradient",
                                            );
                                        });
                                });

                                if self.visual_settings.color_mode
                                    == ColorMappingMode::CustomGradient
                                {
                                    ui.color_edit_button_rgb(
                                        &mut self.visual_settings.custom_color_low,
                                    );
                                    ui.color_edit_button_rgb(
                                        &mut self.visual_settings.custom_color_high,
                                    );
                                }

                                ui.add(
                                    Slider::new(
                                        &mut self.visual_settings.trail_persistence,
                                        0.5..=0.98,
                                    )
                                    .text("Motion Trails"),
                                );
                                ui.checkbox(
                                    &mut self.visual_settings.show_nodal_lines,
                                    "Show Nodal Line Overlay",
                                );
                                if self.visual_settings.show_nodal_lines {
                                    ui.add(
                                        Slider::new(
                                            &mut self.visual_settings.nodal_line_intensity,
                                            0.05..=1.0,
                                        )
                                        .text("Nodal Brightness"),
                                    );
                                }
                            },
                        );

                        ui.separator();

                        // Synth
                        ui.collapsing(RichText::new("🔊 Test Tone Synth").strong(), |ui| {
                            ui.checkbox(&mut self.synth_settings.enabled, "Enable Output Synth");
                            ui.add(
                                Slider::new(
                                    &mut self.synth_settings.frequency_hz,
                                    20.0..=10000.0,
                                )
                                .logarithmic(true)
                                .text("Frequency (Hz)"),
                            );
                            ui.checkbox(
                                &mut self.synth_settings.sweep_enabled,
                                "Frequency Sweep",
                            );
                            if self.synth_settings.sweep_enabled {
                                ui.add(
                                    Slider::new(
                                        &mut self.synth_settings.sweep_speed,
                                        1.0..=60.0,
                                    )
                                    .text("Sweep Speed"),
                                );
                                ui.add(
                                    Slider::new(
                                        &mut self.synth_settings.sweep_min_hz,
                                        20.0..=1000.0,
                                    )
                                    .text("Min Hz"),
                                );
                                ui.add(
                                    Slider::new(
                                        &mut self.synth_settings.sweep_max_hz,
                                        500.0..=12000.0,
                                    )
                                    .text("Max Hz"),
                                );
                            }
                            ui.add(
                                Slider::new(&mut self.synth_settings.volume, 0.0..=1.0)
                                    .text("Volume"),
                            );
                        });
                    });
                });
        } else if self.is_fullscreen {
            egui::Area::new(egui::Id::unique("expand_btn_area_fs"))
                .fixed_pos(egui::pos2(6.0, 6.0))
                .order(egui::Order::Foreground)
                .show(ui.ctx(), |ui| {
                    let settings_btn = egui::Button::new(
                        RichText::new("⚙")
                            .size(11.0)
                            .color(Color32::from_gray(190)),
                    )
                    .fill(Color32::from_rgba_unmultiplied(16, 20, 28, 130))
                    .min_size(egui::vec2(18.0, 18.0))
                    .corner_radius(3);
                    if ui
                        .add(settings_btn)
                        .on_hover_text("Open Settings")
                        .clicked()
                    {
                        self.sidebar_expanded = true;
                    }
                });
        } else {
            egui::Area::new(egui::Id::unique("expand_btn_area"))
                .fixed_pos(egui::pos2(12.0, 12.0))
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("▶ Open Settings").clicked() {
                            self.sidebar_expanded = true;
                        }
                        if ui
                            .button("⛶ Fullscreen")
                            .on_hover_text("Enter True Fullscreen (F11)")
                            .clicked()
                        {
                            self.set_fullscreen(ui.ctx(), true);
                        }
                    });
                });
        }

        if self.is_fullscreen {
            egui::Area::new(egui::Id::unique("fullscreen_exit_btn"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-6.0, 6.0))
                .order(egui::Order::Foreground)
                .show(ui.ctx(), |ui| {
                    let exit_btn = egui::Button::new(
                        RichText::new("✕")
                            .size(11.0)
                            .color(Color32::from_gray(190)),
                    )
                    .fill(Color32::from_rgba_unmultiplied(16, 20, 28, 140))
                    .min_size(egui::vec2(18.0, 18.0))
                    .corner_radius(3);
                    if ui
                        .add(exit_btn)
                        .on_hover_text("Exit Fullscreen (Esc / F11)")
                        .clicked()
                    {
                        self.set_fullscreen(ui.ctx(), false);
                    }
                });
        }

        // 6. Central GPU canvas
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(Color32::from_rgb(4, 6, 10)))
            .show(ui, |ui| {
                let rect = ui.available_rect_before_wrap();
                let (response, painter) =
                    ui.allocate_painter(rect.size(), egui::Sense::hover());

                let view_width = rect.width();
                let view_height = rect.height();
                self.detected_canvas_px = [
                    (view_width * ppp).round().max(1.0) as u32,
                    (view_height * ppp).round().max(1.0) as u32,
                ];

                let drive_energy =
                    self.dsp.smoothed_energy * self.physics_settings.drive_amplitude;
                let use_tri_band = u32::from(
                    self.audio_settings.tracking_mode
                        == FrequencyTrackingMode::TriBandSuperposition
                        && !self.physics_settings.manual_override,
                );
                let active_count = self.visual_settings.particle_count;

                let uniforms = ShaderUniforms {
                    view_size: [view_width, view_height],
                    time: self.start_time.elapsed().as_secs_f32(),
                    dt: dt.clamp(0.0005, 0.05),
                    particle_size: self.visual_settings.particle_size,
                    active_particles: active_count as u32,
                    color_mode: match self.visual_settings.color_mode {
                        ColorMappingMode::VelocityHeatmap => 0,
                        ColorMappingMode::NodalLineGlow => 1,
                        ColorMappingMode::MetallicSand => 2,
                        ColorMappingMode::CyberPhosphor => 3,
                        ColorMappingMode::CustomGradient => 4,
                    },
                    show_nodal_lines: u32::from(self.visual_settings.show_nodal_lines),
                    nodal_intensity: self.visual_settings.nodal_line_intensity,
                    geometry: match self.physics_settings.geometry {
                        PlateGeometry::Square => 0,
                        PlateGeometry::Circular => 1,
                        PlateGeometry::Hexagonal => 2,
                    },
                    solver: match self.physics_settings.solver {
                        ModalEquationSolver::ClassicSymmetric => 0,
                        ModalEquationSolver::AsymmetricMixed => 1,
                        ModalEquationSolver::BesselCircular => 2,
                    },
                    boundary_mode: match self.physics_settings.boundary_mode {
                        EdgeBoundaryMode::Bounce => 0,
                        EdgeBoundaryMode::Wrap => 1,
                        EdgeBoundaryMode::RespawnRandom => 2,
                    },
                    m: self.physics_settings.m,
                    n: self.physics_settings.n,
                    a: self.physics_settings.a,
                    b: self.physics_settings.b,
                    kick_force: self.physics_settings.kick_force,
                    friction: self.physics_settings.friction,
                    brownian_jitter: self.physics_settings.brownian_jitter,
                    drive_energy,
                    bass_m: self.dsp.bass_m,
                    bass_n: self.dsp.bass_n,
                    mid_m: self.dsp.mid_m,
                    mid_n: self.dsp.mid_n,
                    treble_m: self.dsp.treble_m,
                    treble_n: self.dsp.treble_n,
                    e_bass: self.dsp.e_bass,
                    e_mid: self.dsp.e_mid,
                    e_treble: self.dsp.e_treble,
                    use_tri_band,
                    beat_kick: self.dsp.beat_kick,
                    trail_persistence: self.visual_settings.trail_persistence,
                    custom_low: self.visual_settings.custom_color_low,
                    swirl_strength: 0.65,
                    custom_high: self.visual_settings.custom_color_high,
                    use_screen_borders: u32::from(self.physics_settings.use_screen_borders),
                };

                let callback = egui_wgpu::Callback::new_paint_callback(
                    response.rect,
                    ChladniPaintCallback {
                        uniforms,
                        active_particle_count: active_count,
                    },
                );

                painter.add(callback);
            });

        ui.ctx().request_repaint();
    }
}
