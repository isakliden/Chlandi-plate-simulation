use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InputSourceMode {
    #[default]
    SystemLoopback,
    SingleAppProcess,
    Microphone,
    InternalSynth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PlateGeometry {
    #[default]
    Square,
    Circular,
    Hexagonal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ModalEquationSolver {
    #[default]
    ClassicSymmetric,
    AsymmetricMixed,
    BesselCircular,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FrequencyTrackingMode {
    #[default]
    TriBandSuperposition,
    DominantPeak,
    SubBassFocus,
    HarmonicChordBlend,
    MultiBandWeighted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FrequencyMappingCurve {
    Linear,
    #[default]
    LogarithmicOctaves,
    QuantizedHarmonics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColorMappingMode {
    VelocityHeatmap,
    NodalLineGlow,
    #[default]
    MetallicSand,
    CyberPhosphor,
    CustomGradient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EdgeBoundaryMode {
    #[default]
    Bounce,
    Wrap,
    RespawnRandom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub source_mode: InputSourceMode,
    pub input_gain: f32,
    pub noise_gate_threshold: f32,
    pub fft_size: usize,
    pub tracking_mode: FrequencyTrackingMode,
    pub attack_speed: f32,
    pub release_speed: f32,
    pub selected_device_name: Option<String>,
    pub selected_output_device_name: Option<String>,
    pub target_process_name: Option<String>,
    pub target_process_pid: Option<u32>,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            source_mode: InputSourceMode::SystemLoopback,
            input_gain: 2.2,
            noise_gate_threshold: 0.005,
            fft_size: 1024,
            tracking_mode: FrequencyTrackingMode::TriBandSuperposition,
            attack_speed: 0.45,
            release_speed: 0.12,
            selected_device_name: None,
            selected_output_device_name: None,
            target_process_name: None,
            target_process_pid: None,
        }
    }
}

fn default_use_screen_borders() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsSettings {
    pub geometry: PlateGeometry,
    pub solver: ModalEquationSolver,
    pub manual_override: bool,
    pub m: f32,
    pub n: f32,
    pub a: f32,
    pub b: f32,
    pub mapping_curve: FrequencyMappingCurve,
    pub kick_force: f32,
    pub friction: f32,
    pub brownian_jitter: f32,
    pub boundary_mode: EdgeBoundaryMode,
    pub drive_amplitude: f32,
    #[serde(default = "default_use_screen_borders")]
    pub use_screen_borders: bool,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            geometry: PlateGeometry::Square,
            solver: ModalEquationSolver::ClassicSymmetric,
            manual_override: false,
            m: 3.0,
            n: 5.0,
            a: 1.0,
            b: 1.0,
            mapping_curve: FrequencyMappingCurve::LogarithmicOctaves,
            kick_force: 0.0048,
            friction: 0.065,
            brownian_jitter: 0.0012,
            boundary_mode: EdgeBoundaryMode::Bounce,
            drive_amplitude: 1.0,
            use_screen_borders: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualSettings {
    pub particle_count: usize,
    pub particle_size: f32,
    pub color_mode: ColorMappingMode,
    pub trail_persistence: f32,
    pub show_nodal_lines: bool,
    pub nodal_line_intensity: f32,
    pub custom_color_low: [f32; 3],
    pub custom_color_high: [f32; 3],
}

impl Default for VisualSettings {
    fn default() -> Self {
        Self {
            particle_count: 150_000,
            particle_size: 2.2,
            color_mode: ColorMappingMode::MetallicSand,
            trail_persistence: 0.88,
            show_nodal_lines: true,
            nodal_line_intensity: 0.35,
            custom_color_low: [0.1, 0.4, 0.9],
            custom_color_high: [1.0, 0.7, 0.2],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SynthSettings {
    pub enabled: bool,
    pub frequency_hz: f32,
    pub sweep_enabled: bool,
    pub sweep_speed: f32,
    pub sweep_min_hz: f32,
    pub sweep_max_hz: f32,
    pub volume: f32,
}

impl Default for SynthSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            frequency_hz: 440.0,
            sweep_enabled: false,
            sweep_speed: 15.0,
            sweep_min_hz: 60.0,
            sweep_max_hz: 3500.0,
            volume: 0.35,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppPreset {
    pub name: String,
    pub audio: AudioSettings,
    pub physics: PhysicsSettings,
    pub visual: VisualSettings,
    pub synth: SynthSettings,
}

impl AppPreset {
    pub fn built_in_presets() -> Vec<Self> {
        vec![
            Self {
                name: "Fine Brass Dust".to_string(),
                audio: AudioSettings {
                    input_gain: 2.0,
                    fft_size: 1024,
                    tracking_mode: FrequencyTrackingMode::DominantPeak,
                    attack_speed: 0.3,
                    release_speed: 0.08,
                    ..Default::default()
                },
                physics: PhysicsSettings {
                    geometry: PlateGeometry::Square,
                    solver: ModalEquationSolver::ClassicSymmetric,
                    kick_force: 0.0055,
                    friction: 0.075,
                    brownian_jitter: 0.0008,
                    boundary_mode: EdgeBoundaryMode::Bounce,
                    ..Default::default()
                },
                visual: VisualSettings {
                    particle_count: 180_000,
                    particle_size: 2.0,
                    color_mode: ColorMappingMode::MetallicSand,
                    trail_persistence: 0.91,
                    show_nodal_lines: true,
                    nodal_line_intensity: 0.25,
                    ..Default::default()
                },
                synth: SynthSettings::default(),
            },
            Self {
                name: "Heavy Bass Liquid".to_string(),
                audio: AudioSettings {
                    input_gain: 3.5,
                    fft_size: 2048,
                    tracking_mode: FrequencyTrackingMode::SubBassFocus,
                    attack_speed: 0.45,
                    release_speed: 0.12,
                    ..Default::default()
                },
                physics: PhysicsSettings {
                    geometry: PlateGeometry::Circular,
                    solver: ModalEquationSolver::BesselCircular,
                    kick_force: 0.009,
                    friction: 0.04,
                    brownian_jitter: 0.0025,
                    boundary_mode: EdgeBoundaryMode::Wrap,
                    ..Default::default()
                },
                visual: VisualSettings {
                    particle_count: 220_000,
                    particle_size: 2.6,
                    color_mode: ColorMappingMode::VelocityHeatmap,
                    trail_persistence: 0.82,
                    show_nodal_lines: true,
                    nodal_line_intensity: 0.4,
                    ..Default::default()
                },
                synth: SynthSettings::default(),
            },
            Self {
                name: "High-Speed Laser".to_string(),
                audio: AudioSettings {
                    input_gain: 2.5,
                    fft_size: 512,
                    tracking_mode: FrequencyTrackingMode::HarmonicChordBlend,
                    attack_speed: 0.85,
                    release_speed: 0.2,
                    ..Default::default()
                },
                physics: PhysicsSettings {
                    geometry: PlateGeometry::Hexagonal,
                    solver: ModalEquationSolver::AsymmetricMixed,
                    kick_force: 0.007,
                    friction: 0.09,
                    brownian_jitter: 0.0006,
                    boundary_mode: EdgeBoundaryMode::Bounce,
                    ..Default::default()
                },
                visual: VisualSettings {
                    particle_count: 130_000,
                    particle_size: 1.8,
                    color_mode: ColorMappingMode::CyberPhosphor,
                    trail_persistence: 0.95,
                    show_nodal_lines: false,
                    nodal_line_intensity: 0.1,
                    ..Default::default()
                },
                synth: SynthSettings::default(),
            },
            Self {
                name: "Cosmic Glow".to_string(),
                audio: AudioSettings {
                    input_gain: 2.8,
                    fft_size: 1024,
                    tracking_mode: FrequencyTrackingMode::MultiBandWeighted,
                    attack_speed: 0.25,
                    release_speed: 0.06,
                    ..Default::default()
                },
                physics: PhysicsSettings {
                    geometry: PlateGeometry::Circular,
                    solver: ModalEquationSolver::BesselCircular,
                    kick_force: 0.005,
                    friction: 0.05,
                    brownian_jitter: 0.0015,
                    boundary_mode: EdgeBoundaryMode::Bounce,
                    ..Default::default()
                },
                visual: VisualSettings {
                    particle_count: 200_000,
                    particle_size: 2.3,
                    color_mode: ColorMappingMode::NodalLineGlow,
                    trail_persistence: 0.89,
                    show_nodal_lines: true,
                    nodal_line_intensity: 0.55,
                    ..Default::default()
                },
                synth: SynthSettings::default(),
            },
        ]
    }

    pub fn presets_file_path() -> std::path::PathBuf {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            std::path::PathBuf::from(appdata)
                .join("ChladniVisualizer")
                .join("presets.json")
        } else if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
            std::path::PathBuf::from(xdg)
                .join("chladni-visualizer")
                .join("presets.json")
        } else if let Some(home) = std::env::var_os("HOME") {
            #[cfg(target_os = "macos")]
            {
                std::path::PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
                    .join("ChladniVisualizer")
                    .join("presets.json")
            }
            #[cfg(not(target_os = "macos"))]
            {
                std::path::PathBuf::from(home)
                    .join(".config")
                    .join("chladni-visualizer")
                    .join("presets.json")
            }
        } else {
            std::path::PathBuf::from("presets.json")
        }
    }

    pub fn load_from_path(path: &std::path::Path) -> Vec<Self> {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(loaded) = serde_json::from_str::<Vec<Self>>(&contents) {
                return loaded;
            }
        }
        let defaults = Self::built_in_presets();
        let _ = Self::save_to_path(&defaults, path);
        defaults
    }

    pub fn save_to_path(presets: &[Self], path: &std::path::Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create presets directory: {}", e))?;
            }
        }
        let json = serde_json::to_string_pretty(presets)
            .map_err(|e| format!("Failed to serialize presets: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write presets file: {}", e))
    }

    pub fn load_from_disk() -> Vec<Self> {
        Self::load_from_path(&Self::presets_file_path())
    }

    pub fn save_to_disk(presets: &[Self]) -> Result<(), String> {
        Self::save_to_path(presets, &Self::presets_file_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_presets_persist_and_remove_cleanly() {
        let dir = std::env::temp_dir().join(format!(
            "chladni_preset_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("presets.json");

        // Initial load seeds built-in presets on disk
        let mut presets = AppPreset::load_from_path(&path);
        let initial_len = presets.len();
        assert!(initial_len >= 4);
        assert!(path.exists());

        // Add custom preset and persist
        let mut custom = AppPreset {
            name: "Crusher Test".to_string(),
            audio: AudioSettings::default(),
            physics: PhysicsSettings::default(),
            visual: VisualSettings::default(),
            synth: SynthSettings::default(),
        };
        custom.visual.particle_count = 123_456;
        presets.push(custom);
        AppPreset::save_to_path(&presets, &path).expect("save failed");

        // Reload and verify custom preset persists
        let mut reloaded = AppPreset::load_from_path(&path);
        assert_eq!(reloaded.len(), initial_len + 1);
        assert_eq!(reloaded.last().unwrap().name, "Crusher Test");
        assert_eq!(reloaded.last().unwrap().visual.particle_count, 123_456);

        // Remove custom preset, save, and verify removal persists
        reloaded.retain(|p| p.name != "Crusher Test");
        AppPreset::save_to_path(&reloaded, &path).expect("save after remove failed");
        let after_remove = AppPreset::load_from_path(&path);
        assert_eq!(after_remove.len(), initial_len);
        assert!(after_remove.iter().all(|p| p.name != "Crusher Test"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
