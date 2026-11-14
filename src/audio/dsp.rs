use crate::types::{
    AudioSettings, FrequencyMappingCurve, FrequencyTrackingMode, PhysicsSettings,
};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

pub struct AudioDspProcessor {
    fft_size: usize,
    fft: Arc<dyn Fft<f32>>,
    sample_ring: Vec<f32>,
    input_buffer: Vec<Complex<f32>>,
    window: Vec<f32>,
    pub spectrum_magnitudes: Vec<f32>,
    prev_spectrum: Vec<f32>,

    pub detected_frequency: f32,
    pub smoothed_energy: f32,
    pub smoothed_m: f32,
    pub smoothed_n: f32,
    pub smoothed_a: f32,
    pub smoothed_b: f32,

    pub bass_m: f32,
    pub bass_n: f32,
    pub mid_m: f32,
    pub mid_n: f32,
    pub treble_m: f32,
    pub treble_n: f32,

    pub e_bass: f32,
    pub e_mid: f32,
    pub e_treble: f32,

    pub beat_kick: f32,
    energy_avg: f32,
    flux_avg: f32,

    pub vu_peak: f32,
    pub vu_rms: f32,
}

impl AudioDspProcessor {
    pub fn new(fft_size: usize) -> Self {
        let size = fft_size.max(256);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        let window = Self::build_hann_window(size);

        Self {
            fft_size: size,
            fft,
            sample_ring: vec![0.0; size],
            input_buffer: vec![Complex::new(0.0, 0.0); size],
            window,
            spectrum_magnitudes: vec![0.0; size / 2],
            prev_spectrum: vec![0.0; size / 2],
            detected_frequency: 440.0,
            smoothed_energy: 0.0,
            smoothed_m: 3.0,
            smoothed_n: 5.0,
            smoothed_a: 1.0,
            smoothed_b: 1.0,
            bass_m: 2.0,
            bass_n: 3.0,
            mid_m: 4.0,
            mid_n: 7.0,
            treble_m: 8.0,
            treble_n: 13.0,
            e_bass: 0.0,
            e_mid: 0.0,
            e_treble: 0.0,
            beat_kick: 0.0,
            energy_avg: 0.02,
            flux_avg: 0.01,
            vu_peak: 0.0,
            vu_rms: 0.0,
        }
    }

    pub fn resize_fft(&mut self, new_size: usize) {
        let size = new_size.max(256);
        if size == self.fft_size {
            return;
        }

        let mut planner = FftPlanner::new();
        self.fft = planner.plan_fft_forward(size);
        self.window = Self::build_hann_window(size);

        let mut new_ring = vec![0.0; size];
        let copy_len = self.sample_ring.len().min(size);
        let src_start = self.sample_ring.len() - copy_len;
        let dst_start = size - copy_len;
        new_ring[dst_start..].copy_from_slice(&self.sample_ring[src_start..]);

        self.fft_size = size;
        self.sample_ring = new_ring;
        self.input_buffer = vec![Complex::new(0.0, 0.0); size];
        self.spectrum_magnitudes = vec![0.0; size / 2];
        self.prev_spectrum = vec![0.0; size / 2];
    }

    fn build_hann_window(size: usize) -> Vec<f32> {
        let denom = (size.saturating_sub(1)).max(1) as f32;
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / denom).cos()))
            .collect()
    }

    /// Low-pass perceptual tilt: boosts sub-bass/bass around 40-80 Hz (~2.0x-2.7x),
    /// leaves 250 Hz mids at 1.0x, and attenuates 8 kHz hi-hats down to ~0.15x.
    #[inline]
    pub fn perceptual_tilt(freq_hz: f32) -> f32 {
        (250.0 / freq_hz.max(25.0)).powf(0.55)
    }

    pub fn process_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        audio_settings: &AudioSettings,
        physics_settings: &mut PhysicsSettings,
        dt: f32,
    ) {
        let dt = dt.clamp(0.0005, 0.05);
        self.beat_kick *= (-14.0 * dt).exp();

        if samples.is_empty() {
            let rel = audio_settings.release_speed.clamp(0.01, 1.0) * 0.25;
            self.smoothed_energy += (0.0 - self.smoothed_energy) * rel;
            self.e_bass += (0.0 - self.e_bass) * rel;
            self.e_mid += (0.0 - self.e_mid) * rel;
            self.e_treble += (0.0 - self.e_treble) * rel;
            return;
        }

        // Update VU meters and push samples into persistent sliding window
        let gain = audio_settings.input_gain;
        let mut peak = 0.0_f32;
        let mut sq_sum = 0.0_f32;
        for &s in samples {
            let scaled = s * gain;
            peak = peak.max(scaled.abs());
            sq_sum += scaled * scaled;
        }
        self.vu_peak = peak.clamp(0.0, 2.0);
        self.vu_rms = (sq_sum / samples.len() as f32).sqrt().clamp(0.0, 2.0);

        if samples.len() >= self.fft_size {
            let tail = &samples[samples.len() - self.fft_size..];
            for (dst, &src) in self.sample_ring.iter_mut().zip(tail.iter()) {
                *dst = src * gain;
            }
        } else {
            let shift = samples.len();
            self.sample_ring.copy_within(shift.., 0);
            let start = self.fft_size - shift;
            for (dst, &src) in self.sample_ring[start..].iter_mut().zip(samples.iter()) {
                *dst = src * gain;
            }
        }

        // Apply full Hann window across the persistent history buffer and run FFT
        for i in 0..self.fft_size {
            self.input_buffer[i] = Complex::new(self.sample_ring[i] * self.window[i], 0.0);
        }
        self.fft.process(&mut self.input_buffer);

        let sr = sample_rate.max(8000) as f32;
        let bin_hz = sr / self.fft_size as f32;
        let half = self.fft_size / 2;
        let norm = 4.0 / self.fft_size as f32;

        let mut inst_energy_sq = 0.0_f32;
        let mut kick_energy_sq = 0.0_f32;
        let mut pos_flux = 0.0_f32;
        let mut kick_bins = 0usize;

        let mut peak_bin = 1usize;
        let mut peak_mag = 0.0_f32;

        self.spectrum_magnitudes[0] = 0.0;
        self.prev_spectrum[0] = 0.0;

        for i in 1..half {
            let freq_hz = i as f32 * bin_hz;
            let raw_mag = self.input_buffer[i].norm() * norm;
            let tilt = Self::perceptual_tilt(freq_hz);
            let mag = (raw_mag * tilt).min(10.0);

            if (20.0..=12000.0).contains(&freq_hz) {
                inst_energy_sq += mag * mag;
                if mag > peak_mag {
                    peak_mag = mag;
                    peak_bin = i;
                }
            }

            // Beat-kick transient detection uses ONLY bass (25-180 Hz at 1.0x)
            // and low-mid punch (180-400 Hz at 0.2x), ignoring > 1000 Hz hi-hats completely.
            let kick_weight = if (25.0..=180.0).contains(&freq_hz) {
                1.0
            } else if freq_hz > 180.0 && freq_hz <= 400.0 {
                0.2
            } else {
                0.0
            };

            if kick_weight > 0.0 {
                let weighted_mag = mag * kick_weight;
                let weighted_prev = self.prev_spectrum[i] * kick_weight;
                kick_energy_sq += weighted_mag * weighted_mag;
                pos_flux += (weighted_mag - weighted_prev).max(0.0);
                kick_bins += 1;
            }

            self.spectrum_magnitudes[i] = mag;
            self.prev_spectrum[i] = mag;
        }

        // Single fast asymmetric envelope follower (attack=0.45, release=0.12 default)
        let inst_energy = inst_energy_sq.sqrt() * 0.45;
        let target_energy = if inst_energy < audio_settings.noise_gate_threshold {
            0.0
        } else {
            inst_energy.clamp(0.0, 2.0)
        };

        let env_alpha = if target_energy > self.smoothed_energy {
            audio_settings.attack_speed.clamp(0.01, 1.0)
        } else {
            audio_settings.release_speed.clamp(0.01, 1.0)
        };
        self.smoothed_energy += (target_energy - self.smoothed_energy) * env_alpha;

        // Beat kick transient spike
        if target_energy > 0.0 && kick_bins > 0 {
            let kick_energy = kick_energy_sq.sqrt();
            let flux_norm = pos_flux / (kick_bins as f32).sqrt().max(1.0);
            let avg_alpha = (dt * 4.5).clamp(0.02, 0.25);
            self.energy_avg += (kick_energy - self.energy_avg) * avg_alpha;
            self.flux_avg += (flux_norm - self.flux_avg) * avg_alpha;

            let energy_spike = (kick_energy - self.energy_avg * 1.22).max(0.0);
            let flux_spike = (flux_norm - self.flux_avg * 1.35).max(0.0);
            let raw_kick = (energy_spike * 2.2 + flux_spike * 2.8).clamp(0.0, 2.5);
            self.beat_kick = self.beat_kick.max(raw_kick);
        }

        // Sub-bin pitch estimation around dominant bin
        if target_energy > 0.0 && peak_mag > 1e-4 {
            let interp = if peak_bin > 1 && peak_bin + 1 < half {
                let y0 = self.spectrum_magnitudes[peak_bin - 1];
                let y1 = self.spectrum_magnitudes[peak_bin];
                let y2 = self.spectrum_magnitudes[peak_bin + 1];
                let denom = y0 - 2.0 * y1 + y2;
                if denom.abs() > 1e-6 {
                    (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            self.detected_frequency = ((peak_bin as f32 + interp) * bin_hz).clamp(20.0, 16000.0);
        }

        // Tri-band superposition modal extraction
        let curve = physics_settings.mapping_curve;
        let (target_bm, target_bn, b_e) =
            self.extract_band_peak_and_energy(20.0, 200.0, bin_hz, 80.0, curve);
        let (target_mm, target_mn, m_e) =
            self.extract_band_peak_and_energy(200.0, 2500.0, bin_hz, 440.0, curve);
        let (target_tm, target_tn, t_e) =
            self.extract_band_peak_and_energy(2500.0, 12000.0, bin_hz, 4000.0, curve);

        let tri_total = b_e + m_e + t_e + 1e-5;
        let signal_gate = if target_energy <= 0.0 {
            0.0
        } else {
            ((b_e + m_e + t_e) * 4.0).clamp(0.0, 1.0)
        };
        let target_e_bass = (b_e / tri_total) * 1.35 * signal_gate + b_e.min(0.45) * signal_gate;
        let target_e_mid = (m_e / tri_total) * 1.35 * signal_gate + m_e.min(0.45) * signal_gate;
        let target_e_treble = (t_e / tri_total) * 1.35 * signal_gate + t_e.min(0.45) * signal_gate;

        let tri_alpha = |target: f32, cur: f32| -> f32 {
            if target > cur {
                audio_settings.attack_speed.max(0.35).clamp(0.05, 1.0)
            } else {
                audio_settings.release_speed.clamp(0.03, 0.5)
            }
        };

        self.e_bass += (target_e_bass - self.e_bass) * tri_alpha(target_e_bass, self.e_bass);
        self.e_mid += (target_e_mid - self.e_mid) * tri_alpha(target_e_mid, self.e_mid);
        self.e_treble +=
            (target_e_treble - self.e_treble) * tri_alpha(target_e_treble, self.e_treble);

        let mode_alpha = (dt * 12.0).clamp(0.08, 0.45);
        if signal_gate > 0.01 {
            self.bass_m += (target_bm - self.bass_m) * mode_alpha;
            self.bass_n += (target_bn - self.bass_n) * mode_alpha;
            self.mid_m += (target_mm - self.mid_m) * mode_alpha;
            self.mid_n += (target_mn - self.mid_n) * mode_alpha;
            self.treble_m += (target_tm - self.treble_m) * mode_alpha;
            self.treble_n += (target_tn - self.treble_n) * mode_alpha;
        }

        // Single-mode fallback & UI mode parameters
        let (target_m, target_n, target_a, target_b) = match audio_settings.tracking_mode {
            FrequencyTrackingMode::TriBandSuperposition => {
                if self.e_bass >= self.e_mid && self.e_bass >= self.e_treble {
                    (self.bass_m, self.bass_n, 1.0, 1.0)
                } else if self.e_mid >= self.e_treble {
                    (self.mid_m, self.mid_n, 1.0, 1.0)
                } else {
                    (self.treble_m, self.treble_n, 1.0, 1.0)
                }
            }
            FrequencyTrackingMode::DominantPeak => {
                let (m, n) = Self::frequency_to_modal_numbers(self.detected_frequency, curve);
                (m, n, 1.0, 1.0)
            }
            FrequencyTrackingMode::SubBassFocus => {
                let (m, n, _) =
                    self.extract_band_peak_and_energy(20.0, 220.0, bin_hz, 70.0, curve);
                (m, n, 1.0, -1.0)
            }
            FrequencyTrackingMode::HarmonicChordBlend => {
                let b_w = self.e_bass + 0.01;
                let m_w = self.e_mid + 0.01;
                let total = b_w + m_w;
                let m = (self.bass_m * b_w + self.mid_m * m_w) / total;
                let n = (self.bass_n * b_w + self.mid_n * m_w) / total;
                (m, n, 1.0, 0.85)
            }
            FrequencyTrackingMode::MultiBandWeighted => {
                let w_sum = self.e_bass + self.e_mid + self.e_treble + 0.01;
                let m = (self.bass_m * self.e_bass
                    + self.mid_m * self.e_mid
                    + self.treble_m * self.e_treble)
                    / w_sum;
                let n = (self.bass_n * self.e_bass
                    + self.mid_n * self.e_mid
                    + self.treble_n * self.e_treble)
                    / w_sum;
                (m, n, 1.0, 1.0)
            }
        };

        self.smoothed_m += (target_m - self.smoothed_m) * mode_alpha;
        self.smoothed_n += (target_n - self.smoothed_n) * mode_alpha;
        self.smoothed_a += (target_a - self.smoothed_a) * mode_alpha;
        self.smoothed_b += (target_b - self.smoothed_b) * mode_alpha;

        if !physics_settings.manual_override {
            physics_settings.m = self.smoothed_m;
            physics_settings.n = self.smoothed_n;
            physics_settings.a = self.smoothed_a;
            physics_settings.b = self.smoothed_b;
        }
    }

    fn extract_band_peak_and_energy(
        &self,
        min_hz: f32,
        max_hz: f32,
        bin_hz: f32,
        default_hz: f32,
        curve: FrequencyMappingCurve,
    ) -> (f32, f32, f32) {
        let half = self.spectrum_magnitudes.len();
        if half <= 2 {
            let (m, n) = Self::frequency_to_modal_numbers(default_hz, curve);
            return (m, n, 0.0);
        }

        let min_bin = ((min_hz / bin_hz).floor() as usize).clamp(1, half - 1);
        let max_bin = ((max_hz / bin_hz).ceil() as usize).clamp(min_bin, half - 1);

        let mut sum_sq = 0.0_f32;
        let mut peak_bin = min_bin;
        let mut peak_mag = 0.0_f32;

        for (i, &mag) in self
            .spectrum_magnitudes
            .iter()
            .enumerate()
            .take(max_bin + 1)
            .skip(min_bin)
        {
            sum_sq += mag * mag;
            if mag > peak_mag {
                peak_mag = mag;
                peak_bin = i;
            }
        }

        let energy = sum_sq.sqrt();
        let hz = if peak_mag > 1e-4 {
            let interp = if peak_bin > min_bin && peak_bin < max_bin {
                let y0 = self.spectrum_magnitudes[peak_bin - 1];
                let y1 = self.spectrum_magnitudes[peak_bin];
                let y2 = self.spectrum_magnitudes[peak_bin + 1];
                let denom = y0 - 2.0 * y1 + y2;
                if denom.abs() > 1e-6 {
                    (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            ((peak_bin as f32 + interp) * bin_hz).clamp(min_hz, max_hz)
        } else {
            default_hz
        };

        let (m, n) = Self::frequency_to_modal_numbers(hz, curve);
        (m, n, energy)
    }

    pub fn frequency_to_modal_numbers(
        freq_hz: f32,
        curve: FrequencyMappingCurve,
    ) -> (f32, f32) {
        let f = freq_hz.clamp(20.0, 16000.0);
        match curve {
            FrequencyMappingCurve::Linear => {
                let t = ((f - 20.0) / 3980.0).clamp(0.0, 3.0);
                let m = 1.0 + t * 4.5;
                let n = 2.0 + t * 6.0;
                (m, n)
            }
            FrequencyMappingCurve::LogarithmicOctaves => {
                let octaves = (f / 40.0).max(1.0).log2();
                let m = 1.0 + octaves * 1.45;
                let n = 2.0 + octaves * 2.15 + (octaves * 1.7).sin() * 0.45;
                (m, n)
            }
            FrequencyMappingCurve::QuantizedHarmonics => {
                const PAIRS: [(f32, f32); 16] = [
                    (1.0, 2.0),
                    (1.0, 3.0),
                    (2.0, 3.0),
                    (1.0, 4.0),
                    (2.0, 5.0),
                    (3.0, 4.0),
                    (3.0, 5.0),
                    (2.0, 7.0),
                    (4.0, 5.0),
                    (3.0, 7.0),
                    (4.0, 7.0),
                    (5.0, 7.0),
                    (5.0, 8.0),
                    (6.0, 9.0),
                    (7.0, 10.0),
                    (8.0, 11.0),
                ];
                let norm_oct = ((f / 40.0).max(1.0).log2() / 8.0).clamp(0.0, 0.999);
                let idx = (norm_oct * PAIRS.len() as f32) as usize;
                PAIRS[idx.min(PAIRS.len() - 1)]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fft_size_is_1024() {
        let settings = AudioSettings::default();
        assert_eq!(settings.fft_size, 1024);
    }

    #[test]
    fn inverted_tilt_boosts_bass_over_treble() {
        let t_50 = AudioDspProcessor::perceptual_tilt(50.0);
        let t_250 = AudioDspProcessor::perceptual_tilt(250.0);
        let t_8000 = AudioDspProcessor::perceptual_tilt(8000.0);

        assert!(t_50 > 2.0 && t_50 < 2.8, "50 Hz tilt was {}", t_50);
        assert!((t_250 - 1.0).abs() < 1e-5, "250 Hz tilt was {}", t_250);
        assert!(t_8000 < 0.2, "8000 Hz tilt was {}", t_8000);
    }

    #[test]
    fn beat_kick_triggers_on_bass_and_ignores_hi_hats() {
        let audio = AudioSettings::default();
        let mut physics = PhysicsSettings::default();
        let sr = 48_000u32;
        let dt = 1.0 / 60.0;

        // 1. High-frequency 8,000 Hz hi-hat burst should NOT trigger beat_kick
        let mut dsp_hihat = AudioDspProcessor::new(audio.fft_size);
        let silence = vec![0.0_f32; 1024];
        for _ in 0..8 {
            dsp_hihat.process_samples(&silence, sr, &audio, &mut physics, dt);
        }
        let mut hihat = vec![0.0_f32; 1024];
        for (i, s) in hihat.iter_mut().enumerate() {
            let env = (i as f32 / 128.0).min(1.0);
            *s = env * 0.8 * (2.0 * PI * 8000.0 * i as f32 / sr as f32).sin();
        }
        dsp_hihat.process_samples(&hihat, sr, &audio, &mut physics, dt);
        assert!(
            dsp_hihat.beat_kick < 0.05,
            "8 kHz hi-hat triggered beat_kick={}",
            dsp_hihat.beat_kick
        );

        // 2. Low-frequency 60 Hz kick burst MUST trigger beat_kick
        let mut dsp_bass = AudioDspProcessor::new(audio.fft_size);
        for _ in 0..8 {
            dsp_bass.process_samples(&silence, sr, &audio, &mut physics, dt);
        }
        let mut bass_kick = vec![0.0_f32; 1024];
        for (i, s) in bass_kick.iter_mut().enumerate() {
            let env = (i as f32 / 64.0).min(1.0);
            *s = env * 0.8 * (2.0 * PI * 60.0 * i as f32 / sr as f32).sin();
        }
        dsp_bass.process_samples(&bass_kick, sr, &audio, &mut physics, dt);
        assert!(
            dsp_bass.beat_kick > 0.3,
            "60 Hz bass kick failed to trigger strong beat_kick={}",
            dsp_bass.beat_kick
        );
    }
}
