use crate::audio::wasapi::{
    enumerate_audio_devices, find_input_device, find_output_device, list_running_processes,
    AudioDeviceInfo, ProcessItem, ProAudioThreadGuard,
};
use crate::types::{AudioSettings, InputSourceMode, SynthSettings};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct AudioEngine {
    host: cpal::Host,
    pub output_devices: Vec<AudioDeviceInfo>,
    pub input_devices: Vec<AudioDeviceInfo>,
    pub running_processes: Vec<ProcessItem>,
    pub active_stream: Option<Stream>,
    pub synth_stream: Option<Stream>,
    pub sample_consumer: Consumer<f32>,
    sample_producer: Arc<Mutex<Option<Producer<f32>>>>,
    pub sample_rate: Arc<AtomicU32>,
    pub is_capturing: Arc<AtomicBool>,
    pub current_vu: Arc<AtomicU32>,
    last_source_mode: Option<InputSourceMode>,
    last_device_name: Option<String>,
    pub synth_phase: f32,
    synth_time: Instant,
}

impl AudioEngine {
    pub fn new() -> Self {
        let host = cpal::default_host();
        let (output_devices, input_devices) = enumerate_audio_devices(&host);
        let running_processes = list_running_processes();

        let (producer, consumer) = RingBuffer::new(32_768);

        Self {
            host,
            output_devices,
            input_devices,
            running_processes,
            active_stream: None,
            synth_stream: None,
            sample_consumer: consumer,
            sample_producer: Arc::new(Mutex::new(Some(producer))),
            sample_rate: Arc::new(AtomicU32::new(48000)),
            is_capturing: Arc::new(AtomicBool::new(false)),
            current_vu: Arc::new(AtomicU32::new(0)),
            last_source_mode: None,
            last_device_name: None,
            synth_phase: 0.0,
            synth_time: Instant::now(),
        }
    }

    pub fn refresh_devices_and_processes(&mut self) {
        let (output_devices, input_devices) = enumerate_audio_devices(&self.host);
        self.output_devices = output_devices;
        self.input_devices = input_devices;
        self.running_processes = list_running_processes();
    }

    pub fn check_and_update_stream(
        &mut self,
        audio_settings: &AudioSettings,
        synth_settings: &SynthSettings,
    ) {
        let mode = audio_settings.source_mode;
        let device_name = audio_settings.selected_device_name.clone();

        let need_reinit = self.last_source_mode != Some(mode)
            || self.last_device_name != device_name
            || self.active_stream.is_none();

        if need_reinit {
            self.reinit_capture_stream(audio_settings);
            self.last_source_mode = Some(mode);
            self.last_device_name = device_name;
        }

        if synth_settings.enabled && self.synth_stream.is_none() {
            self.start_synth_output_stream(
                synth_settings,
                audio_settings.selected_output_device_name.as_deref(),
            );
        } else if !synth_settings.enabled && self.synth_stream.is_some() {
            self.synth_stream = None;
        }
    }

    fn reinit_capture_stream(&mut self, settings: &AudioSettings) {
        self.active_stream = None;
        self.is_capturing.store(false, Ordering::SeqCst);

        let (producer, consumer) = RingBuffer::new(32_768);
        self.sample_consumer = consumer;
        {
            let mut prod_lock = self
                .sample_producer
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *prod_lock = Some(producer);
        }

        let producer_arc = Arc::clone(&self.sample_producer);
        let sample_rate_atomic = Arc::clone(&self.sample_rate);
        let is_capturing_atomic = Arc::clone(&self.is_capturing);
        let current_vu_atomic = Arc::clone(&self.current_vu);

        match settings.source_mode {
            InputSourceMode::SystemLoopback | InputSourceMode::SingleAppProcess => {
                // On Windows WASAPI loopback: open input stream on OUTPUT device
                if let Some(device) =
                    find_output_device(&self.host, settings.selected_device_name.as_deref())
                {
                    if let Ok(config) = device.default_output_config() {
                        let sample_format = config.sample_format();
                        let stream_config: StreamConfig = config.into();
                        // SampleRate is a type alias for u32 in this cpal version
                        sample_rate_atomic.store(stream_config.sample_rate, Ordering::Relaxed);

                        let stream_res = match sample_format {
                            SampleFormat::F32 => Self::build_input_stream::<f32>(
                                &device,
                                stream_config,
                                producer_arc,
                                is_capturing_atomic,
                                current_vu_atomic,
                            ),
                            SampleFormat::I16 => Self::build_input_stream::<i16>(
                                &device,
                                stream_config,
                                producer_arc,
                                is_capturing_atomic,
                                current_vu_atomic,
                            ),
                            SampleFormat::U16 => Self::build_input_stream::<u16>(
                                &device,
                                stream_config,
                                producer_arc,
                                is_capturing_atomic,
                                current_vu_atomic,
                            ),
                            _ => return,
                        };

                        if let Ok(stream) = stream_res {
                            let _ = stream.play();
                            self.active_stream = Some(stream);
                        }
                    }
                }
            }
            InputSourceMode::Microphone => {
                if let Some(device) =
                    find_input_device(&self.host, settings.selected_device_name.as_deref())
                {
                    if let Ok(config) = device.default_input_config() {
                        let sample_format = config.sample_format();
                        let stream_config: StreamConfig = config.into();
                        sample_rate_atomic.store(stream_config.sample_rate, Ordering::Relaxed);

                        let stream_res = match sample_format {
                            SampleFormat::F32 => Self::build_input_stream::<f32>(
                                &device,
                                stream_config,
                                producer_arc,
                                is_capturing_atomic,
                                current_vu_atomic,
                            ),
                            SampleFormat::I16 => Self::build_input_stream::<i16>(
                                &device,
                                stream_config,
                                producer_arc,
                                is_capturing_atomic,
                                current_vu_atomic,
                            ),
                            SampleFormat::U16 => Self::build_input_stream::<u16>(
                                &device,
                                stream_config,
                                producer_arc,
                                is_capturing_atomic,
                                current_vu_atomic,
                            ),
                            _ => return,
                        };

                        if let Ok(stream) = stream_res {
                            let _ = stream.play();
                            self.active_stream = Some(stream);
                        }
                    }
                }
            }
            InputSourceMode::InternalSynth => {
                self.is_capturing.store(true, Ordering::Relaxed);
            }
        }
    }

    fn build_input_stream<T>(
        device: &cpal::Device,
        config: StreamConfig,
        producer_arc: Arc<Mutex<Option<Producer<f32>>>>,
        is_capturing: Arc<AtomicBool>,
        current_vu: Arc<AtomicU32>,
    ) -> Result<Stream, cpal::Error>
    where
        T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32> + Into<f32> + Send + 'static,
    {
        let channels = config.channels as usize;
        let producer = producer_arc
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let mut producer = match producer {
            Some(p) => p,
            None => {
                return Err(cpal::Error::new(cpal::ErrorKind::InvalidInput));
            }
        };

        device.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let _guard = ProAudioThreadGuard::enter();
                is_capturing.store(true, Ordering::Relaxed);

                let mut max_val = 0.0f32;
                for frame in data.chunks_exact(channels) {
                    let mut sum = 0.0f32;
                    for &s in frame {
                        let val: f32 = s.into();
                        sum += val;
                    }
                    let mono = sum / channels as f32;
                    max_val = max_val.max(mono.abs());
                    let _ = producer.push(mono);
                }
                current_vu.store(max_val.to_bits(), Ordering::Relaxed);
            },
            move |err| {
                eprintln!("Audio Capture Stream Error: {}", err);
            },
            None,
        )
    }

    fn start_synth_output_stream(&mut self, synth: &SynthSettings, out_name: Option<&str>) {
        if let Some(device) = find_output_device(&self.host, out_name) {
            if let Ok(config) = device.default_output_config() {
                let stream_config: StreamConfig = config.into();
                let sample_rate = stream_config.sample_rate as f32;
                let channels = stream_config.channels as usize;

                let mut phase = 0.0f32;
                let freq = synth.frequency_hz;
                let vol = synth.volume;

                let stream = device.build_output_stream(
                    stream_config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let _guard = ProAudioThreadGuard::enter();
                        let phase_inc = (freq * std::f32::consts::TAU) / sample_rate;
                        for frame in data.chunks_exact_mut(channels) {
                            let sample = phase.sin() * vol;
                            phase = (phase + phase_inc) % std::f32::consts::TAU;
                            for s in frame.iter_mut() {
                                *s = sample;
                            }
                        }
                    },
                    |err| eprintln!("Synth Output Stream Error: {}", err),
                    None,
                );

                if let Ok(s) = stream {
                    let _ = s.play();
                    self.synth_stream = Some(s);
                }
            }
        }
    }

    pub fn fetch_samples(&mut self, output: &mut Vec<f32>, max_count: usize) -> usize {
        let mut count = 0;
        while count < max_count {
            if let Ok(sample) = self.sample_consumer.pop() {
                output.push(sample);
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    pub fn generate_synth_samples(
        &mut self,
        output: &mut Vec<f32>,
        count: usize,
        synth: &SynthSettings,
        sample_rate: u32,
    ) {
        let sr = sample_rate as f32;
        let elapsed = self.synth_time.elapsed().as_secs_f32();

        let current_hz = if synth.sweep_enabled {
            let range = synth.sweep_max_hz - synth.sweep_min_hz;
            let cycle = (elapsed * synth.sweep_speed * 0.1).sin() * 0.5 + 0.5;
            synth.sweep_min_hz + cycle * range
        } else {
            synth.frequency_hz
        };

        let phase_inc = (current_hz * std::f32::consts::TAU) / sr;
        for _ in 0..count {
            let s = self.synth_phase.sin() * synth.volume;
            self.synth_phase = (self.synth_phase + phase_inc) % std::f32::consts::TAU;
            output.push(s);
        }
    }
}
