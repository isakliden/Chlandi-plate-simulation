use crate::physics::particles::{GpuParticle, ParticleSystem};
use crate::render::shaders::CHLADNI_WGSL;
use bytemuck::{Pod, Zeroable};
use egui::epaint::PaintCallbackInfo;
use egui_wgpu::wgpu;
use egui_wgpu::wgpu::util::DeviceExt;
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};

pub const DEFAULT_PARTICLE_CAP: usize = 250_000;
pub const MAX_GPU_PARTICLES: usize = 1_000_000;

/// GPU uniform buffer matching the 160-byte WGSL `Uniforms` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShaderUniforms {
    pub view_size: [f32; 2],
    pub time: f32,
    pub dt: f32,

    pub particle_size: f32,
    pub active_particles: u32,
    pub color_mode: u32,
    pub show_nodal_lines: u32,

    pub nodal_intensity: f32,
    pub geometry: u32,
    pub solver: u32,
    pub boundary_mode: u32,

    pub m: f32,
    pub n: f32,
    pub a: f32,
    pub b: f32,

    pub kick_force: f32,
    pub friction: f32,
    pub brownian_jitter: f32,
    pub drive_energy: f32,

    pub bass_m: f32,
    pub bass_n: f32,
    pub mid_m: f32,
    pub mid_n: f32,

    pub treble_m: f32,
    pub treble_n: f32,
    pub e_bass: f32,
    pub e_mid: f32,

    pub e_treble: f32,
    pub use_tri_band: u32,
    pub beat_kick: f32,
    pub trail_persistence: f32,

    pub custom_low: [f32; 3],
    pub swirl_strength: f32,

    pub custom_high: [f32; 3],
    pub use_screen_borders: u32,
}

const _: () = assert!(std::mem::size_of::<ShaderUniforms>() == 160);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    pos: [f32; 2],
}

const QUAD_VERTICES: [QuadVertex; 6] = [
    QuadVertex { pos: [-1.0, -1.0] },
    QuadVertex { pos: [1.0, -1.0] },
    QuadVertex { pos: [-1.0, 1.0] },
    QuadVertex { pos: [-1.0, 1.0] },
    QuadVertex { pos: [1.0, -1.0] },
    QuadVertex { pos: [1.0, 1.0] },
];

pub struct ChladniRenderResources {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub bg_pipeline: wgpu::RenderPipeline,
    pub particle_pipeline: wgpu::RenderPipeline,
    pub blit_pipeline: wgpu::RenderPipeline,

    pub uniform_buffer: wgpu::Buffer,
    pub quad_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,

    pub compute_bind_group: wgpu::BindGroup,
    pub uniform_bind_group: wgpu::BindGroup,

    pub tex_bgl: wgpu::BindGroupLayout,
    pub trail_sampler: wgpu::Sampler,
    pub trail_textures: [wgpu::Texture; 2],
    pub trail_views: [wgpu::TextureView; 2],
    pub trail_bind_groups: [wgpu::BindGroup; 2],
    pub trail_size: [u32; 2],
    pub ping_pong_idx: usize,
}

impl ChladniRenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Chladni Compute + Render Shader"),
            source: wgpu::ShaderSource::Wgsl(CHLADNI_WGSL.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chladni Uniform Buffer"),
            contents: bytemuck::bytes_of(&ShaderUniforms::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chladni Particle Quad Buffer"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let initial_particles = ParticleSystem::new(MAX_GPU_PARTICLES);
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chladni GPU Particle Buffer"),
            contents: bytemuck::cast_slice(&initial_particles.particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
        });

        // Group 0 for compute: uniform (0) + read-write storage particle buffer (1)
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Chladni Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Group 0 for render: uniform (0) only (storage read_write is illegal in VERTEX stage)
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Chladni Render Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 1 for fs_bg and fs_blit: previous-frame trail texture + filtering sampler
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Chladni Trail Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chladni Compute Bind Group"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chladni Render Uniform Bind Group"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let trail_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Chladni Trail Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Chladni Compute Pipeline Layout"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        let bg_blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Chladni Trail Pass Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bgl), Some(&tex_bgl)],
            immediate_size: 0,
        });

        let particle_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Chladni Particle Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Chladni Particle Physics Compute Pipeline"),
            layout: Some(&compute_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let trail_format = wgpu::TextureFormat::Rgba16Float;

        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Chladni Trail Decay Pipeline"),
            layout: Some(&bg_blit_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_bg"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bg"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: trail_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Chladni Additive Particle Pipeline"),
            layout: Some(&particle_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_particles"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuParticle>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: 8,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 20,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    }),
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_particles"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: trail_format,
                    blend: Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Chladni Viewport Composite Pipeline"),
            layout: Some(&bg_blit_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_bg"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let init_size = [512, 512];
        let (tex0, view0, bg0) =
            Self::create_trail_target(device, &tex_bgl, &trail_sampler, init_size[0], init_size[1]);
        let (tex1, view1, bg1) =
            Self::create_trail_target(device, &tex_bgl, &trail_sampler, init_size[0], init_size[1]);

        Self {
            compute_pipeline,
            bg_pipeline,
            particle_pipeline,
            blit_pipeline,
            uniform_buffer,
            quad_buffer,
            instance_buffer,
            compute_bind_group,
            uniform_bind_group,
            tex_bgl,
            trail_sampler,
            trail_textures: [tex0, tex1],
            trail_views: [view0, view1],
            trail_bind_groups: [bg0, bg1],
            trail_size: init_size,
            ping_pong_idx: 0,
        }
    }

    fn create_trail_target(
        device: &wgpu::Device,
        tex_bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Chladni Ping-Pong Trail Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chladni Trail Bind Group"),
            layout: tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        (texture, view, bind_group)
    }

    pub fn ensure_trail_textures(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.clamp(1, 8192);
        let h = height.clamp(1, 8192);
        if self.trail_size == [w, h] {
            return;
        }
        let (tex0, view0, bg0) =
            Self::create_trail_target(device, &self.tex_bgl, &self.trail_sampler, w, h);
        let (tex1, view1, bg1) =
            Self::create_trail_target(device, &self.tex_bgl, &self.trail_sampler, w, h);
        self.trail_textures = [tex0, tex1];
        self.trail_views = [view0, view1];
        self.trail_bind_groups = [bg0, bg1];
        self.trail_size = [w, h];
        self.ping_pong_idx = 0;
    }
}

pub struct ChladniPaintCallback {
    pub uniforms: ShaderUniforms,
    pub active_particle_count: usize,
}

impl CallbackTrait for ChladniPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(resources) = callback_resources.get_mut::<ChladniRenderResources>() else {
            return Vec::new();
        };

        let width = (self.uniforms.view_size[0] * screen_descriptor.pixels_per_point)
            .round()
            .clamp(1.0, 8192.0) as u32;
        let height = (self.uniforms.view_size[1] * screen_descriptor.pixels_per_point)
            .round()
            .clamp(1.0, 8192.0) as u32;
        resources.ensure_trail_textures(device, width, height);

        let count = self.active_particle_count.min(MAX_GPU_PARTICLES);
        let mut uniforms = self.uniforms;
        uniforms.active_particles = count as u32;
        queue.write_buffer(&resources.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Chladni Frame Encoder"),
        });

        if count > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Chladni Particle Physics Compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&resources.compute_pipeline);
            cpass.set_bind_group(0, &resources.compute_bind_group, &[]);
            cpass.dispatch_workgroups((count as u32).div_ceil(256), 1, 1);
        }

        let prev_idx = resources.ping_pong_idx;
        let next_idx = 1 - prev_idx;

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chladni Trail Ping-Pong Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.trail_views[next_idx],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&resources.bg_pipeline);
            rpass.set_bind_group(0, &resources.uniform_bind_group, &[]);
            rpass.set_bind_group(1, &resources.trail_bind_groups[prev_idx], &[]);
            rpass.draw(0..3, 0..1);

            if count > 0 {
                rpass.set_pipeline(&resources.particle_pipeline);
                rpass.set_bind_group(0, &resources.uniform_bind_group, &[]);
                rpass.set_vertex_buffer(0, resources.quad_buffer.slice(..));
                rpass.set_vertex_buffer(1, resources.instance_buffer.slice(..));
                rpass.draw(0..6, 0..count as u32);
            }
        }

        resources.ping_pong_idx = next_idx;
        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let Some(resources) = callback_resources.get::<ChladniRenderResources>() else {
            return;
        };

        render_pass.set_pipeline(&resources.blit_pipeline);
        render_pass.set_bind_group(0, &resources.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &resources.trail_bind_groups[resources.ping_pong_idx], &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn block_on<F: Future>(fut: F) -> F::Output {
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        fn noop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            match Pin::as_mut(&mut fut).poll(&mut cx) {
                Poll::Ready(val) => return val,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn wgsl_and_pipelines_validate() {
        assert_eq!(DEFAULT_PARTICLE_CAP, 250_000);
        assert_eq!(MAX_GPU_PARTICLES, 1_000_000);

        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to acquire a wgpu adapter for validation test");
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("Failed to create wgpu device for validation test");

        let gpu_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let gpu_error_cb = Arc::clone(&gpu_error);
        device.on_uncaptured_error(Arc::new(move |err| {
            *gpu_error_cb.lock().unwrap() = Some(err.to_string());
        }));

        let mut resources =
            ChladniRenderResources::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        resources.ensure_trail_textures(&device, 800, 600);

        let mut uniforms = ShaderUniforms::zeroed();
        uniforms.view_size = [800.0, 600.0];
        uniforms.dt = 1.0 / 60.0;
        uniforms.particle_size = 2.0;
        uniforms.active_particles = 1024;
        uniforms.m = 3.0;
        uniforms.n = 5.0;
        uniforms.a = 1.0;
        uniforms.b = 1.0;
        uniforms.trail_persistence = 0.85;
        uniforms.use_screen_borders = 1;
        queue.write_buffer(&resources.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Validation Test Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Validation Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&resources.compute_pipeline);
            cpass.set_bind_group(0, &resources.compute_bind_group, &[]);
            cpass.dispatch_workgroups(4, 1, 1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Validation Trail Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &resources.trail_views[1],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&resources.bg_pipeline);
            rpass.set_bind_group(0, &resources.uniform_bind_group, &[]);
            rpass.set_bind_group(1, &resources.trail_bind_groups[0], &[]);
            rpass.draw(0..3, 0..1);

            rpass.set_pipeline(&resources.particle_pipeline);
            rpass.set_bind_group(0, &resources.uniform_bind_group, &[]);
            rpass.set_vertex_buffer(0, resources.quad_buffer.slice(..));
            rpass.set_vertex_buffer(1, resources.instance_buffer.slice(..));
            rpass.draw(0..6, 0..1024);
        }
        queue.submit(Some(encoder.finish()));

        let err = gpu_error.lock().unwrap().take();
        if let Some(err) = err {
            panic!("wgpu validation error: {err}");
        }
    }
}
