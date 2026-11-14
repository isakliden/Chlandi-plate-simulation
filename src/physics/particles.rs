use bytemuck::{Pod, Zeroable};
use rand::{rngs::SmallRng, Rng, SeedableRng};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub energy: f32,
    pub life: f32,
}

pub struct ParticleSystem {
    pub particles: Vec<GpuParticle>,
    #[allow(dead_code)]
    pub max_particles: usize,
}

impl ParticleSystem {
    /// Generates initial seeded particle buffer once for upload to GPU storage/vertex buffer.
    /// All ongoing physics simulation runs entirely on the GPU via WGSL compute shaders.
    pub fn new(max_particles: usize) -> Self {
        let mut particles = Vec::with_capacity(max_particles);
        let mut rng = SmallRng::seed_from_u64(0x00C0_FFEE_1AD1_9173);

        for _ in 0..max_particles {
            let px = rng.gen_range(-0.95..0.95);
            let py = rng.gen_range(-0.95..0.95);
            particles.push(GpuParticle {
                position: [px, py],
                velocity: [0.0, 0.0],
                energy: 0.0,
                life: rng.gen_range(0.0..1.0),
            });
        }

        Self {
            particles,
            max_particles,
        }
    }
}
