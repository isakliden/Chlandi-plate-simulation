pub const CHLADNI_WGSL: &str = r#"
struct Uniforms {
    // Block 0
    view_size: vec2<f32>,
    time: f32,
    dt: f32,
    // Block 1
    particle_size: f32,
    active_particles: u32,
    color_mode: u32,
    show_nodal_lines: u32,
    // Block 2
    nodal_intensity: f32,
    geometry: u32,
    solver: u32,
    boundary_mode: u32,
    // Block 3
    m: f32,
    n: f32,
    a: f32,
    b: f32,
    // Block 4
    kick_force: f32,
    friction: f32,
    brownian_jitter: f32,
    drive_energy: f32,
    // Block 5
    bass_m: f32,
    bass_n: f32,
    mid_m: f32,
    mid_n: f32,
    // Block 6
    treble_m: f32,
    treble_n: f32,
    e_bass: f32,
    e_mid: f32,
    // Block 7
    e_treble: f32,
    use_tri_band: u32,
    beat_kick: f32,
    trail_persistence: f32,
    // Block 8
    custom_low: vec3<f32>,
    swirl_strength: f32,
    // Block 9
    custom_high: vec3<f32>,
    use_screen_borders: u32,
};

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    energy: f32,
    life: f32,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

@group(0) @binding(1)
var<storage, read_write> particles: array<Particle>;

@group(1) @binding(0)
var prev_frame_tex: texture_2d<f32>;

@group(1) @binding(1)
var prev_frame_sampler: sampler;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// --- Bessel Functions for Circular Plate Modes ---
fn bessel_j0(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 3.75) {
        let t = x / 3.75;
        let y = t * t;
        return 1.0 + y * (-2.2499997 + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 - y * 0.0039444))));
    } else {
        let y = 3.75 / ax;
        let f0 = 0.79788456 + y * (-0.00000077 + y * (-0.00552740 + y * (0.00009512 + y * (0.00137237 - y * 0.00072805))));
        let theta0 = ax - 0.78539816 + y * (-0.04166397 + y * (-0.00003954 + y * (0.00262573 + y * (-0.00054125 - y * 0.00029333))));
        return (1.0 / sqrt(ax)) * f0 * cos(theta0);
    }
}

fn bessel_j1(x: f32) -> f32 {
    let ax = abs(x);
    var ans: f32 = 0.0;
    if (ax < 3.75) {
        let t = x / 3.75;
        let y = t * t;
        ans = ax * (0.5 + y * (-0.56249985 + y * (0.21093573 + y * (-0.03954289 + y * (0.00443319 - y * 0.00031761)))));
    } else {
        let y = 3.75 / ax;
        let f1 = 0.79788456 + y * (0.00000156 + y * (0.01659667 + y * (0.00017105 + y * (-0.00249511 + y * 0.00113653))));
        let theta1 = ax - 2.35619449 + y * (0.04166397 + y * (-0.00003954 + y * (-0.00262573 + y * (-0.00054125 + y * 0.00029333))));
        ans = (1.0 / sqrt(ax)) * f1 * cos(theta1);
    }
    return select(ans, -ans, x < 0.0);
}

fn bessel_jn(n_in: i32, x: f32) -> f32 {
    let ord = abs(n_in);
    if (ord == 0) {
        return bessel_j0(x);
    } else if (ord == 1) {
        return bessel_j1(x);
    }
    if (abs(x) < 1e-6) {
        return 0.0;
    }
    var j_prev = bessel_j0(x);
    var j_curr = bessel_j1(x);
    for (var k: i32 = 1; k < ord; k = k + 1) {
        let j_next = (2.0 * f32(k) / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }
    return clamp(j_curr, -1.0, 1.0);
}

// --- Chladni Modal Displacement ---
fn to_wave_space(p: vec2<f32>) -> vec2<f32> {
    if (u.use_screen_borders == 0u) {
        return p;
    }
    let aspect = u.view_size.x / max(u.view_size.y, 1.0);
    if (aspect > 1.0) {
        return vec2<f32>(p.x * aspect, p.y);
    } else {
        return vec2<f32>(p.x, p.y / max(aspect, 1e-4));
    }
}

fn eval_mode_w(p: vec2<f32>, m: f32, n: f32, a: f32, b: f32) -> f32 {
    let x = p.x;
    let y = p.y;
    if (u.geometry == 0u) { // Square
        if (u.solver == 0u) {
            return a * cos(n * PI * x) * cos(m * PI * y) - b * cos(m * PI * x) * cos(n * PI * y);
        } else if (u.solver == 1u) {
            return a * sin(n * PI * x) * cos(m * PI * y) + b * cos(m * PI * x) * sin(n * PI * y);
        } else {
            let r = select(min(length(p), 1.0), length(p), u.use_screen_borders != 0u);
            let theta = select(atan2(y, x), 0.0, abs(x) < 1e-6 && abs(y) < 1e-6);
            let j = bessel_jn(i32(round(n)), (m + 1.0) * PI * r);
            return j * cos(n * theta);
        }
    } else if (u.geometry == 1u) { // Circular
        let r = length(p);
        if (u.use_screen_borders == 0u && r > 1.0) {
            return 0.0;
        }
        let theta = select(atan2(y, x), 0.0, abs(x) < 1e-6 && abs(y) < 1e-6);
        if (u.solver == 2u) {
            let ord = clamp(abs(i32(round(n))), 0, 12);
            let k = max(round(m), 1.0) * PI * 1.2;
            let j = bessel_jn(ord, k * r);
            return j * cos(f32(ord) * theta);
        } else if (u.solver == 0u) {
            return a * cos((m + 1.0) * PI * r) * cos(n * theta);
        } else {
            return b * sin((m + 1.0) * PI * r) * sin(n * theta);
        }
    } else { // Hexagonal
        var sum: f32 = 0.0;
        for (var i = 0u; i < 3u; i = i + 1u) {
            let angle = f32(i) * (TAU / 3.0);
            let ca = cos(angle);
            let sa = sin(angle);
            let u_coord = x * ca + y * sa;
            let v_coord = -x * sa + y * ca;
            sum = sum + cos(n * PI * u_coord) * cos(m * PI * v_coord);
        }
        return a * (sum / 3.0);
    }
}

fn chladni_w(p: vec2<f32>) -> f32 {
    let wp = to_wave_space(p);
    if (u.use_tri_band != 0u) {
        let w_bass = eval_mode_w(wp, u.bass_m, u.bass_n, 1.0, 1.0);
        let w_mid = eval_mode_w(wp, u.mid_m, u.mid_n, 1.0, 1.0);
        let w_treble = eval_mode_w(wp, u.treble_m, u.treble_n, 1.0, 1.0);
        return w_bass * u.e_bass + w_mid * u.e_mid + w_treble * u.e_treble;
    }
    return eval_mode_w(wp, u.m, u.n, u.a, u.b);
}

// --- Deterministic Per-Particle Hash RNG ---
fn pcg_hash(val: u32) -> u32 {
    var state = val * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_01(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

fn rand_signed(seed: u32) -> f32 {
    return rand_01(seed) * 2.0 - 1.0;
}

// --- GPU Compute Particle Simulation ---
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.active_particles || idx >= arrayLength(&particles)) {
        return;
    }

    var p = particles[idx];
    var pos = p.position;
    var vel = p.velocity;

    let dt = clamp(u.dt, 0.001, 0.033);
    let eps = 0.002;
    let aspect = u.view_size.x / max(u.view_size.y, 1.0);
    let inv_aspect_x = select(1.0, 1.0 / aspect, u.use_screen_borders != 0u && aspect > 1.0);
    let inv_aspect_y = select(1.0, aspect, u.use_screen_borders != 0u && aspect < 1.0);
    let eps_x = eps * inv_aspect_x;
    let eps_y = eps * inv_aspect_y;

    // 1. Gradient of vibration energy -grad(w^2) = -2 * w * grad(w)
    let w = chladni_w(pos);
    let abs_w = abs(w);
    let dw_dx = (chladni_w(pos + vec2<f32>(eps_x, 0.0)) - chladni_w(pos - vec2<f32>(eps_x, 0.0))) / (2.0 * eps);
    let dw_dy = (chladni_w(pos + vec2<f32>(0.0, eps_y)) - chladni_w(pos - vec2<f32>(0.0, eps_y))) / (2.0 * eps);

    let fx = -2.0 * w * dw_dx;
    let fy = -2.0 * w * dw_dy;

    // Tangential curl / swirl along iso-contours
    let curl_x = -dw_dy * w * u.swirl_strength;
    let curl_y = dw_dx * w * u.swirl_strength;

    let kick_force = u.kick_force * 60.0 * u.drive_energy;
    let friction_factor = pow(1.0 - clamp(u.friction, 0.001, 0.3), dt * 60.0);
    let jitter = u.brownian_jitter * 60.0 * (u.drive_energy + 0.1);

    let frame_seed = idx ^ (u32(u.time * 1000.0) * 1664525u + 1013904223u);
    let jx = rand_signed(frame_seed);
    let jy = rand_signed(frame_seed ^ 0x9e3779b9u);

    // 2. Beat-transient scatter impulse (NOT multiplied by w -> scatters settled w=0 grains!)
    let dist_center = length(pos);
    let radial = pos / max(dist_center, 1e-4);
    let tangent = vec2<f32>(-radial.y, radial.x);
    let bx = rand_signed(frame_seed ^ 0x85ebca6bu);
    let by = rand_signed(frame_seed ^ 0xc2b2ae35u);
    let beat_scatter = (tangent * 0.45 + vec2<f32>(bx, by) * 0.95) * (u.beat_kick * 3.4);

    // 3. Integrate velocity & position
    let ax = (fx + curl_x) * kick_force + jx * jitter * abs_w;
    let ay = (fy + curl_y) * kick_force + jy * jitter * abs_w;

    vel = (vel + vec2<f32>(ax, ay) * dt + beat_scatter * dt) * friction_factor;

    let max_speed = 1.8;
    let speed_sq = dot(vel, vel);
    if (speed_sq > max_speed * max_speed) {
        vel = vel * (max_speed / sqrt(speed_sq));
    }

    pos = pos + vec2<f32>(vel.x * inv_aspect_x, vel.y * inv_aspect_y) * dt;

    // 4. Boundary handling (0=Bounce, 1=Wrap, 2=RespawnRandom)
    if (u.use_screen_borders != 0u) {
        let bound = 0.995;
        if (u.boundary_mode == 0u) {
            if (abs(pos.x) > bound) {
                pos.x = sign(pos.x) * bound;
                vel.x = -vel.x * 0.7;
            }
            if (abs(pos.y) > bound) {
                pos.y = sign(pos.y) * bound;
                vel.y = -vel.y * 0.7;
            }
        } else if (u.boundary_mode == 1u) {
            if (pos.x > bound) { pos.x = -bound; } else if (pos.x < -bound) { pos.x = bound; }
            if (pos.y > bound) { pos.y = -bound; } else if (pos.y < -bound) { pos.y = bound; }
        } else {
            if (abs(pos.x) > bound || abs(pos.y) > bound) {
                pos.x = rand_signed(frame_seed ^ 0x27d4eb2fu) * 0.98;
                pos.y = rand_signed(frame_seed ^ 0x165667b1u) * 0.98;
                vel = vec2<f32>(0.0, 0.0);
            }
        }
    } else if (u.geometry == 0u) { // Square
        let bound = 0.98;
        if (u.boundary_mode == 0u) {
            if (abs(pos.x) > bound) {
                pos.x = sign(pos.x) * bound;
                vel.x = -vel.x * 0.7;
            }
            if (abs(pos.y) > bound) {
                pos.y = sign(pos.y) * bound;
                vel.y = -vel.y * 0.7;
            }
        } else if (u.boundary_mode == 1u) {
            if (pos.x > bound) { pos.x = -bound; } else if (pos.x < -bound) { pos.x = bound; }
            if (pos.y > bound) { pos.y = -bound; } else if (pos.y < -bound) { pos.y = bound; }
        } else {
            if (abs(pos.x) > bound || abs(pos.y) > bound) {
                pos.x = rand_signed(frame_seed ^ 0x27d4eb2fu) * 0.9;
                pos.y = rand_signed(frame_seed ^ 0x165667b1u) * 0.9;
                vel = vec2<f32>(0.0, 0.0);
            }
        }
    } else if (u.geometry == 1u) { // Circular
        let r = length(pos);
        let max_r = 0.98;
        if (r > max_r) {
            if (u.boundary_mode == 0u) {
                let nrm = pos / max(r, 1e-5);
                let d = dot(vel, nrm);
                vel = (vel - 2.0 * d * nrm) * 0.7;
                pos = nrm * max_r;
            } else if (u.boundary_mode == 1u) {
                pos = -pos * 0.95;
            } else {
                let ang = rand_01(frame_seed ^ 0x27d4eb2fu) * TAU;
                let dst = sqrt(rand_01(frame_seed ^ 0x165667b1u)) * 0.85;
                pos = vec2<f32>(cos(ang), sin(ang)) * dst;
                vel = vec2<f32>(0.0, 0.0);
            }
        }
    } else { // Hexagonal
        let hex_bound = 0.96;
        let qx = abs(pos.x);
        let qy = abs(pos.y);
        let is_outside = (qx * 0.866025 + qy * 0.5 > hex_bound) || (qy > hex_bound);
        if (is_outside) {
            if (u.boundary_mode == 0u) {
                pos = pos * 0.95;
                vel = -vel * 0.6;
            } else if (u.boundary_mode == 1u) {
                pos = -pos * 0.9;
            } else {
                pos.x = rand_signed(frame_seed ^ 0x27d4eb2fu) * 0.5;
                pos.y = rand_signed(frame_seed ^ 0x165667b1u) * 0.5;
                vel = vec2<f32>(0.0, 0.0);
            }
        }
    }

    let vel_mag = length(vel);
    p.position = pos;
    p.velocity = vel;
    p.energy = clamp(vel_mag * 5.0 + abs_w * 2.0, 0.0, 1.0);
    p.life = fract(p.life + dt * 0.1);
    particles[idx] = p;
}

// --- Fullscreen Vertex Shader (Background Fade & Blit) ---
struct BgVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tex_uv: vec2<f32>,
};

@vertex
fn vs_bg(@builtin(vertex_index) in_vertex_index: u32) -> BgVertexOutput {
    var out: BgVertexOutput;
    let x = f32((in_vertex_index << 1u) & 2u);
    let y = f32(in_vertex_index & 2u);
    let ndc = vec2<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0);
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = ndc;
    out.tex_uv = vec2<f32>(x, 1.0 - y);
    return out;
}

// Offscreen Ping-Pong Persistence Decay Pass
@fragment
fn fs_bg(in: BgVertexOutput) -> @location(0) vec4<f32> {
    let prev = textureSample(prev_frame_tex, prev_frame_sampler, in.tex_uv).rgb;
    let decayed = prev * clamp(u.trail_persistence, 0.0, 0.99);
    return vec4<f32>(decayed, 1.0);
}

fn compute_plate_bg(ndc_uv: vec2<f32>) -> vec3<f32> {
    if (u.show_nodal_lines == 0u) {
        return vec3<f32>(0.015, 0.018, 0.026);
    }

    var plate_p = ndc_uv;
    if (u.use_screen_borders == 0u) {
        let aspect = u.view_size.x / max(u.view_size.y, 1.0);
        var p = ndc_uv;
        if (aspect > 1.0) {
            p.x = p.x * aspect;
        } else {
            p.y = p.y / aspect;
        }

        let scale = 1.15;
        plate_p = p * scale;

        var inside = false;
        if (u.geometry == 0u) {
            inside = abs(plate_p.x) <= 1.0 && abs(plate_p.y) <= 1.0;
        } else if (u.geometry == 1u) {
            inside = length(plate_p) <= 1.0;
        } else {
            let q_x = abs(plate_p.x);
            let q_y = abs(plate_p.y);
            inside = (q_x * 0.866025 + q_y * 0.5 <= 1.0) && (q_y <= 1.0);
        }

        if (!inside) {
            return vec3<f32>(0.012, 0.014, 0.020);
        }
    }

    let w = chladni_w(plate_p);
    let abs_w = abs(w);
    let line_width = 0.04;
    let line_glow = exp(-abs_w * abs_w / (line_width * line_width));

    let base_bg = vec3<f32>(0.022, 0.026, 0.036);
    let glow_color = vec3<f32>(0.10, 0.34, 0.62) * u.nodal_intensity;

    return mix(base_bg, base_bg + glow_color, line_glow);
}

// Viewport Composite / Blit Fragment Shader
@fragment
fn fs_blit(in: BgVertexOutput) -> @location(0) vec4<f32> {
    let base_bg = compute_plate_bg(in.uv);
    let trail_rgb = textureSample(prev_frame_tex, prev_frame_sampler, in.tex_uv).rgb;
    // Soft highlight rolloff preserves additive glow saturation without flat clipping
    let bloom = trail_rgb / (vec3<f32>(1.0) + trail_rgb * 0.28);
    return vec4<f32>(base_bg + bloom, 1.0);
}

// --- Instanced Particle Pass (Rendered with Additive Blending into Ping-Pong Target) ---
struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    @location(1) particle_pos: vec2<f32>,
    @location(2) particle_vel: vec2<f32>,
    @location(3) particle_energy: f32,
    @location(4) particle_life: f32,
};

struct ParticleVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) quad_coord: vec2<f32>,
    @location(1) energy: f32,
    @location(2) speed: f32,
    @location(3) life: f32,
};

@vertex
fn vs_particles(vertex: VertexInput, instance: InstanceInput) -> ParticleVertexOutput {
    var out: ParticleVertexOutput;

    let aspect = u.view_size.x / max(u.view_size.y, 1.0);
    var plate_pos = instance.particle_pos;
    if (u.use_screen_borders == 0u) {
        let scale = 1.0 / 1.15;
        plate_pos = instance.particle_pos * scale;
        if (aspect > 1.0) {
            plate_pos.x = plate_pos.x / aspect;
        } else {
            plate_pos.y = plate_pos.y * aspect;
        }
    }

    let p_radius = (u.particle_size / max(u.view_size.y, 1.0)) * 2.0;
    var screen_offset = vertex.quad_pos * p_radius;
    screen_offset.x = screen_offset.x / aspect;

    out.position = vec4<f32>(plate_pos + screen_offset, 0.0, 1.0);
    out.quad_coord = vertex.quad_pos;
    out.energy = instance.particle_energy;
    out.speed = length(instance.particle_vel);
    out.life = instance.particle_life;

    return out;
}

@fragment
fn fs_particles(in: ParticleVertexOutput) -> @location(0) vec4<f32> {
    let dist_sq = dot(in.quad_coord, in.quad_coord);
    if (dist_sq > 1.0) {
        discard;
    }

    let alpha = (1.0 - smoothstep(0.25, 1.0, dist_sq)) * 0.24;
    var color = vec3<f32>(1.0);

    // Color Mapping Modes
    if (u.color_mode == 0u) { // Velocity Heatmap
        let t = clamp(in.speed * 4.0 + in.energy * 0.4, 0.0, 1.0);
        let c0 = vec3<f32>(0.1, 0.15, 0.85); // deep indigo
        let c1 = vec3<f32>(0.05, 0.85, 0.95); // cyan
        let c2 = vec3<f32>(0.95, 0.9, 0.1);  // neon yellow
        let c3 = vec3<f32>(1.0, 0.2, 0.05);  // fiery orange-red

        if (t < 0.33) {
            color = mix(c0, c1, t / 0.33);
        } else if (t < 0.66) {
            color = mix(c1, c2, (t - 0.33) / 0.33);
        } else {
            color = mix(c2, c3, (t - 0.66) / 0.34);
        }
    } else if (u.color_mode == 1u) { // Nodal Line Glow
        let nodal_proximity = 1.0 - clamp(in.energy * 1.5, 0.0, 1.0);
        let base_cyan = vec3<f32>(0.1, 0.85, 1.0);
        let excited_magenta = vec3<f32>(1.0, 0.15, 0.7);
        color = mix(excited_magenta, base_cyan, nodal_proximity * nodal_proximity);
    } else if (u.color_mode == 2u) { // Metallic Sand
        let spec = pow(1.0 - dist_sq, 3.0);
        let bronze = vec3<f32>(0.72, 0.52, 0.28);
        let gold = vec3<f32>(0.98, 0.82, 0.42);
        let highlight = vec3<f32>(1.0, 0.98, 0.85);
        let base_metal = mix(bronze, gold, clamp(in.energy * 2.0, 0.0, 1.0));
        color = base_metal + highlight * spec * 0.6;
    } else if (u.color_mode == 3u) { // Cyber Phosphor
        let phosphor_green = vec3<f32>(0.2, 1.0, 0.4);
        let phosphor_bright = vec3<f32>(0.8, 1.0, 0.9);
        let glow = clamp(in.speed * 3.0 + in.energy * 0.5, 0.0, 1.0);
        color = mix(phosphor_green, phosphor_bright, glow);
    } else { // Custom Gradient
        let t = clamp(in.speed * 2.5 + in.energy * 0.5, 0.0, 1.0);
        color = mix(u.custom_low, u.custom_high, t);
    }

    return vec4<f32>(color * alpha, alpha);
}
"#;
