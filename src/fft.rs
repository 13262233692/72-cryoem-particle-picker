use rustfft::FftPlanner;
pub use num_complex::Complex64;
use crate::aligned::AlignedVec;

pub struct FftResult {
    pub spectrum: AlignedVec<f64>,
    pub width: usize,
    pub height: usize,
}

impl FftResult {
    pub fn feature_summary(&self, num_rings: usize) -> FeatureSummary {
        debug_assert!(self.spectrum.is_aligned(), "spectrum must be 64B-aligned");
        let cx = self.width as f64 / 2.0;
        let cy = self.height as f64 / 2.0;
        let max_radius = cx.min(cy);
        let ring_width = max_radius / num_rings as f64;

        let mut ring_sums = vec![0.0f64; num_rings];
        let mut ring_counts = vec![0usize; num_rings];

        for row in 0..self.height {
            for col in 0..self.width {
                let dx = col as f64 - cx;
                let dy = row as f64 - cy;
                let radius = (dx * dx + dy * dy).sqrt();
                let ring_idx = (radius / ring_width).floor() as usize;

                if ring_idx < num_rings {
                    ring_sums[ring_idx] += self.spectrum[row * self.width + col];
                    ring_counts[ring_idx] += 1;
                }
            }
        }

        let radial_profile: Vec<f64> = ring_sums
            .iter()
            .zip(ring_counts.iter())
            .map(|(&sum, &count)| {
                if count > 0 { sum / count as f64 } else { 0.0 }
            })
            .collect();

        let total_energy: f64 = self.spectrum.iter().sum();
        let dc_component = self.spectrum[0];
        let ac_energy = total_energy - dc_component;

        let peak_freq_idx = radial_profile[1..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx + 1)
            .unwrap_or(0);

        FeatureSummary {
            radial_profile,
            total_energy,
            dc_component,
            ac_energy,
            peak_freq_ring: peak_freq_idx,
            width: self.width,
            height: self.height,
        }
    }
}

pub struct FeatureSummary {
    pub radial_profile: Vec<f64>,
    pub total_energy: f64,
    pub dc_component: f64,
    pub ac_energy: f64,
    pub peak_freq_ring: usize,
    pub width: usize,
    pub height: usize,
}

impl std::fmt::Display for FeatureSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║          频域特征向量摘要 (FFT Amplitude Spectrum)          ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ 图像尺寸     : {:>4} x {:<4}                                 ║", self.width, self.height)?;
        writeln!(f, "║ 总能量       : {:>14.4}                                   ║", self.total_energy)?;
        writeln!(f, "║ DC分量       : {:>14.4}                                   ║", self.dc_component)?;
        writeln!(f, "║ AC能量       : {:>14.4}                                   ║", self.ac_energy)?;
        writeln!(f, "║ 峰值频率环   : Ring #{:<4}                                    ║", self.peak_freq_ring)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  径向平均功率谱 (Radial Power Profile)                      ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;

        let max_val = self.radial_profile.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bar_width = 36;

        for (i, &val) in self.radial_profile.iter().enumerate() {
            let normalized = if max_val > 0.0 { val / max_val } else { 0.0 };
            let filled = (normalized * bar_width as f64).round() as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            writeln!(f, "║  Ring {:>2}: {} {:>10.4}  ║", i, bar, val)?;
        }

        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

pub fn fft_2d(data: &[f32], width: usize, height: usize) -> FftResult {
    let data_ptr = data.as_ptr() as usize;
    debug_assert!(
        data_ptr % 64 == 0,
        "Input data must be 64B-aligned for AVX-512 SIMD, got ptr=0x{:x} (align={})",
        data_ptr,
        data_ptr & 63
    );

    let n = width.max(height);
    let padded_size = n.next_power_of_two();

    let mut planner = FftPlanner::new();
    let row_fft = planner.plan_fft_forward(padded_size);
    let col_fft = planner.plan_fft_forward(padded_size);

    let mut buffer: AlignedVec<Complex64> = AlignedVec::zeros(padded_size * padded_size);
    debug_assert!(buffer.is_aligned(), "FFT Complex64 buffer must be 64B-aligned");

    for row in 0..height {
        for col in 0..width {
            let val = data[row * width + col] as f64;
            buffer[row * padded_size + col] = Complex64::new(val, 0.0);
        }
    }

    for row in 0..padded_size {
        let start = row * padded_size;
        row_fft.process(&mut buffer[start..start + padded_size]);
    }

    let mut col_buffer: AlignedVec<Complex64> = AlignedVec::zeros(padded_size);
    debug_assert!(col_buffer.is_aligned());
    for col in 0..padded_size {
        for row in 0..padded_size {
            col_buffer[row] = buffer[row * padded_size + col];
        }
        col_fft.process(&mut col_buffer);
        for row in 0..padded_size {
            buffer[row * padded_size + col] = col_buffer[row];
        }
    }

    let mut spectrum: AlignedVec<f64> = AlignedVec::zeros(padded_size * padded_size);
    debug_assert!(spectrum.is_aligned(), "output spectrum must be 64B-aligned");
    for (i, c) in buffer.iter().enumerate() {
        spectrum[i] = c.norm();
    }

    fft_shift_2d(&mut spectrum, padded_size, padded_size);

    FftResult {
        spectrum,
        width: padded_size,
        height: padded_size,
    }
}

fn fft_shift_2d(data: &mut AlignedVec<f64>, width: usize, height: usize) {
    debug_assert!(data.is_aligned(), "fft_shift input must be 64B-aligned");
    let half_w = width / 2;
    let half_h = height / 2;

    for row in 0..half_h {
        for col in 0..half_w {
            let a = row * width + col;
            let b = (row + half_h) * width + (col + half_w);
            data.swap(a, b);
        }
    }

    for row in 0..half_h {
        for col in half_w..width {
            let a = row * width + col;
            let b = (row + half_h) * width + (col - half_w);
            data.swap(a, b);
        }
    }
}
