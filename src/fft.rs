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

pub struct CrossCorrelationResult {
    pub xcorr: AlignedVec<f64>,
    pub width: usize,
    pub height: usize,
    pub peak_x: usize,
    pub peak_y: usize,
    pub peak_value: f64,
}

pub struct AlignmentResult {
    pub shift_x: f64,
    pub shift_y: f64,
    pub peak_value: f64,
    pub corr_score: f64,
    pub width: usize,
    pub height: usize,
    pub subpixel_precision: bool,
}

impl std::fmt::Display for AlignmentResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           坐标对齐补偿报告 (Alignment Report)                ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ 参考图尺寸   : {:>4} × {:<4}                                 ║", self.width, self.height)?;
        writeln!(f, "║ 对齐精度     : {}                            ",
            if self.subpixel_precision { "亚像素级 (sub-pixel)" } else { "像素级 (pixel)" }
        )?;
        writeln!(f, "║ 峰值互相关值 : {:>14.6}                                   ║", self.peak_value)?;
        writeln!(f, "║ 相关系数     : {:>14.6}                                   ║", self.corr_score)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ 平移偏移量 (Shift Vector)                                   ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  ΔX (水平)   : {:>+10.4} 像素                               ║", self.shift_x)?;
        writeln!(f, "║  ΔY (垂直)   : {:>+10.4} 像素                               ║", self.shift_y)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ 坐标补偿指令 (Target → Reference):                          ║")?;
        writeln!(f, "║  target_x' = target_x {:+}                                ",
            if self.shift_x >= 0.0 { format!("+ {:.4}", self.shift_x) } else { format!("- {:.4}", -self.shift_x) }
        )?;
        writeln!(f, "║  target_y' = target_y {:+}                                ",
            if self.shift_y >= 0.0 { format!("+ {:.4}", self.shift_y) } else { format!("- {:.4}", -self.shift_y) }
        )?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

pub fn fft_2d_raw(data: &[f32], width: usize, height: usize) -> (AlignedVec<Complex64>, usize) {
    let data_ptr = data.as_ptr() as usize;
    debug_assert!(
        data_ptr % 64 == 0,
        "Input data must be 64B-aligned for AVX-512 SIMD"
    );

    let n = width.max(height);
    let padded_size = n.next_power_of_two();

    let mut planner = FftPlanner::new();
    let row_fft = planner.plan_fft_forward(padded_size);
    let col_fft = planner.plan_fft_forward(padded_size);

    let mut buffer: AlignedVec<Complex64> = AlignedVec::zeros(padded_size * padded_size);
    debug_assert!(buffer.is_aligned());

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
    for col in 0..padded_size {
        for row in 0..padded_size {
            col_buffer[row] = buffer[row * padded_size + col];
        }
        col_fft.process(&mut col_buffer);
        for row in 0..padded_size {
            buffer[row * padded_size + col] = col_buffer[row];
        }
    }

    (buffer, padded_size)
}

pub fn ifft_2d_raw(
    freq_data: &mut AlignedVec<Complex64>,
    padded_size: usize,
) -> AlignedVec<f64> {
    debug_assert!(freq_data.is_aligned());

    let mut planner = FftPlanner::new();
    let row_ifft = planner.plan_fft_inverse(padded_size);
    let col_ifft = planner.plan_fft_inverse(padded_size);

    for row in 0..padded_size {
        let start = row * padded_size;
        row_ifft.process(&mut freq_data[start..start + padded_size]);
    }

    let mut col_buffer: AlignedVec<Complex64> = AlignedVec::zeros(padded_size);
    for col in 0..padded_size {
        for row in 0..padded_size {
            col_buffer[row] = freq_data[row * padded_size + col];
        }
        col_ifft.process(&mut col_buffer);
        for row in 0..padded_size {
            freq_data[row * padded_size + col] = col_buffer[row];
        }
    }

    let scale = (padded_size * padded_size) as f64;
    let mut result: AlignedVec<f64> = AlignedVec::zeros(padded_size * padded_size);
    debug_assert!(result.is_aligned());
    for i in 0..padded_size * padded_size {
        result[i] = freq_data[i].re / scale;
    }

    result
}

pub fn cross_correlate(
    reference: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
) -> CrossCorrelationResult {
    let (ref_fft, padded_size) = fft_2d_raw(reference, width, height);
    let (tgt_fft, _) = fft_2d_raw(target, width, height);

    let mut prod: AlignedVec<Complex64> = AlignedVec::zeros(padded_size * padded_size);
    debug_assert!(prod.is_aligned());
    for i in 0..padded_size * padded_size {
        prod[i] = ref_fft[i] * tgt_fft[i].conj();
    }

    let xcorr = ifft_2d_raw(&mut prod, padded_size);
    let mut xcorr = xcorr;

    fft_shift_2d(&mut xcorr, padded_size, padded_size);

    let mut peak_val = f64::NEG_INFINITY;
    let mut peak_x = 0usize;
    let mut peak_y = 0usize;
    for row in 0..padded_size {
        for col in 0..padded_size {
            let v = xcorr[row * padded_size + col];
            if v > peak_val {
                peak_val = v;
                peak_x = col;
                peak_y = row;
            }
        }
    }

    CrossCorrelationResult {
        xcorr,
        width: padded_size,
        height: padded_size,
        peak_x,
        peak_y,
        peak_value: peak_val,
    }
}

pub fn align_images(
    reference: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
) -> AlignmentResult {
    let n = width * height;
    let mut ref_norm: AlignedVec<f32> = AlignedVec::with_capacity(n);
    let mut tgt_norm: AlignedVec<f32> = AlignedVec::with_capacity(n);
    debug_assert!(ref_norm.is_aligned());
    debug_assert!(tgt_norm.is_aligned());

    let ref_mean: f64 = reference.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let tgt_mean: f64 = target.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

    for i in 0..n {
        ref_norm.push((reference[i] as f64 - ref_mean) as f32);
        tgt_norm.push((target[i] as f64 - tgt_mean) as f32);
    }

    let ref_std: f64 = (ref_norm.iter()
        .map(|&v| v as f64 * v as f64)
        .sum::<f64>() / n as f64)
        .sqrt();
    let tgt_std: f64 = (tgt_norm.iter()
        .map(|&v| v as f64 * v as f64)
        .sum::<f64>() / n as f64)
        .sqrt();

    let cc = cross_correlate(&ref_norm, &tgt_norm, width, height);

    let n_padded = (cc.width * cc.height) as f64;
    let corr_score = if ref_std > 0.0 && tgt_std > 0.0 && n_padded > 0.0 {
        cc.peak_value / n_padded / (ref_std * tgt_std)
    } else {
        0.0
    };

    let cx = cc.width as f64 / 2.0;
    let cy = cc.height as f64 / 2.0;

    let (sub_px, sub_py, subpixel_ok) = if cc.peak_x > 0 && cc.peak_x < cc.width - 1
        && cc.peak_y > 0 && cc.peak_y < cc.height - 1
    {
        let px = cc.peak_x;
        let py = cc.peak_y;
        let w = cc.width;

        let y_left = cc.xcorr[py * w + px - 1];
        let y_mid = cc.xcorr[py * w + px];
        let y_right = cc.xcorr[py * w + px + 1];

        let x_up = cc.xcorr[(py - 1) * w + px];
        let x_down = cc.xcorr[(py + 1) * w + px];

        let denom_x = y_left + y_right - 2.0 * y_mid;
        let dx = if denom_x.abs() > 1e-10 {
            0.5 * (y_left - y_right) / denom_x
        } else {
            0.0
        };

        let denom_y = x_up + x_down - 2.0 * y_mid;
        let dy = if denom_y.abs() > 1e-10 {
            0.5 * (x_up - x_down) / denom_y
        } else {
            0.0
        };

        (px as f64 + dx, py as f64 + dy, true)
    } else {
        (cc.peak_x as f64, cc.peak_y as f64, false)
    };

    let shift_x = sub_px - cx;
    let shift_y = sub_py - cy;

    AlignmentResult {
        shift_x,
        shift_y,
        peak_value: cc.peak_value,
        corr_score: corr_score.abs(),
        width: cc.width,
        height: cc.height,
        subpixel_precision: subpixel_ok,
    }
}
