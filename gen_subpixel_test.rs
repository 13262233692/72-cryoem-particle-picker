use std::fs::File;
use std::io::Write;

fn write_mrc(path: &str, nx: i32, ny: i32, data: &[f32]) {
    let mut header = vec![0u8; 1024];
    header[0..4].copy_from_slice(&nx.to_le_bytes());
    header[4..8].copy_from_slice(&ny.to_le_bytes());
    header[8..12].copy_from_slice(&1i32.to_le_bytes());
    header[12..16].copy_from_slice(&2i32.to_le_bytes());

    header[28..32].copy_from_slice(&nx.to_le_bytes());
    header[32..36].copy_from_slice(&ny.to_le_bytes());
    header[36..40].copy_from_slice(&1i32.to_le_bytes());

    let cell = 1.0f32;
    header[40..44].copy_from_slice(&cell.to_le_bytes());
    header[44..48].copy_from_slice(&cell.to_le_bytes());
    header[48..52].copy_from_slice(&cell.to_le_bytes());

    let angle = 90.0f32;
    header[76..80].copy_from_slice(&angle.to_le_bytes());
    header[80..84].copy_from_slice(&angle.to_le_bytes());
    header[84..88].copy_from_slice(&angle.to_le_bytes());

    header[88..92].copy_from_slice(&1i32.to_le_bytes());
    header[92..96].copy_from_slice(&2i32.to_le_bytes());
    header[96..100].copy_from_slice(&3i32.to_le_bytes());

    let mut dmin = f32::MAX;
    let mut dmax = f32::MIN;
    let mut dsum = 0.0f64;
    for &v in data {
        dmin = dmin.min(v);
        dmax = dmax.max(v);
        dsum += v as f64;
    }
    let dmean = (dsum / data.len() as f64) as f32;

    header[100..104].copy_from_slice(&dmin.to_le_bytes());
    header[104..108].copy_from_slice(&dmax.to_le_bytes());
    header[108..112].copy_from_slice(&dmean.to_le_bytes());

    let mut file = File::create(path).unwrap();
    file.write_all(&header).unwrap();
    for &val in data {
        file.write_all(&val.to_le_bytes()).unwrap();
    }
}

fn make_gaussian_image(nx: i32, ny: i32, cx: f32, cy: f32, sigma: f32, amplitude: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity((nx * ny) as usize);
    for y in 0..ny {
        for x in 0..nx {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r2 = dx * dx + dy * dy;
            let g = amplitude * (-r2 / (2.0 * sigma * sigma)).exp();
            data.push(100.0 + g);
        }
    }
    data
}

fn main() {
    let nx = 256;
    let ny = 256;
    let sigma = 12.0f32;
    let amp = 200.0f32;

    let ref_cx = 128.0;
    let ref_cy = 128.0;
    let ref_img = make_gaussian_image(nx, ny, ref_cx, ref_cy, sigma, amp);
    write_mrc("ref_subpixel.mrc", nx, ny, &ref_img);
    println!("Created ref_subpixel.mrc: particle at ({}, {})", ref_cx, ref_cy);

    let shift_x = 3.7f32;
    let shift_y = -2.3f32;
    let tgt_cx = ref_cx + shift_x;
    let tgt_cy = ref_cy + shift_y;
    let tgt_img = make_gaussian_image(nx, ny, tgt_cx, tgt_cy, sigma, amp);
    write_mrc("tgt_subpixel.mrc", nx, ny, &tgt_img);
    println!("Created tgt_subpixel.mrc: particle at ({:.2}, {:.2})", tgt_cx, tgt_cy);
    println!("Ground truth shift: ΔX = {:+.2}, ΔY = {:+.2}", shift_x, shift_y);
    println!("Expected alignment shift: ΔX = {:+.2}, ΔY = {:+.2}", -shift_x, -shift_y);
    println!("(Using larger sigma={} for better sub-pixel accuracy)", sigma);
}
