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

fn main() {
    let nx = 256;
    let ny = 256;

    let mut ref_img = vec![10.0f32; (nx * ny) as usize];
    ref_img[100 * nx as usize + 100] = 1000.0;
    write_mrc("ref_delta.mrc", nx, ny, &ref_img);

    let shift_x = 20;
    let shift_y = 15;
    let mut tgt_img = vec![10.0f32; (nx * ny) as usize];
    let tx = 100 + shift_x;
    let ty = 100 + shift_y;
    if tx < nx && ty < ny {
        tgt_img[ty as usize * nx as usize + tx as usize] = 1000.0;
    }
    write_mrc("tgt_delta.mrc", nx, ny, &tgt_img);

    println!("Reference: bright spot at (100, 100)");
    println!("Target:    bright spot at ({}, {})", 100 + shift_x, 100 + shift_y);
    println!("Ground truth shift: target has +{} X, +{} Y relative to reference", shift_x, shift_y);
    println!("Expected peak offset: -{} X, -{} Y (cross-correlation convention)", shift_x, shift_y);
}
