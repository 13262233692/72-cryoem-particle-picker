use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use crate::aligned::AlignedVec;

pub const MRC_HEADER_SIZE: usize = 1024;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MrcHeader {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub mode: i32,
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,
    pub mx: i32,
    pub my: i32,
    pub mz: i32,
    pub cell_a: [f32; 3],
    pub cell_b: [f32; 3],
    pub cell_c: [f32; 3],
    pub cell_alpha: f32,
    pub cell_beta: f32,
    pub cell_gamma: f32,
    pub mapc: i32,
    pub mapr: i32,
    pub maps: i32,
    pub dmin: f32,
    pub dmax: f32,
    pub dmean: f32,
    pub ispg: i32,
    pub nsymbt: i32,
    pub ext_type: [u8; 4],
    pub nversion: i32,
}

#[derive(Debug, Clone)]
pub enum MrcDataType {
    Int8,
    Int16,
    Float32,
    Complex32,
    Complex64,
    UInt16,
}

impl MrcDataType {
    pub fn from_mode(mode: i32) -> Option<Self> {
        match mode {
            0 => Some(MrcDataType::Int8),
            1 => Some(MrcDataType::Int16),
            2 => Some(MrcDataType::Float32),
            3 => Some(MrcDataType::Complex32),
            4 => Some(MrcDataType::Complex64),
            6 => Some(MrcDataType::UInt16),
            _ => None,
        }
    }

    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            MrcDataType::Int8 => 1,
            MrcDataType::Int16 => 2,
            MrcDataType::Float32 => 4,
            MrcDataType::Complex32 => 8,
            MrcDataType::Complex64 => 16,
            MrcDataType::UInt16 => 2,
        }
    }
}

#[derive(Debug)]
pub struct MrcFile {
    pub header: MrcHeader,
    pub data_type: MrcDataType,
    mmap: Mmap,
}

impl MrcFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("无法打开MRC文件: {}", e))?;
        let metadata = file.metadata().map_err(|e| format!("无法读取文件元数据: {}", e))?;

        if metadata.len() < MRC_HEADER_SIZE as u64 {
            return Err("文件太小，无法包含有效的MRC头".to_string());
        }

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| format!("内存映射失败: {}", e))?
        };

        let header = Self::parse_header(&mmap)?;

        let data_type = MrcDataType::from_mode(header.mode)
            .ok_or_else(|| format!("不支持的MRC数据模式: {}", header.mode))?;

        let expected_data_size = header.nx as u64 * header.ny as u64 * header.nz as u64
            * data_type.bytes_per_pixel() as u64;
        let actual_data_size = metadata.len() - MRC_HEADER_SIZE as u64 - header.nsymbt as u64;

        if actual_data_size < expected_data_size {
            return Err(format!(
                "数据大小不匹配: 期望 {} 字节, 实际 {} 字节",
                expected_data_size, actual_data_size
            ));
        }

        Ok(MrcFile {
            header,
            data_type,
            mmap,
        })
    }

    fn parse_header(data: &[u8]) -> Result<MrcHeader, String> {
        let read_i32 = |offset: usize| -> i32 {
            i32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
        };
        let read_f32 = |offset: usize| -> f32 {
            f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
        };

        let nx = read_i32(0);
        let ny = read_i32(4);
        let nz = read_i32(8);

        if nx <= 0 || ny <= 0 || nz <= 0 {
            return Err(format!("无效的维度: nx={}, ny={}, nz={}", nx, ny, nz));
        }

        let mode = read_i32(12);
        let nxstart = read_i32(16);
        let nystart = read_i32(20);
        let nzstart = read_i32(24);
        let mx = read_i32(28);
        let my = read_i32(32);
        let mz = read_i32(36);

        let cell_a = [read_f32(40), read_f32(44), read_f32(48)];
        let cell_b = [read_f32(52), read_f32(56), read_f32(60)];
        let cell_c = [read_f32(64), read_f32(68), read_f32(72)];
        let cell_alpha = read_f32(76);
        let cell_beta = read_f32(80);
        let cell_gamma = read_f32(84);

        let mapc = read_i32(88);
        let mapr = read_i32(92);
        let maps = read_i32(96);

        let dmin = read_f32(100);
        let dmax = read_f32(104);
        let dmean = read_f32(108);

        let ispg = read_i32(112);
        let nsymbt = read_i32(116);

        let mut ext_type = [0u8; 4];
        ext_type.copy_from_slice(&data[120..124]);

        let nversion = read_i32(128);

        Ok(MrcHeader {
            nx, ny, nz, mode,
            nxstart, nystart, nzstart,
            mx, my, mz,
            cell_a, cell_b, cell_c,
            cell_alpha, cell_beta, cell_gamma,
            mapc, mapr, maps,
            dmin, dmax, dmean,
            ispg, nsymbt,
            ext_type, nversion,
        })
    }

    pub fn extract_frame(&self, section: usize) -> Result<AlignedVec<f32>, String> {
        if section >= self.header.nz as usize {
            return Err(format!("切片索引越界: {} >= {}", section, self.header.nz));
        }

        let nx = self.header.nx as usize;
        let ny = self.header.ny as usize;
        let bpp = self.data_type.bytes_per_pixel();
        let section_size = nx * ny * bpp;
        let data_offset = MRC_HEADER_SIZE + self.header.nsymbt as usize;
        let section_start = data_offset + section * section_size;

        let mut pixels: AlignedVec<f32> = AlignedVec::with_capacity(nx * ny);
        debug_assert!(pixels.is_aligned(), "AlignedVec allocation failed to produce 64B-aligned pointer");

        for row in 0..ny {
            for col in 0..nx {
                let pixel_offset = section_start + (row * nx + col) * bpp;
                let value = match self.data_type {
                    MrcDataType::Int8 => {
                        self.mmap[pixel_offset] as f32
                    }
                    MrcDataType::Int16 => {
                        i16::from_le_bytes(
                            self.mmap[pixel_offset..pixel_offset + 2].try_into().unwrap()
                        ) as f32
                    }
                    MrcDataType::Float32 => {
                        f32::from_le_bytes(
                            self.mmap[pixel_offset..pixel_offset + 4].try_into().unwrap()
                        )
                    }
                    MrcDataType::UInt16 => {
                        u16::from_le_bytes(
                            self.mmap[pixel_offset..pixel_offset + 2].try_into().unwrap()
                        ) as f32
                    }
                    _ => return Err("复数类型暂不支持直接提取为灰度矩阵".to_string()),
                };
                pixels.push(value);
            }
        }

        debug_assert!(pixels.is_aligned());
        Ok(pixels)
    }

    pub fn extract_region(
        &self,
        section: usize,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
    ) -> Result<AlignedVec<f32>, String> {
        let nx = self.header.nx as usize;
        let ny = self.header.ny as usize;

        if x0 + width > nx || y0 + height > ny {
            return Err(format!(
                "区域越界: ({},{})->{}x{} 超出图像尺寸 {}x{}",
                x0, y0, width, height, nx, ny
            ));
        }

        let frame = self.extract_frame(section)?;
        let mut region: AlignedVec<f32> = AlignedVec::with_capacity(width * height);
        debug_assert!(region.is_aligned());

        for row in y0..y0 + height {
            for col in x0..x0 + width {
                region.push(frame[row * nx + col]);
            }
        }

        debug_assert!(region.is_aligned());
        Ok(region)
    }
}

pub fn filter_dead_pixels(data: &mut AlignedVec<f32>, sigma_threshold: f32) -> usize {
    debug_assert!(data.is_aligned(), "filter_dead_pixels requires 64B-aligned input");
    let n = data.len();
    if n == 0 {
        return 0;
    }

    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let variance: f64 = data.iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let lower = mean - sigma_threshold as f64 * std_dev;
    let upper = mean + sigma_threshold as f64 * std_dev;

    let mut filtered_count = 0;
    for val in data.iter_mut() {
        if (*val as f64) < lower || (*val as f64) > upper {
            *val = mean as f32;
            filtered_count += 1;
        }
    }

    filtered_count
}
