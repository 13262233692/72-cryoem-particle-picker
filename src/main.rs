mod mrc;
mod fft;

use clap::{Parser, Subcommand};
use mrc::{MrcFile, filter_dead_pixels};

#[derive(Parser)]
#[command(name = "cryoem-picker")]
#[command(about = "冷冻电镜单颗粒图像分析终端算力工具")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "解析MRC文件并显示头信息与统计")]
    Info {
        #[arg(help = "MRC文件路径")]
        file: String,
    },
    #[command(about = "提取单颗粒区域并进行频域特征分析")]
    Extract {
        #[arg(help = "MRC文件路径")]
        file: String,
        #[arg(long, default_value_t = 0, help = "切片索引(0-based)")]
        section: usize,
        #[arg(long, default_value_t = 0, help = "区域左上角X坐标")]
        x: usize,
        #[arg(long, default_value_t = 0, help = "区域左上角Y坐标")]
        y: usize,
        #[arg(long, default_value_t = 256, help = "提取区域宽度")]
        width: usize,
        #[arg(long, default_value_t = 256, help = "提取区域高度")]
        height: usize,
        #[arg(long, default_value_t = 5.0, help = "死像素过滤sigma阈值")]
        sigma: f32,
        #[arg(long, default_value_t = 16, help = "径向功率谱环数")]
        rings: usize,
    },
    #[command(about = "全图网格扫描: 对整张显微图进行网格切割并批量FFT")]
    Scan {
        #[arg(help = "MRC文件路径")]
        file: String,
        #[arg(long, default_value_t = 0, help = "切片索引(0-based)")]
        section: usize,
        #[arg(long, default_value_t = 128, help = "网格窗口尺寸")]
        window: usize,
        #[arg(long, default_value_t = 64, help = "滑动步长")]
        stride: usize,
        #[arg(long, default_value_t = 5.0, help = "死像素过滤sigma阈值")]
        sigma: f32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { file } => cmd_info(&file),
        Commands::Extract {
            file, section, x, y, width, height, sigma, rings,
        } => cmd_extract(&file, section, x, y, width, height, sigma, rings),
        Commands::Scan {
            file, section, window, stride, sigma,
        } => cmd_scan(&file, section, window, stride, sigma),
    }
}

fn cmd_info(path: &str) {
    eprintln!("► 正在解析MRC文件: {}", path);
    let mrc = match MrcFile::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("✘ 错误: {}", e);
            std::process::exit(1);
        }
    };

    let h = &mrc.header;
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              MRC 文件头信息                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ 维度 (NX×NY×NZ) : {:>5} × {:<5} × {:<5}            ║", h.nx, h.ny, h.nz);
    println!("║ 数据模式         : {} ({})                          ",
        h.mode,
        match h.mode {
            0 => "int8",
            1 => "int16",
            2 => "float32",
            3 => "complex32",
            4 => "complex64",
            6 => "uint16",
            _ => "unknown",
        }
    );
    println!("║ 子区域起始       : ({}, {}, {})                        ║", h.nxstart, h.nystart, h.nzstart);
    println!("║ 网格尺寸 (MX×MY×MZ): {} × {} × {}                    ║", h.mx, h.my, h.mz);
    println!("║ 单元参数 a/b/c   : {:.2} / {:.2} / {:.2} Å              ║", h.cell_a[0], h.cell_a[1], h.cell_a[2]);
    println!("║ 单元角度 α/β/γ   : {:.2} / {:.2} / {:.2}°              ║", h.cell_alpha, h.cell_beta, h.cell_gamma);
    println!("║ 轴映射 C/R/S     : {} / {} / {}                        ║", h.mapc, h.mapr, h.maps);
    println!("║ 密度 min/max/mean : {:.4} / {:.4} / {:.4}            ║", h.dmin, h.dmax, h.dmean);
    println!("║ 空间群           : {}                                    ║", h.ispg);
    println!("║ 扩展头字节数     : {}                                    ║", h.nsymbt);
    println!("║ MRC版本          : {}                                    ║", h.nversion);

    let data_size_bytes = h.nx as u64 * h.ny as u64 * h.nz as u64 * mrc.data_type.bytes_per_pixel() as u64;
    println!("║ 数据体大小       : {} bytes ({:.2} MB)                  ║", data_size_bytes, data_size_bytes as f64 / 1e6);
    println!("╚══════════════════════════════════════════════════════════════╝");

    if h.ny > 0 && h.nz > 0 {
        eprintln!("► 正在提取第0切片进行像素统计...");
        match mrc.extract_frame(0) {
            Ok(frame) => {
                let n = frame.len() as f64;
                let mean: f64 = frame.iter().map(|&v| v as f64).sum::<f64>() / n;
                let variance: f64 = frame.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
                let std_dev = variance.sqrt();
                let snr = if std_dev > 0.0 { mean / std_dev } else { 0.0 };

                println!();
                println!("► 切片#0 像素统计:");
                println!("  均值     = {:.6}", mean);
                println!("  标准差   = {:.6}", std_dev);
                println!("  最小值   = {:.6}", frame.iter().cloned().fold(f32::INFINITY, f32::min));
                println!("  最大值   = {:.6}", frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
                println!("  估计SNR  = {:.6} {}", snr,
                    if snr < 0.1 { "⚠ 极低信噪比" } else if snr < 1.0 { "⚠ 低信噪比" } else { "" }
                );
            }
            Err(e) => eprintln!("✘ 提取切片失败: {}", e),
        }
    }
}

fn cmd_extract(
    path: &str,
    section: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    sigma: f32,
    rings: usize,
) {
    eprintln!("► 正在解析MRC文件: {}", path);
    let mrc = match MrcFile::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("✘ 错误: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!("► 提取区域 ({},{}) {}x{} 切片#{}", x, y, width, height, section);
    let mut region = match mrc.extract_region(section, x, y, width, height) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("✘ 区域提取失败: {}", e);
            std::process::exit(1);
        }
    };

    let n = region.len() as f64;
    let pre_mean: f64 = region.iter().map(|&v| v as f64).sum::<f64>() / n;
    let pre_std: f64 = {
        let v: f64 = region.iter().map(|&v| (v as f64 - pre_mean).powi(2)).sum::<f64>() / n;
        v.sqrt()
    };

    let filtered = filter_dead_pixels(&mut region, sigma);
    eprintln!("► 死像素过滤: sigma阈值={:.1}, 过滤了 {} 个极端像素 ({:.2}%)",
        sigma, filtered, filtered as f64 / n * 100.0);

    let post_mean: f64 = region.iter().map(|&v| v as f64).sum::<f64>() / n;
    let post_std: f64 = {
        let v: f64 = region.iter().map(|&v| (v as f64 - post_mean).powi(2)).sum::<f64>() / n;
        v.sqrt()
    };

    eprintln!("  过滤前: mean={:.4}, std={:.4}", pre_mean, pre_std);
    eprintln!("  过滤后: mean={:.4}, std={:.4}", post_mean, post_std);

    eprintln!("► 执行2D FFT (零填充至2的幂)...");
    let result = fft::fft_2d(&region, width, height);

    eprintln!("► 计算振幅谱特征...");
    let summary = result.feature_summary(rings);
    println!();
    print!("{}", summary);
}

fn cmd_scan(path: &str, section: usize, window: usize, stride: usize, sigma: f32) {
    eprintln!("► 正在解析MRC文件: {}", path);
    let mrc = match MrcFile::open(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("✘ 错误: {}", e);
            std::process::exit(1);
        }
    };

    let nx = mrc.header.nx as usize;
    let ny = mrc.header.ny as usize;

    eprintln!("► 提取切片#{} 全图数据 ({}x{})...", section, nx, ny);
    let frame = match mrc.extract_frame(section) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✘ 提取切片失败: {}", e);
            std::process::exit(1);
        }
    };

    let cols = (nx - window) / stride + 1;
    let rows = (ny - window) / stride + 1;
    let total = cols * rows;

    eprintln!("► 网格扫描: window={}x{}, stride={}, 网格={}x{}={}",
        window, window, stride, cols, rows, total);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        批量频域特征扫描 (Grid Scan)                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ {:>4} {:>4} │ {:>12} │ {:>12} │ {:>12} │ {:>8} ║",
        "Col", "Row", "DC分量", "AC能量", "总能量", "峰值环");
    println!("╠══════════════════════════════════════════════════════════════╣");

    let mut max_ac = f64::NEG_INFINITY;
    let mut best_pos = (0usize, 0usize);

    for grid_row in 0..rows {
        for grid_col in 0..cols {
            let x0 = grid_col * stride;
            let y0 = grid_row * stride;

            let mut patch = Vec::with_capacity(window * window);
            for py in y0..y0 + window {
                for px in x0..x0 + window {
                    patch.push(frame[py * nx + px]);
                }
            }

            let _filtered = filter_dead_pixels(&mut patch, sigma);

            let result = fft::fft_2d(&patch, window, window);
            let summary = result.feature_summary(8);

            println!("║ {:>4} {:>4} │ {:>12.2} │ {:>12.2} │ {:>12.2} │ {:>8} ║",
                grid_col, grid_row, summary.dc_component, summary.ac_energy,
                summary.total_energy, summary.peak_freq_ring);

            if summary.ac_energy > max_ac {
                max_ac = summary.ac_energy;
                best_pos = (grid_col, grid_row);
            }
        }
    }

    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ ★ 最强AC能量位置: Grid({}, {}) AC能量={:.4}           ║",
        best_pos.0, best_pos.1, max_ac);
    println!("╚══════════════════════════════════════════════════════════════╝");
}
