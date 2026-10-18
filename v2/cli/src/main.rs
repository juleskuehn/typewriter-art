//! CLI entry point for the Typewriter Art v2 optimizer.
//!
//! Drop-in replacement for `optimize.py`. Uses clap for argument parsing.
//! Supports `--json` mode for web server integration.

use std::path::PathBuf;
use std::sync::mpsc;use std::thread;

use clap::Parser;
use typewriter_art::{
    align_search, build_json_result, export, optimize, optimize_with_progress, AlignConfig,
    CenterFocus, CropTransform, DitherConfig, DitherMark, DitherMode, InitMode, InkFloor,
    OptimizeConfig, SearchAlgorithm,
};

fn parse_search(s: &str) -> Result<SearchAlgorithm, String> {
    match s.to_lowercase().as_str() {
        "simanneal" | "sim-anneal" | "sa" | "simulated_annealing" => {
            Ok(SearchAlgorithm::SimAnneal)
        }
        "greedy" | "g" => Ok(SearchAlgorithm::Greedy),
        _ => Err(format!(
            "Invalid search algorithm '{}'. Use 'simAnneal' or 'greedy'",
            s
        )),
    }
}

fn parse_init_mode(s: &str) -> Result<InitMode, String> {
    match s.to_lowercase().as_str() {
        "random" | "r" => Ok(InitMode::Random),
        "blank" | "b" => Ok(InitMode::Blank),
        _ => Err(format!(
            "Invalid init mode '{}'. Use 'random' or 'blank'",
            s
        )),
    }
}

fn parse_center_focus(s: &str) -> Result<CenterFocus, String> {
    match s.to_lowercase().as_str() {
        "off" | "none" | "uniform" => Ok(CenterFocus::Off),
        "soft" => Ok(CenterFocus::Soft),
        "strong" => Ok(CenterFocus::Strong),
        _ => Err(format!(
            "Invalid center focus '{}'. Use 'off', 'soft' or 'strong'",
            s
        )),
    }
}

fn parse_ink_floor(s: &str) -> Result<InkFloor, String> {
    if matches!(s.to_lowercase().as_str(), "auto" | "a") {
        return Ok(InkFloor::Auto);
    }
    s.parse::<f32>().map(InkFloor::Fixed).map_err(|_| {
        format!(
            "Invalid ink floor '{}'. Use 'auto' or a brightness value in [0, 1]",
            s
        )
    })
}

fn parse_dither(s: &str) -> Result<DitherMode, String> {
    match s.to_lowercase().as_str() {
        "none" | "off" => Ok(DitherMode::None),
        "bayer4" | "bayer" | "ordered4" => Ok(DitherMode::Bayer4),
        "bayer8" | "ordered8" => Ok(DitherMode::Bayer8),
        "bluenoise" | "blue" | "bn" => Ok(DitherMode::BlueNoise),
        "fs" | "floyd" | "floyd-steinberg" | "floydsteinberg" => Ok(DitherMode::FloydSteinberg),
        _ => Err(format!(
            "Invalid dither '{}'. Use 'none', 'bayer4', 'bayer8', 'bluenoise' or 'fs'",
            s
        )),
    }
}

fn parse_dither_mark(s: &str) -> Result<DitherMark, String> {
    if matches!(s.to_lowercase().as_str(), "auto" | "a") {
        return Ok(DitherMark::Auto);
    }
    s.parse::<u16>()
        .map(DitherMark::Char)
        .map_err(|_| format!("Invalid dither mark '{}'. Use 'auto' or a glyph index", s))
}

fn format_search(s: &SearchAlgorithm) -> String {
    match s {
        SearchAlgorithm::Greedy => "greedy".to_string(),
        SearchAlgorithm::SimAnneal => "simulated annealing".to_string(),
    }
}

fn format_init_mode(m: &InitMode) -> String {
    match m {
        InitMode::Random => "random".to_string(),
        InitMode::Blank => "blank".to_string(),
    }
}

#[derive(Parser, Debug)]
#[command(name = "optimize", about = "Typewriter Art v2 — fast multi-layer ASCII art optimizer")]
struct Args {
    /// Path to charset folder containing config.json and image
    #[arg(short = 'c', long, default_value = "sc-2")]
    charset: String,

    /// Path to target image
    #[arg(short = 't', long, default_value = "images/mwdog_crop.png")]
    target: String,

    /// Number of characters per row (determines output resolution)
    #[arg(short = 'r', long, default_value = "30")]
    row_length: u32,

    /// Number of optimization passes per layer
    #[arg(short = 'n', long, default_value = "15")]
    num_loops: u32,

    /// Initial character assignment: random or blank
    #[arg(short = 'i', long, default_value = "random", value_parser = parse_init_mode)]
    init_mode: InitMode,

    /// Asymmetry of MSE function (penalizes too-light regions)
    #[arg(short = 'a', long, default_value = "0.1")]
    asymmetry: f32,

    /// Center focus: spatial weighting of the loss and alignment score,
    /// emphasizing the middle of the image over edges and margins
    /// (off | soft | strong)
    #[arg(long = "center-focus", default_value = "soft", value_parser = parse_center_focus)]
    center_focus: CenterFocus,

    /// Ink saturation floor: how dark overstriking can get. 'auto' derives it
    /// from the charset scan (darkest glyph pixel = ribbon ink density), or
    /// give a brightness value (advanced)
    #[arg(long = "ink-floor", default_value = "auto", value_parser = parse_ink_floor)]
    ink_floor: InkFloor,

    /// Dithering pattern for light tones (advanced):
    /// none | bayer4 | bayer8 | bluenoise | fs
    #[arg(long = "dither", default_value = "none", value_parser = parse_dither)]
    dither: DitherMode,

    /// Dither light band: only cells whose target ink density is at most this
    /// fraction are dithered (advanced)
    #[arg(long = "dither-band", default_value = "0.15")]
    dither_band: f32,

    /// Dither flip refinement passes after seeding, 0 = seed only (advanced)
    #[arg(long = "dither-refine", default_value = "2")]
    dither_refine: u32,

    /// Dither mark: 'auto' (lightest glyph) or a glyph index (advanced)
    #[arg(long = "dither-mark", default_value = "auto", value_parser = parse_dither_mark)]
    dither_mark: DitherMark,

    /// Search algorithm: simAnneal or greedy
    #[arg(short = 's', long, default_value = "simAnneal", value_parser = parse_search)]
    search: SearchAlgorithm,

    /// Initial temperature for simulated annealing
    #[arg(long = "temp", default_value = "0.001")]
    init_temp: f32,

    /// Layer configuration key from layers.json
    #[arg(short = 'l', long, default_value = "4x2")]
    layers: String,

    /// Disable layer shuffling (default: shuffle enabled)
    #[arg(long = "no-shuffle")]
    no_shuffle: bool,

    /// Output file path for the final mockup PNG
    #[arg(short = 'o', long, default_value = "results/final.png")]
    out_file: String,

    /// Output directory (overrides --out-file parent dir for all outputs)
    #[arg(long = "out-dir")]
    out_dir: Option<String>,

    /// Output result as JSON to stdout (for web server integration).
    /// When set, all log messages go to stderr; only the JSON goes to stdout.
    #[arg(long = "json", default_value = "false")]
    json: bool,

    /// Stream progress as NDJSON lines to stdout (for SSE integration).
    /// Each line is a JSON object with a "type" tag: {"type":"align",
    /// "candidate":3,"total":64} during the auto-alignment sweep, or
    /// {"type":"loop","loop_num":1,"num_loops":15,"layer_num":3,
    /// "num_layers":8,"error":0.042,"mockup_b64":"..."} after each layer pass
    /// (mockup_b64 only on the first layer), followed by a final line
    /// {"type":"complete","result":{...}}. Implies --json.
    #[arg(long = "stream", default_value = "false")]
    stream: bool,

    /// Disable the pre-optimization auto-alignment sweep (enabled by default).
    /// The sweep tests small zoom/shift crops of the target, runs a quick
    /// greedy pass on each, and applies the crop that maximizes SSIM×4 + PSNR.
    #[arg(long = "no-align")]
    no_align: bool,

    /// Only run the auto-alignment sweep: score every crop, write
    /// alignment_scores.json plus one quick mockup per candidate, then exit
    /// without optimizing.
    #[arg(long = "align-only")]
    align_only: bool,

    /// Auto-alignment fine resolution: 1 step = 1/N of a character
    /// (default 8 = ⅛ char; the coarse grid is ¼ char, refined locally)
    #[arg(long = "align-steps", default_value = "8")]
    align_steps: u32,

    /// Quick greedy passes per alignment candidate (paper: 1)
    #[arg(long = "align-passes", default_value = "1")]
    align_passes: u32,

    /// Alignment score weight: score = SSIM × weight + PSNR (paper: 4)
    #[arg(long = "align-ssim-weight", default_value = "4.0")]
    align_ssim_weight: f32,

    /// Two-stage sweep: rescore the top N candidates with the full layer stack
    /// (0 = rank all candidates by the cheap single-pass screen alone)
    #[arg(long = "align-refine-top", default_value = "8")]
    align_refine_top: u32,

    /// Deluxe mode: deep-rerank the top N candidates with a real optimization
    /// run (N = 4 when given as a bare flag; 0 = off). The winner continues to
    /// the full loop budget instead of restarting.
    #[arg(
        long = "align-deluxe",
        num_args = 0..=1,
        default_value = "0",
        default_missing_value = "4"
    )]
    align_deluxe: u32,

    /// Optimization loops per deluxe-reranked candidate
    /// (default: max(3, num_loops / 2))
    #[arg(long = "align-rerank-loops", default_value = "0")]
    align_rerank_loops: u32,

    /// RNG seed for reproducible runs (random init, SA, layer shuffle)
    #[arg(long = "seed")]
    seed: Option<u64>,

    /// Manual crop transform (skips the sweep): zoom step (signed; 1 =
    /// 1/align-steps of a character of boundary growth, negative shrinks)
    #[arg(long = "zoom")]
    zoom: Option<i32>,

    /// Manual crop transform (skips the sweep): horizontal shift step
    /// (signed; positive = left)
    #[arg(long = "shift-x")]
    shift_x: Option<i32>,

    /// Manual crop transform (skips the sweep): vertical shift step
    /// (signed; positive = up)
    #[arg(long = "shift-y")]
    shift_y: Option<i32>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // In JSON or stream mode, send logs to stderr
    let log_target = if args.json || args.stream {
        env_logger::Target::Stderr
    } else {
        env_logger::Target::Stdout
    };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .target(log_target)
        .init();

    // v1 parity: `--out_file` is the path of the final mockup PNG.
    // `--out-dir` (optional) redirects all outputs into a directory.
    let out_file: PathBuf = if let Some(ref dir) = args.out_dir {
        PathBuf::from(dir).join("final.png")
    } else {
        PathBuf::from(&args.out_file)
    };
    let results_dir = out_file
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("results"));

    let align_override = if args.zoom.is_some() || args.shift_x.is_some() || args.shift_y.is_some() {
        Some(CropTransform {
            zoom: args.zoom.unwrap_or(0),
            shift_x: args.shift_x.unwrap_or(0),
            shift_y: args.shift_y.unwrap_or(0),
        })
    } else {
        None
    };

    let config = OptimizeConfig {
        charset: args.charset,
        target: args.target,
        layers: args.layers,
        row_length: args.row_length,
        num_loops: args.num_loops,
        init_mode: args.init_mode,
        asymmetry: args.asymmetry,
        center_focus: args.center_focus,
        ink_floor: args.ink_floor,
        dither: DitherConfig {
            mode: args.dither,
            band_max: args.dither_band.clamp(0.0, 1.0),
            refine_passes: args.dither_refine,
            mark: args.dither_mark,
        },
        search: args.search,
        init_temp: args.init_temp,
        shuffle: !args.no_shuffle,
        out_file: out_file.to_string_lossy().to_string(),
        align: AlignConfig {
            enabled: !args.no_align && align_override.is_none(),
            steps: args.align_steps.max(4),
            passes: args.align_passes.max(1),
            ssim_weight: args.align_ssim_weight,
            refine_top: args.align_refine_top,
            rerank_top: args.align_deluxe,
            rerank_loops: args.align_rerank_loops,
        },
        align_override,
        seed: args.seed,
    };

    if !args.json && !args.stream {
        log::info!("Starting optimization...");
        log::info!("  charset: {}", config.charset);
        log::info!("  target: {}", config.target);
        log::info!("  row_length: {}", config.row_length);
        log::info!("  num_loops: {}", config.num_loops);
        log::info!("  layers: {}", config.layers);
        log::info!("  search: {}", format_search(&config.search));
        log::info!("  init_mode: {}", format_init_mode(&config.init_mode));
    }

    // Alignment-only mode: score every crop, save the sweep results and each
    // candidate's quick mockup, then exit without optimizing (v1's autoCrop).
    if args.align_only {
        let search = align_search(&config, true)?;
        std::fs::create_dir_all(&results_dir)?;
        export::save_alignment(&search.result, &results_dir.join("alignment_scores.json"))?;
        for (t, img) in &search.mockups {
            let name = format!("align_{}_{}_{}.png", t.zoom, t.shift_x, t.shift_y);
            export::save_image(img, &results_dir.join(name))?;
        }
        println!(
            "{:>4} {:>5} {:>5} | {:>8} {:>8} | score",
            "zoom", "shiftX", "shiftY", "PSNR", "SSIM"
        );
        for s in &search.result.scores {
            println!(
                "{:>4} {:>5} {:>5} | {:>8.3} {:>8.4} | {:.3}",
                s.transform.zoom, s.transform.shift_x, s.transform.shift_y, s.psnr, s.ssim, s.score
            );
        }
        let best = search.result.best;
        println!(
            "Best: zoom {}, shift ({}, {}) — scores saved to {}",
            best.zoom,
            best.shift_x,
            best.shift_y,
            results_dir.join("alignment_scores.json").display()
        );
        return Ok(());
    }

    let result = if args.stream {
        // Set up streaming channel
        let (tx, rx) = mpsc::channel::<typewriter_art::ProgressEvent>();

        // Spawn a thread to print NDJSON lines as they arrive
        let stream_thread = thread::spawn(move || {
            for event in rx {
                let json_line = serde_json::to_string(&event).unwrap_or_default();
                println!("{}", json_line);
            }
        });

        let result = optimize_with_progress(config.clone(), tx)?;

        // Wait for the stream thread to finish printing
        stream_thread.join().unwrap();

        result
    } else {
        optimize(config.clone())?
    };

    // Save results to disk (always, so they're available for download)
    std::fs::create_dir_all(&results_dir)?;
    export::save_mockup(&result, &out_file)?;
    export::save_layers(&result, &results_dir)?;
    export::save_choices(&result, &results_dir.join("choices.json"))?;
    export::save_stats(&result, &results_dir.join("stats.txt"))?;
    export::save_error_over_time(&result, &results_dir.join("error_over_time.csv"))?;
    if let Some(ref alignment) = result.alignment {
        export::save_alignment(alignment, &results_dir.join("alignment_scores.json"))?;
    }

    if args.json || args.stream {
        // Output final JSON result to stdout
        let json_result = build_json_result(&result)?;
        let final_event = serde_json::json!({
            "type": "complete",
            "result": json_result
        });
        println!("{}", serde_json::to_string(&final_event)?);
    } else {
        // Human-readable output
        println!(
            "Optimization completed in {:.3} s",
            result.stats.total_time_ns as f64 / 1e9
        );
        println!("Total comparisons: {}", result.stats.total_comparisons);
        if result.stats.total_comparisons > 0 {
            println!(
                "Time per comparison: {:.3} µs",
                result.stats.total_time_ns as f64
                    / result.stats.total_comparisons as f64
                    / 1000.0
            );
        }
        println!("Final error: {:.6}", result.stats.final_error);

        println!("Mockup saved to: {}", out_file.display());
        println!("Layer images saved to: {}", results_dir.display());
        println!("Choices saved to: {}", results_dir.join("choices.json").display());
    }

    Ok(())
}
