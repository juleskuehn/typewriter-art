//! CLI entry point for the Typewriter Art v2 optimizer.
//!
//! Drop-in replacement for `optimize.py`. Uses clap for argument parsing.
//! Supports `--json` mode for web server integration.

use std::path::{Path, PathBuf};

use clap::Parser;
use typewriter_art::{build_json_result, optimize, export, InitMode, OptimizeConfig, SearchAlgorithm};

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
}

fn main() -> anyhow::Result<()> {
    // In JSON mode, send logs to stderr to keep stdout clean
    let log_target = if std::env::args().any(|a| a == "--json") {
        env_logger::Target::Stderr
    } else {
        env_logger::Target::Stdout
    };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .target(log_target)
        .init();

    let args = Args::parse();

    // Determine output directory
    let results_dir = if let Some(ref dir) = args.out_dir {
        PathBuf::from(dir)
    } else {
        let out_path = PathBuf::from(&args.out_file);
        out_path.parent().unwrap_or(Path::new("results")).to_path_buf()
    };

    let config = OptimizeConfig {
        charset: args.charset,
        target: args.target,
        layers: args.layers,
        row_length: args.row_length,
        num_loops: args.num_loops,
        init_mode: args.init_mode,
        asymmetry: args.asymmetry,
        search: args.search,
        init_temp: args.init_temp,
        shuffle: !args.no_shuffle,
        out_file: results_dir.join("final.png").to_string_lossy().to_string(),
    };

    if !args.json {
        log::info!("Starting optimization...");
        log::info!("  charset: {}", config.charset);
        log::info!("  target: {}", config.target);
        log::info!("  row_length: {}", config.row_length);
        log::info!("  num_loops: {}", config.num_loops);
        log::info!("  layers: {}", config.layers);
        log::info!("  search: {}", format_search(&config.search));
        log::info!("  init_mode: {}", format_init_mode(&config.init_mode));
    }

    let result = optimize(config.clone())?;

    // Save results to disk (always, so they're available for download)
    std::fs::create_dir_all(&results_dir)?;
    export::save_mockup(&result, &results_dir.join("final.png"))?;
    export::save_layers(&result, &results_dir)?;
    export::save_choices(&result, &results_dir.join("choices.json"))?;
    export::save_stats(&result, &results_dir.join("stats.txt"))?;
    export::save_error_over_time(&result, &results_dir.join("error_over_time.csv"))?;

    if args.json {
        // Output JSON result to stdout
        let json_result = build_json_result(&result)?;
        let json_str = serde_json::to_string(&json_result)?;
        println!("{}", json_str);
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

        println!("Mockup saved to: {}", results_dir.join("final.png").display());
        println!("Layer images saved to: {}", results_dir.display());
        println!("Choices saved to: {}", results_dir.join("choices.json").display());
    }

    Ok(())
}
