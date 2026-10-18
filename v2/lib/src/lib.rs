//! Typewriter Art — v2 Optimization Engine
//!
//! Converts grayscale photographs into typed characters where multiple layers
//! of typing overlap with sub-character offsets to create continuous tone
//! through multiplicative ink accumulation.
//!
//! ## Quick Usage
//!
//! ```rust,no_run
//! use typewriter_art::{optimize, AlignConfig, CenterFocus, OptimizeConfig, InitMode, SearchAlgorithm};
//!
//! let config = OptimizeConfig {
//!     charset: "hermes-sm".into(),
//!     target: "images/mwdog_crop.png".into(),
//!     row_length: 20,
//!     num_loops: 5,
//!     init_mode: InitMode::Random,
//!     asymmetry: 0.1,
//!     center_focus: CenterFocus::Soft,
//!     search: SearchAlgorithm::SimAnneal,
//!     init_temp: 0.001,
//!     layers: "4x2".into(),
//!     shuffle: true,
//!     out_file: "results/final.png".into(),
//!     align: AlignConfig::default(),
//!     align_override: None,
//! };
//!
//! let result = optimize(config)?;
//! // result.mockup — Array2<f32> grayscale [0.0, 1.0]
//! // result.choices — Array2<u16> char assignments (layers × positions)
//! // result.error_over_time — Vec<f32> after each layer pass
//! // result.stats — timing & comparison counts
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Architecture
//!
//! The core library exposes:
//! - [`optimize()`] — blocking, returns final [`OptimizeResult`]
//! - [`align_search()`] — just the pre-optimization auto-alignment sweep
//! - Configuration via [`OptimizeConfig`] and [`types`]
//! - Export helpers in [`export`] (PNG, JSON, CSV)
//! - Individual modules for lower-level control
//!
//! ## Key performance features
//! - **Batch character evaluation**: all characters computed at once (SIMD-friendly)
//! - **Rayon parallelism**: positions within a layer optimized in parallel
//! - **Coordinate descent**: alternating layer optimization with fixed backgrounds

pub mod align;
pub mod charset;
pub mod chars;
pub mod export;
pub mod layers;
pub mod metrics;
pub mod optimize;
pub mod render;
pub mod resize;
pub mod search;
pub mod types;
pub mod weights;

// Re-export main public API
pub use optimize::{align_search, optimize, optimize_with_progress, ProgressEvent};
pub use types::{
    AlignConfig, AlignmentResult, AlignmentScore, CenterFocus, CharsetConfig, CropTransform,
    InitMode, OptimizeConfig, OptimizeResult, SearchAlgorithm, Stats,
};
pub use export::{build_json_result, image_to_base64_png, JsonResult};
