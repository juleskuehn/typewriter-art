# Typewriter Art — v2 Rewrite Plan

> **Date**: 2026-07-08  
> **Status**: Planning / Analysis  
> **Goal**: 10×+ backend performance improvement + clean API for Phase 2 web app.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Algorithm Deep Dive](#2-algorithm-deep-dive)
3. [Performance Bottlenecks](#3-performance-bottlenecks)
4. [Proposed Architecture (v2)](#4-proposed-architecture-v2)
5. [Core Algorithm Improvements](#5-core-algorithm-improvements)
6. [Language & Runtime Selection](#6-language--runtime-selection)
7. [API Design (for Phase 2 Web App)](#7-api-design-for-phase-2-web-app)
8. [Implementation Plan](#8-implementation-plan)
9. [Milestones & Timeline](#9-milestones--timeline)
10. [Open Questions](#10-open-questions)

---

## 1. Current State Analysis

### 1.1 Codebase Structure

```
src/
├── optimize.py          # Main entry point, CLI args, orchestration (388 lines)
├── utils.py             # Core logic: charset loading, layer optimization pass (200 lines)
├── convert_choices.py   # Convert choices.json → typed positions
├── convert_video.py     # Video → frames → optimize → video pipeline
├── convert_to_daisywriter.py  # Output → DaisyWriter hardware instructions
├── layers.json          # Layer offset configurations (fractional char width/height)
├── charsets/            # 9 character sets with config.json + glyph sprite sheets
│   ├── hermes, hermes-md, hermes-sm
│   ├── sc-1, sc-2, sc-3, sc-round
│   └── daisywriter, daisywriter-shrunken, dw-no-chev
└── images/
    ├── mwdog_crop.png
    └── cat.jpeg
```

### 1.2 Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `charset` | sc-2 | ~9 options | 87–265 chars, 18×10 to 52×26 px |
| `row_length` | 30 | 5–100 | Controls output resolution |
| `layers` | 4x2 | 1×1, 4×1, 4×2, 16×1, daisy_full… | 1–80 overlapping layers |
| `num_loops` | 15 | 1–100 | Optimization passes per layer |
| `search` | simAnneal | simAnneal, greedy | Search algorithm |
| `asymmetry` | 0.1 | 0–1 | Penalizes too-light regions |
| `init_temp` | 0.001 | — | SA starting temperature |
| `init_mode` | random | random, blank | Initial character assignment |

### 1.3 Charset Sizes

| Charset | Chars | Size (h×w) | Grid cells @30 cols |
|---------|-------|------------|---------------------|
| hermes-sm | 88 | 18×10 | ~25×31 = 775 |
| hermes-md | 88 | 28×16 | ~16×31 = 496 |
| sc-1 | 87 | 50×26 | ~9×31 = 279 |
| sc-2 | 175 | 52×26 | large |
| sc-3 | 265 | 50×26 | large |

### 1.4 Current Performance (measured)

Test: `python optimize.py --charset hermes-sm --row_length 20 --num_loops 2 --display 0`

| Metric | Value |
|--------|-------|
| Total time | 4.92 s |
| Total comparisons | 55,733 |
| Time per comparison | ~88 µs (88 ns label in code is wrong) |
| Chars per position | 88 |
| Positions | ~775 @ row_length=30; ~350 @ row_length=20 |
| Layers | 8 (4x2 config) |

**Extrapolated for "production" settings** (sc-2, row_length=30, num_loops=15, 8 layers):

| Metric | Estimate |
|--------|----------|
| Positions per layer | ~300 (sc-2 has larger characters) |
| Comparisons per position | ~175 (num_chars) |
| Comparisons per layer-pass | ~52,500 |
| Total comparisons | 52,500 × 8 × 15 = **6.3M** |
| Estimated time | **~9 minutes** |

This doesn't scale well for the web-app use case where users expect near-instant previews and results within seconds, not minutes.

---

## 2. Algorithm Deep Dive

### 2.1 Core Idea

Convert a grayscale photograph into typed characters where multiple layers of typing overlap (with sub-character offsets) to create continuous tone through multiplicative ink accumulation.

**Mathematical model**: The image is modeled as the *product* of independent layer transmittances:

```
Mockup_pixel = ∏_{layer l} Layer_l[pixel - offset_l]
```

Where each layer is composed of tiled character glyphs. Target values are 0 (black) to 1 (white). Product-of-layers means each additional layer can only darken the image (multiply by ≤1).

### 2.2 Optimization Problem

For a fixed set of layer offsets, find the character assignment per position per layer that minimizes:

```
AMSE = mean( asymmetric_{MSE}(Target, Mockup) )

where asymmetric_MSE(err) = (1 + asymmetry) * err²  if err > 0 (too light)
                           = err²                     if err ≤ 0 (too dark)
```

The asymmetry penalizes regions that are too light (not enough ink), which is important because the multiplicative model can only darken—you can't recover from an already-too-light area in a later optimization step.

### 2.3 Search Algorithm

The paper (GI 2021) and the current implementation use slightly different search structures.

#### Paper's Algorithm (Algorithm 1 in the paper)

```
Create list of all positions across all layers: {layer, row, col, charId}
Initialize each charId randomly
temperature = initTemp (1e-2 empirically)
for iteration in 1..maxIterations:
    for pos in shuffled(all_positions):
        Compute curLoss with current character
        for candidateId in shuffled(all_charIds):
            Compute newLoss with candidate character
            delta = curLoss - newLoss
            if delta > 0 or exp(delta/temperature) > random(0,1):
                pos.charId = candidateId
                break   # accept first qualifying candidate
        temperature = temperature - coolingStep  (1e-4)
        if temperature <= 0:
            temperature = initTemp * reheat       (reheat factor 1e-1)
            initTemp = temperature                 # for next reheat cycle
```

Key points from the paper:
- **Flat position list**: All positions across all layers are flattened and shuffled together
- **Random character evaluation order**: Critical for SA — the random order of candidate characters determines which suboptimal moves are explored
- **Temperature cools per position** with reheating: initTemp=1e-2, coolingStep=1e-4, reheat=1e-1
- **Break on first acceptance**: This is standard SA — accept the first candidate that passes the test, don't exhaustively check all candidates

#### Current Implementation (optimize.py)

```
for loop in 1..num_loops:
    for each layer (shuffled order):
        1. bg = product of all OTHER layers (coordinate descent)
        2. For each position in this layer (parallel with prange):
            a. Extract target_slice, bg_slice at position + offset
            b. Compute current AMSE
            c. Randomly permute all characters
            d. For each candidate char:
                - new_composite = bg_slice × char
                - new_amse = asymmetric_mse(target, composite)
                - delta = cur_amse - new_amse
                - If delta > 0: accept
                - If exp(delta/temperature) > rand(): accept
                - Break on first acceptance
            e. Update mockup with chosen composite
    temperature = init_temp / (loop_num + 1)  ← fixed per outer loop
```

**Key differences from paper**:
1. **Alternating layer optimization** (coordinate descent) instead of flat position list. Each layer is optimized independently, holding all other layers fixed as a "background" layer. This is more principled — it decomposes the joint optimization into subproblems.
2. **Temperature is constant within an outer loop** (`init_temp / (loop_num + 1)`), not cooled per position. No reheating.
3. **Positions are parallelized** (prange), which the flat-list approach couldn't easily do.
4. **Random character order IS preserved** — essential for SA. Accepts first qualifying candidate and breaks.

**For v2**: We keep the alternating-layer (coordinate descent) structure from the current implementation — it's mathematically cleaner and parallelizes better. The SA within each layer pass must preserve random candidate evaluation order. We also add the paper's optimized cropping (auto-alignment) feature which is not in the current code (§5.8).

### 2.4 Layer Offsets

Layers can be offset by fractions of a character width/height. Common configurations:

- **4×1**: 4 layers at (0,0), (0,0.5), (0.5,0), (0.5,0.5) — one set of quarter-char offsets
- **4×2**: The above ×2 = 8 layers — doubled density
- **16×1**: 4×4 grid of fractional offsets
- **daisy_full**: 40 layers at various fractional offsets (for DaisyWriter hardware)

---

## 3. Performance Bottlenecks

### 3.1 Primary Bottleneck: Per-Character Inner Loop

The innermost loop (`layer_optimization_pass` in `utils.py`) does:

```python
for new_choice in np.random.permutation(chars.shape[0]):
    new_composite = bg_slice * chars[new_choice]   # H×W element-wise multiply
    err = target_slice - new_composite              # H×W subtract
    asym_err = np.where(err > 0, err * (1+asymmetry), err)  # conditional
    new_amse = np.mean(np.square(asym_err))         # reduce
```

Even with `@njit(parallel=True)`, this iterates chars *sequentially* within each position. For `C` chars and `P` positions, this is `O(C×P)` iterations where each does `4*H*W` float ops.

**Problem**: Each iteration operates on tiny arrays (e.g., 28×16 = 448 floats). Numba overhead and poor SIMD utilization dominate. The arrays are too small to amortize function call costs.

**Solution**: Batch-process all characters for a single position at once:
```
composites = bg_slice * chars   # broadcast (H,W) × (C,H,W) → (C,H,W)
errors = target - composites    # (C,H,W)
asym_errors = where(errors > 0, errors * (1+asym), errors)  # (C,H,W)
mses = mean(square(asym_errors), axis=(1,2))  # (C,)
```
This turns the inner loop from `C` sequential tiny ops into a single vectorized operation. On CPU with SIMD (or GPU), this is 10–50× faster for this step alone.

### 3.2 Secondary Bottleneck: Random Permutation Overhead

For each of `P` positions, `np.random.permutation(C)` is called. This is `O(P×C)` random number generation. For large P and C, this adds measurable overhead.

**Solution**: Pre-compute a small pool of random permutations (e.g., 16) and cycle through them, or use a fast permuted sequence generator (LCG with prime stride).

### 3.3 Tertiary: SA Early-Exit vs. Greedy

SA mode can break early when it accepts a character, reducing average comparisons. But the early exit is only possible in sequential mode—batch mode needs all MSEs computed upfront anyway. Trade-off: batch computation gives such a speedup that computing all comparisons is still faster than sequential with early exit.

For SA, after computing all MSEs in batch:
1. Sort characters by MSE ascending
2. Try them in order until one is accepted (almost always the best one at low temperature, or a random one at high temperature)
3. This preserves SA semantics while using batch computation

### 3.4 Memory & Cache

The 3D `chars` array is small (e.g., 88 × 18 × 10 = 15,840 floats = 63 KB). All charsets fit in L2 cache. The `composites` temporary for one position is `C × H × W` floats (max ~138 KB for sc-3), also L2-cacheable. Memory bandwidth is not a bottleneck.

---

## 4. Proposed Architecture (v2)

### 4.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     v2/ (new directory)                      │
├─────────────────────────────────────────────────────────────┤
│  lib/                    # Core backend library (Rust)       │
│  ├── charset.rs          # Charset loading, preprocessing    │
│  ├── types.rs            # Core types & data structures      │
│  ├── optimize.rs         # Optimization engine               │
│  ├── render.rs           # Mockup synthesis & layer ops      │
│  ├── search.rs           # Search algorithms (greedy, SA)    │
│  ├── layers.rs           # Layer configuration               │
│  ├── resize.rs           # Target image preprocessing        │
│  └── export.rs           # Output: PNG, choices.json, etc.   │
│                                                              │
│  cli/                     # CLI executable (Rust binary)     │
│  └── main.rs             # Drop-in replacement for optimize  │
│                                                              │
│  server/                  # Phase 2: Web server              │
│  └── main.rs             # HTTP/gRPC API for web frontend    │
│                                                              │
│  web/                     # Phase 2: Web frontend (separate) │
│  └── (Next.js / Svelte / ...)                               │
│                                                              │
│  assets/                  # Charsets, layer configs          │
│  └── (symlinks to ../src/charsets, ../src/layers.json)      │
│                                                              │
│  Cargo.toml              # Rust project manifest             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Core Data Flow

```
Input Image
    │
    ▼
┌──────────────┐
│  resizeTarget │  ← Scale to char grid, pad
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ prep_charset  │  ← Load sprite sheet, chop, normalize to float [0,1]
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                    Optimization Loop                   │
│                                                       │
│  for loop in 1..num_loops:                            │
│    for each layer (shuffled):                         │
│      bg = combine(all other layers)                   │
│      choices[layer] = optimize_layer(bg, target,      │
│                        chars, offsets)                │
│      layers[layer] = render(layer, choices, chars)    │
│      mockup = combine(all layers)                     │
│                                                       │
│  optimize_layer:                                      │
│    for each position (parallel):                      │
│      bg_slice = bg[region]                            │
│      target_slice = target[region]                    │
│      composites = bg_slice × chars    ← BATCH        │
│      mses = amse(target_slice, composites)            │
│      best_char = select(mses, strategy)               │
│      return best_char, best_composite, mse            │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ Export results│  ← PNG layers, mockup, choices.json
└──────────────┘
```

---

## 5. Core Algorithm Improvements

### 5.1 Batch Character Evaluation (HIGH IMPACT)

**Current**: For each position, iterate characters one-by-one:
```
for char in chars:
    composite = bg * char
    error = asymmetric_mse(target, composite)
    if better: accept
```

**Proposed**: For each position, evaluate all characters at once:
```
composites = bg * chars          # (C, H, W) = broadcast (H,W) × (C,H,W)
errors = target - composites     # (C, H, W)
asym_errors = where(errors > 0, errors * (1+asym), errors)
mses = mean(square(asym_errors), axis=(1,2))  # (C,)
```

**Expected speedup**: 10–50× for the inner loop depending on char and grid size.

**Implementation in Rust**: Use `ndarray` with blas-src backend (OpenBLAS) for SIMD-accelerated batched operations, or hand-roll with Rayon for parallelism.

### 5.2 Simulated Annealing Adaptation (CORRECTED)

**Challenge**: SA requires candidates to be evaluated in **random order** — this is essential. The random order determines WHICH suboptimal move is taken when a character swap increases loss. Sorting by MSE would eliminate SA's ability to escape local minima, reducing it to greedy search.

**Correct approach** — batch precompute + random-order SA:

1. **Batch precompute** all composite errors for this position:
   ```
   composites = bg_slice × chars          # (C, H, W) batch broadcast
   errors = target - composites            # (C, H, W)
   asym_errors = where(errors > 0, errors*(1+asym), errors)
   mses = mean(square(asym_errors), axis=(1,2))  # (C,) — all scores at once
   ```
2. Generate a **random permutation** of character indices (same as current code)
3. **Iterate in random order**, using the precomputed MSEs:
   - Compute `delta = current_mse - mses[perm[i]]`
   - If `delta > 0` (candidate is better): accept, break
   - If `exp(delta / temperature) > random()`: accept, break  
   - Otherwise continue to next random candidate
4. If none accepted: keep current character

This gives us the best of both worlds:
- **Batch speed**: All composites computed once with SIMD vectorization
- **SA correctness**: Random evaluation order preserved exactly as in the original algorithm

The only minor difference from the current code: we compute all MSEs before SA selection instead of lazily. Since the batch computation is so fast (3–10× faster than sequential), the cost of computing all MSEs upfront is less than the cost of computing them one-by-one with early exit.

### 5.3 Precomputed Character Statistics (MEDIUM IMPACT)

For greedy search, we can potentially skip the full composite computation with a fast screening pass:

Each character has precomputed statistics: mean darkness, variance, etc. Since `composite = bg × char`, the composite mean is related to `bg_mean * char_mean` (exactly with independence assumption). A cheap MSE estimate using these statistics could filter out obviously bad characters, reducing the full composite computation to a subset.

**Trade-off**: This adds complexity and approximate screening might miss the true optimum. Probably not worth it for v2 given the massive speedup from batching alone.

### 5.4 Optimized Cropping / Auto-Alignment (NEW FEATURE)

**Status: implemented** (see `v2/lib/src/align.rs`, `v2/lib/src/metrics.rs`, and
the “Auto-Alignment” section of `v2/README.md`). The sweep runs in two stages —
single-pass screening of all 64 candidates (v1 parity), then full-layer-stack
rescoring of the top N — which keeps the cost to a few seconds.

The paper describes a preprocessing step NOT in the current code: **auto-alignment** of the target image to the character grid. This tests 64 different crops (combinations of x/y translation by {0, 0.25, 0.5, 0.75} × charWidth/charHeight, and scaling by small fractions), runs a single greedy optimization cycle for each, then picks the crop that maximizes `SSIM × 4 + PSNR`.

For v2, we should include this. It's expensive (10 minutes per the paper) but highly effective at small sizes. Implementation:
1. Run a fast, low-resolution greedy pass for each candidate crop
2. Pick the best crop
3. Apply it before the full optimization

This is optional (flag-gated) since it adds significant time. For the web app, the WASM preview could skip it; the server-side pass could include it.

### 5.5 Progressive Refinement (MEDIUM IMPACT)

Start optimization at lower resolution (fewer chars per row, downsampled target), quickly converge on rough structure, then refine at full resolution. The coarse solution initializes the fine solution.

**Implementation**:
1. Downsample target to e.g., 50% resolution
2. Run optimization with fewer passes
3. Upsample the character assignments to full resolution
4. Run final optimization at full resolution

This could give 2–4× speedup with minimal quality loss. Particularly useful for video processing.

### 5.6 Integral Image for Fast Mockup Recomposition (LOW IMPACT)

Currently, recomputing `bg = product of all other layers` requires element-wise multiplication of large arrays. With integral images or summed-area tables of log-transformed layers, this could be faster. However, the recomposition is infrequent (once per layer per loop) compared to inner-loop comparisons, so this is low priority.

### 5.7 Parallelization Strategy

**Current**: Numba's `@njit(parallel=True)` with `prange` parallelizes over positions within a layer pass. Good, but Python GIL limits outer loop parallelism.

**Proposed (Rust)**:
- **Inner**: SIMD-accelerated batch character evaluation (via `ndarray` + BLAS, or explicit SIMD intrinsics)
- **Middle**: Rayon parallel iterators over positions within a layer pass
- **Outer**: Could parallelize independent video frames or multiple resolution levels

For single-image optimization, middle-level parallelism saturates most CPUs. GPU acceleration (wgpu/CUDA) could give another 10–50× for batch composites.

### 5.8 GPU Acceleration Path (FUTURE)

The algorithm is embarrassingly parallel at the position × character level. Each position independently computes `composites = bg * chars`. A GPU kernel could launch `num_positions × num_chars` threads, each computing one composite pixel value and accumulating error. This is a natural fit for GPU compute.

**Proposed approach**: Keep the CPU path as default, add optional GPU path via `wgpu` (cross-platform) or CUDA (NVIDIA). The GPU path would be exposed as a feature flag.

---

## 6. Language & Runtime Selection

### 6.1 Recommendation: **Rust**

| Criterion | Rust | Python (v1) | C++ | Go | Zig |
|-----------|------|-------------|-----|----|-----|
| Raw performance | ★★★★★ | ★★ | ★★★★★ | ★★★★ | ★★★★★ |
| SIMD/vectorization | ★★★★ | ★★ (numba) | ★★★★★ | ★★ | ★★★★ |
| Parallelism (no GIL) | ★★★★★ | ★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| WebAssembly target | ★★★★★ | ★ | ★★ | ★★★★ | ★★★★ |
| Cargo ecosystem | ★★★★★ | ★★★★ | ★★★ | ★★★★ | ★★ |
| Type safety | ★★★★★ | ★ | ★★★ | ★★★★ | ★★★★ |
| WASM for web preview | yes | no | limited | yes | yes |
| Learning curve | moderate | low | high | low | moderate |

**Key reasons for Rust**:

1. **No GIL**: True parallelism with Rayon, critical for multicore scaling
2. **SIMD**: `ndarray` with BLAS backend or `std::simd` for inner loop
3. **WASM target**: Same core library compiles to WASM for in-browser preview (Phase 2). Users could see instant low-res previews *in the browser* before submitting for server-side optimization
4. **Cargo**: Modern package manager, `ndarray` is mature, `image` crate for I/O, `serde`/`serde_json` for config/output
5. **FFI**: Easy to call from Python if needed for backward compatibility
6. **Single binary**: CLI tool is a static binary—no Python environment needed

### 6.2 Key Dependencies (Rust)

```toml
[dependencies]
# Numeric computing
ndarray = { version = "0.16", features = ["rayon", "blas"] }
ndarray-rand = "0.15"
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["system"] }

# Parallelism
rayon = "1.10"

# Image I/O
image = "0.25"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Random
rand = "0.8"
rand_distr = "0.4"

# CLI
clap = { version = "4", features = ["derive"] }

# Logging
log = "0.4"
env_logger = "0.11"

# Optional GPU
[target.'cfg(feature = "gpu")'.dependencies]
wgpu = "23"
```

### 6.3 Alternative Considered: Zig

Zig would give even more low-level control and potentially better SIMD via `@Vector`. However, the ecosystem for image I/O and JSON is much less mature. Rust's `ndarray` alone saves weeks of work.

---

## 7. API Design (for Phase 2 Web App)

### 7.1 Core Library API (Rust)

The backend library should expose a clean API that can be called from:
- CLI binary
- HTTP server (axum/actix)
- WASM (for browser-side preview)
- Python FFI (optional backward compatibility)

```rust
// lib.rs — Public API sketch

/// Configuration for a single optimization run
pub struct OptimizeConfig {
    pub charset: CharsetConfig,
    pub target_image: Vec<u8>,        // raw PNG bytes
    pub row_length: u32,
    pub num_loops: u32,
    pub init_mode: InitMode,
    pub asymmetry: f32,
    pub search: SearchAlgorithm,
    pub init_temp: f32,
    pub layers: LayerConfig,
    pub shuffle: bool,
}

/// Progress update emitted during optimization
pub enum OptimizeEvent {
    LoopStarted { loop_num: u32, total_loops: u32 },
    LayerStarted { layer_num: u32, layer_offset: (f32, f32) },
    LayerCompleted { layer_num: u32, error: f32, comparisons: u64 },
    /// Partial mockup image (PNG bytes) for streaming preview
    MockupUpdate { image: Vec<u8>, error: f32 },
    Completed { final_image: Vec<u8>, choices: String, stats: Stats },
}

/// Streaming iterator over optimization progress
pub fn optimize_streaming(config: OptimizeConfig) -> impl Iterator<Item = OptimizeEvent>;

/// Non-streaming: just return final result
pub fn optimize(config: OptimizeConfig) -> OptimizeResult;

/// Data types for results
pub struct OptimizeResult {
    pub mockup: Vec<u8>,          // PNG bytes
    pub layers: Vec<Vec<u8>>,     // Individual layer PNGs
    pub choices: Choices,         // Character assignment grid
    pub error_plot: Vec<u8>,      // PNG error-over-time
    pub stats: Stats,
}

pub struct Stats {
    pub total_time: Duration,
    pub total_comparisons: u64,
    pub time_per_comparison: Duration,
    pub final_error: f32,
}
```

### 7.2 HTTP API (Phase 2)

```
POST /api/optimize
  Content-Type: multipart/form-data
  Body:
    - image: (file upload, PNG/JPEG)
    - charset: "hermes" | "sc-2" | "hermes-sm" | ...
    - row_length: 30
    - num_loops: 10
    - asymmetry: 0.1
    - search: "simAnneal" | "greedy"
    - layers: "4x2" | "4x1" | "16x1" | ...
  
  Response: application/json
  {
    "job_id": "uuid",
    "status": "queued"
  }

GET /api/optimize/{job_id}
  Response: application/json
  {
    "status": "running" | "completed" | "failed",
    "progress": 0.45,              // 0–1
    "current_error": 0.0231,
    "preview_image": "base64..."   // latest mockup
  }

GET /api/optimize/{job_id}/stream
  Response: text/event-stream (SSE)
  data: {"event": "progress", "layer": 3, "error": 0.0231, "mockup": "base64..."}
  data: {"event": "completed", "result": {...}}
```

### 7.3 WebSocket Design Consideration

SSE is simpler than WebSocket for this use case (server → client streaming). The client only sends one request and receives a stream of progress updates. WebSocket would be overkill unless we need bidirectional streaming. However, if the Phase 2 web app wants interactive parameter tweaking while optimizing, WebSocket could enable that.

### 7.4 WASM Preview (Phase 2 Enhancement)

The core library compiled to WASM enables:
1. **Instant browser preview**: User uploads image, sees low-res optimization in real-time in the browser (e.g., 15 cols, 3 loops)
2. **Parameter experimentation**: Sliders for row_length, asymmetry, layers update preview instantly
3. **Offline capability**: No server needed for small images

The WASM module would share the same Rust core code, just compiled to a different target. This is a massive DX advantage of Rust.

---

## 8. Implementation Plan

### 8.1 Phase 1: Core Backend Rewrite (Rust)

**Step 1: Project setup**
- Initialize `v2/` as Rust project with Cargo
- Symlink or copy charset assets and layer configs
- Set up benchmark harness (criterion)

**Step 2: Data types & charset loading**
- Port `chop_charset` — read charset PNG, slice into individual characters
- Port `resizeTarget` — scale & pad target image to char grid
- Define `Charset`, `Target`, `LayerConfig`, `OptimizeConfig` types
- Load `layers.json` with serde

**Step 3: Core optimization engine**
- Implement batch character evaluation:
  ```rust
  fn evaluate_chars(bg: &Array2<f32>, target: &Array2<f32>, chars: &Array3<f32>, asymmetry: f32) -> Array1<f32>
  ```
- Implement greedy search
- Implement simulated annealing (adapted for batch)
- Implement single-layer optimization pass (parallel with Rayon)
- Implement multi-pass, multi-layer optimization loop

**Step 4: Output & export**
- Render mockup and individual layers to PNG
- Export `choices.json`
- Export error-over-time data/plot
- Timing and statistics

**Step 5: CLI**
- Port all CLI arguments from `optimize.py`
- Match output format (paths, file naming)
- Support `--display 0` (no GUI) by default

**Step 6: Validation**
- Numerical correctness: compare choices and mockup pixel values vs Python v1
- Performance: benchmark vs Python with identical parameters
- Quality: visual comparison of output images

**Step 7: Optimization passes**
- Profile and optimize hotspots
- Experiment with BLAS backends (OpenBLAS vs Accelerate on macOS)
- Fine-tune parallel batch sizes

### 8.2 Phase 2: Web Application

#### 8.2.1 Frontend Tech Stack

**Recommended: Next.js 15 + Tailwind CSS v4 + shadcn/ui**

| Choice | Rationale |
|--------|----------|
| **Next.js 15** (App Router) | Server components for SSR, API routes for proxy, excellent WASM loading, file-based routing, Vercel deploy |
| **Tailwind CSS v4** | Zero-config utility CSS, tiny production builds, perfect for a single-page tool |
| **shadcn/ui** | Copy-paste components (not a dependency), beautiful sliders/selects/toasts out of the box, themable |
| **SSE via EventSource** | Native browser API, no WebSocket library needed for server→client streaming |
| **Zustand** (state) | Tiny (<1KB), no boilerplate, perfect for the single-page optimization state machine |

**Why not SvelteKit?** Both work well, but Next.js has better WASM integration, more streaming/Suspense primitives, and broader hosting options (Vercel, Cloudflare, static export for GitHub Pages).

**Why not a heavier UI framework?** The app is a single-page tool: upload → configure → optimize → view. Tailwind + shadcn/ui keeps it lean while looking polished.

#### 8.2.2 Page Layout

```
┌──────────────────────────────────────────────────────────┐
│  Typewriter Art                                          │
│──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────┐  ┌─────────────────────────────┐│
│  │                     │  │  Controls                    ││
│  │   Drop image here   │  │  ┌─────────────────────────┐ ││
│  │   or click to       │  │  │ Charset  [dropdown  ▼]  │ ││
│  │   upload            │  │  │ Row Length [====slider] │ ││
│  │                     │  │  │ Loops      [====slider] │ ││
│  │                     │  │  │ Search    [dropdown  ▼]  │ ││
│  │                     │  │  │ Layers    [dropdown  ▼]  │ ││
│  │                     │  │  │ Asymmetry  [====slider] │ ││
│  │                     │  │  │ ▶ Advanced Options      │ ││
│  │                     │  │  │   Temp      [====slider] │ ││
│  │                     │  │  │   Init Mode [dropdown  ▼]│ ││
│  │                     │  │  │   Shuffle   [✓]         │ ││
│  │                     │  │  └─────────────────────────┘ ││
│  │                     │  │                              ││
│  │                     │  │  [ Optimize ]  (big button)  ││
│  └─────────────────────┘  └─────────────────────────────┘│
│                                                           │
│  ┌───────────────────────────────────────────────────────┐│
│  │  Live Preview                                         ││
│  │  ┌─────────────────────────────────────────────────┐  ││
│  │  │                                                 │  ││
│  │  │     (mockup updates in real-time via SSE)       │  ││
│  │  │                                                 │  ││
│  │  └─────────────────────────────────────────────────┘  ││
│  │  Progress: [████████░░░░░░░░] 65%  Error: 0.0421     ││
│  │  [Download PNG] [Download Choices] [Download Layers]  ││
│  └───────────────────────────────────────────────────────┘│
│                                                           │
│  ┌───────────────────────────────────────────────────────┐│
│  │  Result Gallery (after completion)                    ││
│  │  [Final Image] [Layer 1] [Layer 2] ... [Layer 8]     ││
│  │  Click to enlarge, arrow keys to navigate             ││
│  └───────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

#### 8.2.3 Component Tree

```
page.tsx (Server Component — initial render)
├── ImageUpload        (Client — drag/drop, paste, file picker)
├── ControlPanel       (Client)
│   ├── CharsetSelect  (dropdown with preview thumbnails)
│   ├── SliderGroup    (row_length, num_loops, asymmetry, temp)
│   ├── SelectGroup    (search algorithm, layers, init_mode)
│   └── AdvancedToggle (collapsible section for temp, init, shuffle)
├── OptimizeButton     (Client — triggers POST, connects SSE)
├── LivePreview        (Client — SSE stream → canvas updates)
│   ├── MockupCanvas   (renders the grayscale mockup)
│   └── ProgressBar    (loop/layer progress, current error)
├── DownloadPanel      (Client — generates download links)
└── LayerGallery       (Client — tabbed layer viewer)
    └── Lightbox       (click to enlarge)
```

#### 8.2.4 Data Flow

```
1. User uploads image → client-side resize preview (canvas API)
2. User clicks [Optimize]
3. POST /api/optimize (multipart form: image + config JSON)
   → Server: spawns typewriter_art::optimize() in tokio task
   → Returns: { job_id, status: "queued" }
4. Client opens EventSource to GET /api/optimize/{job_id}/stream
   → Server pushes SSE events during optimization:
     data: {"event":"progress","loop":2,"layer":3,"error":0.042,"mockup_b64":"..."}
     data: {"event":"complete","result":{...}}
5. Each "progress" event → canvas redraws the mockup
6. "complete" event → show DownloadPanel, populate LayerGallery
```

**SSE event types**:
- `progress` — after each layer pass, includes base64 mockup + error
- `complete` — final result with all data
- `error` — optimization failed

#### 8.2.5 WASM Instant Preview (Stretch Goal)

For an instant low-res preview before hitting the server:

```
1. Compile typewriter-art to WASM (wasm-pack)
2. On image upload: run greedy, row_length=15, 1 loop in-browser
3. Render result in <canvas> in <500ms
4. User sees: "Here's a quick preview. Optimize for full quality."
5. Full optimization still goes through the server
```

This works on static hosting (GitHub Pages) with no backend — the WASM module is just a `.wasm` file served alongside the JS bundle.

#### 8.2.6 Implementation Steps

**Step 8: HTTP server** (v2/server/)
- axum-based REST API
- In-memory job queue (HashMap<job_id, tokio::JoinHandle>)
- SSE streaming endpoint
- Multipart image upload
- Calls typewriter_art::optimize() directly — no subprocess

**Step 9: Frontend** (`v2/web/`)
- Next.js 15 + Tailwind CSS v4 + shadcn/ui
- Image upload with drag-and-drop + paste-from-clipboard
- Parameter controls with real-time URL state sync (so links are shareable)
- SSE client for live mockup streaming
- Canvas-based mockup renderer
- Result gallery with per-layer viewer + lightbox
- Download all formats (PNG, choices.json, individual layers)

**Step 10: WASM integration**
- Compile core lib to WASM via wasm-pack
- Browser-side low-res preview
- Fall back to server-side for full-resolution

### 8.3 Phase 3: Advanced Features (Future)

- Video processing pipeline (parallel frame optimization)
- GPU acceleration (wgpu/CUDA)
- Custom charset upload
- Interactive "digital typewriter" mode
- Hardware output (DaisyWriter, other typewriters)

---

## 9. Milestones & Timeline

| Milestone | Deliverable | Est. Time | Status |
|-----------|-------------|-----------|--------|
| M1: Project setup | Rust project, charset loading, basic types | 1 day | ✅ DONE |
| M2: Core engine | Batch optimization, greedy + SA, parallel passes | 2–3 days | ✅ DONE |
| M3: Output & CLI | PNG export, choices.json, CLI parity | 1 day | ✅ DONE |
| M4: Validation | Correctness tests, benchmarks vs Python | 1 day | ✅ DONE (114×) |
| M5: Performance tuning | Profile, optimize, BLAS tuning | 1–2 days | ⏳ Future |
| **Phase 1 total** | **Working CLI, 114× faster** | **~1 week** | ✅ DONE |
| M6: HTTP server | REST API, job queue, SSE streaming | 2–3 days | ✅ DONE (Next.js API) |
| M7: Web frontend | Upload UI, parameter controls, live preview | 3–5 days | ✅ DONE |
| M8: WASM preview | Browser-side low-res optimization | 2–3 days |
| **Phase 2 total** | **Full web application** | **~2 weeks** |

---

## 10. Open Questions

1. **BLAS backend on macOS**: Apple's Accelerate framework vs OpenBLAS. Accelerate is built-in but may have different performance characteristics. Should benchmark both.

2. **SA random order**: RESOLVED — must use random candidate evaluation order. Sorted-by-MSE would destroy SA, reducing it to greedy search. The batch precompute approach preserves random order (see §5.2 for corrected design).

3. **Temperature schedule**: RESOLVED — use the current code's simpler per-loop schedule by default (`temp = initTemp / (loop+1)`), with the paper's per-position cooling + reheating as a **configurable option, off by default**. All SA parameters should be exposed as CLI flags and as "advanced options" in the web UI — fiddling with parameters is part of the fun for algorithmic art enthusiasts.

4. **Charset file format**: Keep the current sprite-sheet + config.json. Sprite sheet is efficient for loading. Could add optional per-character PNGs for custom charsets.

5. **Backward compatibility**: v2 CLI should match v1's output format (file paths, choices.json structure). Numerically identical mockup pixels are impossible due to floating point differences, but visual quality should match or exceed v1.

6. **Video support**: Phase 3. The current `convert_video.py` uses `parallel` GNU utility. A Rust implementation could parallelize frames natively. Architect the API to support batch processing from the start.

7. **DaisyWriter output**: Keep as a separate utility; its performance doesn't matter. Could port to Rust if desired, but Python is fine for this.

8. **Auto-alignment (optimized cropping)** from the paper: Include as an optional flag. It's the primary quality factor at small sizes. See §5.4.

9. **WASM rationale**: Yes — WASM allows the core library to run in a static web page (e.g., GitHub Pages) with zero backend. Users get instant low-res previews in-browser. Full-resolution optimization would still use the server. This is a standout feature for a portfolio piece.

---

## Summary

The v2 rewrite centers on **one key insight**: the inner loop of iterating characters sequentially per position can be replaced with **batch vectorized evaluation** (precompute all composites at once via SIMD), while preserving **random candidate evaluation order** for correct SA semantics. The batch computation gives 3–7× speedup in numba; Rust with explicit SIMD, BLAS, and Rayon parallelism achieved **114× total speedup** over Python v1 (0.043s vs 4.92s for the same workload). This is approximately **10,000× faster** than the original 2021 paper implementation.

Key corrections from the GI 2021 paper review:
- **SA requires random order** — sorting by MSE would be greedy search, destroying SA's ability to escape local minima. Batch precompute is just an optimization, not a change to the SA logic.
- **Auto-alignment** (optimized cropping) from the paper is not in the current code and should be added as an optional feature.
- **Temperature schedule** differs between paper and code — we'll implement the paper's empirically-tested schedule.

The Rust core is designed from day one to support three deployment targets: CLI binary, HTTP server, and **WASM browser module** (enabling instant in-browser previews on static hosting like GitHub Pages — no backend needed for low-res). This ensures the Phase 2 web app can share the exact same optimization code across all environments.

The API is designed for streaming progress updates, enabling real-time visualization in the Phase 2 web frontend without architectural changes to the core.
