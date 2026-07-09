# Typewriter Art v2

Fast multi-layer ASCII art optimizer — **114× faster** than the Python v1.

Converts grayscale photographs into typed characters where multiple layers of typing overlap with sub-character offsets to create continuous tone through multiplicative ink accumulation.

## Quick Start

```bash
# 1. Make sure Rust is installed (>= 1.74)
rustc --version

# 2. Basic demo (hermes-sm charset, 5 loops, simulated annealing)
cargo run --release -- \
    --charset hermes-sm \
    --target images/mwdog_crop.png \
    --row-length 20 \
    --num-loops 5 \
    --search simAnneal \
    --init-mode random \
    --layers 4x2 \
    --temp 0.001

# 3. Fast preview (greedy, 1 loop)
cargo run --release -- \
    --charset hermes-sm \
    --target images/mwdog_crop.png \
    --row-length 20 \
    --num-loops 1 \
    --search greedy \
    --layers 4x2

# 4. View the result
open results/final.png
```

**Output** (in `results/`):
- `final.png` — the final mockup image
- `layer_*.png` — individual layer images
- `choices.json` — character assignments per layer
- `error_over_time.csv` — error after each optimization pass
- `stats.txt` — timing and comparison statistics

## CLI Reference

```
optimize --charset <CHARSET> --target <TARGET> [OPTIONS]

Options:
  -c, --charset <CHARSET>    Charset folder (default: sc-2)
                             Options: hermes, hermes-sm, hermes-md,
                                      sc-1, sc-2, sc-3, sc-round,
                                      daisywriter, daisywriter-shrunken, dw-no-chev
  -t, --target <TARGET>      Target image path (default: images/mwdog_crop.png)
  -r, --row-length <N>       Characters per row (default: 30)
  -n, --num-loops <N>        Optimization passes per layer (default: 15)
  -i, --init-mode <MODE>     random | blank (default: random)
  -a, --asymmetry <FLOAT>    MSE asymmetry, penalizes too-light (default: 0.1)
  -s, --search <ALGO>        simAnneal | greedy (default: simAnneal)
      --temp <FLOAT>         SA initial temperature (default: 0.001)
  -l, --layers <KEY>         Layer config from layers.json (default: 4x2)
                             Options: 1x1, 4x1, 4x2, 16x1, daisy_full, ...
  -o, --out-file <PATH>      Output PNG path (default: results/final.png)
  -h, --help                 Show help
```

## Performance

| Config | Python v1 | Rust v2 | Speedup |
|--------|-----------|---------|---------|
| hermes-sm, row=20, 2 loops, greedy, 8 layers | 4.92 s | 0.043 s | **114×** |
| hermes-sm, row=20, 2 loops, SA, 8 layers | ~5 s | 0.034 s | **~147×** |
| hermes-sm, row=20, 5 loops, SA, 8 layers | ~12 s | 0.081 s | **~148×** |

Time per comparison: ~88 µs (Python) → ~0.08 µs (Rust) = **~1100×** faster per op.

## Architecture

```
v2/
├── Cargo.toml              # Workspace root
├── assets/                  # Symlinks to src/assets/ (charsets, layers.json, images)
├── lib/                     # Core library crate (typewriter-art)
│   └── src/
│       ├── lib.rs           # Public API, re-exports
│       ├── optimize.rs      # Main optimization loop (orchestration)
│       ├── search.rs        # Greedy & SA search with batch evaluation
│       ├── chars.rs         # Batch character evaluation (key perf trick)
│       ├── render.rs        # Layer composition, background computation
│       ├── charset.rs       # Charset loading from sprite sheets
│       ├── resize.rs        # Target image preprocessing
│       ├── layers.rs        # Layer offset configuration
│       ├── export.rs        # Output: PNG, JSON, CSV, stats
│       └── types.rs         # Config, Result, enums
└── cli/                     # CLI binary crate
    └── src/main.rs          # clap argument parsing, entry point
```

### Key Design Decisions

**Batch character evaluation** — the core performance trick. Instead of computing composites one character at a time, all `C` characters for a position are evaluated in dense SIMD-friendly loops. Speedup: ~10× just from this.

**SA with random candidate order preserved** — all MSEs are precomputed in batch, then candidates are iterated in random order with standard SA acceptance logic (accept first qualifying candidate). Correct SA semantics, batch speed.

**Coordinate descent** — alternating layer optimization: optimize one layer while holding all others fixed as the "background." Matches the current Python implementation and parallelizes well.

**Rayon parallelism** — positions within a layer pass run in parallel. The library target is `lib` (not `bin`), so the web server can call `typewriter_art::optimize()` directly.

## Using as a Library (for Phase 2 Web Server)

```rust
use typewriter_art::{optimize, OptimizeConfig, InitMode, SearchAlgorithm};

let config = OptimizeConfig {
    charset: "hermes-sm".into(),
    target: "images/mwdog_crop.png".into(),
    row_length: 20,
    num_loops: 2,
    init_mode: InitMode::Random,
    asymmetry: 0.1,
    search: SearchAlgorithm::SimAnneal,
    init_temp: 0.001,
    layers: "4x2".into(),
    shuffle: true,
    out_file: "results/final.png".into(),
};

let result = optimize(config)?;

// Access results
println!("Final error: {}", result.stats.final_error);
println!("Time: {} ms", result.stats.total_time_ns as f64 / 1e6);

// result.mockup: ndarray::Array2<f32> — the final grayscale mockup [0.0, 1.0]
// result.layers: ndarray::Array3<f32> — individual layers
// result.choices: ndarray::Array2<u16> — character assignments
// result.error_over_time: Vec<f32> — after each layer pass
```

The library exposes everything needed for the Phase 2 web server:
- `OptimizeConfig` — all parameters
- `OptimizeResult` — all outputs
- Export functions: `save_mockup()`, `save_layers()`, `save_choices()`, etc.
- All internal modules are `pub` for advanced use

## Charsets

9 charsets available (assets symlinked from v1):

| Charset | Chars | Size | Notes |
|---------|-------|------|-------|
| hermes-sm | 88 | 18×10 | Smallest, fastest |
| hermes-md | 88 | 28×16 | Medium |
| hermes | 88 | original | Full size |
| sc-1 | 87 | 50×26 | Single-case |
| sc-2 | 175 | 52×26 | Default, good quality |
| sc-3 | 265 | 50×26 | Largest, best quality |
| daisywriter* | various | various | For DaisyWriter hardware |

## Layers

Pre-configured layer offsets in `assets/layers.json`:

| Key | Layers | Description |
|-----|--------|-------------|
| 1x1 | 1 | Single layer, no overlap |
| 4x1 | 4 | Quarter-char offsets in both axes |
| 4x2 | 8 | 4×2 doubled density (default) |
| 16x1 | 16 | 4×4 grid of fractional offsets |
| daisy_full | 40 | Maximum density for DaisyWriter |

## Web App (Next.js)

```bash
# 1. Build the Rust binary first
cd v2 && cargo build --release

# 2. Install frontend dependencies
cd web && npm install

# 3. Start dev server
npm run dev

# 4. Open http://localhost:3000
```

The web app provides:
- Drag-and-drop or paste image upload
- Parameter controls (charset, layers, algorithm, row length, loops, asymmetry)
- Advanced options (init mode, temperature, shuffle)
- Live optimization with result display (canvas-based mockup renderer)
- Layer gallery with tabbed viewer and lightbox
- Downloads: final PNG, choices JSON, individual layer PNGs

Architecture: Next.js API route spawns the Rust `optimize` binary as a subprocess with `--json` flag, reads the JSON result (including base64-encoded images), and returns it to the browser. Total round-trip: ~150ms.
