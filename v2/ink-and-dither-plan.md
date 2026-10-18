# Plan: ink-faithful compositing + dithering for light tones

Two quality improvements, aimed at opposite ends of the tonal range:

1. **Saturating ink compositing** — overstrikes can't reach black (dark end)
   — **implemented** (advanced option: `--ink-floor`, web Advanced panel)
2. **Dither options** — light tones need spatial tone, not just overlap (light end)

They partition the range cleanly: dark regions get their control from overlap
(the new ceiling fixes its realism), light regions get dithering, the middle is
already served by the existing machinery.

---

## What the current model does (and where it misleads)

The paper's choice (gi-paper: *"We simulated the effect of overlapping
characters with multiplicative compositing"*) and ours: mockup brightness =
product of layer brightnesses. `evaluate_all_chars` scores candidates with
`composite = bg * char`, `combine_layers` multiplies the layers, and that is
the whole model.

Measured against the sc-2 scan (default charset, "Smith-Corona, 2 weights",
326 non-blank glyphs):

| fact | value | consequence |
|---|---|---|
| darkest single-strike pixel | **0.102** (median char min: 0.329) | one strike tops out at mid-grey |
| product model, 8 overstrikes of a 0.5 pixel | **0.0039** (≈ black) | unreachable — real typing saturates |
| per-char darkest pixels | families at ~0.2 / 0.3 / 0.4 … | the scan already contains 2 strike-force weights |
| total-ink quantiles of glyphs | p5 0.007, p25 0.015, p50 0.034, p75 0.10 | coarse steps in the light band |

So two concrete failures:

- **Over-promised darkness.** The preview says "black", the typewriter prints
  grey. The optimizer also *chases* impossible darkness — piling 8-strike
  stacks into dark targets when that ink would be better spent on shape.
- **Flat light tones.** For a tile at 4 % ink, per-cell MSE prefers blank
  (MSE ≈ 4·10⁻⁴) over scattering the lightest mark (MSE ≈ 3·10⁻³, ~8× worse)
  even though a scattered pattern matches the tone and blank does not. The
  gradient vanishes. MSE *pointwise* cannot dither; the benefit of dithering
  lives in the **local mean**, which the current objective never looks at.

---

## Proposal 1 — Saturating ink compositing ("ink floor")

### Model

Replace the product with a product **in floor-shifted whiteness**:

```
τ(c) = clamp((c − f) / (1 − f), 0, 1)        // "effective whiteness", precomputed per char
composite = f + (1 − f) · Π τ(layers)
```

Properties (all verified algebraically):

- **Single strikes are exact**: one layer → composite = c. The model changes
  *only* overlap behaviour; scanned glyph fidelity is untouched.
- **Overstrikes saturate at `f`**: k strikes of value c →
  `f + (1−f)·((c−f)/(1−f))^k` — exponential approach to the floor, never black.
- **`f = 0` reproduces today's model exactly** — a strict generalization, so it
  is trivially A/B-able and safe to roll out.

Example, a 0.5-grey pixel overstruck (f = 0.10):

| strikes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| product (today) | 0.50 | 0.25 | 0.063 | **0.004** |
| saturating | 0.50 | 0.278 | 0.135 | **0.101** |

### Default floor: measured, not guessed

`--ink-floor auto` (default) = **the darkest solid-ink pixel in the charset
scan** = the ribbon's ink density. Physical story: typing more ink can only
raise a pixel's *coverage* up to 100 %; brightness then bottoms out at the ink
material's own density — and the darkest fully-inked pixel in the scan is
exactly that measurement. For sc-2: f = 0.102. Charsets scanned from lighter
ribbons get lighter floors automatically, and the constraint *f ≤ every
single-strike pixel* (required for single-strike exactness) holds by
construction.

Overridable: `--ink-floor <0..1>` (calibrated value) and optionally
`--ink-floor-scale <k>` (e.g. 0.9 for a safety margin).

### Calibration worksheet (concrete deliverable)

Feel < measurement. A small generator emits a typing sheet: "strike key × at
the marked spots 1…8 times", plus a measurement script that scans it and fits
`f` by least squares on `b_k = f + (1−f)((b₁−f)/(1−f))^k`. One minute at the
typewriter, once per ribbon. (Also naturally detects the weight families and
can later feed per-weight floors.)

### Code changes (choke points, small)

| file | change |
|---|---|
| `chars.rs::evaluate_all_chars` | precomputed `τ(chars)`; `composite = f + (1−f)·bg_t·τ(c)` (one fma/pixel) |
| `render.rs::combine_layers`, `compute_background` | map layers through `τ` before the product (backgrounds become τ-space products) |
| `search.rs` greedy/SA composite builders | share one `composite(bg_t, char_t, f)` helper |
| `align.rs::quick_mockup` | inherits it (uses the same three functions) |
| `types.rs`, CLI, `web/src/lib/types.ts` | `ink_floor: auto \| f`, plumbing |

Layers/exports stay in brightness space; only compositing works in τ-space.
Cost: negligible (τ precomputed once per charset; the hot loop gains one fma).

Notes: dark targets below the floor become irreducible error — honest, and the
asymmetric loss already prefers getting as dark as possible there. Document as
a deliberate deviation from the paper in the README. Optional later: per-char
floors (mixed weights), misregistration softening (jitter blur on re-strikes —
breaks per-pixel decomposability, so it is future work).

---

## Proposal 2 — Dither options for the light end

### Why dithering must be explicit

As shown above, per-cell MSE ranks blank ≫ scattered marks for any target
lighter than the lightest mark. Two things are therefore needed:

1. a **spatial pattern decision** (which cells carry a mark), and
2. an **objective that sees the local mean**, so the pattern *survives* the
   optimizer instead of being erased by blank.

### Three phases (A+B recommended first)

**A — Halftone seeding.** For cells whose target ink density is in the light
band (`--dither-band 0…0.15`), decide mark/no-mark from the target density with
a chosen pattern:

- `bayer4` / `bayer8` — ordered, regular. **Recommended default**: the output
  is typed by hand and regular grids are followable on a real typewriter.
- `bluenoise` — organic, best-looking, hard to type manually.
- `fs` — Floyd–Steinberg error diffusion, natural but irregular.

Mark character: `--dither-mark auto` (the lightest non-blank glyph, or the
glyph whose ink best matches one "dither step") or an explicit index. Seeds are
placed per (layer, cell) — the light band typically uses one layer only.

**B — Low-frequency flip refinement (the piece that makes dithering survive).**
After the main loops, box-blur target and mockup (σ ≈ 1–1.5 char cells, integral
images), then greedily flip light-band positions between {blank, light marks},
accepting any flip that improves the **blurred** error; 2–3 passes
(`--dither-refine`). Each flip's delta is O(1) via integral images; the pass is
parallel over positions with sequential application (same pattern as
`optimize_layer_pass`). This is a bounded finishing stage — the fast main loop
is untouched.

**C — (optional) unified objective.** Fold a `λ · MSE(blur(target),
blur(mockup))` term into `evaluate_all_chars` (halo'd tiles). Most principled,
most work; only if A+B underdeliver.

### Config surface

```
--dither none|bayer4|bayer8|bluenoise|fs     (default: bayer4? or none initially)
--dither-band 0.0-0.15                       (target ink density treated as "light")
--dither-refine 2                            (LF flip passes; 0 = seed only)
--dither-mark auto|<char index>
```

plus web dropdown/sliders. No export changes: `choices.json` already carries
every typed mark.

### Interactions and notes

- Light band = few overstrikes → the ink floor is irrelevant there; dark band =
  dither is irrelevant (overlap is the control). The features don't interfere.
- Alignment sweep stays on undithered quick mockups (speed; dithering is a
  finishing stage on the final run).
- The paper's own answer to light tones was **strike force** (*"Varying the
  strike force produces a much smoother tonal range … especially in
  lighter-tone regions"*). The scans already contain 2 weights; dithering adds
  spatial tone where even the lightest weight is too heavy. More scanned
  weights (the force axis) would complement dither later.

---

## Validation

**Ink model**
- Unit test: k = 1…8 overstrikes of a grey pixel converge to `f`, single
  strike unchanged; `f = 0` bit-identical to today.
- mwdog_crop @ 4×2: no mockup pixel below the floor; side-by-side vs old
  render (dark regions visibly "greyer" = truthful); dark-region error rises
  slightly — expected and honest.

**Dither**
- Synthetic light ramp (white → 15 % ink): before = flat/stepped, after =
  smooth perceived gradient. Quantify with **LF-PSNR** (blur σ = 2) — must
  improve markedly in the light band.
- Visual check on a portrait's light background; confirm the pattern reads as
  tone, not objects.

**Perf**: ink model ≈ 0 overhead; dither post-pass < 1 s. Bench on sc-2/row-30
against the current 7.4 s sweep.

## Suggested order

1. **Ink floor** — small, self-contained, immediately fixes previews and
   optimizer behavior; with `auto` default, unit tests, README deviation note.
2. **Dither A+B** — new `dither.rs` module (density field, patterns, LF flip
   pass) + config + web controls.
3. Optional: calibration worksheet generator, Phase C, per-weight floors.
