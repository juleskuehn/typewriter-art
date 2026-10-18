// Browserless integration test of the actual client code (api.ts + download-zip.ts)
// against a live Next server.
//
// Usage (from v2/web):
//   npm run build && npm run start -- -p 3123   # in one shell
//   node smoke-test.mjs                         # in another
//
// Verifies: SSE progress parsing, result parsing, hang-guard on stream end,
// and the real ZIP download contents (writes the zip to /tmp/ui-test.zip).
import { readFileSync, writeFileSync } from "fs";

const realFetch = globalThis.fetch;
globalThis.fetch = (url, init) =>
  realFetch(new URL(url, "http://localhost:3123"), init);

const { runOptimization } = await import("./src/lib/api.ts");

// 1. Load a test image as a data URL (like the FileReader path in the UI)
const img = readFileSync("../assets/images/mwdog_crop.png");
const dataUrl = `data:image/png;base64,${img.toString("base64")}`;

const config = {
  charset: "sc-2",
  row_length: 10,
  num_loops: 2,
  init_mode: "random",
  asymmetry: 0.1,
  search: "simAnneal",
  init_temp: 0.001,
  layers: "1x2",
  shuffle: true,
  align_enabled: true,
  align_deluxe: true,
  center_focus: "soft",
  ink_floor: "auto",
  dither_mode: "bayer4",
  dither_band: 0.15,
  dither_refine: 2,
};

// 2. Run through the real client SSE code
const events = [];
const result = await runOptimization(dataUrl, config, (e) => events.push(e));

console.log("progress events:", events.length);
console.log(
  "align events:",
  events.filter((e) => e.type === "align").length
);
console.log("mockup previews:", events.filter((e) => e.mockup_b64).length);
console.log("errors:", events.filter((e) => e.type !== "align").map((e) => e.error));
console.log("layers:", result.layers_base64.length,
  "offsets:", JSON.stringify(result.fractional_offsets),
  "grid:", result.num_rows + "x" + result.num_cols,
  "eot:", result.error_over_time.length);
console.log("mockup png magic ok:",
  Buffer.from(result.mockup_base64, "base64").subarray(0, 4).toString("hex") === "89504e47");
console.log("stats:", JSON.stringify(result.stats));
console.log(
  "alignment:",
  result.alignment
    ? `best ${JSON.stringify(result.alignment.best)} of ${result.alignment.scores.length}` +
        (result.alignment.rerank
          ? `, reranked ${result.alignment.rerank.length}`
          : "")
    : "none"
);

// 3. Run the real ZIP download code, intercepting the browser download
globalThis.document = {
  createElement: () => ({ href: "", download: "", click() {} }),
};
let zipped = null;
const realCreate = URL.createObjectURL;
URL.createObjectURL = (blob) => { zipped = blob; return "blob:stub"; };
const { downloadZip } = await import("./src/components/download-zip.ts");
await downloadZip(result);
URL.createObjectURL = realCreate;
writeFileSync("/tmp/ui-test.zip", Buffer.from(await zipped.arrayBuffer()));
console.log("zip bytes:", zipped.size);

// 4. Negative test: stream that ends without a complete event must reject, not hang
globalThis.fetch = async () =>
  new Response(new Blob(["event: progress\ndata: {\"loop_num\":1,\"num_loops\":1,\"layer_num\":1,\"num_layers\":1,\"error\":0.5}\n\n"]), {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
try {
  await runOptimization(dataUrl, config);
  console.log("hang-guard: FAIL (resolved without complete event)");
} catch (e) {
  console.log("hang-guard: OK ->", e.message);
}
