import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { Readable } from "stream";
import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import crypto from "crypto";

import type { OptimizeConfig } from "@/lib/types";

// This route spawns the optimizer binary and streams progress.
export const runtime = "nodejs";

function buildArgs(config: OptimizeConfig, imagePath: string, outDir: string): string[] {
  const args = [
    "--charset", config.charset,
    "--target", imagePath,
    "--row-length", String(config.row_length),
    "--num-loops", String(config.num_loops),
    "--init-mode", config.init_mode,
    "--asymmetry", String(config.asymmetry),
    "--search", config.search === "simAnneal" ? "simAnneal" : "greedy",
    "--temp", String(config.init_temp),
    "--layers", config.layers,
    "--out-dir", outDir,
    "--stream",
  ];
  if (config.shuffle === false) args.push("--no-shuffle");
  // Older persisted configs lack these fields; default to on/soft/no-deluxe.
  if (config.align_enabled === false) args.push("--no-align");
  if (config.align_deluxe === true) args.push("--align-deluxe");
  args.push("--center-focus", config.center_focus ?? "soft");
  args.push("--ink-floor", config.ink_floor === undefined ? "auto" : String(config.ink_floor));
  args.push("--dither", config.dither_mode ?? "none");
  args.push("--dither-band", String(config.dither_band ?? 0.15));
  args.push("--dither-refine", String(config.dither_refine ?? 2));
  return args;
}

function findBinary(): string {
  const webDir = process.cwd();
  const exe = process.platform === "win32" ? "optimize.exe" : "optimize";
  const relativePath = path.join(webDir, "..", "target", "release", exe);
  if (existsSync(relativePath)) return relativePath;
  throw new Error(
    `Rust binary not found at ${relativePath}. ` +
    `Please run "cargo build --release" in the v2/ directory first.`
  );
}

async function parseFormData(req: Request): Promise<{
  imageBuffer: Buffer;
  filename: string;
  config: OptimizeConfig;
}> {
  const formData = await req.formData();
  const imageFile = formData.get("image");
  if (!imageFile || !(imageFile instanceof File)) {
    throw new Error("Missing or invalid 'image' field in FormData");
  }
  const configStr = formData.get("config");
  if (!configStr || typeof configStr !== "string") {
    throw new Error("Missing or invalid 'config' field in FormData");
  }
  const imageBuffer = Buffer.from(await imageFile.arrayBuffer());
  const config: OptimizeConfig = JSON.parse(configStr);
  const filename = imageFile.name || "input.png";
  return { imageBuffer, filename, config };
}

/**
 * POST /api/optimize — accepts image + config, returns SSE stream of progress events.
 */
export async function POST(request: Request) {
  const jobId = crypto.randomUUID().slice(0, 8);
  const v2Root = path.join(process.cwd(), "..");
  const uploadsDir = path.join(v2Root, "uploads");
  const resultsDir = path.join(v2Root, "results", "jobs", jobId);

  try {
    await mkdir(uploadsDir, { recursive: true });
    await mkdir(resultsDir, { recursive: true });

    const { imageBuffer, filename, config } = await parseFormData(request);

    const ext = path.extname(filename) || ".png";
    const imagePath = path.join(uploadsDir, `${jobId}${ext}`);
    await writeFile(imagePath, imageBuffer);

    const relativeImagePath = `uploads/${jobId}${ext}`;
    const relativeOutDir = `results/jobs/${jobId}`;
    const binary = findBinary();
    const args = buildArgs(config, relativeImagePath, relativeOutDir);

    console.log(`[optimize ${jobId}] Streaming: ${binary} ${args.join(" ")}`);

    const stream = createSSEStream(binary, args, v2Root, jobId);
    return new Response(Readable.toWeb(stream) as ReadableStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });

  } catch (err) {
    console.error(`[optimize ${jobId}] Error:`, err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

/**
 * Create a readable stream that converts NDJSON lines from the Rust binary
 * into SSE (Server-Sent Events) format.
 */
function createSSEStream(
  binary: string,
  args: string[],
  cwd: string,
  jobId: string
): Readable {
  const child = spawn(binary, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });

  const stream = new Readable({
    read() { /* push is called manually */ },
  });

  let buffer = "";
  let ended = false;
  function endStream() {
    if (!ended) {
      ended = true;
      stream.push(null); // end stream
    }
  }

  function emitSSE(event: string, data: string) {
    if (ended) return;
    stream.push(`event: ${event}\ndata: ${data}\n\n`);
  }

  child.stdout.on("data", (data: Buffer) => {
    buffer += data.toString();
    const lines = buffer.split("\n");
    // Keep last incomplete line in buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const parsed = JSON.parse(line);

        if (parsed.type === "complete") {
          emitSSE("complete", JSON.stringify(parsed.result));
          endStream();
        } else if (parsed.type === "align" || parsed.type === "loop") {
          // Progress event (alignment sweep or layer pass)
          emitSSE("progress", line);
        }
      } catch {
        // Skip non-JSON lines (e.g., log output to stderr)
      }
    }
  });

  child.stderr.on("data", (data: Buffer) => {
    console.error(`[optimize ${jobId}] stderr: ${data.toString().trim()}`);
  });

  child.on("error", (err) => {
    emitSSE("error", JSON.stringify({ message: err.message }));
    endStream();
  });

  child.on("close", (code) => {
    if (code !== 0 && !ended) {
      emitSSE("error", JSON.stringify({ message: `Exit code ${code}` }));
    }
    // Flush any remaining buffer
    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer);
        if (parsed.type === "complete") {
          emitSSE("complete", JSON.stringify(parsed.result));
        }
      } catch { /* ignore */ }
    }
    endStream();
  });

  return stream;
}
