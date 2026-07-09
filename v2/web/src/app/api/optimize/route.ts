import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { writeFile, mkdir, readFile, rm } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";

import type { OptimizeConfig, OptimizeResult } from "@/lib/types";

/**
 * Build the CLI argument array from an OptimizeConfig + image path.
 */
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
    "--json",
  ];
  if (!config.shuffle) {
    args.push("--no-shuffle");
  }
  return args;
}

/**
 * Find the Rust binary. Looks for:
 * 1. ../target/release/optimize (relative to web/ dir)
 * 2. The project root's target/release/optimize
 */
function findBinary(): string {
  const webDir = process.cwd();

  // When running "next dev", CWD is v2/web/
  const relativePath = path.join(webDir, "..", "target", "release", "optimize");
  if (existsSync(relativePath)) return relativePath;

  // Fallback: try with absolute resolution
  throw new Error(
    `Rust binary not found at ${relativePath}. ` +
    `Please run "cargo build --release" in the v2/ directory first.`
  );
}

/**
 * Convert a Request body with FormData into an image Blob + config object.
 */
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

  // Use the original filename (or fall back to .png)
  const filename = imageFile.name || "input.png";

  return { imageBuffer, filename, config };
}

export async function POST(request: Request) {
  const jobId = crypto.randomUUID().slice(0, 8);
  const v2Root = path.join(process.cwd(), ".."); // v2/ dir
  const uploadsDir = path.join(v2Root, "uploads");
  const resultsDir = path.join(v2Root, "results", "jobs", jobId);

  try {
    // Ensure directories exist
    await mkdir(uploadsDir, { recursive: true });
    await mkdir(resultsDir, { recursive: true });

    // Parse the request
    const { imageBuffer, filename, config } = await parseFormData(request);

    // Save the uploaded image, preserving original extension
    const ext = path.extname(filename) || ".png";
    const imagePath = path.join(uploadsDir, `${jobId}${ext}`);
    await writeFile(imagePath, imageBuffer);

    // The image path relative to v2/ dir (binary's CWD)
    const relativeImagePath = `uploads/${jobId}${ext}`;
    const relativeOutDir = `results/jobs/${jobId}`;

    // Build args and find binary
    const binary = findBinary();
    const args = buildArgs(config, relativeImagePath, relativeOutDir);

    console.log(`[optimize ${jobId}] Spawning: ${binary} ${args.join(" ")}`);

    // Run the optimizer
    const result = await runBinary(binary, args, v2Root);

    console.log(`[optimize ${jobId}] Completed in ${result.stats.total_time_ns / 1e6}ms`);

    return NextResponse.json(result);
  } catch (err) {
    console.error(`[optimize ${jobId}] Error:`, err);
    return NextResponse.json(
      { error: String(err) },
      { status: 500 }
    );
  }
}

/**
 * Spawn the Rust binary and collect its JSON output.
 */
function runBinary(
  binary: string,
  args: string[],
  cwd: string
): Promise<OptimizeResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    child.on("error", (err) => {
      reject(new Error(`Failed to start optimizer: ${err.message}`));
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Optimizer exited with code ${code}\n${stderr}`));
        return;
      }

      try {
        const result = JSON.parse(stdout.trim()) as OptimizeResult;
        resolve(result);
      } catch (err) {
        reject(new Error(`Failed to parse optimizer output: ${err}\nstdout: ${stdout.slice(0, 500)}`));
      }
    });
  });
}
