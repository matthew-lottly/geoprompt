import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const previewUrl = "http://127.0.0.1:4173";
const outputPath = path.join(repoRoot, "assets", "dashboard-live-screenshot.png");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";


function runCommand(args) {
  return new Promise((resolve, reject) => {
    const child = spawn([npmCommand, ...args].join(" "), [], {
      cwd: repoRoot,
      stdio: "inherit",
      shell: true,
    });

    child.on("exit", (code) => {
      if (code === 0) {
        resolve();
        return;
      }

      reject(new Error(`Command failed: ${args.join(" ")}`));
    });

    child.on("error", reject);
  });
}


function startPreviewServer() {
  return spawn(`${npmCommand} run preview -- --host 127.0.0.1 --port 4173`, [], {
    cwd: repoRoot,
    stdio: "pipe",
    shell: true,
  });
}


async function waitForPreview() {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      const response = await fetch(previewUrl);
      if (response.ok) {
        return;
      }
    } catch {
      // Preview server is still starting.
    }

    await delay(1000);
  }

  throw new Error("Preview server did not become available at http://127.0.0.1:4173.");
}


async function captureScreenshot() {
  const browser = await chromium.launch({ headless: true });

  try {
    const page = await browser.newPage({
      viewport: { width: 1480, height: 1320 },
      deviceScaleFactor: 1.5,
    });

    await page.goto(previewUrl, { waitUntil: "networkidle" });
    await page.waitForSelector(".map-surface");
    await page.locator("label select").nth(0).selectOption("West");
    await page.locator("label select").nth(1).selectOption("review");
    await page.waitForTimeout(1500);
    await page.locator(".page-shell").screenshot({ path: outputPath });
  } finally {
    await browser.close();
  }
}


async function main() {
  await runCommand(["run", "build"]);

  const previewServer = startPreviewServer();

  previewServer.stdout.on("data", (chunk) => {
    process.stdout.write(chunk);
  });

  previewServer.stderr.on("data", (chunk) => {
    process.stderr.write(chunk);
  });

  try {
    await waitForPreview();
    await captureScreenshot();
  } finally {
    previewServer.kill();
  }
}


main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});