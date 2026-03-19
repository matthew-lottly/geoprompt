import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const outputPath = path.join(repoRoot, "assets", "dashboard-live-screenshot.png");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";


function getAvailablePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();

    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();

      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("Could not determine preview port.")));
        return;
      }

      server.close(() => resolve(address.port));
    });
  });
}


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


function startPreviewServer(port) {
  return spawn(`${npmCommand} run preview -- --host 127.0.0.1 --port ${port} --strictPort`, [], {
    cwd: repoRoot,
    stdio: "pipe",
    shell: true,
  });
}


async function waitForPreview(previewUrl) {
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

  throw new Error(`Preview server did not become available at ${previewUrl}.`);
}


async function captureScreenshot(previewUrl) {
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

  const port = await getAvailablePort();
  const previewUrl = `http://127.0.0.1:${port}`;

  const previewServer = startPreviewServer(port);

  previewServer.stdout.on("data", (chunk) => {
    process.stdout.write(chunk);
  });

  previewServer.stderr.on("data", (chunk) => {
    process.stderr.write(chunk);
  });

  try {
    await waitForPreview(previewUrl);
    await captureScreenshot(previewUrl);
  } finally {
    previewServer.kill();
  }
}


main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});