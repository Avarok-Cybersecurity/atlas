// =============================================================================
// Playwright config for the "Ask the codebase" E2E suite.
// - webServer: production build then vite preview on its default port (4173),
//   both under bun's runtime (node on this box is v18; vite 8 needs 20+).
// - Two projects: desktop chromium and a 390x844 mobile viewport (the modal
//   turns into a full-bleed sheet at <=860px).
// - @live tests (real corpus URL / real OpenRouter key) are excluded unless
//   LIVE=1 — `bun run test:e2e:live` sets it.
// - serviceWorkers blocked: the site ships a precaching SW that would let
//   requests bypass route interception and leak cache state between tests.
// =============================================================================

import { defineConfig, devices } from '@playwright/test';

const PORT = 4173; // vite preview's default port, pinned with --strictPort

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [['list']],
  grepInvert: process.env.LIVE ? undefined : /@live/,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    serviceWorkers: 'block',
    trace: 'retain-on-failure'
  },
  webServer: {
    command: `bun x --bun vite build && bun x --bun vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 240_000
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      // The phone sheet: chromium engine, 390x844 viewport, touch on.
      name: 'mobile',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 390, height: 844 },
        hasTouch: true
      }
    }
  ]
});
