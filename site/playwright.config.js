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
    // Never reused, not even locally. `reuseExistingServer: !CI` sounds like a
    // developer convenience and behaves like a trap: a `vite preview` left
    // running from an earlier session keeps answering on 4173, so the whole
    // suite runs against whatever bundle that process built — days old, and
    // silently. The run goes green and proves nothing about the working tree,
    // which is the worst outcome a test suite has.
    //
    // The cost is that a leftover preview now collides instead of being used:
    // `--strictPort` makes vite refuse, and the fix is to stop the process on
    // 4173. A suite that will not start beats one that starts and lies.
    //
    // Set ALLOW_STALE_PREVIEW=1 to opt back in when you know the server on 4173
    // is yours and current. Naming it that way means whoever turns it on has
    // read what they are turning on. Never honoured in CI, where "the server
    // someone left running" is not a thing that should exist.
    reuseExistingServer: !process.env.CI && process.env.ALLOW_STALE_PREVIEW === '1',
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
