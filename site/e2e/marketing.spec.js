// =============================================================================
// marketing.spec.js — the marketing site, in a browser
// -----------------------------------------------------------------------------
// The unit suite proves the data is consistent (src/lib/content/site.test.js).
// This proves the things only a browser can: that menus open, that one tab
// panel shows at a time, that the calculator recomputes, that the theme
// persists, that every page renders with one h1, and that the front page makes
// no request to anyone else's server, which the Lighthouse gate also demands.
//
// Runs in both projects: chromium (desktop) and mobile (390x844). The nav has
// two implementations, a mega menu and a drawer, and each test says which one
// it is for.
// =============================================================================

import { test, expect } from '@playwright/test';
import { pages, routes } from '../src/lib/content/index.js';

const isMobile = (testInfo) => testInfo.project.name === 'mobile';

test.describe('front page', () => {
  test('says what the product is, and every call to action goes to the demo', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Avarok, the inference economics platform/);
    await expect(page.locator('h1')).toHaveCount(1);
    await expect(page.locator('h1')).toContainText('Faster inference');
    await expect(page.locator('h1')).toContainText('A fraction of what you pay today');
    await expect(page.locator('.av-hero-kicker')).toHaveText('Inference economics, reimagined.');
    await expect(page.locator('.av-hero-actions a').first()).toHaveAttribute('href', routes.demoForm);
    await expect(page.locator('.av-hero-claim')).toContainText('70% or less');
  });

  test('the hero poster is a real image and the clip attaches once it is in view', async ({ page }) => {
    await page.goto('/');
    const poster = page.locator('.av-hero .av-video img');
    await expect(poster).toBeVisible();
    expect(await poster.evaluate((img) => img.complete && img.naturalWidth)).toBeGreaterThan(600);
    await expect(page.locator('.av-hero .av-video video source[type="video/mp4"]')).toHaveAttribute('src', '/media/console-ask.mp4');
  });

  test('the hero clip actually plays: its clock advances and the poster gives way', async ({ page }) => {
    await page.goto('/');
    const playing = () => page.evaluate(() => {
      const v = document.querySelector('.av-hero .av-video video');
      return !!v && !v.paused && v.currentTime > 0.2 && v.readyState >= 2;
    });
    await expect.poll(playing, { timeout: 20_000 }).toBe(true);
    await expect(page.locator('.av-hero .av-video img')).toHaveClass(/is-hidden/);
  });

  test('makes no request to a third party', async ({ page, baseURL }) => {
    const foreign = [];
    page.on('request', (r) => {
      const u = new URL(r.url());
      if (u.protocol.startsWith('http') && u.origin !== new URL(baseURL).origin) foreign.push(r.url());
    });
    await page.goto('/', { waitUntil: 'networkidle' });
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);
    expect(foreign).toEqual([]);
  });

  test('the console tour shows exactly one panel, whichever tab is chosen', async ({ page }) => {
    await page.goto('/#tour');
    const panels = page.locator('#tour [role="tabpanel"]');
    await expect(panels).toHaveCount(5);
    await expect(page.locator('#tour [role="tabpanel"]:visible')).toHaveCount(1);
    await page.getByRole('tab', { name: 'Economics' }).click();
    await expect(page.locator('#tour [role="tabpanel"]:visible')).toHaveCount(1);
    await expect(page.locator('#tour-panel-economics')).toBeVisible();
    await expect(page.locator('#tour-panel-console')).toBeHidden();
    // roving tabindex: arrow keys move the selection
    await page.getByRole('tab', { name: 'Economics' }).press('ArrowRight');
    await expect(page.getByRole('tab', { name: 'Governance' })).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('#tour-panel-governance')).toBeVisible();
  });

  test('a tab change in the tour paints at once: every poster is ready before the first click', async ({ page }) => {
    await page.goto('/#tour');
    const posters = page.locator('#tour .av-tabpanel .av-video img');
    // held until the page is idle, then released together
    await expect(posters).toHaveCount(5, { timeout: 15_000 });
    await expect
      .poll(() => posters.evaluateAll((imgs) => imgs.every((i) => i.complete && i.naturalWidth > 600)), { timeout: 15_000 })
      .toBe(true);
    await page.getByRole('tab', { name: 'Governance' }).click();
    const shown = page.locator('#tour-panel-governance .av-video img');
    await expect(shown).toBeVisible();
    expect(await shown.evaluate((i) => i.complete && i.naturalWidth)).toBeGreaterThan(600);
    // only the chosen tab plays, and the one left behind rewinds under its poster
    await expect.poll(() => page.locator('#tour-panel-governance video').evaluate((v) => !v.paused && v.currentTime > 0.2), { timeout: 20_000 }).toBe(true);
    const others = await page.locator('#tour .av-tabpanel.is-off video').evaluateAll((vs) => vs.every((v) => v.paused));
    expect(others).toBe(true);
  });

  test('the architecture diagram keeps every label inside its box, and every box on the canvas', async ({ page }) => {
    await page.goto('/platform');
    const problems = await page.locator('.av-diagram svg').first().evaluate((svg) => {
      const vb = svg.viewBox.baseVal;
      const boxes = [...svg.querySelectorAll('rect.box')].map((r) => r.getBBox());
      const out = [];
      for (const r of boxes) if (r.x < vb.x || r.y < vb.y || r.x + r.width > vb.x + vb.width || r.y + r.height > vb.y + vb.height) out.push('a box leaves the canvas');
      for (const t of svg.querySelectorAll('text')) {
        const b = t.getBBox();
        const cx = b.x + b.width / 2;
        const cy = b.y + b.height / 2;
        const host = boxes.filter((r) => cx >= r.x && cx <= r.x + r.width && cy >= r.y && cy <= r.y + r.height).sort((a, c) => a.width * a.height - c.width * c.height)[0];
        if (!host) out.push('no box holds: ' + t.textContent);
        else if (b.x < host.x + 8 || b.x + b.width > host.x + host.width - 8) out.push('overflows its box: ' + t.textContent);
      }
      return out;
    });
    expect(problems).toEqual([]);
  });

  test('a question opens to its answer', async ({ page }) => {
    await page.goto('/');
    const first = page.locator('.av-accordion details').first();
    await first.locator('summary').click();
    await expect(first).toHaveAttribute('open', '');
  });
});

test.describe('navigation', () => {
  test('desktop: a mega menu opens, closes on Escape, and its links navigate', async ({ page }, testInfo) => {
    test.skip(isMobile(testInfo), 'the drawer covers mobile');
    await page.goto('/');
    const button = page.getByRole('button', { name: 'Platform' });
    await button.click();
    await expect(button).toHaveAttribute('aria-expanded', 'true');
    const engine = page.locator('.av-mega:visible a', { hasText: 'Avarok Engine' }).first();
    await expect(engine).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(button).toHaveAttribute('aria-expanded', 'false');
    await button.click();
    await engine.click();
    await expect(page).toHaveURL(/\/platform\/engine/);
    await expect(page.locator('h1')).toBeVisible();
  });

  test('mobile: the drawer opens, a group expands, and a link navigates and closes it', async ({ page }, testInfo) => {
    test.skip(!isMobile(testInfo), 'the mega menu covers desktop');
    await page.goto('/');
    await page.getByRole('button', { name: 'Open menu' }).click();
    await expect(page.locator('#av-drawer')).toBeVisible();
    await page.locator('.av-drawer-head', { hasText: 'Solutions' }).click();
    await page.locator('.av-drawer-items a', { hasText: 'Healthcare' }).click();
    await expect(page).toHaveURL(/\/solutions\/healthcare/);
    await expect(page.locator('#av-drawer')).toBeHidden();
  });

  test('the logo goes home and the footer reaches every section', async ({ page }) => {
    await page.goto(routes.pricing);
    await expect(page.locator('.av-brand')).toHaveAttribute('href', '/');
    for (const label of ['Platform', 'Solutions', 'Resources', 'Company']) {
      await expect(page.locator('.av-footer h2', { hasText: label }).first()).toBeVisible();
    }
  });
});

test.describe('pricing', () => {
  test('the payback model recomputes when an input changes', async ({ page }) => {
    await page.goto(`${routes.pricing}#payback`);
    const out = page.locator('#payback .av-calc-hero').first();
    const before = await out.innerText();
    expect(before).toMatch(/month/i);
    const gpus = page.locator('#payback input[type="number"]').first();
    await gpus.fill('8');
    await expect(out).not.toHaveText(before);
  });

  test('the second scenario is the one behind the 70% claim', async ({ page }) => {
    await page.goto(`${routes.pricing}#payback`);
    await page.locator('#payback [role="tab"]').nth(1).click();
    await expect(page.locator('#payback [role="tabpanel"]:visible')).toHaveCount(1);
    await expect(page.locator('#payback .av-calc-hero')).toContainText('%');
  });

  test('every proposed price is labelled as proposed', async ({ page }) => {
    await page.goto(routes.pricing);
    expect(await page.locator('.av-evidence.is-proposed, .av-badge-proposed, :text("PROPOSED")').count()).toBeGreaterThan(0);
  });
});

test.describe('theme', () => {
  test('the toggle switches the theme and the choice survives a reload', async ({ page }) => {
    await page.goto('/');
    const html = page.locator('html');
    const start = await html.getAttribute('data-theme');
    await page.locator('.av-header .theme-toggle, .av-header [aria-label*="theme" i]').first().click();
    const flipped = start === 'light' ? 'dark' : 'light';
    await expect(html).toHaveAttribute('data-theme', flipped);
    await page.reload();
    await expect(html).toHaveAttribute('data-theme', flipped);
  });
});

test.describe('footer', () => {
  test('the social marks sit on one line', async ({ page }) => {
    await page.goto('/');
    await page.locator('.av-footer-social').scrollIntoViewIfNeeded();
    const centres = await page.locator('.av-footer-social a').evaluateAll((links) =>
      links.map((a) => {
        const ink = a.querySelector('path').getBoundingClientRect();
        return (ink.top + ink.bottom) / 2 - a.getBoundingClientRect().top;
      })
    );
    expect(centres).toHaveLength(3);
    expect(Math.max(...centres) - Math.min(...centres)).toBeLessThan(1);
  });
});

test.describe('the film', () => {
  test('has a player nothing is laid over, and plays when started', async ({ page }) => {
    await page.goto(`${routes.demo}#film`);
    const film = page.locator('#film video');
    await expect(film).toBeVisible({ timeout: 20_000 });
    await expect(film).toHaveAttribute('controls', '');
    await expect(film).toHaveAttribute('poster', '/media/reel.webp');
    // The poster image must not sit on top of the controls.
    await expect(page.locator('#film img')).toHaveCount(0);
    expect(await film.evaluate((v) => v.paused)).toBe(true);
    await film.evaluate((v) => v.play());
    await expect.poll(() => film.evaluate((v) => !v.paused && v.currentTime > 0.2), { timeout: 20_000 }).toBe(true);
  });
});

test.describe('demo request', () => {
  test('without a form endpoint the form composes an email to sales and confirms', async ({ page }) => {
    await page.goto(routes.demo);
    for (const input of await page.locator('form.av-form [required]').all()) {
      const type = await input.getAttribute('type');
      await input.fill(type === 'email' ? 'buyer@example.com' : 'Test value');
    }
    const [request] = await Promise.all([
      page.waitForEvent('request', { predicate: (r) => r.url().startsWith('mailto:'), timeout: 5000 }).catch(() => null),
      page.locator('form.av-form button[type="submit"]').click()
    ]);
    await expect(page.locator('form.av-form [role="status"]')).toBeVisible();
    if (request) expect(decodeURIComponent(request.url())).toContain('Avarok working session');
  });
});

test.describe('calls to action land on the form', () => {
  test('a link to the booking form scrolls to it and puts the caret in the first field', async ({ page }) => {
    await page.goto('/pricing');
    await page.locator(`a[href="${routes.demoForm}"]`).first().click();
    await expect(page).toHaveURL(/\/demo#book$/);
    await expect(page.locator('#book form')).toBeInViewport();
    await expect(page.locator('#book input').first()).toBeFocused();
  });

  test('the Community Edition is a waitlist, not an install button', async ({ page }) => {
    await page.goto('/pricing');
    const tier = page.locator('.av-tier', { hasText: 'Community Edition' });
    await expect(tier).toContainText('Waitlist open');
    await tier.getByRole('link', { name: 'Join the waitlist' }).click();
    await expect(page).toHaveURL(/\/waitlist$/);
    await expect(page.locator('h1')).toContainText('not out yet');
    await page.locator('#waitlist-email').fill('dev@example.com');
    const [request] = await Promise.all([
      page.waitForEvent('request', { predicate: (r) => r.url().startsWith('mailto:'), timeout: 5000 }).catch(() => null),
      page.locator('form.av-form button[type="submit"]').click()
    ]);
    await expect(page.locator('form.av-form [role="status"]')).toContainText('on the list');
    if (request) expect(decodeURIComponent(request.url())).toContain('Community Edition waitlist');
  });

  test('no page still offers to install an edition that is not released', async ({ page }) => {
    for (const path of ['/', '/platform/engine', '/pricing', '/resources']) {
      await page.goto(path);
      await expect(page.locator('main')).not.toContainText(/install the community edition/i);
    }
  });
});

test.describe('logos', () => {
  test('the wall shows both emblems and every partner logo paints on either theme', async ({ page }) => {
    await page.goto('/');
    await page.locator('.av-wall').scrollIntoViewIfNeeded();
    const broken = async () =>
      page.locator('.av-wall img').evaluateAll((imgs) => imgs.filter((i) => i.offsetParent !== null && !(i.complete && i.naturalWidth > 0)).map((i) => i.getAttribute('src')));
    await expect(page.locator('.av-wall .has-emblem img')).toHaveCount(2);
    await expect.poll(broken, { timeout: 15_000 }).toEqual([]);
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light'));
    await expect.poll(broken, { timeout: 15_000 }).toEqual([]);
    await expect(page.locator('.av-wall-note')).toContainText('does not imply or constitute DoD endorsement');
  });
});

test.describe('installed art', () => {
  const samples = [
    ['/why-avarok', '/media/art/art-prisms.webp'],
    ['/solutions/financial-services', '/media/art/art-finance.webp'],
    ['/platform/security', '/media/art/art-enclave.webp'],
    ['/pricing', '/media/art/art-desk-box.webp'],
    ['/solutions/healthcare', '/media/art/art-health.webp'],
    ['/platform/economics', '/media/art/art-power.webp'],
    ['/labs', '/media/art/art-research.webp']
  ];
  for (const [path, src] of samples) {
    test(`${path} paints its still`, async ({ page }) => {
      await page.goto(path);
      const img = page.locator('.av-page-hero-art img');
      await expect(page.locator('.av-page-hero.has-art')).toBeVisible();
      await expect(img).toBeVisible();
      await expect(img).toHaveAttribute('src', src);
      expect(await img.evaluate((el) => el.complete && el.naturalWidth)).toBeGreaterThan(400);
    });
  }
});

test.describe('every page', () => {
  for (const p of pages.filter((x) => x.path !== '/404')) {
    test(`${p.path} renders with its own title and one h1`, async ({ page }) => {
      const res = await page.goto(p.path);
      expect(res.status()).toBe(200);
      await expect(page).toHaveTitle(p.title);
      await expect(page.locator('h1')).toHaveCount(1);
      // The old interactive console is not part of the public site.
      expect(await page.locator('a[href="/console"]').count()).toBe(0);
      // No heading skips a level. Cards that open with an h3 straight after the
      // h1 cost 18 templates their outline once, on pages the Lighthouse gate
      // does not visit. A section with no visible title takes an `av-sr` h2.
      const skipped = await page.locator('h1, h2, h3, h4, h5, h6').evaluateAll((hs) => {
        const out = [];
        let last = 0;
        for (const h of hs) {
          const level = Number(h.tagName[1]);
          if (last && level > last + 1) out.push(`h${last} then h${level}: ${h.textContent.trim().slice(0, 50)}`);
          last = level;
        }
        return out;
      });
      expect(skipped).toEqual([]);
    });
  }

  test('the developer pages still mount under the new header', async ({ page }) => {
    for (const path of ['/engine', '/control']) {
      await page.goto(path);
      await expect(page.locator('.av-header')).toBeVisible();
      await expect(page.locator('.av-footer')).toHaveCount(1);
    }
  });
});
