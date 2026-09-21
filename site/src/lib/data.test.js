// SPDX-License-Identifier: AGPL-3.0-only

// `flagshipRecipe` existed, was referenced by nothing, and its value sat
// hardcoded eleven lines below inside `runCommandRaw` — the command the site
// tells people to copy. Changing the flagship recipe would have updated the
// obvious place and silently left the page advertising the old one.

import { expect, test } from 'bun:test';
import { flagshipRecipe, runCommandRaw, runCommand, installerUrl } from './data.js';

test('the command the page shows is built from the flagship recipe, not a copy of it', () => {
  expect(runCommandRaw).toContain(flagshipRecipe);
  expect(runCommandRaw).toBe(`atlasctl run ${flagshipRecipe}`);
});

test('the same rule holds for the installer URL it already applied to', () => {
  expect(runCommand).toContain(installerUrl);
});

// A recipe name reaches a terminal, so it must not carry anything a shell
// would act on. This is data, not user input, but it is data that is rendered
// into a copyable command.
test('the flagship recipe is a plain recipe name', () => {
  expect(flagshipRecipe).toMatch(/^[a-z0-9][a-z0-9._-]*$/);
  expect(runCommandRaw).not.toMatch(/[;&|`$(){}<>]/);
});
