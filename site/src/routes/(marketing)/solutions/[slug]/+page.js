// SPDX-License-Identifier: AGPL-3.0-only
import { error } from '@sveltejs/kit';
import { solutions } from '$lib/content/solutions.js';
import { industries } from '$lib/content/brand.js';

export const prerender = true;
export const entries = () => industries.map((i) => ({ slug: i.slug }));

export function load({ params }) {
  const s = solutions[params.slug];
  if (!s) error(404, 'No such solution');
  return { s };
}
