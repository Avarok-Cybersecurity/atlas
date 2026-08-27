// SPDX-License-Identifier: AGPL-3.0-only

import { error } from '@sveltejs/kit';
import { posts, getPost } from '$lib/blog.js';

export const prerender = true;

export function entries() {
  return posts.map((p) => ({ slug: p.slug }));
}

export function load({ params }) {
  const post = getPost(params.slug);
  if (!post) error(404, 'Not found');
  return { post };
}
