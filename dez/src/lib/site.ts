// SPDX-License-Identifier: AGPL-3.0-only

/**
 * Single source of truth for every externally-visible string and URL on the
 * site. Nothing here is a placeholder claim: if a fact is not verifiable from
 * the Atlas repository it does not belong in this file.
 */

export const NAME = 'Dez';

/** Verbatim product tagline. Do not paraphrase. */
export const TAGLINE =
  'Dez: The free and open-source IDE for the local inference-first paradigm';

export const DESCRIPTION =
  'An IDE built on the Atlas Inference Engine, WebGPU, Rust and 100% WebAssembly. ' +
  'Models run locally in your browser — no server round-trip, no API key.';

/** Honest development status. This is a placeholder site, not a launch. */
export const STATUS = {
  label: 'In development',
  detail:
    'Dez is an early work in progress. There is nothing to download yet, and no release date. ' +
    'This page exists so you can follow the work as it happens.'
} as const;

export const LINKS = {
  atlasRepo: 'https://github.com/Avarok-Cybersecurity/atlas',
  atlasSite: 'https://atlasinference.io',
  atlasLicense: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/LICENSE',
  discord: 'https://discord.gg/6vDbKaKrKD',
  webgpu: 'https://www.w3.org/TR/webgpu/'
} as const;

export interface Feature {
  readonly id: string;
  readonly title: string;
  readonly body: string;
}

export const FEATURES: readonly Feature[] = [
  {
    id: 'local',
    title: 'Local-first, not local-optional',
    body:
      'Inference runs on the machine in front of you. Your source, your prompts and your ' +
      'weights are never uploaded, because there is nowhere to upload them to. No account, ' +
      'no API key, no per-token bill.'
  },
  {
    id: 'webgpu',
    title: 'WebGPU for compute',
    body:
      'Dez targets the WebGPU standard rather than a single vendor toolchain, so the same ' +
      'build talks to the GPU already in your laptop — whoever made it.'
  },
  {
    id: 'wasm',
    title: '100% WebAssembly',
    body:
      'The engine compiles to WebAssembly and ships as static files. Nothing is installed, ' +
      'nothing runs as a privileged process, and the browser sandbox stays the trust boundary.'
  },
  {
    id: 'oss',
    title: 'Free and open source',
    body:
      'Dez follows Atlas: source-available to read, fork and audit under the AGPL-3.0. ' +
      'A tool you run on your own hardware should be a tool you can inspect.'
  }
] as const;

export interface PipelineStage {
  readonly step: string;
  readonly title: string;
  readonly body: string;
}

export const PIPELINE: readonly PipelineStage[] = [
  {
    step: '01',
    title: 'Atlas, in Rust',
    body:
      'Atlas is a pure-Rust LLM inference engine: scheduler, model graph and hardware-specific ' +
      'kernels behind tight trait boundaries, with no Python runtime anywhere in the request path.'
  },
  {
    step: '02',
    title: 'Compiled to WebAssembly',
    body:
      'That same Rust core is compiled to WebAssembly instead of a native binary. The scheduling ' +
      'and tokenisation logic is the code Atlas already runs on servers — retargeted, not rewritten.'
  },
  {
    step: '03',
    title: 'Dispatched over WebGPU',
    body:
      'Atlas is built around swappable hardware backends. In the browser that backend is WebGPU: ' +
      'compute shaders do the matrix work, and WebAssembly drives them.'
  },
  {
    step: '04',
    title: 'The editor is the client',
    body:
      'The IDE talks to the engine in-process. Completion and chat are function calls into ' +
      'WebAssembly, not HTTPS requests to somebody else’s datacentre.'
  }
] as const;
