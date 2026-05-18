// Central data source for all site content.
// Any copy changes land here. Components are just presentation.

export const discordUrl = 'https://discord.gg/DwF3brBMpw';
// The post that started it all.
export const firstPostUrl = 'https://www.reddit.com/r/LocalLLaMA/comments/1rkefjw/solved_the_dgx_spark_102_stable_toks_qwen3535ba3b/';
// Second Atlas post; testimonial comments came from this thread.
export const redditUrl = 'https://www.reddit.com/r/LocalLLaMA/comments/1rmvxo3/';

export const heroStats = [
  { value: '130', label: 'tok/s peak (Qwen3.5-35B)' },
  { value: '~2.5GB', label: 'total image size' },
  { value: '<2min', label: 'cold start time' },
  { value: '3.1x', label: 'faster than vLLM' }
];

export const advantage = {
  atlas: [
    { label: 'Image size', value: '~2.5 GB', tone: 'good' },
    { label: 'Cold start', value: '<2 min', tone: 'good' },
    { label: 'Runtime', value: 'Rust + CUDA', tone: 'good' },
    { label: 'Dependencies', value: 'None', tone: 'good' }
  ],
  vllm: [
    { label: 'Image size', value: '20+ GB', tone: 'bad' },
    { label: 'Cold start', value: '~10 min', tone: 'bad' },
    { label: 'Runtime', value: 'Python + PyTorch', tone: 'neutral' },
    { label: 'Dependencies', value: '200+ packages', tone: 'bad' }
  ]
};

export const pillars = [
  {
    icon: '⚡',
    title: 'Pure Rust + CUDA',
    body: 'Compiled from HTTP to kernel dispatch. No interpreter, no GIL, no JIT warm-up.'
  },
  {
    icon: '🔧',
    title: 'Custom CUDA Kernels',
    body: 'Hand-tuned attention, MoE, GDN, and Mamba-2 kernels for Blackwell SM120/121. NVFP4 and FP8 with native tensor cores.'
  },
  {
    icon: '🔮',
    title: 'MTP Speculative Decoding',
    body: 'Multi-Token Prediction generates multiple tokens per forward pass. Up to 3x throughput over single-token decoding.'
  }
];

export const benchmarks = [
  {
    title: 'Qwen3.5-35B (NVFP4) on DGX Spark',
    sub: 'Single GPU, batch=1. Atlas with MTP K=2.',
    pairs: [
      {
        label: 'Average (diverse workloads)',
        atlas: { value: '111.4 tok/s', width: 100, speedup: '3.0x' },
        vllm: { value: '37.5 tok/s', width: 33.6 }
      },
      {
        label: 'Peak (short context)',
        atlas: { value: '130 tok/s', width: 100, speedup: '3.3x' },
        vllm: { value: '~38 tok/s', width: 30 }
      }
    ]
  },
  {
    title: 'Qwen3.5-122B (NVFP4) on a single DGX Spark',
    sub: '122B parameter model, single node. ~54 tok/s with EP=2.',
    pairs: [
      {
        label: 'Decode throughput',
        atlas: { value: '~50 tok/s', width: 100, speedup: '3.3x' },
        vllm: { value: '~15 tok/s', width: 30 }
      }
    ]
  }
];

export const models = [
  { name: 'Qwen3.6-35B-A3B', badges: ['MTP', 'FP8'], params: '35B (3B active)', quant: 'FP8', arch: 'GDN + Attention + MoE (Vision)', tps: '~71 tok/s' },
  { name: 'Qwen3.5-35B-A3B', badges: ['MTP', 'FP8'], params: '35B (3B active)', quant: 'NVFP4 / FP8', arch: 'GDN + Attention + MoE', tps: '~130 tok/s' },
  { name: 'Qwen3.5-122B-A10B', badges: ['MTP', 'EP2'], params: '122B (10B active)', quant: 'NVFP4', arch: 'GDN + Attention + MoE', tps: '~38 tok/s' },
  { name: 'MiniMax M2.7', badges: ['EP2'], params: '229B (10B active)', quant: 'NVFP4', arch: 'Attention + MoE', tps: '~15 tok/s' },
  { name: 'Qwen3.5-27B', badges: [], params: '27B (dense)', quant: 'NVFP4', arch: 'GDN + Attention (Dense)', tps: '~15 tok/s' },
  { name: 'Qwen3-Next-80B-A3B', badges: ['MTP'], params: '80B (3B active)', quant: 'NVFP4', arch: 'SSM + Attention + MoE', tps: '~87 tok/s' },
  { name: 'Qwen3-Coder-Next', badges: ['FP8'], params: '80B (3B active)', quant: 'FP8', arch: 'SSM + Attention + MoE', tps: '~45 tok/s' },
  { name: 'Qwen3-VL-30B', badges: [], params: '30B (3B active)', quant: 'NVFP4', arch: 'Attention + MoE (Vision)', tps: '~68 tok/s' },
  { name: 'Gemma 4 31B', badges: [], params: '31B (dense)', quant: 'NVFP4', arch: 'Dense Transformer', tps: '~11 tok/s' },
  { name: 'Gemma 4 26B', badges: [], params: '26B (3.8B active)', quant: 'NVFP4', arch: 'MoE (128 experts, top-8)', tps: '~73 tok/s' },
  { name: 'Nemotron-3 Super 120B', badges: ['FP8'], params: '120B (12B active)', quant: 'NVFP4 / FP8', arch: 'Mamba-2 + MoE', tps: '~27 tok/s' },
  { name: 'Nemotron-3 Nano 30B', badges: ['FP8'], params: '30B (3.5B active)', quant: 'NVFP4 / FP8', arch: 'Mamba-2 + MoE', tps: '~88 tok/s' },
  { name: 'Mistral Small 4 119B', badges: [], params: '119B (6.5B active)', quant: 'NVFP4', arch: 'MLA + MoE', tps: '~30 tok/s' }
];

export const roadmap = [
  {
    icon: '🌐',
    title: 'Hardware Expansion',
    body: 'Optimized for DGX Spark today. ASUS Ascent GX10 compatibility confirmed by the community. Strix Halo port in exploration. RTX 6000 Pro Blackwell on the horizon. Same kernel philosophy, adapted per chip.'
  },
  {
    icon: '💡',
    title: 'Kernel Philosophy',
    body: 'Every model gets its own hand-tuned CUDA kernels. No generic fallbacks. We profile, optimize, and validate at the register level. If a model matters to you, it matters to us.'
  },
  {
    icon: '📢',
    title: 'Community-Driven',
    body: "MiniMax M2.7 just landed. Model support is driven entirely by what the community asks for. We're in Discord every day listening. Tell us what you're running and we'll optimize for your use case."
  },
  {
    icon: '🛠',
    title: 'Open Source',
    body: "Free and open source release coming soon. We want to make sure what we release is something people can actually build on, not just a dump."
  },
  {
    icon: '🎨',
    title: 'Multimodal',
    body: 'Vision support live for Qwen3-VL. Audio and additional modalities on the roadmap. The goal is proper kernel-level support for each modality.'
  },
  {
    icon: '🎯',
    title: 'Agentic-Ready',
    body: 'OpenAI + Anthropic API compatibility on the same port. Tool calling, structured output, multi-turn. Works with Claude Code, Cline, OpenCode, and Open WebUI out of the box.'
  }
];

export const testimonials = [
  {
    quote: "103 tok/s sustained on the 35B, startup in 15 seconds. Night and day compared to vLLM's 10-minute torch.compile cycle. Then tried the 122B, 43.8 tok/s with MTP, a 41% speedup over our vLLM hybrid, same hardware, 2-minute startup.",
    author: 'ronald_15496',
    source: '#general',
    sourceUrl: discordUrl
  },
  {
    quote: 'Testing atlas-qwen3.5-35b for over an hour on a PNY DGX Spark in an agentic workflow. Super impressed. Spark is actually awesome with Atlas.',
    author: 'PersonWhoThinks',
    source: 'r/LocalLLaMA',
    sourceUrl: redditUrl
  },
  {
    quote: "I've grown tired of vLLM and have been hoping for something. I was really surprised and impressed. I'm so glad I bought Spark because I came across this.",
    author: 'tetsuro59',
    source: '#general',
    sourceUrl: discordUrl
  },
  {
    quote: '115 tok/s on Spark is actually nuts. This speed is insane, amazing work.',
    author: 'ikkiho, Waste_Ad9929',
    source: 'r/LocalLLaMA',
    sourceUrl: redditUrl
  }
];

// Hero shows the install step. TryIt below shows install + run.
// sparkrun pulls & runs the avarok/atlas-gb10:latest image for you
// (the recipe declares `container:`); it uses an existing Docker/Podman
// + NVIDIA container runtime — it does not install the container engine.
export const quickInstall = `uvx sparkrun setup install`;

export const dockerCommand = `uvx sparkrun setup install

sparkrun run @atlas/qwen3.6-35b-a3b-fp8-mtp-atlas`;
