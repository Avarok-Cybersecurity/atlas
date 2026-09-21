// =============================================================================
// make-corpus.mjs — generates the checked-in test corpus for the E2E suite:
//   corpus-small.jsonl       valid lattice-jsonl v1, dim 8, 20 points
//   corpus-small.jsonl.gz    the gzip the tests serve for CORPUS_GZ_URL
//   corpus-small.meta.json   the manifest the tests serve for CORPUS_META_URL
//
// Fully deterministic: fixed commit sha, fixed timestamp, vectors from the
// shared fake embedder (embed.mjs), every JSON line written with sorted keys
// (matching lattice-core's interchange serializer). Re-run with
//   bun e2e/fixtures/make-corpus.mjs
// and the output bytes are identical unless the chunk table below changes.
// =============================================================================

import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { embedText } from './embed.mjs';

const OUT_DIR = dirname(fileURLToPath(import.meta.url));
const DIM = 8;
const REPO = 'Avarok-Cybersecurity/atlas';
const COMMIT = 'a3f9c1e7b2d84056917e2c3a4b5d6f7089abcde1';
const GENERATED_AT = '2026-08-16T00:00:00Z';
const MODEL = 'nvidia/llama-nemotron-embed-vl-1b-v2:free';

// ~20 realistic Atlas-flavored chunks (Rust engine + CUDA kernels + docs).
// path / language / start_line / body.
const CHUNKS = [
  ['src/scheduler/batch.rs', 'rust', 41, `impl BatchScheduler {
    /// Admit waiting requests into the running decode batch. Prefill-heavy
    /// requests are chunked so a long prompt never stalls decode for the
    /// whole batch (chunked prefill, 2048-token slices).
    pub fn admit(&mut self, now: Instant) -> Vec<RequestId> {
        let mut admitted = Vec::new();
        while let Some(req) = self.waiting.peek() {
            if self.kv_pool.free_blocks() < req.blocks_needed() { break; }
            admitted.push(self.waiting.pop().unwrap().id);
        }
        admitted
    }
}`],
  ['src/scheduler/batch.rs', 'rust', 88, `    /// A request joins the decode batch only once its prefill is complete
    /// and a KV block reservation is held. Preemption evicts the youngest
    /// request first so long-running generations keep their cache.
    fn join_decode(&mut self, id: RequestId) {
        debug_assert!(self.prefill_done.contains(&id));
        self.decode_batch.push(id);
        self.decode_batch.sort_by_key(|r| self.arrival[r]);
    }`],
  ['src/decode/mtp.rs', 'rust', 12, `//! Multi-token prediction (MTP) speculative decoding.
//! The draft head proposes N tokens per step; the verifier scores them in a
//! single batched forward pass and keeps the longest accepted prefix.
pub struct MtpConfig {
    /// Draft tokens proposed per decode step (2 on GB10 by default).
    pub num_speculative_tokens: usize,
    /// Reject drafts whose verifier logprob falls below this margin.
    pub acceptance_margin: f32,
}`],
  ['src/decode/mtp.rs', 'rust', 47, `impl MtpEngine {
    /// Keep draft tokens while verifier argmax matches the draft; the first
    /// mismatch truncates the accepted prefix and resamples from the
    /// verifier distribution at that position.
    pub fn accept_prefix(&self, draft: &[TokenId], verified: &[Logits]) -> usize {
        draft.iter().zip(verified).take_while(|(d, v)| v.argmax() == **d).count()
    }
}`],
  ['kernels/nvfp4/gemm_sm121.cu', 'cuda', 5, `// NVFP4 (E2M1) block-scaled GEMM for GB10 (SM121). SM121 lacks the
// cvt.rn.satfinite.e2m1x2.f32 PTX instruction, so quantization uses a
// software bit-twiddle path (__float_as_uint, shift, mask) before the MMA.
template <int TileM, int TileN, int TileK>
__global__ void nvfp4_gemm_sm121(const uint8_t* __restrict__ a,
                                 const uint8_t* __restrict__ b,
                                 const float* __restrict__ scales,
                                 half* __restrict__ out) {
  constexpr int kWarps = 8; // cooperative schedule, AtomLayout<4,2,1>
  // ...
}`],
  ['kernels/nvfp4/dispatch.rs', 'rust', 19, `/// Pick the NVFP4 GEMM backend for the current device. GB10 (SM121) routes
/// through the SM120 codepath: CUTLASS block-scaled kernels with the
/// software E2M1 conversion patch; Marlin handles the W4A16 dequant path.
pub fn select_backend(cc: ComputeCapability) -> GemmBackend {
    match (cc.major, cc.minor) {
        (12, 1) | (12, 0) => GemmBackend::CutlassSm120,
        (10, _) => GemmBackend::CutlassSm100,
        _ => GemmBackend::Cublas,
    }
}`],
  ['src/kv/pool.rs', 'rust', 30, `/// Paged KV cache pool. Blocks are fixed 16-token pages; the FP8 KV path
/// halves bytes per token so a 4k context fits ~2x the sequences.
pub struct KvPool {
    block_bytes: usize,
    free_list: Vec<BlockId>,
    /// Copy-on-write reference counts for prefix-shared blocks.
    refcounts: Vec<u16>,
}`],
  ['src/kv/pool.rs', 'rust', 72, `impl KvPool {
    pub fn allocate(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if self.free_list.len() < n { return None; }
        Some(self.free_list.split_off(self.free_list.len() - n))
    }
    pub fn release(&mut self, blocks: &[BlockId]) {
        for b in blocks {
            if self.decref(*b) == 0 { self.free_list.push(*b); }
        }
    }
}`],
  ['src/prefill/chunked.rs', 'rust', 9, `/// Chunked prefill: long prompts are sliced into fixed-size chunks that
/// interleave with decode steps, bounding TTFT for everyone in the batch.
/// The chunk size trades prefill throughput against decode stall time.
pub const PREFILL_CHUNK_TOKENS: usize = 2048;

pub fn plan_chunks(prompt_len: usize) -> impl Iterator<Item = Range<usize>> {
    (0..prompt_len).step_by(PREFILL_CHUNK_TOKENS)
        .map(move |s| s..(s + PREFILL_CHUNK_TOKENS).min(prompt_len))
}`],
  ['src/attention/flash.rs', 'rust', 25, `/// FlashAttention-style fused attention for SM120/SM121. Softmax runs in
/// registers with an online max/sum rescale; KV loads are staged through
/// shared memory with cp.async double buffering.
pub struct FlashAttnParams {
    pub head_dim: usize,
    pub kv_cache_dtype: KvDtype, // Fp16 | Fp8E4M3
    pub causal: bool,
}`],
  ['src/server/openai.rs', 'rust', 140, `/// OpenAI-compatible /v1/chat/completions handler. Tool calls are parsed
/// from the model stream with the hermes-format parser; SSE chunks flush on
/// every token so TTFT is one forward pass, not one buffer.
async fn chat_completions(State(engine): State<Engine>, Json(req): Json<ChatRequest>)
    -> Result<Sse<TokenStream>, ApiError> {
    let stream = engine.submit(req.into_generation()?).await?;
    Ok(Sse::new(stream.map(to_chunk)))
}`],
  ['src/server/openai.rs', 'rust', 203, `/// Streaming tool-call assembly: fragments arrive interleaved with text;
/// the parser tracks brace depth per call id and only emits a tool_call
/// delta once its JSON argument fragment is balanced.
fn assemble_tool_calls(deltas: &mut ToolCallBuffer, frag: &str) -> Vec<ToolCallDelta> {
    deltas.push(frag);
    deltas.drain_balanced()
}`],
  ['src/quant/nvfp4.rs', 'rust', 16, `/// NVFP4 tensor layout: 4-bit E2M1 values packed two per byte, with one
/// FP8 (E4M3) scale per 16-value block. Weights are pre-swizzled at load
/// time so the GEMM reads scales through TMA without a gather.
pub const NVFP4_BLOCK: usize = 16;

pub fn pack_e2m1(vals: &[f32; NVFP4_BLOCK], scale: f32) -> [u8; NVFP4_BLOCK / 2] {
    let mut out = [0u8; NVFP4_BLOCK / 2];
    for (i, pair) in vals.chunks_exact(2).enumerate() {
        out[i] = encode_e2m1(pair[0] / scale) | (encode_e2m1(pair[1] / scale) << 4);
    }
    out
}`],
  ['src/quant/nvfp4.rs', 'rust', 58, `/// Software E2M1 encode for SM121: round-to-nearest-even into the 8-value
/// E2M1 lattice {0, .5, 1, 1.5, 2, 3, 4, 6} with saturation, sign in bit 3.
fn encode_e2m1(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) as u8) << 3;
    let mag = f32::from_bits(bits & 0x7fff_ffff);
    sign | E2M1_TABLE.partition_point(|t| *t < mag).min(7) as u8
}`],
  ['src/engine/mod.rs', 'rust', 1, `//! Atlas engine core: request lifecycle from tokenize -> prefill -> decode
//! -> detokenize. The engine owns the scheduler, the KV pool, and the model
//! runner; everything above it (HTTP, metrics) is stateless.
pub mod runner;
pub mod sampler;
pub struct Engine {
    scheduler: Mutex<BatchScheduler>,
    runner: ModelRunner,
}`],
  ['src/engine/sampler.rs', 'rust', 34, `/// Fused sampling: temperature scale, top-k mask, top-p renormalize, then
/// a single Gumbel draw — one kernel launch per decode step for the whole
/// batch instead of four.
pub fn sample_batch(logits: &Tensor, params: &[SamplingParams]) -> Vec<TokenId> {
    fused_sample_kernel(logits, params)
}`],
  ['docs/architecture.md', 'markdown', 1, `# Atlas architecture

Atlas is a single-node inference engine for DGX Spark (GB10). The serving
path is: HTTP front door -> tokenizer -> batch scheduler (continuous
batching, chunked prefill) -> model runner (CUDA graphs per batch shape) ->
sampler -> SSE stream. NVFP4 weights ride CUTLASS SM120 kernels with a
software E2M1 conversion, since SM121 lacks the hardware cvt instruction.`],
  ['docs/benchmarks.md', 'markdown', 12, `## MLPerf-edge agentic reference

The flagship 35B config is gated on webserver_ok 10/10, BFCL subset ST-995
against the recorded baseline, and a warm-TTFT guard (median within 3%) on
any perf-path diff. Decode throughput on GB10 tops out near the LPDDR5X
bandwidth ceiling; MTP speculative decoding is what pushes past it.`],
  ['src/runner/cuda_graph.rs', 'rust', 21, `/// One captured CUDA graph per (batch_size, num_draft_tokens) shape. Decode
/// replays the graph; a shape miss falls back to eager and captures the new
/// shape in the background so steady state is always graph-replay.
pub struct GraphCache {
    graphs: HashMap<GraphKey, CudaGraphExec>,
    capture_stream: Stream,
}`],
  ['src/ttft/warmup.rs', 'rust', 8, `/// Warm-path TTFT: the first token cost is dominated by prompt prefill, so
/// the server pre-captures decode graphs and pre-touches KV pages at boot.
/// Cold TTFT is reported separately — never average the two populations.
pub fn warm_boot(engine: &Engine) {
    engine.capture_common_shapes();
    engine.kv_pool_pretouch();
}`]
];

// --- serialization helpers ---------------------------------------------------

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map((k) => [k, sortKeys(value[k])])
    );
  }
  return value;
}

const jsonLine = (obj) => JSON.stringify(sortKeys(obj));

// --- build the dump ----------------------------------------------------------

const points = CHUNKS.map(([path, language, startLine, body], i) => {
  const text = body;
  const endLine = startLine + text.split('\n').length - 1;
  return {
    t: 'point',
    id: i + 1,
    vector: embedText(text, DIM),
    payload: {
      chunk_hash: createHash('sha256').update(text).digest('hex').slice(0, 16),
      commit_sha: COMMIT,
      end_line: endLine,
      language,
      path,
      repo: REPO,
      start_line: startLine,
      text
    }
  };
});

const header = {
  t: 'header',
  format: 'lattice-jsonl',
  version: 1,
  config: {
    name: 'atlas-code',
    vectors: { size: DIM, distance: 'Cosine' },
    hnsw: { m: 16, m0: 32, ml: 0.36067376022224085, ef: 50, ef_construction: 200 },
    relations: {},
    durability: 'ephemeral'
  },
  vectors: 'inline',
  dim: DIM,
  points: points.length,
  edges: 0
};

const jsonl = [header, ...points].map(jsonLine).join('\n') + '\n';
const jsonlBytes = Buffer.from(jsonl, 'utf8');
// gzip with fixed level; node's gzip writes mtime=0, so the bytes are stable.
const gz = gzipSync(jsonlBytes, { level: 9 });

const meta = {
  bytes: jsonlBytes.byteLength,
  commit_sha: COMMIT,
  dim: DIM,
  files: new Set(CHUNKS.map(([p]) => p)).size,
  generated_at: GENERATED_AT,
  gz_bytes: gz.byteLength,
  model: MODEL,
  points: points.length,
  repo: REPO,
  sha256: createHash('sha256').update(jsonlBytes).digest('hex')
};

writeFileSync(join(OUT_DIR, 'corpus-small.jsonl'), jsonlBytes);
writeFileSync(join(OUT_DIR, 'corpus-small.jsonl.gz'), gz);
writeFileSync(join(OUT_DIR, 'corpus-small.meta.json'), JSON.stringify(sortKeys(meta), null, 2) + '\n');

console.log(`corpus-small: ${points.length} points, ${meta.files} files, ` +
  `${meta.bytes} bytes (${meta.gz_bytes} gz), commit ${COMMIT.slice(0, 7)}`);
