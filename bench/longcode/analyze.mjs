// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-sample structural analyzer for the long-code degeneration harness.
// Input: a raw sample JSON {seed, content, reasoning, finish_reason}.
// Output (stdout): metrics JSON. Uses node's own parser via `node --check`
// for syntactic validity and acorn for scope-level duplicate-declaration
// detection (the primary degeneration signal: duplicate var/const blocks).
//
// This is a measurement tool, not production code; it intentionally does
// file I/O. Determinism only — no network, no model knowledge.

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

// acorn 8.x ships CJS-only (no ESM `exports`); a bare ESM import fails.
// Resolve it through CJS require, which walks the global node lib path.
const _req = createRequire(import.meta.url);
// acorn-loose yields a best-effort AST even for syntactically broken
// code. Essential here: degenerate output usually does NOT parse
// strictly, and a strict-then-longest-valid-prefix fallback would
// exclude the degenerate tail and mask the very duplicates we measure.
const acornLoose = _req("acorn-loose");

const sample = JSON.parse(readFileSync(process.argv[2], "utf8"));
const content = sample.content || "";
const finish = sample.finish_reason;

// --- 1. Extract the JS to validate ----------------------------------------
// The model's wrapping is unreliable — and crucially, DEGENERATE output
// usually never emits the closing ``` (or </script>), so a closed-only
// extractor would return "" exactly on the failing samples and mask the
// signal. Handle: closed fences, an UNCLOSED trailing fence, no fence at
// all (raw HTML), and unclosed <script>.
function fenceBlock(text) {
  let block = "";
  let lang = "";
  for (const m of text.matchAll(/```([a-zA-Z0-9]*)\n([\s\S]*?)```/g)) {
    if (m[2].length > block.length) {
      block = m[2];
      lang = (m[1] || "").toLowerCase();
    }
  }
  // Unclosed trailing fence: opener with no matching closer → to EOF.
  const open = text.match(/```([a-zA-Z0-9]*)\n([\s\S]*)$/);
  if (open && open[2].length > block.length && !open[2].includes("```")) {
    block = open[2];
    lang = (open[1] || "").toLowerCase();
  }
  // No fence at all but clearly HTML/JS → treat whole text as the block.
  if (!block && /<!DOCTYPE|<html|<script|new\s+THREE\./i.test(text)) {
    block = text;
  }
  return { block, lang };
}

function extractJs(text) {
  const { block, lang } = fenceBlock(text);
  if (!block) return "";
  const looksHtml = lang === "html" || /<script[\s>]/i.test(block);
  if (looksHtml) {
    const out = [];
    // Closed <script>…</script> bodies (skip external src=).
    for (const s of block.matchAll(
      /<script\b([^>]*)>([\s\S]*?)<\/script>/gi
    )) {
      if (!/\bsrc\s*=/.test(s[1])) out.push(s[2]);
    }
    // Unclosed trailing <script> (truncated mid-script) → to EOF.
    const lastOpen = block.match(/<script\b([^>]*)>([\s\S]*)$/i);
    if (
      lastOpen &&
      !/\bsrc\s*=/.test(lastOpen[1]) &&
      !/<\/script>/i.test(lastOpen[2])
    ) {
      out.push(lastOpen[2]);
    }
    if (out.length) return out.join("\n;\n");
  }
  return block;
}

const js = extractJs(content);
const tmp = mkdtempSync(join(tmpdir(), "lc-"));

// `node --check` on a JS string; returns true iff it parses.
function checks(src) {
  const f = join(tmp, "c.js");
  writeFileSync(f, src);
  try {
    execFileSync(process.execPath, ["--check", f], { stdio: "pipe" });
    return true;
  } catch {
    return false;
  }
}

// Longest line-prefix that `node --check` accepts (binary search on lines).
function validLineCount(src) {
  const lines = src.split("\n");
  if (checks(src)) return lines.length;
  let lo = 0;
  let hi = lines.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (checks(lines.slice(0, mid).join("\n"))) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// --- 2. Duplicate-declaration count (acorn scope walk) --------------------
// var/function -> nearest function|program scope; let/const/class ->
// nearest block scope. A name re-declared in the same bucket is the
// gross degeneration symptom (duplicate `var board`, repeated funcs).
function dupDeclCount(src) {
  // Loose parse over the FULL source (incl. the degenerate tail) so
  // late duplicates are never masked by an early syntax error.
  const ast = acornLoose.parse(src, { ecmaVersion: "latest" });
  let dups = 0;
  let firstDupPos = null;
  const fnStack = [{ names: new Set() }]; // function/program scope
  const blkStack = [{ names: new Set() }]; // block scope

  function decl(name, kind, pos) {
    const bucket =
      kind === "var" || kind === "function"
        ? fnStack[fnStack.length - 1]
        : blkStack[blkStack.length - 1];
    if (bucket.names.has(name)) {
      dups += 1;
      if (firstDupPos === null) firstDupPos = pos;
    } else {
      bucket.names.add(name);
    }
  }

  function names(id, kind, pos) {
    if (!id) return;
    if (id.type === "Identifier") decl(id.name, kind, pos);
    else if (id.type === "ObjectPattern")
      id.properties.forEach((p) =>
        names(p.value || p.argument, kind, pos)
      );
    else if (id.type === "ArrayPattern")
      id.elements.forEach((e) => e && names(e, kind, pos));
    else if (id.type === "AssignmentPattern") names(id.left, kind, pos);
    else if (id.type === "RestElement") names(id.argument, kind, pos);
  }

  function walk(node) {
    if (!node || typeof node.type !== "string") return;
    const isFn =
      node.type === "FunctionDeclaration" ||
      node.type === "FunctionExpression" ||
      node.type === "ArrowFunctionExpression";
    const isBlk = node.type === "BlockStatement" || isFn;
    if (node.type === "VariableDeclaration")
      node.declarations.forEach((d) =>
        names(d.id, node.kind, d.start)
      );
    if (node.type === "FunctionDeclaration" && node.id)
      decl(node.id.name, "function", node.start);
    if (isFn) fnStack.push({ names: new Set() });
    if (isBlk) blkStack.push({ names: new Set() });
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (Array.isArray(v)) v.forEach((c) => c && walk(c));
      else if (v && typeof v.type === "string") walk(v);
    }
    if (isBlk) blkStack.pop();
    if (isFn) fnStack.pop();
  }
  walk(ast);
  return { dups, firstDupPos };
}

// --- 3. Metrics -----------------------------------------------------------
const validLines = js ? validLineCount(js) : 0;
const totalLines = js ? js.split("\n").length : 0;
const { dups, firstDupPos } = js
  ? dupDeclCount(js)
  : { dups: 0, firstDupPos: null };

const badHex = [...content.matchAll(/0x(?![0-9a-fA-F])|0x[0-9a-fA-F]*[g-zG-Z]/g)];
const firstBadHex = badHex.length ? badHex[0].index : null;

// Char-offset proxy for "tokens to first degeneration" (no per-token
// offsets from the API; offset is stable and paired-comparable).
const degCandidates = [];
if (firstDupPos !== null) {
  // firstDupPos is into `js`; map approximately via indexOf in content.
  const frag = js.slice(Math.max(0, firstDupPos - 12), firstDupPos + 12);
  const at = frag ? content.indexOf(frag.trim().split("\n")[0]) : -1;
  degCandidates.push(at >= 0 ? at : firstDupPos);
}
if (firstBadHex !== null) degCandidates.push(firstBadHex);
const closedHtml =
  content.includes("</script>") && content.includes("</html>");
if (finish !== "stop" || !closedHtml) degCandidates.push(content.length);
const tokensToFirstDegen = degCandidates.length
  ? Math.min(...degCandidates)
  : null;

const parsesClean = js ? checks(js) : false;
const hasScene = /new\s+THREE\.Scene\s*\(/.test(content);
const hasLoop =
  /requestAnimationFrame\s*\(/.test(content) ||
  /\.render\s*\(\s*scene/.test(content);
const completenessPass =
  parsesClean &&
  hasScene &&
  hasLoop &&
  closedHtml &&
  finish === "stop" &&
  dups === 0;

process.stdout.write(
  JSON.stringify({
    valid_js_line_count: validLines,
    total_js_line_count: totalLines,
    duplicate_declaration_count: dups,
    malformed_hex_count: badHex.length,
    tokens_to_first_degeneration: tokensToFirstDegen,
    parses_clean: parsesClean,
    has_scene: hasScene,
    has_render_loop: hasLoop,
    closed_html: closedHtml,
    completeness_pass: completenessPass,
  })
);
