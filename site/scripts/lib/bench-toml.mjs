// SPDX-License-Identifier: AGPL-3.0-only

// bench-toml.mjs — read the gate limits declared in kernels/gb10/*/BENCH.toml.
//
// The site's generators run under Node with no third-party dependencies, and
// Node has no TOML reader. A regex over the file is not an option either: every
// BENCH.toml carries long `note = """…"""` strings that QUOTE table headers
// (`[benchmarks.metrics.c1_aggregate_tok_s]`) and `min = …` lines as prose,
// so anything that does not track string boundaries reads a comment as a
// bound. This is a small TOML parser covering the grammar those files use —
// tables, arrays of tables, dotted keys, basic/literal/multi-line strings,
// numbers, booleans, arrays and inline tables — and it REFUSES anything it
// does not understand rather than guessing past it. `bench-toml.test.js`
// cross-checks its output against Bun's TOML parser on every real BENCH.toml.

const WS = /[ \t]/;
const BARE_KEY = /[A-Za-z0-9_-]/;

class Cursor {
  constructor(text) {
    this.s = text;
    this.i = 0;
  }
  get done() {
    return this.i >= this.s.length;
  }
  peek(n = 0) {
    return this.s[this.i + n];
  }
  startsWith(str) {
    return this.s.startsWith(str, this.i);
  }
  fail(msg) {
    const line = this.s.slice(0, this.i).split('\n').length;
    throw new SyntaxError(`bench-toml: ${msg} at line ${line}`);
  }
  skipWs() {
    while (!this.done && WS.test(this.peek())) this.i += 1;
  }
  /** Whitespace, newlines and comments — between statements. */
  skipBlank() {
    for (;;) {
      this.skipWs();
      if (this.peek() === '#') {
        while (!this.done && this.peek() !== '\n') this.i += 1;
      } else if (this.peek() === '\n' || this.peek() === '\r') {
        this.i += 1;
      } else {
        return;
      }
    }
  }
  /** After a value: only whitespace, a comment, and the line end may follow. */
  endOfLine() {
    this.skipWs();
    if (this.peek() === '#') while (!this.done && this.peek() !== '\n') this.i += 1;
    if (this.done) return;
    if (this.peek() === '\r') this.i += 1;
    if (this.peek() !== '\n') this.fail(`expected end of line, got ${JSON.stringify(this.peek())}`);
    this.i += 1;
  }
}

const ESCAPES = { b: '\b', t: '\t', n: '\n', f: '\f', r: '\r', '"': '"', '\\': '\\' };

function readEscape(c) {
  const e = c.peek(1);
  if (e in ESCAPES) {
    c.i += 2;
    return ESCAPES[e];
  }
  if (e === 'u' || e === 'U') {
    const len = e === 'u' ? 4 : 8;
    const hex = c.s.slice(c.i + 2, c.i + 2 + len);
    if (!/^[0-9A-Fa-f]+$/.test(hex) || hex.length !== len) c.fail('bad unicode escape');
    c.i += 2 + len;
    return String.fromCodePoint(parseInt(hex, 16));
  }
  return c.fail(`unknown escape \\${e}`);
}

function readBasicString(c) {
  c.i += 1;
  let out = '';
  for (;;) {
    if (c.done || c.peek() === '\n') c.fail('unterminated string');
    const ch = c.peek();
    if (ch === '"') {
      c.i += 1;
      return out;
    }
    if (ch === '\\') out += readEscape(c);
    else {
      out += ch;
      c.i += 1;
    }
  }
}

function readMultilineBasic(c) {
  c.i += 3;
  if (c.peek() === '\r') c.i += 1;
  if (c.peek() === '\n') c.i += 1;
  let out = '';
  for (;;) {
    if (c.done) c.fail('unterminated multi-line string');
    if (c.startsWith('"""')) {
      // Up to two extra quotes may precede the closing delimiter.
      let extra = 0;
      while (c.peek(3 + extra) === '"' && extra < 2) extra += 1;
      c.i += 3 + extra;
      return out + '"'.repeat(extra);
    }
    const ch = c.peek();
    if (ch === '\\') {
      // Line-ending backslash: trim it and every whitespace/newline after it.
      let j = c.i + 1;
      while (j < c.s.length && WS.test(c.s[j])) j += 1;
      if (c.s[j] === '\n' || c.s[j] === '\r') {
        while (j < c.s.length && /[ \t\r\n]/.test(c.s[j])) j += 1;
        c.i = j;
        continue;
      }
      out += readEscape(c);
    } else {
      out += ch;
      c.i += 1;
    }
  }
}

function readLiteralString(c) {
  c.i += 1;
  const end = c.s.indexOf("'", c.i);
  if (end < 0 || c.s.slice(c.i, end).includes('\n')) c.fail('unterminated literal string');
  const out = c.s.slice(c.i, end);
  c.i = end + 1;
  return out;
}

function readMultilineLiteral(c) {
  c.i += 3;
  if (c.peek() === '\r') c.i += 1;
  if (c.peek() === '\n') c.i += 1;
  const end = c.s.indexOf("'''", c.i);
  if (end < 0) c.fail('unterminated multi-line literal string');
  let stop = end;
  while (c.s[stop + 3] === "'" && stop - end < 2) stop += 1;
  const out = c.s.slice(c.i, stop);
  c.i = stop + 3;
  return out;
}

function readScalarToken(c) {
  const start = c.i;
  while (!c.done && /[A-Za-z0-9_+\-.:]/.test(c.peek())) c.i += 1;
  if (c.i === start) c.fail(`unexpected ${JSON.stringify(c.peek())}`);
  return c.s.slice(start, c.i);
}

function scalarOf(token, c) {
  if (token === 'true') return true;
  if (token === 'false') return false;
  if (/^[+-]?(inf|nan)$/.test(token)) return token.endsWith('nan') ? NaN : token.startsWith('-') ? -Infinity : Infinity;
  const num = token.replace(/_/g, '');
  if (/^[+-]?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$/.test(num)) return Number(num);
  if (/^0x[0-9A-Fa-f]+$/.test(num)) return parseInt(num.slice(2), 16);
  if (/^0o[0-7]+$/.test(num)) return parseInt(num.slice(2), 8);
  if (/^0b[01]+$/.test(num)) return parseInt(num.slice(2), 2);
  return c.fail(`unsupported value ${JSON.stringify(token)} (dates are not read)`);
}

function readValue(c) {
  const ch = c.peek();
  if (c.startsWith('"""')) return readMultilineBasic(c);
  if (ch === '"') return readBasicString(c);
  if (c.startsWith("'''")) return readMultilineLiteral(c);
  if (ch === "'") return readLiteralString(c);
  if (ch === '[') return readArray(c);
  if (ch === '{') return readInlineTable(c);
  return scalarOf(readScalarToken(c), c);
}

function readArray(c) {
  c.i += 1;
  const out = [];
  for (;;) {
    c.skipBlank();
    if (c.peek() === ']') {
      c.i += 1;
      return out;
    }
    out.push(readValue(c));
    c.skipBlank();
    if (c.peek() === ',') c.i += 1;
    else if (c.peek() !== ']') c.fail('expected , or ] in array');
  }
}

function readInlineTable(c) {
  c.i += 1;
  const out = {};
  c.skipWs();
  if (c.peek() === '}') {
    c.i += 1;
    return out;
  }
  for (;;) {
    c.skipWs();
    const path = readKeyPath(c);
    c.skipWs();
    if (c.peek() !== '=') c.fail('expected = in inline table');
    c.i += 1;
    c.skipWs();
    assign(out, path, readValue(c), c);
    c.skipWs();
    if (c.peek() === ',') {
      c.i += 1;
      continue;
    }
    if (c.peek() === '}') {
      c.i += 1;
      return out;
    }
    c.fail('expected , or } in inline table');
  }
}

function readKeyPath(c) {
  const path = [];
  for (;;) {
    c.skipWs();
    const ch = c.peek();
    let key;
    if (ch === '"') key = readBasicString(c);
    else if (ch === "'") key = readLiteralString(c);
    else {
      const start = c.i;
      while (!c.done && BARE_KEY.test(c.peek())) c.i += 1;
      if (c.i === start) c.fail('expected a key');
      key = c.s.slice(start, c.i);
    }
    path.push(key);
    c.skipWs();
    if (c.peek() !== '.') return path;
    c.i += 1;
  }
}

const isTable = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

/** Walk a dotted path, creating tables; an array-of-tables segment means its last element. */
function descend(root, path, c) {
  let cur = root;
  for (const key of path) {
    let next = cur[key];
    if (next === undefined) next = cur[key] = {};
    else if (Array.isArray(next)) next = next[next.length - 1];
    if (!isTable(next)) c.fail(`key ${key} is not a table`);
    cur = next;
  }
  return cur;
}

function assign(table, path, value, c) {
  const parent = descend(table, path.slice(0, -1), c);
  const key = path[path.length - 1];
  if (key in parent) c.fail(`key ${path.join('.')} defined twice`);
  parent[key] = value;
}

/**
 * Parse TOML text into plain data. Throws on any syntax this reader does not
 * cover (dates, for one) — a bound that parsed wrongly would draw a wrong
 * line, so refusing is the only safe failure.
 *
 * @param {string} text
 * @returns {Record<string, unknown>}
 */
export function parseToml(text) {
  const c = new Cursor(text);
  const root = {};
  let current = root;
  for (;;) {
    c.skipBlank();
    if (c.done) return root;
    if (c.startsWith('[[')) {
      c.i += 2;
      const path = readKeyPath(c);
      if (!c.startsWith(']]')) c.fail('expected ]]');
      c.i += 2;
      const parent = descend(root, path.slice(0, -1), c);
      const key = path[path.length - 1];
      const arr = (parent[key] ??= []);
      if (!Array.isArray(arr)) c.fail(`key ${key} is not an array of tables`);
      current = {};
      arr.push(current);
      c.endOfLine();
    } else if (c.peek() === '[') {
      c.i += 1;
      const path = readKeyPath(c);
      if (c.peek() !== ']') c.fail('expected ]');
      c.i += 1;
      current = descend(root, path, c);
      c.endOfLine();
    } else {
      const path = readKeyPath(c);
      if (c.peek() !== '=') c.fail('expected =');
      c.i += 1;
      c.skipWs();
      assign(current, path, readValue(c), c);
      c.endOfLine();
    }
  }
}

/**
 * The gate limits one BENCH.toml declares, keyed by gate id then checkpoint:
 *   { [gate]: { [checkpoint]: { [metric]: { min?: number, max?: number } } } }
 *
 * Only entries with a `[benchmarks.metrics]` table contribute — an
 * `unmeasured` entry has none on purpose, and its absence must stay absent.
 * A metric row without `min` or `max` is dropped: `noise` alone is not a
 * limit. Two entries declaring the same (gate, checkpoint) are refused: the
 * chart would have no way to say which bound is in force.
 *
 * @param {Record<string, unknown>} parsed output of `parseToml`
 * @param {string} [source] for the error message
 */
export function declaredLimitsOf(parsed, source = 'BENCH.toml') {
  const out = {};
  for (const entry of parsed.benchmarks ?? []) {
    const { gate, checkpoint, metrics } = entry;
    if (typeof gate !== 'string' || typeof checkpoint !== 'string') {
      throw new TypeError(`${source}: a [[benchmarks]] entry lacks a string gate or checkpoint`);
    }
    if (!isTable(metrics)) continue;
    const limits = {};
    for (const [name, row] of Object.entries(metrics)) {
      if (!isTable(row)) throw new TypeError(`${source}: metrics.${name} is not a table`);
      const lim = {};
      for (const bound of ['min', 'max']) {
        if (row[bound] === undefined) continue;
        if (typeof row[bound] !== 'number' || !Number.isFinite(row[bound])) {
          throw new TypeError(`${source}: ${gate} ${name}.${bound} is not a finite number`);
        }
        lim[bound] = row[bound];
      }
      if (Object.keys(lim).length > 0) limits[name] = lim;
    }
    if (Object.keys(limits).length === 0) continue;
    const byCheckpoint = (out[gate] ??= {});
    if (byCheckpoint[checkpoint]) {
      throw new TypeError(`${source}: ${gate} declares limits for ${checkpoint} twice`);
    }
    byCheckpoint[checkpoint] = limits;
  }
  return out;
}

/**
 * Merge per-file limit tables into one; the same (gate, checkpoint) declared
 * in two files is refused for the reason above.
 */
export function mergeDeclaredLimits(tables) {
  const out = {};
  for (const [source, table] of tables) {
    for (const [gate, byCheckpoint] of Object.entries(table)) {
      const target = (out[gate] ??= {});
      for (const [checkpoint, limits] of Object.entries(byCheckpoint)) {
        if (target[checkpoint]) {
          throw new TypeError(`${source}: ${gate} limits for ${checkpoint} already declared elsewhere`);
        }
        target[checkpoint] = limits;
      }
    }
  }
  return out;
}
