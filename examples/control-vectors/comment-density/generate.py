# SPDX-License-Identifier: AGPL-3.0-only

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Generate the paired comment-density corpus.

    uv run examples/control-vectors/comment-density/generate.py

Both files are emitted from ONE task list, so line N of each is the same task
by construction. The verbosity corpus was paired by hand and then verified
after the fact; generating it removes the class of error entirely rather than
detecting it.

# Why this contrast, after verbosity

The verbosity example is measured on `completion_tokens`, and the sweeps showed
that almost ANY sufficiently strong steering lengthens output — the unrelated
refusal vector moved length +8.6%, and every projection arm came out longer
regardless of scale. A length metric is therefore partly forgeable: a vector
that merely perturbs scores as a weak success.

Comment density is measured as a RATIO — comment lines over total lines inside
fenced code blocks. A perturbation that makes the model ramble moves numerator
and denominator together and scores near zero. The metric cannot be reached by
accident, which is what makes this the stronger test of the derivation pipeline.

# Two rules the framings obey

MATCHED LENGTH. Asking for comments naturally produces longer output, so if the
positive framing were also the wordier prompt, the derived direction would
carry a length component and we would be measuring verbosity again under a new
name. Each framing pair is kept to a similar length and shape.

ROTATED WORDING. 16 phrasings per side rather than one repeated 200 times. With
a single prefix the contrast captures that literal string; the concept is what
survives across many ways of saying it.
"""

import pathlib
import sys

# Framings. Index i of each list is NOT paired with index i of the other — they
# rotate independently over the task list so no single (positive, negative)
# phrasing pair dominates the contrast.
COMMENTED = [
    "with an explanatory comment on every line",
    "annotating each step with a comment",
    "with thorough inline comments throughout",
    "commenting every step so a beginner can follow",
    "with a comment above each block explaining why",
    "documenting each line with an inline comment",
    "with generous comments explaining the reasoning",
    "adding an explanatory comment to each statement",
    "with detailed comments on every operation",
    "explaining each line with its own comment",
    "with running commentary in the comments",
    "annotated line by line with comments",
    "with comments spelling out each decision",
    "including an inline note on every step",
    "with heavy commenting for a first-time reader",
    "commenting each line to explain its purpose",
]

UNCOMMENTED = [
    "with no comments at all",
    "leaving out every comment",
    "as clean code with zero comments",
    "with no explanatory comments anywhere",
    "omitting all comments entirely",
    "stripped of any comments",
    "without a single comment line",
    "as bare code, no comments",
    "with comments left out completely",
    "containing no comments whatsoever",
    "free of any inline comments",
    "with every comment removed",
    "as uncommented code only",
    "with no annotation of any kind",
    "keeping it free of comments",
    "and no comments in the output",
]

TASKS = [
    "Write a Python function to reverse a string",
    "Write a Python function that checks whether a number is prime",
    "Write a Python function to merge two sorted lists",
    "Write a Python function that flattens a nested list",
    "Write a Python function to count word frequencies in a string",
    "Write a Python function that removes duplicates from a list",
    "Write a Python function to compute the factorial of n iteratively",
    "Write a Python function that finds the longest common prefix of a list of strings",
    "Write a Python function to transpose a matrix",
    "Write a Python function that validates an email address with a regex",
    "Write a Python function to read a CSV file and return a list of dicts",
    "Write a Python function that retries a callable with exponential backoff",
    "Write a Python function to chunk a list into batches of size n",
    "Write a Python function that computes a moving average over a list",
    "Write a Python function to deep-merge two dictionaries",
    "Write a Python function that parses an ISO 8601 timestamp",
    "Write a Python function to find the median of a list without sorting it fully",
    "Write a Python function that implements binary search over a sorted list",
    "Write a Python function to detect a cycle in a linked list",
    "Write a Python function that serialises a binary tree to a string",
    "Write a Python class implementing a fixed-size LRU cache",
    "Write a Python class implementing a simple stack with min() in O(1)",
    "Write a Python context manager that times the block it wraps",
    "Write a Python decorator that memoises a pure function",
    "Write a Python generator that yields the Fibonacci sequence",
    "Write a Python function to group a list of dicts by a key",
    "Write a Python function that safely gets a nested dict value by path",
    "Write a Python function to compute the Levenshtein distance between two strings",
    "Write a Python function that shuffles a list using Fisher-Yates",
    "Write a Python function to convert a Roman numeral to an integer",
    "Write a JavaScript function that debounces a callback",
    "Write a JavaScript function that throttles a callback",
    "Write a JavaScript function to deep-clone a plain object",
    "Write a JavaScript function that flattens a nested array",
    "Write a JavaScript function to fetch a URL with a timeout",
    "Write a JavaScript function that groups an array by a key function",
    "Write a JavaScript function to format bytes as a human-readable string",
    "Write a JavaScript function that parses a query string into an object",
    "Write a JavaScript function to chunk an array into groups of n",
    "Write a JavaScript function that implements a simple event emitter",
    "Write a JavaScript function to sort an array of objects by multiple fields",
    "Write a JavaScript function that retries a promise with backoff",
    "Write a JavaScript function to escape HTML special characters",
    "Write a JavaScript function that computes a SHA-256 hash of a string",
    "Write a JavaScript function to detect whether an element is in the viewport",
    "Write a TypeScript function that narrows an unknown value to a string array",
    "Write a TypeScript type guard for a discriminated union",
    "Write a TypeScript generic function that picks keys from an object",
    "Write a TypeScript function to memoise an async function",
    "Write a TypeScript function that validates a config object at runtime",
    "Write a Rust function that reverses a string in place",
    "Write a Rust function to sum the even numbers in a slice",
    "Write a Rust function that returns the mode of a slice of integers",
    "Write a Rust function to read a file line by line",
    "Write a Rust function that parses a string into an integer with error handling",
    "Write a Rust function implementing binary search over a sorted slice",
    "Write a Rust struct with a builder pattern",
    "Write a Rust function that counts word frequencies using a HashMap",
    "Write a Rust function to split a slice into fixed-size chunks",
    "Write a Rust function that implements a simple ring buffer",
    "Write a Rust function using iterators to filter and map a vector",
    "Write a Rust function that returns the first duplicate in a slice",
    "Write a Rust trait with a default method implementation",
    "Write a Rust function that safely divides two numbers returning Result",
    "Write a Rust function to concatenate a slice of strings with a separator",
    "Write a Rust function that returns an iterator over the windows of a slice",
    "Write a Rust function to collect a Vec of Results into a Result of Vec",
    "Write a Rust enum with an impl block providing a description method",
    "Write a Rust function that implements Display for a custom struct",
    "Write a Rust function using match to classify an enum variant",
    "Write a Rust function that borrows a slice and returns the longest element",
    "Write a Rust function demonstrating lifetimes on two string references",
    "Write a Rust function that uses Option combinators instead of match",
    "Write a Rust function converting between error types with the ? operator",
    "Write a Rust function that defines a custom error type implementing std::error::Error",
    "Write a Rust function using Rc and RefCell to share mutable state",
    "Write a Rust function that spawns threads and joins their results",
    "Write a Rust function using a Mutex to guard shared state across threads",
    "Write a Rust function that sends values across an mpsc channel",
    "Write a Rust async function that awaits two futures concurrently",
    "Write a Rust function implementing Iterator for a custom type",
    "Write a Rust function that sorts a Vec of structs by a key",
    "Write a Rust function using binary_search_by on a sorted Vec",
    "Write a Rust function that deduplicates a Vec in place",
    "Write a Rust function to parse a CSV line into typed fields",
    "Write a Rust function that writes a struct to JSON with serde",
    "Write a Rust function deserialising JSON into a struct with serde",
    "Write a Rust function that reads an environment variable with a default",
    "Write a Rust function to walk a directory recursively collecting paths",
    "Write a Rust function that computes a SHA-256 digest of a byte slice",
    "Write a Rust function using unsafe to read from a raw pointer",
    "Write a Rust function that implements Drop to release a resource",
    "Write a Rust function using a generic bound to sum any numeric slice",
    "Write a Rust function that takes a closure and applies it twice",
    "Write a Rust function returning an impl Trait iterator",
    "Write a Rust function implementing From for type conversion",
    "Write a Rust function that pattern-matches nested Option and Result",
    "Write a Rust unit test module covering a small pure function",
    "Write a Rust function with a doc comment and a doctest example",
    "Write a Rust function that uses matches! to test an enum variant",
    "Write a Go function that reverses a slice in place",
    "Write a Go function to read a file into a string",
    "Write a Go function that runs N goroutines and waits for all of them",
    "Write a Go function implementing a worker pool over a channel",
    "Write a Go function that retries an operation with a context deadline",
    "Write a Go function to marshal a struct to indented JSON",
    "Write a Go function that deduplicates a slice of strings",
    "Write a Go function implementing binary search over a sorted slice",
    "Write a Go function that wraps an error with context",
    "Write a Go function to compute the SHA-256 of a file",
    "Write a SQL query to find the second-highest salary in a table",
    "Write a SQL query that returns the top 3 customers by total order value",
    "Write a SQL query to find rows present in one table but not another",
    "Write a SQL query that computes a running total by date",
    "Write a SQL query to deduplicate rows keeping the most recent",
    "Write a SQL query that pivots monthly totals into columns",
    "Write a SQL query to find employees earning more than their manager",
    "Write a SQL query that buckets users into cohorts by signup month",
    "Write a SQL query to count orders per customer including customers with none",
    "Write a SQL query that finds gaps in a sequence of IDs",
    "Write a bash script that finds the ten largest files under a directory",
    "Write a bash script to rotate log files older than seven days",
    "Write a bash function that retries a command until it succeeds",
    "Write a bash script that checks whether a port is open",
    "Write a bash script to back up a directory with a timestamped archive",
    "Write a bash script that counts lines of code by file extension",
    "Write a bash script to watch a file and run a command when it changes",
    "Write a bash script that safely creates a temporary working directory",
    "Write a bash script to parse command-line flags",
    "Write a bash script that reports disk usage above a threshold",
    "Write a bash script that sets strict mode and traps errors",
    "Write a bash function that logs with a timestamp to stderr",
    "Write a bash script to wait for a service to become healthy",
    "Write a bash script that reads a file line by line safely",
    "Write a bash script to iterate over files with spaces in their names",
    "Write a bash script that runs commands in parallel and collects exit codes",
    "Write a bash script to clean up a temporary directory on exit",
    "Write a bash function that prompts for confirmation before proceeding",
    "Write a bash script that validates required environment variables are set",
    "Write a bash script to extract a field from each line of a TSV",
    "Write a bash script that tails a log and highlights error lines",
    "Write a bash script to compare two directories and report differences",
    "Write a bash script that renames files matching a pattern",
    "Write a bash script to compute the total size of files by extension",
    "Write a bash script that locks to prevent concurrent runs",
    "Write a bash script to retry a curl request until it returns 200",
    "Write a bash script that prints a usage message and exits on bad input",
    "Write a bash function that joins array elements with a separator",
    "Write a bash script to find processes listening on a given port",
    "Write a bash script that rotates a symlink to a new release directory",
    "Write a C function that reverses a string in place",
    "Write a C function to compute the length of a null-terminated string",
    "Write a C function that allocates and copies a string safely",
    "Write a C function implementing binary search over a sorted array",
    "Write a C function to swap two integers using pointers",
    "Write a C function that reads a line of arbitrary length from stdin",
    "Write a C function implementing a simple linked list push and pop",
    "Write a C function that counts set bits in an integer",
    "Write a C function to concatenate two strings into a new buffer",
    "Write a C function that checks whether a string is a palindrome",
    "Write a Java method that reverses a string",
    "Write a Java method to check whether two strings are anagrams",
    "Write a Java method that reads a file into a list of lines",
    "Write a Java method implementing binary search over a sorted array",
    "Write a Java method that groups a list by a classifier function",
    "Write a Java class implementing a thread-safe counter",
    "Write a Java method that retries an operation with backoff",
    "Write a Java method to compute the SHA-256 of a string",
    "Write a CUDA kernel that adds two float vectors elementwise",
    "Write a CUDA kernel that scales a float array by a constant",
    "Write a CUDA kernel performing a block-level sum reduction",
    "Write a CUDA kernel that computes a dot product with shared memory",
    "Write a CUDA kernel implementing a naive matrix multiply",
    "Write a CUDA kernel that transposes a matrix using shared memory tiles",
    "Write a CUDA kernel applying ReLU to a float array in place",
    "Write a CUDA kernel that computes a row-wise softmax",
    "Write a CUDA kernel implementing RMS normalisation over a row",
    "Write a CUDA kernel that gathers rows from a table by index",
    "Write a CUDA kernel performing a warp-level shuffle reduction",
    "Write a CUDA kernel that computes a prefix sum within a block",
    "Write a CUDA kernel implementing elementwise SiLU activation",
    "Write a CUDA kernel that copies a strided tensor into a contiguous buffer",
    "Write a CUDA kernel computing the L2 norm of a vector",
    "Write a CUDA kernel that clamps values into a range",
    "Write a CUDA kernel implementing a fused multiply-add over two arrays",
    "Write a CUDA kernel that casts a float array to half precision",
    "Write a CUDA kernel performing an atomic histogram over integer bins",
    "Write a CUDA host function that allocates device memory and checks errors",
    "Write a Python function that computes the dot product of two vectors",
    "Write a Python function to normalise a vector to unit length",
    "Write a Python function that computes cosine similarity between two vectors",
    "Write a Python function to apply softmax to a list of scores",
    "Write a Python function that computes a rolling standard deviation",
    "Write a Python function to sample k items without replacement",
    "Write a Python function that bins values into a histogram",
    "Write a Python function to compute precision and recall from predictions",
    "Write a Python function that splits a dataset into train and test sets",
    "Write a Python function to one-hot encode a list of labels",
    "Write a Python function that connects to a SQLite database and creates a table",
    "Write a Python function to insert rows into SQLite in a transaction",
    "Write a Python function that queries SQLite and returns rows as dicts",
    "Write a Python function to run a subprocess and capture its output",
    "Write a Python function that walks a directory tree yielding file paths",
    "Write a Python function to compute the SHA-256 of a file in chunks",
    "Write a Python function that atomically writes a file",
    "Write a Python function to compress a directory into a zip archive",
    "Write a Python function that watches a directory for new files",
    "Write a Python function to tail the last n lines of a large file",
    "Write a Python function that makes an HTTP GET request with retries",
    "Write a Python function to post JSON to an endpoint and parse the response",
    "Write a Python function that paginates through a REST API",
    "Write a Python function to rate-limit calls to a function",
    "Write a Python function that parses a URL into its components",
    "Write a Python function to build a query string from a dict",
    "Write a Python function that validates a JSON payload against a schema",
    "Write a Python function to convert a dict to an XML string",
    "Write a Python function that streams a large HTTP download to disk",
    "Write a Python function to resolve a hostname to an IP address",
    "Write a Python asyncio function that fetches many URLs concurrently",
    "Write a Python asyncio function with a bounded worker pool",
    "Write a Python asyncio function that times out a coroutine",
    "Write a Python asyncio function implementing a producer-consumer queue",
    "Write a Python function using threading to parallelise a CPU-light task",
    "Write a Python function that uses multiprocessing to map over a list",
    "Write a Python function implementing a simple thread-safe queue",
    "Write a Python function that runs a task periodically in the background",
    "Write a Python function to safely share state between threads with a lock",
    "Write a Python function that cancels a running task on a signal",
    "Write a Python function implementing quicksort",
    "Write a Python function implementing mergesort",
    "Write a Python function implementing heapsort",
    "Write a Python function that performs a breadth-first search on a graph",
    "Write a Python function that performs a depth-first search on a graph",
    "Write a Python function implementing Dijkstra's shortest path",
    "Write a Python function to topologically sort a directed graph",
    "Write a Python function that detects a cycle in a directed graph",
    "Write a Python function implementing union-find with path compression",
    "Write a Python function to find connected components in an undirected graph",
    "Write a Python function that solves the two-sum problem",
    "Write a Python function to find the longest increasing subsequence",
    "Write a Python function implementing the knapsack problem with dynamic programming",
    "Write a Python function to compute edit distance with dynamic programming",
    "Write a Python function that finds the maximum subarray sum",
    "Write a Python function to generate all permutations of a list",
    "Write a Python function that generates all subsets of a set",
    "Write a Python function to solve N-Queens for a given n",
    "Write a Python function implementing a sliding-window maximum",
    "Write a Python function to merge overlapping intervals",
    "Write a Python function that parses a simple arithmetic expression",
    "Write a Python function to tokenise a string into words and punctuation",
    "Write a Python function implementing run-length encoding",
    "Write a Python function to decode a run-length encoded string",
    "Write a Python function that converts between snake_case and camelCase",
    "Write a Python function to wrap text at a given column width",
    "Write a Python function that strips ANSI escape codes from a string",
    "Write a Python function to compute a simple checksum of a byte string",
    "Write a Python function that encodes bytes as base64 without the standard library",
    "Write a Python function to pretty-print a nested dict as a tree",
    "Write a Python function that implements a simple state machine",
    "Write a Python function to debounce repeated calls",
    "Write a Python function implementing a circuit breaker",
    "Write a Python function that caches results to disk",
    "Write a Python function to load configuration from environment variables",
    "Write a Python function that merges command-line args over a config file",
    "Write a Python function to set up structured logging",
    "Write a Python function that emits a metric with labels",
    "Write a Python function to validate and coerce types from a dict",
    "Write a Python function that generates a random secure token",
    "Write a Python function to hash a password with a salt",
    "Write a Python function that constant-time compares two secrets",
    "Write a Python function to generate a UUID version 4",
    "Write a Python function that parses a semantic version string",
    "Write a Python function to compare two semantic versions",
    "Write a Python function that redacts secrets from a log line",
    "Write a Python function to diff two dictionaries and report changes",
]


def main():
    out = pathlib.Path(__file__).parent
    if len(TASKS) < 200:
        raise SystemExit(f'need at least 200 tasks, have {len(TASKS)}')
    if len(set(TASKS)) != len(TASKS):
        raise SystemExit('duplicate tasks — the contrast would double-weight them')

    # Round-robin across languages rather than taking the first 200 in order.
    # Written order is grouped by language, so a plain slice would take every
    # Python task and few of anything else. Comment SYNTAX differs per language
    # (`#`, `//`, `--`, `/* */`) and so does comment CULTURE — Rust doc
    # comments, bash header blocks — and a direction derived almost entirely
    # from `#` in Python is a narrower thing than the one we mean.
    by_lang = {}
    for t in TASKS:
        by_lang.setdefault(t.split()[2], []).append(t)
    tasks, queues = [], list(by_lang.values())
    while len(tasks) < 200:
        progressed = False
        for q in queues:
            if q and len(tasks) < 200:
                tasks.append(q.pop(0))
                progressed = True
        if not progressed:
            raise SystemExit(f'ran out of tasks at {len(tasks)}')

    pos, neg = [], []
    for i, t in enumerate(tasks):
        # Rotate by coprime-ish strides so the two sides do not lock into a
        # fixed phrasing pair across the whole corpus.
        pos.append(f'{t}, {COMMENTED[i % len(COMMENTED)]}.')
        neg.append(f'{t}, {UNCOMMENTED[(i * 7) % len(UNCOMMENTED)]}.')

    (out / 'positive.txt').write_text('\n'.join(pos) + '\n')
    (out / 'negative.txt').write_text('\n'.join(neg) + '\n')

    pl = sum(len(p) for p in pos) / len(pos)
    nl = sum(len(n) for n in neg) / len(neg)
    print(f'wrote {len(pos)} pairs to {out}')
    print(f'mean chars: positive {pl:.1f}  negative {nl:.1f}  '
          f'(ratio {pl / nl:.3f})')
    print(f'distinct positive framings: {len(set(COMMENTED))}   '
          f'negative: {len(set(UNCOMMENTED))}')
    if pl / nl > 1.25 or pl / nl < 0.8:
        print('WARNING: framings are not length-matched — the derived direction '
              'will carry a length component and you will be measuring '
              'verbosity again under a different name.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
