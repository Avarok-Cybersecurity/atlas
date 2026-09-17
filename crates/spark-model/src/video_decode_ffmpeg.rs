// SPDX-License-Identifier: AGPL-3.0-only

//! Frame extraction for real-world containers, via ffmpeg.
//!
//! # Why a subprocess and not a linked decoder
//!
//! The alternative evaluated was `openh264` plus a pure-Rust MP4 demuxer. It
//! builds quickly (~9 s on aarch64) and needs no runtime dependency, but it
//! covers **H.264 only** — no H.265, VP9 or AV1, which is a large share of
//! what people actually send — and Cisco's royalty-free patent grant covers
//! the binaries *Cisco* distributes, not a source build redistributed by a
//! third party. That is a licensing question for the project to answer
//! deliberately, not one to settle by adding a dependency.
//!
//! ffmpeg as a subprocess needs no build or link dependency, decodes
//! everything, and can be swapped for an in-process decoder later without
//! changing this module's signature. The cost is a runtime binary and a
//! process spawn per request, so it is OPT-IN and the absence of the binary
//! is reported by name rather than as a decode failure.
//!
//! # What is bounded
//!
//! Everything the caller controls, because the input is an untrusted byte
//! blob from an HTTP request:
//!
//! - **No shell.** Arguments are passed as argv. Nothing is interpolated into
//!   a command string, so no input can become a flag or a second command.
//! - **No temp file.** The container goes in over stdin, so there is no path
//!   to traverse, collide on, or leave behind.
//! - **`-nostdin`, and stdin is the pipe** — ffmpeg cannot reach for a
//!   terminal or block waiting on one.
//! - **Frame count** capped with `-frames:v`, so a long clip cannot decode
//!   forever.
//! - **Output size** capped while reading, and the child is killed the moment
//!   the cap is passed.
//! - **Wall clock** capped by a watchdog that kills the child; a decoder that
//!   hangs must not hold a request thread indefinitely.
//! - **Protocol whitelist** is moot because the input is a pipe, but
//!   `-f image2pipe` output and a `pipe:0` input mean ffmpeg is never asked
//!   to open a URL — the SSRF path that `remote_image` guards for stills
//!   simply does not exist here.

use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// Operator policy for subprocess decoding.
#[derive(Debug, Clone)]
pub struct FfmpegPolicy {
    pub enabled: bool,
    /// Binary to run. A name is resolved on PATH; an absolute path is used
    /// as given, so a deployment can pin a known build.
    pub binary: String,
    pub max_frames: usize,
    pub max_output_bytes: usize,
    pub timeout_secs: u64,
}

impl Default for FfmpegPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            binary: "ffmpeg".to_string(),
            max_frames: 768,
            // 768 frames of 1280x720 PNG is comfortably under this; it exists
            // to bound a pathological stream, not to size a normal one.
            max_output_bytes: 512 * 1024 * 1024,
            timeout_secs: 120,
        }
    }
}

/// The 8-byte PNG signature. Frames arrive concatenated on one stream, so
/// this is how they are separated.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// What a startup probe found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// Usable, with the version string it reported.
    Ready(String),
    /// Configured but not runnable, with why.
    Missing(String),
    /// Not asked for.
    Disabled,
}

/// Check at BOOT whether the configured decoder can actually run.
///
/// Worth doing eagerly rather than discovering it on the first video request:
/// a deployment that enabled video decoding and does not have the binary is
/// misconfigured, and the operator should learn that while reading the
/// startup log — not from a user's failed request an hour later. The check is
/// one `-version` invocation, so it costs nothing at boot.
pub fn probe(policy: &FfmpegPolicy) -> Availability {
    if !policy.enabled {
        return Availability::Disabled;
    }
    match Command::new(&policy.binary)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        Ok(out) if out.status.success() => {
            let first = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("unknown version")
                .trim()
                .to_string();
            Availability::Ready(first)
        }
        Ok(out) => {
            Availability::Missing(format!("{:?} ran but exited {}", policy.binary, out.status))
        }
        Err(e) => Availability::Missing(format!("{:?} could not be run: {e}", policy.binary)),
    }
}

/// Why the decoder could not be started, without guessing.
///
/// The message this replaced said "is ffmpeg installed and on PATH?" for EVERY
/// spawn error. That is a diagnosis, and only one of the errno values it
/// covered had ever been checked. On 2026-09-06 a CI job failed here against a
/// binary the test had just written and chmod'd 0755 in `/tmp` — installed,
/// present, executable — and the operator was told to install ffmpeg. The real
/// errno never reached the log at all.
///
/// So: keep the hint where it is actually implied (`NotFound`), and otherwise
/// say what the OS said. `PermissionDenied` means the mode or a `noexec` mount;
/// `ExecutableFileBusy` (ETXTBSY) means something still holds a write handle to
/// it, which on a busy multi-job runner is a race, not a misconfiguration.
fn spawn_failure(binary: &str, err: std::io::Error) -> anyhow::Error {
    let hint = match err.kind() {
        std::io::ErrorKind::NotFound => {
            " — is ffmpeg installed and on PATH? (set --video-ffmpeg-path to point at it)"
        }
        std::io::ErrorKind::PermissionDenied => {
            " — not executable by this user, or its filesystem is mounted noexec"
        }
        std::io::ErrorKind::ExecutableFileBusy => {
            " — another process still holds it open for writing (ETXTBSY)"
        }
        _ => "",
    };
    anyhow::anyhow!("could not run {binary:?}: {err}{hint}")
}

/// Source duration in seconds, or `None` when it cannot be determined.
///
/// ★ WHY THIS EXISTS. `decode_frames` asks ffmpeg for `-vf fps={fps}` and then
/// `-frames:v {max_frames}`. ffmpeg resamples to `fps` and hands back the FIRST
/// `max_frames` of that stream — so any clip longer than `max_frames / fps` is
/// silently TRUNCATED TO ITS PREFIX. At the defaults (768 frames, 2 fps) that
/// is 6.4 minutes: a ten-minute video is answered from its first six and a half,
/// and nothing in the reply or the logs says so. Knowing the duration lets the
/// caller lower `fps` instead, so the sample spans the WHOLE clip.
///
/// Uses ffmpeg itself rather than ffprobe: ffprobe is a separate binary that a
/// deployment may not have, and `-f null -` costs a full decode but no PNG
/// encode, which is the expensive half. Callers should treat `None` as "assume
/// it fits" rather than refusing — a probe failure must not take down a clip
/// that would have decoded.
///
/// 🪤 Shares the policy's timeout rather than adding a second one, so a
/// pathological container cannot spend 2x the operator's wall budget.
pub fn probe_duration_secs(bytes: &[u8], policy: &FfmpegPolicy) -> Option<f32> {
    if !policy.enabled {
        return None;
    }
    // 🪤 THIS IS A SECOND FFMPEG WITH A SECOND WATCHDOG, SO ITS TIMEOUT MUST
    // COME OUT OF THE CALLER'S BUDGET AND NEVER BE ADDED TO IT.
    //
    // A quarter of the budget is the slice. Below four seconds that quarter
    // rounds to zero, and the honest move is to decline rather than round it
    // up: a one-second policy that rounded up spent one second here and one
    // more in the decode, so a hung input took 2-3x the timeout the caller
    // configured. Declining restores the pre-probe behaviour exactly, which
    // is the correct answer for a budget this small — `decode_frames` then
    // keeps the requested fps, the same as any other probe failure.
    let probe_secs = policy.timeout_secs / 4;
    if probe_secs == 0 {
        return None;
    }
    use std::io::Write;
    let mut child = Command::new(&policy.binary)
        .args(["-v", "error", "-nostdin", "-i", "pipe:0", "-f", "null", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    // Feed on a thread: a large clip will not fit the pipe buffer, and writing
    // inline would deadlock against ffmpeg's own output.
    if let Some(mut sin) = child.stdin.take() {
        let owned = bytes.to_vec();
        std::thread::spawn(move || {
            let _ = sin.write_all(&owned);
        });
    }
    // 🪤 A BOUNDED WAIT, and the budget is SHARED not stacked. An unbounded
    // `wait_with_output()` here means a hanging container blocks in the PROBE,
    // before the decoder's own watchdog can ever run — the probe becomes a new
    // way to hang, which is the opposite of its purpose. (Caught by
    // `a_hanging_decoder_is_killed_at_timeout`: the kill took 11s instead of
    // being prompt.)
    //
    // The probe gets a quarter of the operator's wall, min 1s: it decodes but
    // does not PNG-encode, so it is far cheaper than the real pass, and what it
    // spends comes OUT of the budget rather than being added to it. On expiry
    // we kill and return None — "assume it fits" — which is exactly the
    // pre-probe behaviour.
    //
    // ★ stderr is taken BEFORE the child goes into the mutex. Reading it while
    // holding the lock would starve the watchdog of the very lock it needs to
    // kill, turning the guard into the deadlock it exists to prevent.
    let mut stderr = child.stderr.take();
    let child = std::sync::Arc::new(std::sync::Mutex::new(child));
    {
        let child = std::sync::Arc::clone(&child);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(probe_secs);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(50));
                let mut guard = match child.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                match guard.try_wait() {
                    Ok(Some(_)) | Err(_) => return,
                    Ok(None) => {}
                }
                if std::time::Instant::now() >= deadline {
                    let _ = guard.kill();
                    return;
                }
            }
        });
    }
    // Read outside the lock; the kill closes the pipe and ends this read.
    let mut buf = Vec::new();
    if let Some(se) = stderr.as_mut() {
        use std::io::Read;
        let _ = se.read_to_end(&mut buf);
    }
    if let Ok(mut g) = child.lock() {
        let _ = g.wait();
    }
    let err = String::from_utf8_lossy(&buf);

    // ffmpeg reports progress on stderr; the last `time=HH:MM:SS.ss` it prints
    // is the decoded length.
    let mut best: Option<f32> = None;
    for tok in err.split_whitespace() {
        if let Some(t) = tok.strip_prefix("time=") {
            let parts: Vec<&str> = t.split(':').collect();
            if parts.len() == 3
                && let (Ok(h), Ok(m), Ok(sec)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                )
            {
                best = Some(h * 3600.0 + m * 60.0 + sec);
            }
        }
    }
    best.filter(|d| d.is_finite() && *d > 0.0)
}

/// Decode `bytes` to RGB frames sampled at `target_fps`.
///
/// ffmpeg performs the temporal sampling itself (`-vf fps=`), which is both
/// faster and more accurate than decoding everything and discarding most of
/// it — and it means the caller does not have to know the source frame rate.
/// The returned frames are therefore ALREADY at `target_fps`.
pub fn decode_frames(
    bytes: &[u8],
    target_fps: f32,
    policy: &FfmpegPolicy,
) -> Result<Vec<RgbImage>> {
    ensure!(
        policy.enabled,
        "this container needs ffmpeg to decode and subprocess decoding is disabled; \
         pass --video-allow-ffmpeg to enable it, or send an animated GIF"
    );
    let requested_fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        2.0
    };
    // ── Span the clip instead of truncating it ──
    //
    // `-frames:v max_frames` keeps the FIRST max_frames of the resampled
    // stream, so at the requested fps a long clip loses its tail entirely.
    // Lower the sample rate just enough that max_frames covers the whole
    // duration: the caller asked for "this many frames at most", not "the
    // first this-many".
    //
    // A probe failure leaves `requested_fps` untouched, which is exactly the
    // previous behaviour — this can only ever widen coverage, never narrow it.
    let probe_started = std::time::Instant::now();
    let fps = match probe_duration_secs(bytes, policy) {
        Some(dur) if dur > 0.0 && policy.max_frames > 0 => {
            let spanning = policy.max_frames as f32 / dur;
            if spanning < requested_fps {
                tracing::info!(
                    duration_s = dur,
                    requested_fps,
                    effective_fps = spanning,
                    max_frames = policy.max_frames,
                    "video longer than the frame budget: lowering fps so the sample \
                     spans the whole clip instead of truncating to its first {:.1}s",
                    policy.max_frames as f32 / requested_fps
                );
                spanning
            } else {
                requested_fps
            }
        }
        _ => requested_fps,
    };
    // ── The decode gets what the probe left, so `timeout_secs` still bounds
    //    THIS FUNCTION and not merely its last subprocess ──
    //
    // Without this subtraction the probe's watchdog runs BESIDE the decoder's
    // rather than inside it, and a hung input pays both. That is not a
    // tightness question: a caller who sets 120s and waits 150s has been told
    // something untrue by the flag. Measured as a 1s policy taking 3.0-3.1s
    // before the fix, flaky 2 runs in 3 against the suite's 3s assertion.
    //
    // The floor is one second: a probe that consumed the whole budget still
    // owes the decode a chance to report a real error rather than a timeout
    // it never had the time to avoid.
    let spent = probe_started.elapsed().as_secs();
    let trimmed;
    let policy = if spent == 0 {
        policy
    } else {
        trimmed = FfmpegPolicy {
            timeout_secs: policy.timeout_secs.saturating_sub(spent).max(1),
            ..policy.clone()
        };
        &trimmed
    };

    let mut child = Command::new(&policy.binary)
        .args([
            "-v",
            "error",
            // Never touch a terminal: without this ffmpeg can block on a
            // prompt when it thinks stdin is interactive.
            "-nostdin",
            "-i",
            "pipe:0",
            "-vf",
            &format!("fps={fps}"),
            "-frames:v",
            &policy.max_frames.to_string(),
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| spawn_failure(&policy.binary, err))?;

    let mut stdin = child.stdin.take().context("no stdin pipe")?;
    let mut stdout = child.stdout.take().context("no stdout pipe")?;
    let mut stderr = child.stderr.take().context("no stderr pipe")?;

    // Feed the container on a thread. It MUST be concurrent with reading:
    // ffmpeg writes output while still consuming input, so a write-then-read
    // sequence deadlocks as soon as the output pipe buffer fills.
    let input = bytes.to_vec();
    let writer = std::thread::spawn(move || {
        // A broken pipe here is normal — ffmpeg stops reading once it has the
        // frames it was asked for — so the error is deliberately dropped.
        let _ = stdin.write_all(&input);
        drop(stdin);
    });

    let child = Arc::new(Mutex::new(child));
    let watchdog = {
        let child = Arc::clone(&child);
        let secs = policy.timeout_secs.max(1);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let mut guard = match child.lock() {
                    Ok(g) => g,
                    Err(_) => return false,
                };
                match guard.try_wait() {
                    Ok(Some(_)) => return false, // exited on its own
                    Ok(None) => {}
                    Err(_) => return false,
                }
                if std::time::Instant::now() >= deadline {
                    let _ = guard.kill();
                    return true; // we killed it
                }
            }
        })
    };

    // Read with a hard cap. `take` bounds it without trusting the child.
    let mut out = Vec::new();
    let read_res = (&mut stdout)
        .take(policy.max_output_bytes as u64 + 1)
        .read_to_end(&mut out);

    // KILL FIRST, then drain stderr. Order matters and getting it wrong
    // deadlocks: once the cap is hit we stop reading stdout, so the child
    // blocks writing into a full pipe and never closes stderr — and a
    // `read_to_string` on stderr then waits for an EOF that only the watchdog
    // will ever cause. Killing here turns a 120-second hang into an immediate
    // error. (Draining stdout instead would defeat the cap, which is the
    // thing being enforced.)
    let over_cap = out.len() > policy.max_output_bytes;
    if over_cap && let Ok(mut g) = child.lock() {
        let _ = g.kill();
    }

    let mut err_text = String::new();
    let _ = stderr.read_to_string(&mut err_text);
    let _ = writer.join();

    let status = {
        let mut g = child
            .lock()
            .map_err(|_| anyhow::anyhow!("decoder lock poisoned"))?;
        g.wait().context("waiting for the decoder")?
    };
    let timed_out = watchdog.join().unwrap_or(false);

    read_res.context("reading decoded frames")?;
    ensure!(
        !timed_out,
        "decoding exceeded {}s and was stopped",
        policy.timeout_secs
    );
    ensure!(
        !over_cap,
        "decoded output exceeded the {}-byte cap",
        policy.max_output_bytes
    );
    if !status.success() {
        let why = err_text.lines().next_back().unwrap_or("no detail").trim();
        bail!("decoder failed: {why}");
    }

    let frames = split_png_stream(&out)?;
    ensure!(
        !frames.is_empty(),
        "the container decoded to zero frames (is there a video stream?)"
    );
    Ok(frames)
}

/// Split a concatenated PNG stream into images.
///
/// Splitting on the signature rather than parsing IEND chunks: a PNG's
/// payload can legitimately contain the IEND byte pattern, whereas the
/// 8-byte signature only appears at a file start in a well-formed stream
/// produced by `image2pipe`.
fn split_png_stream(buf: &[u8]) -> Result<Vec<RgbImage>> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + PNG_MAGIC.len() <= buf.len() {
        if buf[i..i + PNG_MAGIC.len()] == PNG_MAGIC {
            starts.push(i);
            i += PNG_MAGIC.len();
        } else {
            i += 1;
        }
    }
    let mut frames = Vec::with_capacity(starts.len());
    for (n, &s) in starts.iter().enumerate() {
        let e = starts.get(n + 1).copied().unwrap_or(buf.len());
        let img = image::load_from_memory_with_format(&buf[s..e], image::ImageFormat::Png)
            .with_context(|| format!("frame {n} did not decode as PNG"))?;
        frames.push(img.to_rgb8());
    }
    Ok(frames)
}

#[cfg(test)]
#[path = "video_decode_ffmpeg_tests.rs"]
mod tests;
