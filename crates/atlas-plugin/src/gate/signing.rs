// SPDX-License-Identifier: AGPL-3.0-only
//! Ed25519 signatures binding a gate record to the commit it measured.
//!
//! # What this proves, and what it does not
//!
//! The identity is generated on the box, unprompted, so **a signature does not
//! prove who ran the campaign** — anyone can mint a keypair. Saying otherwise
//! would be the security theatre this module is trying to avoid.
//!
//! What it does prove is what the gate actually needs: every record in a pull
//! request was produced by ONE signer at ONE commit, and none was altered after
//! it was written. A record cannot be hand-edited, and it cannot be lifted from
//! one commit and presented as evidence for another — the commit sha is inside
//! the signed message, so re-pointing it invalidates the signature.
//!
//! The trust decision is deliberately moved to the one moment a human is already
//! looking: the FIRST time a fingerprint appears, it arrives as a one-line
//! addition under `.github/record-signers/`, in the diff, covered by the seal.
//! After that the box is silent forever and the operator never thinks about keys.
//!
//! # Why `ring`
//!
//! It is already in `Cargo.lock` (via `ureq` → `rustls`), so declaring it here
//! adds a dependency EDGE and zero packages — the constraint this design was
//! given. Ed25519 is compiled unconditionally, and `Apache-2.0 AND ISC` already
//! clears `deny.toml`'s allow-list.
use anyhow::{Context, Result, bail};
use ring::signature::{ED25519, Ed25519KeyPair, KeyPair, UnparsedPublicKey};
use std::path::{Path, PathBuf};

/// Where a signer's public key is published, relative to the repo root.
pub const REGISTRY_DIR: &str = ".github/record-signers";

/// Sidecar format version. Bumping it is a breaking change to every committed
/// `.sig`, so it is a constant rather than a magic number in two places.
const SIG_VERSION: u32 = 1;

/// A record whose commit is older than this is accepted without a signature.
///
/// ★ The migration, made explicit rather than implied. 600-odd records predate
/// signing; demanding a `.sig` for all of them would fail every gate on day one,
/// and back-signing them would assert something nobody witnessed. So: records
/// recorded before the cutover verify as they always did, records after it must
/// carry a valid signature, and the window closes by itself as records age out.
/// `signing_tests::the_cutover_is_pinned` fails if this moves, so it cannot drift
/// quietly into "signatures optional forever".
/// The value is one minute past the newest record that predates signing
/// (`1788268309`, video-fidelity, 2026-09-01T13:11:49Z). Not a round number on
/// purpose: picking a comfortable-looking hour in the future would have exempted
/// the records THIS feature's own certification produces, shipping a signature
/// check that had never once been exercised by the thing it guards.
pub const SIGNATURE_REQUIRED_AFTER: u64 = 1_788_268_400; // 2026-09-01T13:13:20Z

/// The signing identity for this machine.
pub struct Identity {
    keypair: Ed25519KeyPair,
    fingerprint: String,
}

impl Identity {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn public_key_bytes(&self) -> &[u8] {
        self.keypair.public_key().as_ref()
    }
}

/// Short, stable name for a public key: the first 16 hex of its SHA-256.
///
/// Not a security boundary — the full key is what verifies. This is the handle
/// that appears in a filename and in an error message, and 64 bits is far more
/// than enough to keep a handful of build boxes distinct.
pub fn fingerprint_of(public_key: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(public_key);
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// `<record>.json` → `<record>.json.sig`.
///
/// APPENDS. `variant_slug` deliberately keeps `.` in a model name, so records
/// like `2026-08-05-abc-qwen3.8-27b.json` exist and `with_extension` would eat
/// the `8-27b.json` tail.
pub fn sig_path(record: &Path) -> PathBuf {
    let mut s = record.to_path_buf().into_os_string();
    s.push(".sig");
    PathBuf::from(s)
}

/// The bytes a signature covers: the record file exactly as written, then the
/// commit sha.
///
/// The sha is also inside the file, so appending it is redundant for integrity —
/// it is domain separation, and it makes the binding explicit at the one place a
/// reader looks to find out what was signed.
fn message(record_bytes: &[u8], git_sha: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(record_bytes.len() + git_sha.len());
    msg.extend_from_slice(record_bytes);
    msg.extend_from_slice(git_sha.as_bytes());
    msg
}

/// Load this machine's identity, creating one on first use.
///
/// Silent on success: an operator should never learn this exists. The only time
/// this speaks is when it cannot write, and then it says why rather than
/// producing an unsigned record that fails a gate an hour later.
pub fn load_or_create(atlas_home: &Path) -> Result<Identity> {
    let dir = atlas_home.join("identity");
    let key_path = dir.join("ed25519.pk8");

    let pkcs8 = if key_path.exists() {
        std::fs::read(&key_path).with_context(|| format!("reading {}", key_path.display()))?
    } else {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let rng = ring::rand::SystemRandom::new();
        let doc = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|_| anyhow::anyhow!("generating an Ed25519 key"))?;
        write_private(&key_path, doc.as_ref())?;
        doc.as_ref().to_vec()
    };

    let keypair = Ed25519KeyPair::from_pkcs8(&pkcs8)
        .map_err(|e| anyhow::anyhow!("{} is not a usable Ed25519 key: {e}", key_path.display()))?;
    let fingerprint = fingerprint_of(keypair.public_key().as_ref());
    Ok(Identity {
        keypair,
        fingerprint,
    })
}

/// Write key material at `0600`.
///
/// The mode is set through `OpenOptions` rather than after the fact: a
/// `write`-then-`set_permissions` leaves the key world-readable for the window
/// between the two calls.
#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    f.write_all(bytes)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

/// Publish this identity's public key into the repo, if it is not already there.
///
/// Idempotent, and returns whether it wrote — the caller tells the operator to
/// commit a new signer exactly once, not on every run.
pub fn register(root: &Path, identity: &Identity) -> Result<bool> {
    let dir = root.join(REGISTRY_DIR);
    let path = dir.join(format!("{}.pub", identity.fingerprint()));
    if path.exists() {
        return Ok(false);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let armored = format!(
        "# Atlas record signer {}\n# Ed25519 public key, base64. Added automatically on first use.\n{}\n",
        identity.fingerprint(),
        b64(identity.public_key_bytes())
    );
    std::fs::write(&path, armored).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

/// Sign a record already on disk. Returns the sidecar path.
pub fn sign_record(identity: &Identity, record: &Path, git_sha: &str) -> Result<PathBuf> {
    let bytes = std::fs::read(record).with_context(|| format!("reading {}", record.display()))?;
    let sig = identity.keypair.sign(&message(&bytes, git_sha));
    let out = sig_path(record);
    let body = format!(
        "{{\"v\":{SIG_VERSION},\"key\":\"{}\",\"sig\":\"{}\"}}\n",
        identity.fingerprint(),
        b64(sig.as_ref())
    );
    std::fs::write(&out, body).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

/// What a signature turned out to be. `Exempt` is not a pass with a shrug — it
/// names the one condition under which an unsigned record is still evidence.
#[derive(Debug, PartialEq, Eq)]
pub enum Verified {
    /// Signed, and the signature checks out under a registered key.
    Signed { fingerprint: String },
    /// Recorded before the cutover, so no signature was ever required.
    Exempt,
}

/// Verify a record's sidecar.
///
/// `recorded_at` is the record's own timestamp, used only for the migration
/// exemption. Everything else is a hard error: a record after the cutover with
/// no sidecar, a bad signature, or a key nobody has registered all fail, and the
/// message says which so the fix is obvious.
pub fn verify_record(
    root: &Path,
    record: &Path,
    git_sha: &str,
    recorded_at: u64,
) -> Result<Verified> {
    let sidecar = sig_path(record);
    if !sidecar.exists() {
        if recorded_at < SIGNATURE_REQUIRED_AFTER {
            return Ok(Verified::Exempt);
        }
        bail!(
            "{} has no signature ({}). Re-run the benchmark, or commit the .sig \
             the CLI wrote beside the record.",
            record.file_name().unwrap_or_default().to_string_lossy(),
            sidecar.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    let raw = std::fs::read_to_string(&sidecar)
        .with_context(|| format!("reading {}", sidecar.display()))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", sidecar.display()))?;
    let version = parsed.get("v").and_then(serde_json::Value::as_u64);
    if version != Some(u64::from(SIG_VERSION)) {
        bail!(
            "{} is signature format v{:?}; this build understands v{SIG_VERSION}",
            sidecar.display(),
            version
        );
    }
    let fingerprint = parsed
        .get("key")
        .and_then(serde_json::Value::as_str)
        .context("signature names no key")?;
    let sig = unb64(
        parsed
            .get("sig")
            .and_then(serde_json::Value::as_str)
            .context("signature has no sig field")?,
    )?;

    let key_path = root.join(REGISTRY_DIR).join(format!("{fingerprint}.pub"));
    if !key_path.exists() {
        bail!(
            "{} is signed by {fingerprint}, which is not in {REGISTRY_DIR}. A new \
             signer must be committed there — it is one line, and reviewing it is \
             the point.",
            record.file_name().unwrap_or_default().to_string_lossy()
        );
    }
    let public = read_registered_key(&key_path)?;

    let bytes = std::fs::read(record).with_context(|| format!("reading {}", record.display()))?;
    UnparsedPublicKey::new(&ED25519, &public)
        .verify(&message(&bytes, git_sha), &sig)
        .map_err(|_| {
            anyhow::anyhow!(
                "{} does not match its signature. The record, or the commit it \
                 names, was changed after it was measured.",
                record.file_name().unwrap_or_default().to_string_lossy()
            )
        })?;

    Ok(Verified::Signed {
        fingerprint: fingerprint.to_string(),
    })
}

/// Read a registered key, skipping the `#` header lines.
fn read_registered_key(path: &Path) -> Result<Vec<u8>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .with_context(|| format!("{} holds no key line", path.display()))?;
    unb64(line)
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn unb64(s: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s.trim())
        .context("decoding base64")
}
