// SPDX-License-Identifier: AGPL-3.0-only

use std::path::{Component, Path, PathBuf};

use anyhow::{Result, bail};

/// Lexical, not `canonicalize`: the target usually does not exist yet, and a
/// canonicalize-then-compare check silently passes on a missing path.
///
/// An absolute path is accepted **only** when it is already inside the sandbox.
/// opencode's file tools ask for absolute paths and its environment block hands
/// the model the working directory to build them from, so rejecting every
/// absolute path would fail a prompt-compliant call.
///
/// Lexical containment is necessary and not sufficient: `ln -s / esc` inside
/// the sandbox makes `esc/etc/passwd` a lexically clean path that a file tool
/// would follow out. This is defense in depth rather than a privilege boundary:
/// `bash` is unconfined by construction, and a symlink swapped after this check
/// could still win.
pub fn resolve(sandbox: &Path, path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    let path = match path.strip_prefix(sandbox) {
        Ok(inside) => inside,
        Err(_) if path.is_absolute() => bail!(
            "path must be inside the project directory {}: {}",
            sandbox.display(),
            path.display()
        ),
        Err(_) => path,
    };
    let mut out = sandbox.to_path_buf();
    for component in path.components() {
        match component {
            Component::Normal(c) => out.push(c),
            Component::CurDir => {}
            Component::ParentDir => bail!("path must not leave the project directory"),
            Component::RootDir | Component::Prefix(_) => bail!("absolute paths are not allowed"),
        }
    }
    if leaves_via_symlink(sandbox, &out) {
        bail!("path must not leave the project directory through a symlink");
    }
    Ok(out)
}

/// Does `out`, already lexically inside `sandbox`, resolve somewhere else?
fn leaves_via_symlink(sandbox: &Path, out: &Path) -> bool {
    let Ok(root) = std::fs::canonicalize(sandbox) else {
        return false;
    };
    deepest_existing_real(out, 40).is_none_or(|real| !real.starts_with(&root))
}

/// Resolve `path`, or the deepest ancestor whose destination can be resolved.
fn deepest_existing_real(path: &Path, hops: usize) -> Option<PathBuf> {
    let mut probe = path;
    loop {
        if let Ok(real) = std::fs::canonicalize(probe) {
            return Some(real);
        }
        if std::fs::symlink_metadata(probe).is_ok_and(|meta| meta.file_type().is_symlink()) {
            if hops == 0 {
                return None;
            }
            let target = std::fs::read_link(probe).ok()?;
            let target = match target.is_absolute() {
                true => target,
                false => probe.parent()?.join(target),
            };
            return deepest_existing_real(&target, hops - 1);
        }
        probe = probe.parent()?;
    }
}
