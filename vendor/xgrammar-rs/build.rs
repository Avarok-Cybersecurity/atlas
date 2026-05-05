use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self, copy, create_dir_all},
    path::{Path, PathBuf},
    process::Command,
};

use cmake::Config as CMakeConfig;
use walkdir::WalkDir;

const DEFAULT_XGRAMMAR_GIT_URL: &str = "https://github.com/mlc-ai/xgrammar.git";
const DEFAULT_XGRAMMAR_GIT_REF: &str =
    "19a6893f1114ce9bd7ac171e19261a5bc55d1acc";

fn abs_path<P: AsRef<Path>>(p: P) -> PathBuf {
    if p.as_ref().is_absolute() {
        p.as_ref().to_path_buf()
    } else {
        env::current_dir().expect("current_dir failed").join(p)
    }
}

fn parse_include_search_list(stderr: &str) -> Vec<String> {
    let mut includes = Vec::new();
    let mut in_section = false;
    for line in stderr.lines() {
        if line.contains("#include <...> search starts here:") {
            in_section = true;
            continue;
        }
        if in_section {
            if line.contains("End of search list") {
                break;
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed.starts_with('/') {
                includes.push(trimmed.to_string());
            }
        }
    }
    includes
}

fn normalize_include_path(path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        match fs::canonicalize(p) {
            Ok(p) => p.display().to_string(),
            Err(_) => path.to_string(),
        }
    } else {
        path.to_string()
    }
}

fn gcc_multiarch_triple(target: &str) -> Option<String> {
    if let Ok(out) = Command::new("gcc").arg("-print-multiarch").output() {
        if let Ok(triple) = String::from_utf8(out.stdout) {
            let t = triple.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    if !target.is_empty() {
        return Some(target.to_string());
    }
    None
}

fn gcc_version() -> Option<String> {
    if let Ok(out) = Command::new("gcc").arg("-dumpversion").output() {
        if let Ok(v) = String::from_utf8(out.stdout) {
            let v = v.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn probe_compiler_includes(
    compiler: &str,
    target: &str,
) -> Vec<String> {
    let mut args = vec!["-E", "-x", "c++", "-", "-v"];
    if !target.is_empty() {
        args.push("-target");
        args.push(target);
    }

    let output = Command::new(compiler)
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();

    match output {
        Ok(out) => {
            parse_include_search_list(&String::from_utf8_lossy(&out.stderr))
                .into_iter()
                .map(|p| normalize_include_path(&p))
                .collect()
        },
        Err(_) => Vec::new(),
    }
}

fn fallback_include_paths(target: &str) -> Vec<String> {
    let mut paths = Vec::new();

    let mut libc_dirs = Vec::new();
    if let Some(triple) = gcc_multiarch_triple(target) {
        libc_dirs.push(format!("/usr/include/{}", triple));
    }
    libc_dirs.push("/usr/include".to_string());
    libc_dirs.push("/usr/local/include".to_string());

    if let (Some(triple), Some(version)) =
        (gcc_multiarch_triple(target), gcc_version())
    {
        paths.push(format!("/usr/lib/gcc/{}/{}/include", triple, version));
        paths
            .push(format!("/usr/lib/gcc/{}/{}/include-fixed", triple, version));
        paths.push(format!("/usr/include/c++/{}", version));
        paths.push(format!("/usr/include/{}/c++/{}", triple, version));
        paths.extend(libc_dirs.into_iter());
    } else {
        paths.extend(libc_dirs.into_iter());
    }

    if let Ok(out) = Command::new("gcc").arg("-print-libgcc-file-name").output()
    {
        if let Ok(path) = String::from_utf8(out.stdout) {
            let p = PathBuf::from(path.trim());
            if let Some(include_dir) =
                p.parent().and_then(|p| p.parent()).map(|p| p.join("include"))
            {
                paths.push(normalize_include_path(
                    &include_dir.display().to_string(),
                ));
            }
        }
    }

    paths
}

fn collect_system_include_args(target: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut seen = HashSet::new();

    let mut compilers = vec!["clang++", "g++", "gcc", "c++", "cc"];
    if let Ok(cxx) = env::var("CXX") {
        compilers.insert(0, Box::leak(cxx.into_boxed_str()));
    }

    for compiler in compilers {
        let includes = probe_compiler_includes(compiler, target);
        if !includes.is_empty() {
            for path in includes {
                if seen.insert(path.clone()) {
                    args.push(format!(
                        "-isystem{}",
                        normalize_include_path(&path)
                    ));
                }
            }
        }
    }

    // Always add a conservative fallback set to ensure glibc headers are visible,
    // even if probing succeeded but missed multiarch include dirs.
    for path in fallback_include_paths(target) {
        if seen.insert(path.clone()) {
            args.push(format!("-isystem{}", normalize_include_path(&path)));
        }
    }

    args
}

fn windows_target_clang_args(target: &str) -> Vec<String> {
    let mut args = Vec::new();
    if target.contains("windows") {
        if target.contains("aarch64") {
            args.push("--target=aarch64-pc-windows-msvc".to_string());
        } else if target.contains("x86_64") {
            args.push("--target=x86_64-pc-windows-msvc".to_string());
        }
    }
    args
}

fn apple_target_clang_args(target: &str) -> Vec<String> {
    let mut args = Vec::new();
    if target.contains("apple-ios-sim") || target.contains("x86_64-apple-ios") {
        let arch = if target.contains("aarch64") {
            "arm64"
        } else {
            "x86_64"
        };
        let version = env::var("IPHONEOS_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "17.0".into());
        args.push(format!("--target={}-apple-ios{}-simulator", arch, version));
        if let Ok(sdkroot) = env::var("SDKROOT") {
            args.push(format!("-isysroot{}", sdkroot));
        }
    }
    args
}

fn linux_clang_include_args(target: &str) -> Vec<String> {
    let mut args = vec!["--sysroot=/".to_string()];
    args.extend(collect_system_include_args(target));
    args
}

fn looks_like_xgrammar_repo_root(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").exists()
        && dir.join("include").exists()
        && dir.join("cpp").exists()
}

fn is_truthy_env(name: &str) -> bool {
    let Ok(v) = env::var(name) else {
        return false;
    };
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn cargo_offline() -> bool {
    is_truthy_env("CARGO_NET_OFFLINE") || is_truthy_env("XGRAMMAR_RS_OFFLINE")
}

fn submodule_cache_dir(out_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("XGRAMMAR_RS_CACHE_DIR") {
        return abs_path(p);
    }

    if let Ok(p) = env::var("CARGO_HOME") {
        return abs_path(p).join("xgrammar-rs-cache");
    }

    if let Ok(p) = env::var("HOME") {
        return PathBuf::from(p).join(".cache/xgrammar-rs");
    }
    if let Ok(p) = env::var("LOCALAPPDATA") {
        return PathBuf::from(p).join("xgrammar-rs");
    }

    out_dir.join("xgrammar-rs-cache")
}

fn run_checked(
    mut cmd: Command,
    what: &str,
) {
    let output = cmd.output().unwrap_or_else(|e| {
        panic!("Failed to run {}: {}", what, e);
    });
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "{} failed (exit={:?})\n--- stdout ---\n{}\n--- stderr ---\n{}\n",
            what,
            output.status.code(),
            stdout,
            stderr
        );
    }
}

fn pins_toml_path(manifest_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("XGRAMMAR_RS_PINS_TOML") {
        return abs_path(p);
    }
    // Backward compatibility with the previous env var name.
    if let Ok(p) = env::var("XGRAMMAR_RS_SUBMODULES_TOML") {
        return abs_path(p);
    }
    manifest_dir.join("xgrammar-pins.toml")
}

#[derive(Debug, Clone)]
struct Pins {
    pins_path: PathBuf,
    repo_url: Option<String>,
    repo_ref: Option<String>,
    submodules: HashMap<String, (String, String)>,
}

fn parse_pins(pins_path: &Path) -> Pins {
    let contents = fs::read_to_string(pins_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read pins file at {}: {}. \
             Update the pins file or set XGRAMMAR_RS_PINS_TOML to a valid path.",
            pins_path.display(),
            e
        )
    });

    #[derive(Debug)]
    enum Section {
        None,
        Repo,
        Submodule(String),
    }

    let mut section = Section::None;
    let mut pins = Pins {
        pins_path: pins_path.to_path_buf(),
        repo_url: None,
        repo_ref: None,
        submodules: HashMap::new(),
    };

    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let header = &line[1..line.len() - 1];
            if header == "repo" {
                section = Section::Repo;
            } else if let Some(name) = header.strip_prefix("submodules.") {
                section = Section::Submodule(name.trim().to_string());
            } else {
                section = Section::None;
            }
            continue;
        }

        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let mut val = v.trim();
        if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
            val = &val[1..val.len() - 1];
        }

        match &mut section {
            Section::Repo => match key {
                "url" => pins.repo_url = Some(val.to_string()),
                "ref" | "rev" => pins.repo_ref = Some(val.to_string()),
                _ => {},
            },
            Section::Submodule(name) => {
                let entry = pins
                    .submodules
                    .entry(name.clone())
                    .or_insert_with(|| (String::new(), String::new()));
                match key {
                    "url" => entry.0 = val.to_string(),
                    "ref" | "rev" => entry.1 = val.to_string(),
                    _ => {},
                }
            },
            Section::None => {},
        }
    }

    pins
}

fn pins_submodule(
    pins: &Pins,
    name: &str,
) -> (String, String) {
    if let Some((url, rev)) = pins.submodules.get(name) {
        if url.is_empty() || rev.is_empty() {
            panic!(
                "Pins file {} has an incomplete entry for submodule '{}'",
                pins.pins_path.display(),
                name
            );
        }
        return (url.clone(), rev.clone());
    }
    panic!(
        "Missing submodule '{}' in pins file {}",
        name,
        pins.pins_path.display()
    );
}

fn maybe_clear_cmake_build_dir(
    build_dir: &Path,
    source_dir: &Path,
) {
    let cache = build_dir.join("CMakeCache.txt");
    let Ok(contents) = fs::read_to_string(&cache) else {
        return;
    };
    let src =
        source_dir.canonicalize().unwrap_or_else(|_| source_dir.to_path_buf());
    for line in contents.lines() {
        if !line.starts_with("CMAKE_HOME_DIRECTORY") {
            continue;
        }
        let needs_cleanup = match line.split('=').next_back() {
            Some(cmake_home) => fs::canonicalize(cmake_home)
                .ok()
                .map(|cmake_home| cmake_home != src)
                .unwrap_or(true),
            None => true,
        };
        if needs_cleanup {
            let _ = fs::remove_dir_all(build_dir);
        }
        break;
    }
}

fn copy_dir_recursive_filtered(
    src: &Path,
    dst: &Path,
    should_skip: impl Fn(&Path) -> bool,
) {
    for entry in WalkDir::new(src).into_iter().filter_map(Result::ok) {
        let p = entry.path();
        let rel = p.strip_prefix(src).expect("strip_prefix failed");
        if should_skip(rel) {
            continue;
        }
        let out_path = dst.join(rel);
        if entry.file_type().is_dir() {
            create_dir_all(&out_path).ok();
            continue;
        }
        if entry.file_type().is_file() {
            if let Some(parent) = out_path.parent() {
                create_dir_all(parent).ok();
            }
            let _ = fs::copy(p, &out_path);
        }
    }
}

fn write_if_changed(
    path: &Path,
    contents: &[u8],
) {
    if let Ok(existing) = fs::read(path) {
        if existing == contents {
            return;
        }
    }
    if let Some(parent) = path.parent() {
        create_dir_all(parent).ok();
    }
    fs::write(path, contents).unwrap_or_else(|e| {
        panic!("Failed to write {}: {}", path.display(), e)
    });
}

fn copy_if_changed(
    src: &Path,
    dst: &Path,
) {
    if let (Ok(src_bytes), Ok(dst_bytes)) = (fs::read(src), fs::read(dst)) {
        if src_bytes == dst_bytes {
            return;
        }
    }
    if let Some(parent) = dst.parent() {
        create_dir_all(parent).ok();
    }
    let _ = copy(src, dst);
}

fn ensure_git_checkout_cached(
    name: &str,
    url: &str,
    rev: &str,
    cache_dir: &Path,
) -> PathBuf {
    let checkout_dir = cache_dir.join(format!("{}-{}", name, rev));
    let marker = checkout_dir.join(".xgrammar_rs_fetched");
    if marker.exists() {
        return checkout_dir;
    }

    if checkout_dir.exists() {
        let _ = fs::remove_dir_all(&checkout_dir);
    }
    create_dir_all(cache_dir).expect("Failed to create cache dir");

    run_checked(
        {
            let mut c = Command::new("git");
            c.arg("clone").arg(url).arg(&checkout_dir);
            c
        },
        &format!("git clone {} into cache", name),
    );
    run_checked(
        {
            let mut c = Command::new("git");
            c.arg("-C").arg(&checkout_dir).arg("checkout").arg(rev);
            c
        },
        &format!("git checkout {}@{}", name, rev),
    );

    let _ = fs::write(&marker, rev);
    checkout_dir
}

fn default_xgrammar_git_ref() -> String {
    env::var("CARGO_PKG_VERSION")
        .map(|v| format!("v{}", v))
        .unwrap_or_else(|_| DEFAULT_XGRAMMAR_GIT_REF.to_string())
}

fn pinned_xgrammar_git(pins: &Pins) -> (String, String) {
    let url = env::var("XGRAMMAR_GIT_URL")
        .ok()
        .or_else(|| pins.repo_url.clone())
        .unwrap_or_else(|| DEFAULT_XGRAMMAR_GIT_URL.to_string());
    let rev = env::var("XGRAMMAR_GIT_REF")
        .ok()
        .or_else(|| pins.repo_ref.clone())
        .unwrap_or_else(|| default_xgrammar_git_ref());
    (url, rev)
}

fn ensure_xgrammar_repo(
    out_dir: &Path,
    repo_url: &str,
    repo_ref: &str,
) -> PathBuf {
    if let Ok(p) = env::var("XGRAMMAR_SRC_DIR") {
        let p = abs_path(p);
        if !looks_like_xgrammar_repo_root(&p) {
            panic!(
                "XGRAMMAR_SRC_DIR={} does not look like an XGrammar repo root (expected CMakeLists.txt + include/ + cpp/)",
                p.display()
            );
        }
        return p;
    }

    if cargo_offline() {
        panic!(
            "XGrammar sources not found locally and Cargo is offline. \
             Set XGRAMMAR_SRC_DIR to a checked-out XGrammar repo or build with network access."
        );
    }

    let cache_dir = submodule_cache_dir(out_dir);
    println!(
        "cargo:warning=xgrammar-rs: fetching XGrammar {}@{} into {}",
        repo_url,
        repo_ref,
        cache_dir.display()
    );
    ensure_git_checkout_cached("xgrammar", repo_url, repo_ref, &cache_dir)
}

fn prepare_xgrammar_source_tree(
    xgrammar_repo_dir: &Path,
    out_dir: &Path,
    pins: &Pins,
) -> PathBuf {
    let dlpack_header =
        xgrammar_repo_dir.join("3rdparty/dlpack/include/dlpack/dlpack.h");
    if dlpack_header.exists() {
        return xgrammar_repo_dir.to_path_buf();
    }

    if cargo_offline() {
        panic!(
            "Required git submodule `3rdparty/dlpack` is missing (expected {}). \
             Cargo is in offline mode. Either:\n\
             - build with network access, or\n\
             - build from a git checkout with submodules initialized, or\n\
             - set XGRAMMAR_SRC_DIR to an XGrammar repo root that already has submodules.",
            dlpack_header.display()
        );
    }

    let cache_dir = submodule_cache_dir(out_dir);
    println!(
        "cargo:warning=xgrammar-rs: dlpack submodule missing; fetching into cache at {}",
        cache_dir.display()
    );

    let (dlpack_url, dlpack_rev) = pins_submodule(pins, "dlpack");
    let dlpack_checkout = ensure_git_checkout_cached(
        "dlpack",
        &dlpack_url,
        &dlpack_rev,
        &cache_dir,
    );

    let work_dir = out_dir.join("xgrammar-src");

    let marker_path = work_dir.join(".xgrammar_rs_source_marker");
    let marker_content = format!(
        "repo={}\ndlpack={}\n",
        xgrammar_repo_dir.display(),
        dlpack_checkout.display(),
    );
    if let Ok(existing) = fs::read_to_string(&marker_path) {
        if existing == marker_content {
            return work_dir;
        }
    }

    if work_dir.exists() {
        let _ = fs::remove_dir_all(&work_dir);
    }

    // Copy the minimal set of XGrammar sources needed for the CMake build.
    let to_copy =
        ["CMakeLists.txt", "cmake", "cpp", "include", "3rdparty/picojson"];
    for rel in to_copy {
        let src = xgrammar_repo_dir.join(rel);
        let dst = work_dir.join(rel);
        if src.is_dir() {
            copy_dir_recursive_filtered(&src, &dst, |_| false);
        } else if src.is_file() {
            if let Some(parent) = dst.parent() {
                create_dir_all(parent).ok();
            }
            let _ = fs::copy(&src, &dst);
        }
    }

    let dlpack_dst = work_dir.join("3rdparty/dlpack");
    copy_dir_recursive_filtered(&dlpack_checkout, &dlpack_dst, |rel| {
        rel.components().any(|c| c.as_os_str() == ".git")
    });

    let dlpack_header_work =
        work_dir.join("3rdparty/dlpack/include/dlpack/dlpack.h");
    if !dlpack_header_work.exists() {
        panic!(
            "Fetched dlpack but the expected header was not found at {}",
            dlpack_header_work.display()
        );
    }

    let _ = fs::write(&marker_path, marker_content);

    work_dir
}

fn find_xgrammar_lib_dir(root: &Path) -> Option<PathBuf> {
    let static_candidates = ["libxgrammar.a", "xgrammar.lib"];

    for entry in
        WalkDir::new(root).max_depth(6).into_iter().filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let name = entry.file_name().to_string_lossy();
        if static_candidates.iter().any(|c| name == *c) {
            return entry.path().parent().map(|p| p.to_path_buf());
        }
    }

    None
}

fn strip_autocxx_generated_doc_comments(out_dir: &Path) {
    let debug = env::var("XGRAMMAR_RS_DEBUG_DOCSTRIP").is_ok();
    let rs_dir = out_dir.join("autocxx-build-dir/rs");
    if debug {
        println!("cargo:warning=docstrip: scanning {}", rs_dir.display());
    }
    let Ok(rd) = std::fs::read_dir(&rs_dir) else {
        if debug {
            println!("cargo:warning=docstrip: rs dir missing");
        }
        return;
    };
    let entries: Vec<_> = rd.flatten().collect();
    if debug {
        let mut names: Vec<String> = entries
            .iter()
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        names.sort();
        println!("cargo:warning=docstrip: entries={}", names.join(", "));
    }
    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !file_name.starts_with("autocxx-") || !file_name.ends_with("-gen.rs")
        {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(&path) else {
            if debug {
                println!(
                    "cargo:warning=docstrip: failed to read {}",
                    path.display()
                );
            }
            continue;
        };
        if debug {
            let count = contents.matches("#[doc =").count();
            println!(
                "cargo:warning=docstrip: {} contains {} #[doc =] lines",
                file_name, count
            );
        }
        let mut changed = false;
        let mut removed = 0usize;
        let mut out = String::with_capacity(contents.len());
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#[doc =") {
                changed = true;
                removed += 1;
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        if changed {
            if debug {
                println!(
                    "cargo:warning=docstrip: {} removed {} doc lines",
                    file_name, removed
                );
            }
            let _ = std::fs::write(&path, out);
        }
    }
}

#[cfg(target_os = "windows")]
fn find_libclang_windows() -> Option<PathBuf> {
    let vswhere = PathBuf::from(
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
    );

    let mut candidates: Vec<PathBuf> = Vec::new();

    if vswhere.exists() {
        let args = [
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Llvm.Clang",
            "-property",
            "installationPath",
        ];

        if let Ok(out) = Command::new(&vswhere).args(args).output() {
            if out.status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
                    let base = PathBuf::from(line.trim());
                    // Prefer host x64 libclang; fall back to ARM64 if present.
                    candidates.push(base.join(r"VC\Tools\Llvm\x64\bin"));
                    candidates.push(base.join(r"VC\Tools\Llvm\bin"));
                    candidates.push(base.join(r"VC\Tools\Llvm\ARM64\bin"));
                }
            }
        }
    }

    for edition in ["Community", "Professional", "Enterprise"] {
        candidates.push(PathBuf::from(format!(
            r"C:\Program Files\Microsoft Visual Studio\2022\{}\VC\Tools\Llvm\x64\bin",
            edition
        )));
        candidates.push(PathBuf::from(format!(
            r"C:\Program Files\Microsoft Visual Studio\2022\{}\VC\Tools\Llvm\bin",
            edition
        )));
        candidates.push(PathBuf::from(format!(
            r"C:\Program Files\Microsoft Visual Studio\2022\{}\VC\Tools\Llvm\ARM64\bin",
            edition
        )));
    }

    candidates.push(PathBuf::from(r"C:\Program Files\LLVM\bin"));

    for dir in candidates {
        if dir.join("libclang.dll").exists() {
            return Some(dir);
        }
    }

    None
}

#[cfg(not(target_os = "windows"))]
fn find_libclang_windows() -> Option<PathBuf> {
    None
}

// ============================================================================
// Build orchestration
// ============================================================================

#[derive(Debug, Clone)]
struct BuildContext {
    manifest_dir: PathBuf,
    xgrammar_src_dir: PathBuf,
    out_dir: PathBuf,

    src_include_dir: PathBuf,
    xgrammar_include_dir: PathBuf,
    dlpack_include_dir: PathBuf,
    picojson_include_dir: PathBuf,

    target: String,
}

fn configure_libclang_windows(_target: &str) {
    if env::var("LIBCLANG_PATH").is_err() && cfg!(target_os = "windows") {
        if let Some(dir) = find_libclang_windows() {
            let host_is_arm64 = cfg!(target_arch = "aarch64");
            let base = dir.parent().and_then(|p| p.parent());
            let mut candidates: Vec<PathBuf> = Vec::new();
            if let Some(base) = base {
                if host_is_arm64 {
                    candidates.push(base.join("ARM64").join("bin"));
                    candidates.push(base.join("x64").join("bin"));
                } else {
                    candidates.push(base.join("x64").join("bin"));
                    candidates.push(base.join("ARM64").join("bin"));
                }
            }
            candidates.push(dir.clone());

            let chosen = candidates
                .into_iter()
                .find(|p| p.join("libclang.dll").exists())
                .unwrap_or_else(|| dir.clone());

            unsafe { env::set_var("LIBCLANG_PATH", &chosen) };
            println!("cargo:rustc-env=LIBCLANG_PATH={}", chosen.display());
        }
    }
}

fn collect_build_context() -> BuildContext {
    println!("cargo:rerun-if-env-changed=XGRAMMAR_SRC_DIR");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_PINS_TOML");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_SUBMODULES_TOML");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_GIT_URL");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_GIT_REF");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=XGRAMMAR_RS_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_HOME");

    let manifest_dir = abs_path(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let pins_path = pins_toml_path(&manifest_dir);
    println!("cargo:rerun-if-changed={}", pins_path.display());
    let pins = parse_pins(&pins_path);

    let (repo_url, repo_ref) = pinned_xgrammar_git(&pins);
    let xgrammar_repo_dir =
        ensure_xgrammar_repo(&out_dir, &repo_url, &repo_ref);

    println!(
        "cargo:rerun-if-changed={}",
        xgrammar_repo_dir.join("include").display()
    );
    println!("cargo:rerun-if-changed={}/cpp", xgrammar_repo_dir.display());
    println!("cargo:rerun-if-changed={}/3rdparty", xgrammar_repo_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        xgrammar_repo_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        xgrammar_repo_dir.join(".gitmodules").display()
    );

    let xgrammar_src_dir =
        prepare_xgrammar_source_tree(&xgrammar_repo_dir, &out_dir, &pins);

    let xgrammar_include_dir = xgrammar_src_dir.join("include");
    let dlpack_include_dir = xgrammar_src_dir.join("3rdparty/dlpack/include");
    let picojson_include_dir = xgrammar_src_dir.join("3rdparty/picojson");
    let src_include_dir = manifest_dir.join("src");

    let target = env::var("TARGET").unwrap_or_default();

    BuildContext {
        manifest_dir,
        xgrammar_src_dir,
        out_dir,
        src_include_dir,
        xgrammar_include_dir,
        dlpack_include_dir,
        picojson_include_dir,
        target,
    }
}

fn build_xgrammar_cmake(ctx: &BuildContext) -> PathBuf {
    let cmake_build_dir = ctx.out_dir.join("build");
    maybe_clear_cmake_build_dir(&cmake_build_dir, &ctx.xgrammar_src_dir);
    create_dir_all(&cmake_build_dir).ok();

    let config_cmake_path = cmake_build_dir.join("config.cmake");
    write_if_changed(
        &config_cmake_path,
        b"set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n\
          set(XGRAMMAR_BUILD_CXX_TESTS OFF)\n\
          set(XGRAMMAR_ENABLE_CPPTRACE OFF)\n\
          set(CMAKE_BUILD_TYPE RelWithDebInfo)\n",
    );

    let mut cmake_config = CMakeConfig::new(&ctx.xgrammar_src_dir);
    cmake_config.out_dir(&ctx.out_dir);
    cmake_config.define("XGRAMMAR_BUILD_PYTHON_BINDINGS", "OFF");
    cmake_config.define("XGRAMMAR_BUILD_CXX_TESTS", "OFF");
    cmake_config.define("XGRAMMAR_ENABLE_CPPTRACE", "OFF");
    cmake_config.define("CMAKE_CXX_STANDARD", "17");
    cmake_config.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");
    cmake_config.define("CMAKE_CXX_EXTENSIONS", "OFF");

    cmake_config.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "OFF");

    let is_msvc = ctx.target.contains("msvc");
    if !is_msvc {
        cmake_config.cflag("-fno-lto");
        cmake_config.cxxflag("-fno-lto");
    } else {
        cmake_config.cxxflag("/EHsc");
    }

    let build_profile =
        match env::var("PROFILE").unwrap_or_else(|_| "release".into()).as_str()
        {
            "debug" => "Debug",
            "release" => "Release",
            other => {
                eprintln!(
                    "Unknown cargo PROFILE '{}' -> using RelWithDebInfo",
                    other
                );
                "RelWithDebInfo"
            },
        };
    cmake_config.profile(build_profile);

    if !ctx.target.is_empty() {
        if ctx.target.contains("apple-darwin") {
            let arch = if ctx.target.contains("aarch64") {
                "arm64"
            } else {
                "x86_64"
            };
            cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
            // Keep CMake builds aligned with Rust's default macOS minimum; avoids
            // linking objects built for a newer SDK than rustc links against.
            let dep_target = env::var("MACOSX_DEPLOYMENT_TARGET")
                .unwrap_or_else(|_| "11.0".into());
            cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", dep_target);
        } else if ctx.target.contains("apple-ios")
            || ctx.target.contains("apple-ios-sim")
        {
            let is_sim = ctx.target.contains("apple-ios-sim")
                || ctx.target.contains("x86_64-apple-ios");
            let arch = if ctx.target.contains("aarch64") {
                "arm64"
            } else {
                "x86_64"
            };
            let sysroot = if is_sim {
                "iphonesimulator"
            } else {
                "iphoneos"
            };
            cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
            cmake_config.define("CMAKE_OSX_SYSROOT", sysroot);
            if let Ok(dep_target) = env::var("IPHONEOS_DEPLOYMENT_TARGET") {
                cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", dep_target);
            }
        }
    }

    if ctx.target.contains("msvc") {
        // Force the release CRT (/MD) even for debug builds to avoid
        // _ITERATOR_DEBUG_LEVEL and runtime mismatches when linking with Rust.
        cmake_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
        cmake_config.define("CMAKE_C_FLAGS_DEBUG", "/MD");
        cmake_config.define("CMAKE_CXX_FLAGS_DEBUG", "/MD");
    }

    cmake_config.build_target("xgrammar").build()
}

fn link_xgrammar_static(
    ctx: &BuildContext,
    destination_path: &Path,
) {
    let cmake_build_dir = ctx.out_dir.join("build");
    let lib_search_dir = find_xgrammar_lib_dir(&cmake_build_dir)
        .or_else(|| find_xgrammar_lib_dir(destination_path))
        .unwrap_or_else(|| destination_path.join("lib"));
    println!("cargo:rustc-link-search=native={}", lib_search_dir.display());
    println!("cargo:rustc-link-lib=static=xgrammar");
}

fn build_autocxx_bridge(ctx: &BuildContext) {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cxx_utils.hpp");
    println!("cargo:rerun-if-changed=src/cxx_utils");

    let mut extra_clang_args = vec!["-std=c++17".to_string()];
    extra_clang_args.extend(windows_target_clang_args(&ctx.target));
    extra_clang_args.extend(apple_target_clang_args(&ctx.target));
    if ctx.target.contains("linux") {
        extra_clang_args.extend(linux_clang_include_args(&ctx.target));
    }
    if is_truthy_env("XGRAMMAR_RS_DEBUG_INCLUDES") {
        println!(
            "cargo:warning=xgrammar-rs: extra_clang_args={}",
            extra_clang_args.join(" ")
        );
    }

    let extra_clang_args_refs: Vec<&str> =
        extra_clang_args.iter().map(|s| s.as_str()).collect();

    let mut autocxx_builder = autocxx_build::Builder::new(
        "src/lib.rs",
        &[
            &ctx.src_include_dir,
            &ctx.xgrammar_include_dir,
            &ctx.dlpack_include_dir,
            &ctx.picojson_include_dir,
            &ctx.xgrammar_src_dir,
        ],
    )
    .extra_clang_args(&extra_clang_args_refs)
    .build()
    .expect("autocxx build failed");

    autocxx_builder
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .include(&ctx.src_include_dir)
        .include(&ctx.xgrammar_include_dir)
        .include(&ctx.dlpack_include_dir)
        .include(&ctx.picojson_include_dir)
        .include(&ctx.xgrammar_src_dir)
        .include(&ctx.manifest_dir);

    autocxx_builder.compile("xgrammar_rs_bridge");
}

fn copy_headers_for_generated_rust_code(ctx: &BuildContext) {
    let rs_dir = ctx.out_dir.join("autocxx-build-dir/rs");

    let gen_include_dir = ctx.out_dir.join("autocxx-build-dir/include");
    copy_if_changed(
        &gen_include_dir.join("autocxxgen_ffi.h"),
        &rs_dir.join("autocxxgen_ffi.h"),
    );

    let rs_xgrammar_dir = rs_dir.join("xgrammar");
    create_dir_all(&rs_xgrammar_dir).ok();
    copy_if_changed(
        &ctx.xgrammar_include_dir.join("xgrammar/xgrammar.h"),
        &rs_xgrammar_dir.join("xgrammar.h"),
    );

    let rs_dlpack_dir = rs_dir.join("dlpack");
    create_dir_all(&rs_dlpack_dir).ok();
    copy_if_changed(
        &ctx.dlpack_include_dir.join("dlpack/dlpack.h"),
        &rs_dlpack_dir.join("dlpack.h"),
    );
}

fn format_generated_bindings_optional(out_dir: &Path) {
    let gen_rs =
        out_dir.join("autocxx-build-dir/rs/autocxx-ffi-default-gen.rs");
    if gen_rs.exists() {
        match Command::new("rustfmt").arg(&gen_rs).status() {
            Ok(status) => {
                if !status.success() {
                    eprintln!(
                        "rustfmt returned non-zero status on {}",
                        gen_rs.display()
                    );
                }
            },
            Err(err) => {
                eprintln!("rustfmt not executed: {}", err);
            },
        }
    }
}

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    configure_libclang_windows(&target);
    let build_context = collect_build_context();
    let destination_path = build_xgrammar_cmake(&build_context);
    link_xgrammar_static(&build_context, &destination_path);
    build_autocxx_bridge(&build_context);
    copy_headers_for_generated_rust_code(&build_context);
    format_generated_bindings_optional(&build_context.out_dir);
    strip_autocxx_generated_doc_comments(&build_context.out_dir);
}
