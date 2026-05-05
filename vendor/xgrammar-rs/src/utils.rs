use std::ffi::c_char;

#[cfg(not(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86"),
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86"),
    all(target_os = "macos", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
#[inline]
pub fn bytes_as_c_char_ptr(bytes: &[u8]) -> *const c_char {
    bytes.as_ptr() as *const c_char
}

#[cfg(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "x86"),
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86"),
    all(target_os = "macos", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
#[inline]
pub fn bytes_as_c_char_ptr(bytes: &[u8]) -> *const c_char {
    bytes.as_ptr() as *const i8
}
