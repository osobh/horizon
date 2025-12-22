use std::env;

fn main() {
    // Get kernel headers path
    let kernel_headers = env::var("KERNEL_HEADERS")
        .unwrap_or_else(|_| format!("/lib/modules/{}/build", uname_release()));

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=build.rs");

    // Include kernel headers
    println!("cargo:include={}/include", kernel_headers);
}

fn uname_release() -> String {
    std::process::Command::new("uname")
        .arg("-r")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "6.14.0".to_string())
}
