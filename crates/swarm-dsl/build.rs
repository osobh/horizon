fn main() {
    // This ensures that the pest grammar file is included in the build
    println!("cargo:rerun-if-changed=src/swarm.pest");
}
