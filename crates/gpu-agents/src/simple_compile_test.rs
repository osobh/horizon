//! Simple compilation test to isolate issues

use anyhow::Result;

pub fn simple_test() -> Result<()> {
    println!("Testing basic compilation");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compilation() {
        simple_test().unwrap();
    }
}