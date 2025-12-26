//! TDD Test: Array Size Constraints
//!
//! This test will fail until we fix array size mismatches

use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that all benchmark patterns have exactly 56 bytes
    #[test]
    fn test_benchmark_pattern_sizes() -> Result<()> {
        // This test will fail until all patterns are exactly 56 bytes
        let patterns = vec![
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            b"The quick brown fox jumps over the lazy dog.            ",
            b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123ABCD",
            b"{ \"key\": \"value\", \"number\": 42 } { \"key\": \"value\" }     ",
        ];

        for (i, pattern) in patterns.iter().enumerate() {
            assert_eq!(
                pattern.len(),
                56,
                "Pattern {} has length {}, expected 56. Pattern: {:?}",
                i,
                pattern.len(),
                std::str::from_utf8(pattern).unwrap_or("invalid utf8")
            );
        }

        println!("All patterns have correct size of 56 bytes");
        Ok(())
    }

    /// Test that patterns can be used in arrays
    #[test]
    fn test_pattern_array_compilation() -> Result<()> {
        // This test validates that patterns can be stored in same-sized arrays
        let pattern1: &[u8; 56] = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let pattern2: &[u8; 56] = b"The quick brown fox jumps over the lazy dog.            ";
        let pattern3: &[u8; 56] = b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123";
        let pattern4: &[u8; 56] = b"{ \"key\": \"value\", \"number\": 42 } { \"key\": \"value\" }";

        // If this compiles, arrays are same size
        let _patterns: [&[u8; 56]; 4] = [pattern1, pattern2, pattern3, pattern4];

        println!("Pattern arrays compile successfully");
        Ok(())
    }

    /// Test specific problematic pattern
    #[test]
    fn test_hex_pattern_size() -> Result<()> {
        // This specifically tests the hex pattern that was causing issues
        let hex_pattern = b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123ABCD";

        assert_eq!(
            hex_pattern.len(),
            56,
            "Hex pattern has length {}, expected 56. Pattern: {:?}",
            hex_pattern.len(),
            std::str::from_utf8(hex_pattern)?
        );

        // Verify it has the right content structure
        assert!(hex_pattern.iter().all(|&b| b.is_ascii_alphanumeric()));

        println!("Hex pattern has correct size and content");
        Ok(())
    }

    /// Test that streaming benchmark patterns work
    #[test]
    fn test_streaming_benchmark_integration() -> Result<()> {
        // Test that the patterns work in the actual context they're used

        // Simulate the benchmark pattern usage
        let patterns = vec![
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            b"The quick brown fox jumps over the lazy dog.            ",
            b"0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123ABCD",
            b"{ \"key\": \"value\", \"number\": 42 } { \"key\": \"value\" }     ",
        ];

        // Test data generation like in the benchmark
        let size = 1024;
        let mut data = vec![0u8; size];
        let mut offset = 0;

        while offset < size {
            let pattern = patterns[offset % patterns.len()];
            let copy_len = pattern.len().min(size - offset);
            data[offset..offset + copy_len].copy_from_slice(&pattern[..copy_len]);
            offset += copy_len;
        }

        assert_eq!(data.len(), size);
        println!("Streaming benchmark pattern integration works");
        Ok(())
    }
}
