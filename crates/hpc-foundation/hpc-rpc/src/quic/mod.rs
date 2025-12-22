//! QUIC endpoint and stream abstractions
//!
//! This module provides low-latency, multiplexed streaming via QUIC protocol.

mod endpoint;
mod streams;

pub use endpoint::QuicEndpoint;
pub use streams::{QuicBiStream, QuicUniStream};

// Re-export quinn types for convenience
pub use quinn::{Connection, RecvStream, SendStream, VarInt};
