//! gRPC server and client abstractions
//!
//! This module provides type-safe wrappers around tonic for building
//! gRPC servers and clients with optional mTLS support.

mod server;
mod client;

pub use server::{GrpcServerBuilder, GrpcServer};
pub use client::GrpcClientBuilder;
