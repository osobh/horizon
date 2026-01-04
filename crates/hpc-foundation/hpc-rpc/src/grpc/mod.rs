//! gRPC server and client abstractions
//!
//! This module provides type-safe wrappers around tonic for building
//! gRPC servers and clients with optional mTLS support.

mod client;
mod server;

pub use client::GrpcClientBuilder;
pub use server::{GrpcServer, GrpcServerBuilder};
