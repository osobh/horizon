//! CPU-GPU communication bridge

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Message types between CPU and GPU agents
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpuGpuMessage {
    pub msg_type: u32,
    pub sender_id: u64,
    pub target_id: u64,
    pub payload: [f32; 32],
}

/// Bridge for CPU-GPU agent communication
pub struct GpuCpuBridge {
    to_gpu: Arc<Mutex<VecDeque<CpuGpuMessage>>>,
    from_gpu: Arc<Mutex<VecDeque<CpuGpuMessage>>>,
}

impl GpuCpuBridge {
    pub fn new() -> Self {
        Self {
            to_gpu: Arc::new(Mutex::new(VecDeque::new())),
            from_gpu: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Send message from CPU to GPU
    pub fn send_to_gpu(&self, msg: CpuGpuMessage) -> Result<()> {
        self.to_gpu.lock().map_err(|e| anyhow::anyhow!("Failed to lock mutex: {}", e))?.push_back(msg);
        Ok(())
    }

    /// Receive message from GPU
    pub fn receive_from_gpu(&self) -> Option<CpuGpuMessage> {
        self.from_gpu.lock().ok()?.pop_front()
    }

    /// Get pending messages for GPU
    pub fn drain_to_gpu(&self) -> Vec<CpuGpuMessage> {
        self.to_gpu.lock().ok().map(|mut guard| guard.drain(..).collect()).unwrap_or_default()
    }

    /// Push results from GPU
    pub fn push_from_gpu(&self, msgs: Vec<CpuGpuMessage>) {
        if let Ok(mut queue) = self.from_gpu.lock() {
            queue.extend(msgs);
        }
    }
}
