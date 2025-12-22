use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct Topology {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct Node {
    pub id: Uuid,
    pub hostname: String,
    pub numa_nodes: Vec<NumaNode>,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct NumaNode {
    pub id: u32,
    pub gpus: Vec<Gpu>,
    pub cpu_cores: Vec<u32>,
    pub memory_gb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct Gpu {
    pub id: Uuid,
    pub device_id: String,
    pub gpu_type: String,
    pub memory_gb: u64,
    pub available: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub enum LinkType {
    NvLink,
    PciE,
    Network,
}
