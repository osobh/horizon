//! Mock implementations for missing dependencies
//! TODO: Remove when actual crates are available

// Mock cluster-mesh types
pub mod cluster_mesh {
    use serde::{Deserialize, Serialize};
    use std::error::Error;

    #[derive(Debug, Clone)]
    pub struct ClusterMesh;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeId(pub u64);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Message {
        Custom(Vec<u8>),
    }

    impl ClusterMesh {
        pub fn new_mock() -> Self {
            Self
        }

        pub async fn broadcast(
            &self,
            _message: Message,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            Ok(())
        }
    }
}

// Mock memory types
pub mod memory {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
    pub enum MemoryTier {
        Gpu,
        Cpu,
        Nvme,
        Ssd,
        Hdd,
    }

    #[derive(Debug)]
    pub struct TierManager;

    impl Default for TierManager {
        fn default() -> Self {
            Self::new()
        }
    }

    impl TierManager {
        pub fn new() -> Self {
            Self
        }
    }
}

// Mock net types
pub mod net {
    // Add network mocks if needed
}
