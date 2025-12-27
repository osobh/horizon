//! GPU Consensus Module Tests
//!
//! Following TDD principles - tests written FIRST before implementation

use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

// Import types from parent module
use crate::consensus::{ConsensusState, Proposal, Vote};

#[cfg(test)]
mod voting_tests {
    use super::*;
    use crate::consensus::voting::*;

    #[test]
    fn test_gpu_vote_aggregation_basic() {
        // Arrange - TDD RED PHASE: Test will fail as implementation doesn't exist
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let num_agents = 1000;
        let proposal_id = 42;

        // Create test votes
        let mut votes = vec![];
        for i in 0..num_agents {
            votes.push(Vote {
                agent_id: i as u32,
                proposal_id,
                value: (i % 3) as u32, // Distribute votes across 3 options
                timestamp: 1000 + i as u64,
            });
        }

        // Act - Call the GPU voting aggregation (will fail in RED phase)
        let gpu_voting = GpuVoting::new(Arc::clone(&device), num_agents)?;
        let vote_counts = gpu_voting.aggregate_votes(&votes, 3)?;

        // Assert
        assert_eq!(vote_counts.len(), 3);
        assert_eq!(vote_counts[0], 334); // Agents 0, 3, 6, ... voted for option 0
        assert_eq!(vote_counts[1], 333); // Agents 1, 4, 7, ... voted for option 1
        assert_eq!(vote_counts[2], 333); // Agents 2, 5, 8, ... voted for option 2
    }

    #[test]
    fn test_gpu_vote_aggregation_performance() {
        // Test that consensus is achieved in <100μs as required
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let num_agents = 10_000;

        let mut votes = vec![];
        for i in 0..num_agents {
            votes.push(Vote {
                agent_id: i as u32,
                proposal_id: 1,
                value: (i % 5) as u32,
                timestamp: 1000 + i as u64,
            });
        }

        let gpu_voting = GpuVoting::new(Arc::clone(&device), num_agents)?;

        // Warm up
        let _ = gpu_voting.aggregate_votes(&votes, 5)?;

        // Measure performance
        let start = Instant::now();
        let _ = gpu_voting.aggregate_votes(&votes, 5)?;
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_micros() < 100,
            "Vote aggregation took {}μs, expected <100μs",
            elapsed.as_micros()
        );
    }

    #[test]
    fn test_atomic_vote_updates() {
        // Test atomic operations for concurrent vote updates
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let num_agents = 5000;
        let num_proposals = 10;

        let gpu_voting = GpuVoting::new(Arc::clone(&device), num_agents)?;

        // Submit votes for multiple proposals concurrently
        let mut all_votes = vec![];
        for proposal_id in 0..num_proposals {
            for i in 0..num_agents / num_proposals {
                all_votes.push(Vote {
                    agent_id: (proposal_id * num_agents / num_proposals + i) as u32,
                    proposal_id: proposal_id as u32,
                    value: (i % 2) as u32,
                    timestamp: 1000 + i as u64,
                });
            }
        }

        let results = gpu_voting
            .aggregate_votes_concurrent(&all_votes, 2, num_proposals)
            .unwrap();

        // Each proposal should have exactly num_agents/num_proposals votes
        for proposal_results in results {
            let total_votes: u32 = proposal_results.iter().sum();
            assert_eq!(total_votes, (num_agents / num_proposals) as u32);
        }
    }
}

#[cfg(test)]
mod leader_tests {
    use super::*;
    use crate::consensus::leader::*;

    #[test]
    fn test_gpu_leader_election() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let num_agents = 100;

        let gpu_leader = GpuLeaderElection::new(Arc::clone(&device), num_agents)?;

        // Initial leader election
        let leader_id = gpu_leader.elect_leader(0)?;
        assert!(leader_id < num_agents as u32);

        // Leader should remain stable in same round
        let leader_id2 = gpu_leader.elect_leader(0)?;
        assert_eq!(leader_id, leader_id2);

        // New round should potentially elect new leader
        let leader_id3 = gpu_leader.elect_leader(1)?;
        // Leader could be same or different, but must be valid
        assert!(leader_id3 < num_agents as u32);
    }

    #[test]
    fn test_heartbeat_tracking() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let num_agents = 50;

        let gpu_leader = GpuLeaderElection::new(Arc::clone(&device), num_agents)?;

        // Set initial heartbeats
        let mut heartbeats = vec![1000u64; num_agents];
        gpu_leader.update_heartbeats(&heartbeats)?;

        // Update some heartbeats
        for i in 0..10 {
            heartbeats[i] = 2000;
        }
        gpu_leader.update_heartbeats(&heartbeats)?;

        // Check if leader failover occurs when current leader stops heartbeating
        let current_leader = gpu_leader.elect_leader(0)?;
        heartbeats[current_leader as usize] = 0; // Simulate leader failure
        gpu_leader.update_heartbeats(&heartbeats)?;

        let new_leader = gpu_leader
            .check_leader_health(current_leader, 1500)
            .unwrap();
        assert_ne!(
            new_leader, current_leader,
            "Failed leader should be replaced"
        );
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::consensus::persistence::*;
    use std::path::Path;

    #[test]
    fn test_nvme_consensus_log() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");

        // Use test directory instead of production path
        let test_path = "/tmp/test_gpu_consensus";
        std::fs::create_dir_all(test_path)?;

        let persistence = GpuConsensusPersistence::new(
            Arc::clone(&device),
            test_path,
            1024 * 1024, // 1MB log size for testing
        )
        .unwrap();

        // Create test consensus state
        let state = ConsensusState {
            current_round: 5,
            leader_id: 42,
            vote_count: 100,
            decision: 1,
        };

        // Write state to NVMe
        persistence.checkpoint_state(&state)?;

        // Read back and verify
        let recovered_state = persistence.recover_state()?;
        assert_eq!(recovered_state.current_round, state.current_round);
        assert_eq!(recovered_state.leader_id, state.leader_id);
        assert_eq!(recovered_state.vote_count, state.vote_count);
        assert_eq!(recovered_state.decision, state.decision);

        // Test recovery time < 1 second
        let start = Instant::now();
        let _ = persistence.recover_state()?;
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 1000,
            "Recovery took {}ms, expected <1000ms",
            elapsed.as_millis()
        );

        // Cleanup
        std::fs::remove_dir_all(test_path).ok();
    }
}

#[cfg(test)]
mod multi_gpu_tests {
    use super::*;
    use crate::consensus::multi_gpu::*;

    #[test]
    fn test_multi_gpu_vote_aggregation() {
        // Skip if only one GPU available
        let device_count = CudaDevice::count()?;
        if device_count < 2 {
            println!(
                "Skipping multi-GPU test - only {} GPU(s) available",
                device_count
            );
            return;
        }

        let num_agents_per_gpu = 1000;
        let multi_gpu = MultiGpuConsensus::new(2, num_agents_per_gpu)?;

        // Create votes distributed across GPUs
        let mut all_votes = vec![];
        for gpu_id in 0..2 {
            for i in 0..num_agents_per_gpu {
                all_votes.push(Vote {
                    agent_id: (gpu_id * num_agents_per_gpu + i) as u32,
                    proposal_id: 1,
                    value: (i % 3) as u32,
                    timestamp: 1000 + i as u64,
                });
            }
        }

        // Aggregate across all GPUs
        let aggregated = multi_gpu.aggregate_cross_gpu(&all_votes, 3)?;

        // Verify aggregation
        let total_votes: u32 = aggregated.iter().sum();
        assert_eq!(total_votes, (2 * num_agents_per_gpu) as u32);

        // Test CUDA IPC communication
        multi_gpu.test_ipc_communication()?;
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::consensus::*;

    #[test]
    fn test_complete_consensus_workflow() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let num_agents = 1000;

        // Initialize consensus module
        let consensus = GpuConsensusModule::new(Arc::clone(&device), num_agents)?;

        // Create a proposal
        let proposal = Proposal {
            id: 1,
            proposer_id: 42,
            value: 100,
            round: 1,
        };

        // Submit votes
        let mut votes = vec![];
        for i in 0..num_agents {
            votes.push(Vote {
                agent_id: i as u32,
                proposal_id: proposal.id,
                value: if i < 600 { 1 } else { 0 }, // 60% vote yes
                timestamp: 1000 + i as u64,
            });
        }

        // Run consensus
        let start = Instant::now();
        let decision = consensus.run_consensus_round(proposal, &votes)?;
        let elapsed = start.elapsed();

        // Verify results
        assert_eq!(decision.value, 1, "Majority vote should win");
        assert_eq!(decision.vote_count, num_agents as u32);
        assert!(
            elapsed.as_micros() < 100,
            "Consensus took {}μs, expected <100μs",
            elapsed.as_micros()
        );
    }
}
