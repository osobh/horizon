//! Time Navigation Components
//!
//! Handles navigation through evolution timeline

use super::EvolutionSnapshot;
use std::collections::HashMap;

/// Time navigation for evolution debugging
#[derive(Debug)]
pub struct TimeNavigator {
    current_generation: usize,
    snapshots: HashMap<usize, EvolutionSnapshot>,
}

impl TimeNavigator {
    /// Create new time navigator
    pub fn new() -> Self {
        Self {
            current_generation: 0,
            snapshots: HashMap::new(),
        }
    }

    /// Get current generation
    pub fn get_current_generation(&self) -> usize {
        self.current_generation
    }

    /// Set current generation
    pub fn set_current_generation(&mut self, generation: usize) {
        self.current_generation = generation;
    }

    /// Add snapshot for navigation
    pub fn add_snapshot(&mut self, snapshot: EvolutionSnapshot) {
        self.snapshots.insert(snapshot.generation, snapshot);
    }

    /// Get snapshot by generation
    pub fn get_snapshot(&self, generation: usize) -> Option<&EvolutionSnapshot> {
        self.snapshots.get(&generation)
    }

    /// Navigate to previous generation
    pub fn navigate_previous(&mut self) -> Option<usize> {
        if self.current_generation > 0 {
            self.current_generation -= 1;
            Some(self.current_generation)
        } else {
            None
        }
    }

    /// Navigate to next generation
    pub fn navigate_next(&mut self) -> Option<usize> {
        if self.snapshots.contains_key(&(self.current_generation + 1)) {
            self.current_generation += 1;
            Some(self.current_generation)
        } else {
            None
        }
    }
}
