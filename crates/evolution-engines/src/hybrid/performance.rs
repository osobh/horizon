//! Performance tracking for hybrid evolution engines

/// Performance metrics for individual engines
#[derive(Debug, Clone)]
pub struct EnginePerformance {
    /// Engine name
    pub name: String,
    /// Number of generations run
    pub generations_run: u32,
    /// Average fitness improvement per generation
    pub avg_improvement: f64,
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Total time spent in nanoseconds
    pub time_spent_ns: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
}

impl Default for EnginePerformance {
    fn default() -> Self {
        Self {
            name: String::new(),
            generations_run: 0,
            avg_improvement: 0.0,
            best_fitness: 0.0,
            time_spent_ns: 0,
            success_rate: 0.0,
        }
    }
}

impl EnginePerformance {
    /// Create new performance tracker for an engine
    pub fn new(name: String) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    /// Update performance metrics
    pub fn update(&mut self, improvement: f64, runtime_ns: u64, success: bool, best_fitness: f64) {
        self.generations_run += 1;
        self.time_spent_ns += runtime_ns;

        // Update best fitness
        if best_fitness > self.best_fitness {
            self.best_fitness = best_fitness;
        }

        // Update average improvement with running average
        let n = self.generations_run as f64;
        self.avg_improvement = (self.avg_improvement * (n - 1.0) + improvement) / n;

        // Update success rate
        let success_value = if success { 1.0 } else { 0.0 };
        self.success_rate = (self.success_rate * (n - 1.0) + success_value) / n;
    }

    /// Get average runtime per generation
    pub fn avg_runtime_ns(&self) -> u64 {
        if self.generations_run > 0 {
            self.time_spent_ns / self.generations_run as u64
        } else {
            0
        }
    }

    /// Get performance score combining improvement and success rate
    pub fn performance_score(&self) -> f64 {
        self.avg_improvement * self.success_rate
    }
}
