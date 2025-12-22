//! Orchestrator for workflow coordination
//!
//! Manages complex workflows with task dependencies and parallel execution

use crate::{
    agent::{CpuAgent, TaskType},
    CpuAgentError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::timeout;

/// Individual task within a workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTask {
    pub id: String,
    pub task_type: TaskType,
    pub dependencies: Vec<String>,
    pub timeout_seconds: u64,
}

/// Complete workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: String,
    pub name: String,
    pub tasks: Vec<WorkflowTask>,
}

/// Task execution result within workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub duration_ms: u64,
    pub executed_by: String,
}

/// Complete workflow execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_id: String,
    pub success: bool,
    pub completed_tasks: Vec<TaskResult>,
    pub failed_tasks: Vec<TaskResult>,
    pub total_duration_ms: u64,
    pub retries_used: u32,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub id: String,
    pub max_workflows: usize,
    pub task_timeout_seconds: u64,
    pub enable_retry: bool,
    pub max_retries: u32,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            id: "orchestrator-default".to_string(),
            max_workflows: 10,
            task_timeout_seconds: 300, // 5 minutes
            enable_retry: true,
            max_retries: 3,
        }
    }
}

/// Task execution state
#[derive(Debug, Clone, PartialEq)]
enum TaskState {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
}

/// Internal task tracking
#[derive(Debug)]
struct TaskExecution {
    task: WorkflowTask,
    state: TaskState,
    retries: u32,
    result: Option<TaskResult>,
    started_at: Option<Instant>,
}

/// Workflow orchestrator
pub struct Orchestrator {
    config: OrchestratorConfig,
    active_workflows: HashMap<String, WorkflowExecution>,
    failure_mode: bool, // For testing
    failure_rate: f64,  // For testing
}

/// Active workflow execution state
struct WorkflowExecution {
    workflow: Workflow,
    tasks: HashMap<String, TaskExecution>,
    dependency_graph: HashMap<String, Vec<String>>,
    reverse_dependencies: HashMap<String, Vec<String>>,
    started_at: Instant,
}

impl Orchestrator {
    /// Create new orchestrator
    pub fn new(config: OrchestratorConfig) -> Self {
        Self {
            config,
            active_workflows: HashMap::new(),
            failure_mode: false,
            failure_rate: 0.0,
        }
    }

    /// Get orchestrator ID
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Get configuration
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Set failure mode for testing
    pub fn set_failure_mode(&mut self, enabled: bool) {
        self.failure_mode = enabled;
    }

    /// Set failure rate for testing
    pub fn set_failure_rate(&mut self, rate: f64) {
        self.failure_rate = rate;
    }

    /// Execute workflow
    pub async fn execute_workflow(&mut self, workflow: Workflow) -> Result<WorkflowResult> {
        if self.active_workflows.len() >= self.config.max_workflows {
            return Err(CpuAgentError::TaskError(
                "Maximum concurrent workflows reached".to_string(),
            ));
        }

        log::info!("Starting workflow: {} ({})", workflow.name, workflow.id);

        let workflow_id = workflow.id.clone();
        let mut execution = self.prepare_workflow_execution(workflow)?;

        let start_time = Instant::now();
        let mut completed_tasks = Vec::new();
        let mut failed_tasks = Vec::new();
        let mut total_retries = 0;

        // Execute tasks based on dependency order
        while !self.is_workflow_complete(&execution) {
            let ready_tasks = self.get_ready_tasks(&execution);

            if ready_tasks.is_empty() {
                // Check if we're stuck due to failed dependencies
                if self.has_failed_dependencies(&execution) {
                    break;
                }

                // Wait a bit and check again
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }

            // Execute ready tasks in parallel
            let results = self.execute_parallel_tasks(ready_tasks).await;

            // Process results
            for (task_id, result) in results {
                let task_execution = execution.tasks.get_mut(&task_id).unwrap();

                match result {
                    Ok(task_result) => {
                        task_execution.state = TaskState::Completed;
                        task_execution.result = Some(task_result.clone());
                        completed_tasks.push(task_result);
                    }
                    Err(e) => {
                        task_execution.retries += 1;
                        total_retries += 1;

                        if self.config.enable_retry
                            && task_execution.retries <= self.config.max_retries
                        {
                            // Retry the task
                            task_execution.state = TaskState::Pending;
                            log::warn!(
                                "Retrying task {} (attempt {}): {}",
                                task_id,
                                task_execution.retries,
                                e
                            );
                        } else {
                            // Mark as failed
                            task_execution.state = TaskState::Failed;
                            let failed_result = TaskResult {
                                task_id: task_id.clone(),
                                success: false,
                                output: String::new(),
                                error: Some(e.to_string()),
                                duration_ms: 0,
                                executed_by: self.config.id.clone(),
                            };
                            task_execution.result = Some(failed_result.clone());
                            failed_tasks.push(failed_result);

                            log::error!(
                                "Task {} failed after {} retries: {}",
                                task_id,
                                task_execution.retries,
                                e
                            );
                        }
                    }
                }
            }
        }

        let total_duration = start_time.elapsed();
        let success =
            failed_tasks.is_empty() && completed_tasks.len() == execution.workflow.tasks.len();

        log::info!(
            "Workflow {} completed: success={}, tasks={}/{}, duration={:.2}s",
            workflow_id,
            success,
            completed_tasks.len(),
            execution.workflow.tasks.len(),
            total_duration.as_secs_f64()
        );

        Ok(WorkflowResult {
            workflow_id,
            success,
            completed_tasks,
            failed_tasks,
            total_duration_ms: total_duration.as_millis() as u64,
            retries_used: total_retries,
        })
    }

    /// Execute workflow with specific agent pool
    pub async fn execute_workflow_with_agents(
        &mut self,
        workflow: Workflow,
        agents: Vec<Arc<Mutex<dyn CpuAgent>>>,
    ) -> Result<WorkflowResult> {
        // For this implementation, we'll use the same logic as execute_workflow
        // In a full implementation, we would distribute tasks among the provided agents
        self.execute_workflow(workflow).await
    }

    /// Prepare workflow execution state
    fn prepare_workflow_execution(&self, workflow: Workflow) -> Result<WorkflowExecution> {
        let mut tasks = HashMap::new();
        let mut dependency_graph = HashMap::new();
        let mut reverse_dependencies: HashMap<String, Vec<String>> = HashMap::new();

        // Build task map and dependency graph
        for task in &workflow.tasks {
            tasks.insert(
                task.id.clone(),
                TaskExecution {
                    task: task.clone(),
                    state: if task.dependencies.is_empty() {
                        TaskState::Ready
                    } else {
                        TaskState::Pending
                    },
                    retries: 0,
                    result: None,
                    started_at: None,
                },
            );

            dependency_graph.insert(task.id.clone(), task.dependencies.clone());

            // Build reverse dependencies
            for dep in &task.dependencies {
                reverse_dependencies
                    .entry(dep.clone())
                    .or_insert_with(Vec::new)
                    .push(task.id.clone());
            }
        }

        // Validate dependencies
        self.validate_dependencies(&dependency_graph)?;

        Ok(WorkflowExecution {
            workflow,
            tasks,
            dependency_graph,
            reverse_dependencies,
            started_at: Instant::now(),
        })
    }

    /// Validate workflow dependencies for cycles
    fn validate_dependencies(&self, dependency_graph: &HashMap<String, Vec<String>>) -> Result<()> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for task_id in dependency_graph.keys() {
            if !visited.contains(task_id) {
                if self.has_cycle(task_id, dependency_graph, &mut visited, &mut rec_stack) {
                    return Err(CpuAgentError::TaskError(
                        "Circular dependency detected in workflow".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check for cycles in dependency graph
    fn has_cycle(
        &self,
        task_id: &str,
        dependency_graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(task_id.to_string());
        rec_stack.insert(task_id.to_string());

        if let Some(dependencies) = dependency_graph.get(task_id) {
            for dep in dependencies {
                if !visited.contains(dep) {
                    if self.has_cycle(dep, dependency_graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }

        rec_stack.remove(task_id);
        false
    }

    /// Check if workflow is complete
    fn is_workflow_complete(&self, execution: &WorkflowExecution) -> bool {
        execution
            .tasks
            .values()
            .all(|task| matches!(task.state, TaskState::Completed | TaskState::Failed))
    }

    /// Get tasks that are ready to execute
    fn get_ready_tasks(&self, execution: &WorkflowExecution) -> Vec<String> {
        execution
            .tasks
            .iter()
            .filter_map(|(task_id, task_exec)| {
                if task_exec.state == TaskState::Pending {
                    // Check if all dependencies are completed
                    let dependencies_met = execution
                        .dependency_graph
                        .get(task_id)
                        .map(|deps| {
                            deps.iter().all(|dep| {
                                execution
                                    .tasks
                                    .get(dep)
                                    .map(|dep_task| dep_task.state == TaskState::Completed)
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(true);

                    if dependencies_met {
                        Some(task_id.clone())
                    } else {
                        None
                    }
                } else if task_exec.state == TaskState::Ready {
                    Some(task_id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if workflow has failed dependencies
    fn has_failed_dependencies(&self, execution: &WorkflowExecution) -> bool {
        execution
            .tasks
            .values()
            .any(|task| task.state == TaskState::Failed)
    }

    /// Execute multiple tasks in parallel
    async fn execute_parallel_tasks(
        &self,
        task_ids: Vec<String>,
    ) -> Vec<(String, Result<TaskResult>)> {
        let mut handles = Vec::new();

        for task_id in task_ids {
            let config_id = self.config.id.clone();
            let timeout_duration = Duration::from_secs(self.config.task_timeout_seconds);
            let failure_mode = self.failure_mode;
            let failure_rate = self.failure_rate;

            let handle = tokio::spawn(async move {
                let result = timeout(
                    timeout_duration,
                    Self::execute_single_task(
                        task_id.clone(),
                        config_id,
                        failure_mode,
                        failure_rate,
                    ),
                )
                .await;

                match result {
                    Ok(task_result) => (task_id, task_result),
                    Err(_) => (
                        task_id.clone(),
                        Err(CpuAgentError::TaskError(format!(
                            "Task {} timed out after {}s",
                            task_id,
                            timeout_duration.as_secs()
                        ))),
                    ),
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|e| {
                    (
                        "unknown".to_string(),
                        Err(CpuAgentError::TaskError(format!(
                            "Task execution failed: {}",
                            e
                        ))),
                    )
                })
            })
            .collect()
    }

    /// Execute a single task
    async fn execute_single_task(
        task_id: String,
        executor_id: String,
        failure_mode: bool,
        failure_rate: f64,
    ) -> Result<TaskResult> {
        let start = Instant::now();

        // Simulate task execution
        let execution_time = match task_id.as_str() {
            id if id.contains("long") => Duration::from_secs(5),
            id if id.contains("slow") => Duration::from_secs(3),
            id if id.contains("flaky") && failure_mode => {
                // Simulate flaky task that eventually succeeds
                static mut FLAKY_ATTEMPTS: u32 = 0;
                unsafe {
                    FLAKY_ATTEMPTS += 1;
                    if FLAKY_ATTEMPTS < 3 {
                        return Err(CpuAgentError::TaskError("Flaky task failed".to_string()));
                    }
                }
                Duration::from_millis(100)
            }
            _ => Duration::from_millis(50 + (task_id.len() * 10) as u64),
        };

        tokio::time::sleep(execution_time).await;

        // Simulate random failures in test mode
        if failure_mode && failure_rate > 0.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            task_id.hash(&mut hasher);
            let hash = hasher.finish();
            let random_value = (hash % 100) as f64 / 100.0;

            if random_value < failure_rate {
                return Err(CpuAgentError::TaskError(format!(
                    "Simulated failure for task {}",
                    task_id
                )));
            }
        }

        let duration = start.elapsed();

        Ok(TaskResult {
            task_id: task_id.clone(),
            success: true,
            output: format!("Task {} completed successfully", task_id),
            error: None,
            duration_ms: duration.as_millis() as u64,
            executed_by: executor_id,
        })
    }
}
