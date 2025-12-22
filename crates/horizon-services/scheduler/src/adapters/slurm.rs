use crate::models::{Job, Priority};
use crate::Result;

/// Slurm compatibility adapter
pub struct SlurmAdapter;

impl SlurmAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Parse sbatch script and extract job parameters
    pub fn parse_sbatch(&self, script: &str) -> Result<Job> {
        let user_id = "slurm_user";
        let mut gpu_count = 1;
        let mut priority = Priority::Normal;
        let mut job_name = None;
        let mut _time_limit = None;

        for line in script.lines() {
            let line = line.trim();

            if line.starts_with("#SBATCH") {
                if line.contains("--gres=gpu:") {
                    if let Some(count_str) = line.split("--gres=gpu:").nth(1) {
                        if let Ok(count) = count_str.trim().parse::<usize>() {
                            gpu_count = count;
                        }
                    }
                } else if line.contains("--job-name=") {
                    if let Some(name) = line.split("--job-name=").nth(1) {
                        job_name = Some(name.trim().to_string());
                    }
                } else if line.contains("--time=") {
                    if let Some(time) = line.split("--time=").nth(1) {
                        _time_limit = Some(time.trim().to_string());
                    }
                } else if line.contains("--priority=") {
                    if let Some(prio_str) = line.split("--priority=").nth(1) {
                        priority = match prio_str.trim().to_lowercase().as_str() {
                            "high" => Priority::High,
                            "low" => Priority::Low,
                            _ => Priority::Normal,
                        };
                    }
                }
            }
        }

        let mut builder = Job::builder().user_id(user_id).gpu_count(gpu_count).priority(priority);

        if let Some(name) = job_name {
            builder = builder.job_name(name);
        }

        builder.build()
    }

    /// Format job status for squeue-like output
    pub fn format_squeue(&self, jobs: &[Job]) -> String {
        let mut output = String::from("JOBID     USER       STATE      GPUS\n");

        for job in jobs {
            let gpu_count = job.resources.get_gpu_spec()
                .map(|s| s.amount as usize)
                .unwrap_or(0);

            output.push_str(&format!(
                "{:<10} {:<10} {:<10} {}\n",
                &job.id.to_string()[..8],
                &job.user_id,
                format!("{:?}", job.state),
                gpu_count
            ));
        }

        output
    }
}

impl Default for SlurmAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sbatch() {
        let adapter = SlurmAdapter::new();

        let script = r#"
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=training
python train.py
        "#;

        let job = adapter.parse_sbatch(script).unwrap();
        let gpu_count = job.resources.get_gpu_spec()
            .map(|s| s.amount as usize)
            .unwrap_or(0);
        assert_eq!(gpu_count, 4);
        assert_eq!(job.job_name, Some("training".to_string()));
    }

    #[test]
    fn test_format_squeue() {
        let adapter = SlurmAdapter::new();

        let jobs = vec![
            Job::builder().user_id("user1").gpu_count(2).build().unwrap(),
            Job::builder().user_id("user2").gpu_count(4).build().unwrap(),
        ];

        let output = adapter.format_squeue(&jobs);
        assert!(output.contains("JOBID"));
        assert!(output.contains("user1"));
        assert!(output.contains("user2"));
    }
}
