//! Interactive project picker using dialoguer
//!
//! Provides an interactive terminal UI for selecting projects to deploy.

use anyhow::Result;
use console::style;
use dialoguer::{theme::ColorfulTheme, Confirm, MultiSelect, Select};

use crate::core::project::{get_all_projects, get_deployable_projects, resolve_dependencies, Project, ProjectCategory};

/// Deployment target options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployTarget {
    Local,
    Cluster,
}

impl DeployTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cluster => "cluster",
        }
    }
}

/// Environment options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployEnvironment {
    Dev,
    Staging,
    Prod,
}

impl DeployEnvironment {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Staging => "staging",
            Self::Prod => "prod",
        }
    }
}

/// Result of interactive project selection
#[derive(Debug)]
pub struct DeploySelection {
    /// Selected projects (with dependencies resolved)
    pub projects: Vec<String>,
    /// Original user selection (before dependency resolution)
    pub user_selected: Vec<String>,
    /// Target (local/cluster)
    pub target: DeployTarget,
    /// Environment
    pub environment: DeployEnvironment,
    /// Whether to perform dry run
    pub dry_run: bool,
}

/// Preset deployment profiles
pub struct DeployProfile {
    pub name: &'static str,
    pub description: &'static str,
    pub projects: &'static [&'static str],
}

/// Get available deployment profiles
pub fn get_profiles() -> Vec<DeployProfile> {
    vec![
        DeployProfile {
            name: "ml-training",
            description: "GPU ML training stack (rnccl, slai, torch, argus)",
            projects: &["rnccl", "slai", "torch", "argus"],
        },
        DeployProfile {
            name: "data-processing",
            description: "Distributed data processing (rmpi, spark, warp, argus)",
            projects: &["rmpi", "spark", "warp", "argus"],
        },
        DeployProfile {
            name: "full-stack",
            description: "Complete HPC-AI platform",
            projects: &["rnccl", "slai", "torch", "spark", "warp", "stratoswarm", "nebula", "vortex", "argus"],
        },
        DeployProfile {
            name: "monitoring",
            description: "Observability only (argus, slai)",
            projects: &["argus", "slai"],
        },
        DeployProfile {
            name: "minimal",
            description: "Minimal stack (stratoswarm, argus)",
            projects: &["stratoswarm", "argus"],
        },
    ]
}

/// Interactive project picker
pub struct ProjectPicker {
    theme: ColorfulTheme,
    target: DeployTarget,
    environment: DeployEnvironment,
}

impl Default for ProjectPicker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectPicker {
    /// Create a new project picker
    pub fn new() -> Self {
        Self {
            theme: ColorfulTheme::default(),
            target: DeployTarget::Local,
            environment: DeployEnvironment::Dev,
        }
    }

    /// Set the deployment target
    pub fn with_target(mut self, target: &str) -> Self {
        self.target = match target {
            "cluster" => DeployTarget::Cluster,
            _ => DeployTarget::Local,
        };
        self
    }

    /// Set the environment
    pub fn with_environment(mut self, env: &str) -> Self {
        self.environment = match env {
            "staging" => DeployEnvironment::Staging,
            "prod" => DeployEnvironment::Prod,
            _ => DeployEnvironment::Dev,
        };
        self
    }

    /// Run the interactive picker flow
    pub fn run(&self) -> Result<Option<DeploySelection>> {
        self.print_header();

        // Step 1: Choose selection mode
        let mode = self.select_mode()?;

        let user_selected = match mode {
            SelectionMode::Profile => self.select_profile()?,
            SelectionMode::Category => self.select_by_category()?,
            SelectionMode::Individual => self.select_individual()?,
            SelectionMode::Cancel => return Ok(None),
        };

        if user_selected.is_empty() {
            println!();
            println!("{}", style("No projects selected. Aborting.").yellow());
            return Ok(None);
        }

        // Step 2: Resolve dependencies
        let resolved = resolve_dependencies(&user_selected);
        let added_deps: Vec<_> = resolved
            .iter()
            .filter(|p| !user_selected.contains(p))
            .cloned()
            .collect();

        // Step 3: Show resolved projects and confirm
        self.show_deployment_summary(&user_selected, &added_deps, &resolved)?;

        // Step 4: Confirm deployment
        let dry_run = self.confirm_deployment()?;

        match dry_run {
            Some(dry) => Ok(Some(DeploySelection {
                projects: resolved,
                user_selected,
                target: self.target,
                environment: self.environment,
                dry_run: dry,
            })),
            None => Ok(None),
        }
    }

    fn print_header(&self) {
        println!();
        println!("{}", style("HPC-AI Interactive Deploy").cyan().bold());
        println!("{}", style("=".repeat(40)).dim());
        println!();
        println!(
            "Target:      {}",
            style(self.target.as_str()).green()
        );
        println!(
            "Environment: {}",
            style(self.environment.as_str()).green()
        );
        println!();
    }

    fn select_mode(&self) -> Result<SelectionMode> {
        let options = vec![
            "Use a preset profile (ml-training, data-processing, etc.)",
            "Select by category (GPU, Networking, etc.)",
            "Pick individual projects",
            "Cancel",
        ];

        let selection = Select::with_theme(&self.theme)
            .with_prompt("How would you like to select projects?")
            .items(&options)
            .default(0)
            .interact()?;

        Ok(match selection {
            0 => SelectionMode::Profile,
            1 => SelectionMode::Category,
            2 => SelectionMode::Individual,
            _ => SelectionMode::Cancel,
        })
    }

    fn select_profile(&self) -> Result<Vec<String>> {
        let profiles = get_profiles();
        let items: Vec<String> = profiles
            .iter()
            .map(|p| format!("{}: {}", style(&p.name).cyan(), p.description))
            .collect();

        let selection = Select::with_theme(&self.theme)
            .with_prompt("Select a deployment profile")
            .items(&items)
            .default(0)
            .interact()?;

        Ok(profiles[selection]
            .projects
            .iter()
            .map(|s| s.to_string())
            .collect())
    }

    fn select_by_category(&self) -> Result<Vec<String>> {
        let categories = vec![
            (ProjectCategory::Storage, "Storage & Communication (parcode, rmpi, rnccl)"),
            (ProjectCategory::Gpu, "GPU Management (slai)"),
            (ProjectCategory::Orchestration, "Orchestration (warp, stratoswarm)"),
            (ProjectCategory::DataProcessing, "Data Processing & ML (spark, torch)"),
            (ProjectCategory::Networking, "Networking (vortex, nebula)"),
            (ProjectCategory::Observability, "Observability (argus)"),
        ];

        let items: Vec<&str> = categories.iter().map(|(_, desc)| *desc).collect();

        let selections = MultiSelect::with_theme(&self.theme)
            .with_prompt("Select categories (Space to toggle, Enter to confirm)")
            .items(&items)
            .interact()?;

        if selections.is_empty() {
            return Ok(vec![]);
        }

        // Get all projects from selected categories
        let all_projects = get_all_projects();
        let mut selected_projects = Vec::new();

        for idx in selections {
            let category = categories[idx].0;
            for project in &all_projects {
                if project.category == category && project.deployable {
                    selected_projects.push(project.id.clone());
                }
            }
        }

        // Let user refine within categories
        if !selected_projects.is_empty() {
            println!();
            println!("{}", style("Refine selection within chosen categories:").dim());

            let items: Vec<String> = selected_projects
                .iter()
                .map(|id| {
                    let proj = all_projects.iter().find(|p| &p.id == id).unwrap();
                    format!("{}: {}", style(&proj.id).cyan(), proj.description)
                })
                .collect();

            let refined = MultiSelect::with_theme(&self.theme)
                .with_prompt("Select projects (Space to toggle)")
                .items(&items)
                .defaults(&vec![true; items.len()])
                .interact()?;

            selected_projects = refined
                .iter()
                .map(|&idx| selected_projects[idx].clone())
                .collect();
        }

        Ok(selected_projects)
    }

    fn select_individual(&self) -> Result<Vec<String>> {
        let deployable = get_deployable_projects();

        let items: Vec<String> = deployable
            .iter()
            .map(|p| {
                let deps = if p.dependencies.is_empty() {
                    String::new()
                } else {
                    format!(" [deps: {}]", p.dependencies.join(", "))
                };
                format!(
                    "{}: {}{}",
                    style(&p.id).cyan(),
                    p.description,
                    style(deps).dim()
                )
            })
            .collect();

        let selections = MultiSelect::with_theme(&self.theme)
            .with_prompt("Select projects to deploy (Space to toggle, Enter to confirm)")
            .items(&items)
            .interact()?;

        Ok(selections
            .iter()
            .map(|&idx| deployable[idx].id.clone())
            .collect())
    }

    fn show_deployment_summary(
        &self,
        user_selected: &[String],
        added_deps: &[String],
        all_resolved: &[String],
    ) -> Result<()> {
        println!();
        println!("{}", style("Deployment Summary").cyan().bold());
        println!("{}", style("-".repeat(40)).dim());
        println!();

        println!("{}:", style("Selected projects").green());
        for proj in user_selected {
            println!("  {} {}", style("+").green(), proj);
        }

        if !added_deps.is_empty() {
            println!();
            println!("{}:", style("Dependencies (auto-added)").yellow());
            for dep in added_deps {
                println!("  {} {} (dependency)", style("+").yellow(), dep);
            }
        }

        println!();
        println!(
            "Total: {} projects will be deployed",
            style(all_resolved.len()).cyan().bold()
        );
        println!();

        Ok(())
    }

    fn confirm_deployment(&self) -> Result<Option<bool>> {
        let options = vec![
            "Deploy now",
            "Dry run (show what would happen)",
            "Cancel",
        ];

        let selection = Select::with_theme(&self.theme)
            .with_prompt("Proceed with deployment?")
            .items(&options)
            .default(0)
            .interact()?;

        Ok(match selection {
            0 => Some(false), // Deploy (not dry run)
            1 => Some(true),  // Dry run
            _ => None,        // Cancel
        })
    }
}

enum SelectionMode {
    Profile,
    Category,
    Individual,
    Cancel,
}

/// Execute a deployment based on selection
pub fn execute_deployment(selection: &DeploySelection) -> Result<()> {
    let all_projects = get_all_projects();

    println!();
    if selection.dry_run {
        println!("{}", style("[DRY RUN] Would deploy:").yellow().bold());
    } else {
        println!("{}", style("Deploying projects:").green().bold());
    }
    println!();

    for (idx, project_id) in selection.projects.iter().enumerate() {
        let project = all_projects.iter().find(|p| &p.id == project_id);
        let name = project.map(|p| p.name.as_str()).unwrap_or(project_id);
        let port_info = project
            .and_then(|p| p.default_port)
            .map(|p| format!(" (port {})", p))
            .unwrap_or_default();

        let prefix = if selection.dry_run { "  " } else { "  " };
        let status = if selection.dry_run {
            style("would deploy").yellow()
        } else {
            style("deploying...").green()
        };

        println!(
            "{}[{}/{}] {} {}{}",
            prefix,
            idx + 1,
            selection.projects.len(),
            status,
            style(name).cyan(),
            style(port_info).dim()
        );
    }

    println!();

    if selection.dry_run {
        println!("{}", style("Dry run complete. No changes made.").yellow());
    } else {
        println!(
            "{}",
            style("Note: Actual deployment requires Docker or StratoSwarm.").dim()
        );
        println!();
        println!("Next steps:");
        println!(
            "  {} Start services with docker-compose",
            style("1.").cyan()
        );
        println!(
            "  {} Monitor with 'hpc deploy status'",
            style("2.").cyan()
        );
        println!(
            "  {} View logs with 'hpc argus logs'",
            style("3.").cyan()
        );
    }

    Ok(())
}

/// Quick select from preset profiles without full interactive flow
pub fn quick_profile_select() -> Result<Option<Vec<String>>> {
    let profiles = get_profiles();
    let items: Vec<String> = profiles
        .iter()
        .map(|p| format!("{}: {}", p.name, p.description))
        .collect();

    let theme = ColorfulTheme::default();
    let selection = Select::with_theme(&theme)
        .with_prompt("Quick profile selection")
        .items(&items)
        .default(0)
        .interact_opt()?;

    Ok(selection.map(|idx| {
        profiles[idx]
            .projects
            .iter()
            .map(|s| s.to_string())
            .collect()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiles() {
        let profiles = get_profiles();
        assert!(!profiles.is_empty());
        assert!(profiles.iter().any(|p| p.name == "ml-training"));
    }

    #[test]
    fn test_deploy_target() {
        assert_eq!(DeployTarget::Local.as_str(), "local");
        assert_eq!(DeployTarget::Cluster.as_str(), "cluster");
    }

    #[test]
    fn test_deploy_environment() {
        assert_eq!(DeployEnvironment::Dev.as_str(), "dev");
        assert_eq!(DeployEnvironment::Staging.as_str(), "staging");
        assert_eq!(DeployEnvironment::Prod.as_str(), "prod");
    }

    #[test]
    fn test_picker_builder() {
        let picker = ProjectPicker::new()
            .with_target("cluster")
            .with_environment("staging");

        assert_eq!(picker.target, DeployTarget::Cluster);
        assert_eq!(picker.environment, DeployEnvironment::Staging);
    }
}
