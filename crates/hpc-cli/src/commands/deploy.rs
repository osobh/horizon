//! Quick deploy command
//!
//! Shorthand for common deployment operations.

use clap::Subcommand;

use super::picker::{execute_deployment, ProjectPicker};
use crate::core::project::resolve_dependencies;

/// Deploy commands for quick project deployment
#[derive(Subcommand)]
pub enum DeployCommands {
    /// Deploy specific projects
    Projects {
        /// Projects to deploy (comma-separated)
        projects: String,

        /// Deployment target (local, cluster)
        #[arg(short, long, default_value = "local")]
        target: String,

        /// Environment (dev, staging, prod)
        #[arg(short, long, default_value = "dev")]
        env: String,

        /// Dry run mode
        #[arg(long)]
        dry_run: bool,
    },

    /// Deploy using a preset profile
    Profile {
        /// Profile name (ml-training, data-processing, full-stack, monitoring)
        name: String,

        /// Deployment target
        #[arg(short, long, default_value = "local")]
        target: String,

        /// Environment
        #[arg(short, long, default_value = "dev")]
        env: String,

        /// Dry run mode
        #[arg(long)]
        dry_run: bool,
    },

    /// Interactive project selection (launches TUI picker)
    Interactive {
        /// Deployment target
        #[arg(short, long, default_value = "local")]
        target: String,

        /// Environment
        #[arg(short, long, default_value = "dev")]
        env: String,
    },

    /// Deploy local development stack
    Local {
        /// Projects to deploy (comma-separated, defaults to stratoswarm,argus)
        #[arg(short, long)]
        projects: Option<String>,

        /// Dry run mode
        #[arg(long)]
        dry_run: bool,
    },

    /// Deploy to StratoSwarm cluster
    Cluster {
        /// Cluster endpoint or name
        cluster: String,

        /// Projects to deploy
        #[arg(short, long)]
        projects: Option<String>,

        /// Namespace
        #[arg(short, long, default_value = "hpc-ai")]
        namespace: String,
    },

    /// Show deployment status
    Status {
        /// Show all environments
        #[arg(short, long)]
        all: bool,
    },
}

impl DeployCommands {
    pub async fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Projects { projects, target, env, dry_run } => {
                use console::style;

                let project_list: Vec<String> = projects
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();

                println!();
                println!("{}", style("Project Deployment").cyan().bold());
                println!("{}", style("=".repeat(40)).dim());
                println!();
                println!("Projects:    {}", style(projects.replace(',', ", ")).green());
                println!("Target:      {}", style(&target).green());
                println!("Environment: {}", style(&env).green());
                println!();

                // Resolve dependencies using core module
                println!("Resolving dependencies...");
                let resolved = resolve_dependencies(&project_list);
                let added: Vec<_> = resolved
                    .iter()
                    .filter(|p| !project_list.contains(p))
                    .collect();

                if !added.is_empty() {
                    println!(
                        "  {} Adding dependencies: {}",
                        style("+").yellow(),
                        style(added.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")).yellow()
                    );
                }
                println!();

                // Create selection and execute
                let selection = super::picker::DeploySelection {
                    projects: resolved.clone(),
                    user_selected: project_list,
                    target: if target == "cluster" {
                        super::picker::DeployTarget::Cluster
                    } else {
                        super::picker::DeployTarget::Local
                    },
                    environment: match env.as_str() {
                        "staging" => super::picker::DeployEnvironment::Staging,
                        "prod" => super::picker::DeployEnvironment::Prod,
                        _ => super::picker::DeployEnvironment::Dev,
                    },
                    dry_run,
                };

                execute_deployment(&selection)?;
                Ok(())
            }

            Self::Profile { name, target, env, dry_run } => {
                use console::style;
                use super::picker::get_profiles;

                let profiles = get_profiles();
                let profile = profiles.iter().find(|p| p.name == name);

                let projects: Vec<String> = match profile {
                    Some(p) => p.projects.iter().map(|s| s.to_string()).collect(),
                    None => {
                        println!();
                        println!("{}: {}", style("Unknown profile").red(), name);
                        println!();
                        println!("Available profiles:");
                        for p in &profiles {
                            println!(
                                "  {} {}: {}",
                                style("-").dim(),
                                style(p.name).cyan(),
                                p.description
                            );
                        }
                        return Ok(());
                    }
                };

                println!();
                println!("{}", style("Profile Deployment").cyan().bold());
                println!("{}", style("=".repeat(40)).dim());
                println!();
                println!("Profile:     {}", style(&name).green());
                println!("Target:      {}", style(&target).green());
                println!("Environment: {}", style(&env).green());
                println!();

                // Resolve dependencies
                let resolved = resolve_dependencies(&projects);

                let selection = super::picker::DeploySelection {
                    projects: resolved,
                    user_selected: projects,
                    target: if target == "cluster" {
                        super::picker::DeployTarget::Cluster
                    } else {
                        super::picker::DeployTarget::Local
                    },
                    environment: match env.as_str() {
                        "staging" => super::picker::DeployEnvironment::Staging,
                        "prod" => super::picker::DeployEnvironment::Prod,
                        _ => super::picker::DeployEnvironment::Dev,
                    },
                    dry_run,
                };

                execute_deployment(&selection)?;
                Ok(())
            }

            Self::Interactive { target, env } => {
                let picker = ProjectPicker::new()
                    .with_target(&target)
                    .with_environment(&env);

                match picker.run()? {
                    Some(selection) => {
                        execute_deployment(&selection)?;
                    }
                    None => {
                        println!();
                        println!("Deployment cancelled.");
                    }
                }
                Ok(())
            }

            Self::Local { projects, dry_run } => {
                let project_list = projects.as_deref().unwrap_or("stratoswarm,argus");

                println!("Local Deployment");
                println!("================");
                println!();
                println!("Projects: {}", project_list);
                println!("Target:   local (docker-compose)");
                println!();

                if dry_run {
                    println!("[DRY RUN] Would execute:");
                    println!("  docker-compose up -d {}", project_list.replace(',', " "));
                    return Ok(());
                }

                println!("Note: Local deployment requires Docker to be running.");
                println!("      Use 'hpc stack up --target local' for managed deployment.");
                Ok(())
            }

            Self::Cluster { cluster, projects, namespace } => {
                println!("Cluster Deployment");
                println!("==================");
                println!();
                println!("Cluster:   {}", cluster);
                println!("Namespace: {}", namespace);
                if let Some(proj) = &projects {
                    println!("Projects:  {}", proj);
                } else {
                    println!("Projects:  (all from stack config)");
                }
                println!();

                println!("Connecting to cluster...");
                println!();
                println!("Note: Cluster deployment requires StratoSwarm integration.");
                println!("      Use 'hpc swarm deploy' for direct cluster management.");
                Ok(())
            }

            Self::Status { all } => {
                println!("Deployment Status");
                println!("=================");
                println!();

                if all {
                    println!("All Environments:");
                    println!();
                    println!("  Dev:");
                    println!("    No active deployments");
                    println!();
                    println!("  Staging:");
                    println!("    No active deployments");
                    println!();
                    println!("  Production:");
                    println!("    No active deployments");
                } else {
                    println!("Current Environment: dev");
                    println!();
                    println!("No active deployments.");
                }
                println!();
                println!("Use 'hpc deploy local' to start a local deployment.");
                Ok(())
            }
        }
    }
}
