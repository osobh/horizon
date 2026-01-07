//! RMPI CLI
//!
//! Rust MPI utilities.

use clap::Subcommand;

#[derive(Subcommand)]
pub enum RmpiCommands {
    /// Show RMPI info
    Info,

    /// Run a distributed program
    Run {
        /// Number of processes
        #[arg(short = 'n', long, default_value = "4")]
        np: u32,
        /// Hostfile
        #[arg(long)]
        hostfile: Option<String>,
        /// Program to run
        program: String,
        /// Program arguments
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Show cluster topology
    Topology,

    /// Benchmark collective operations
    Bench {
        /// Operation (allreduce, broadcast, gather, scatter)
        #[arg(long, default_value = "allreduce")]
        op: String,
        /// Message size
        #[arg(long, default_value = "1M")]
        size: String,
    },
}

impl RmpiCommands {
    pub async fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Info => {
                println!("RMPI Information");
                println!("================");
                println!("Version:    0.1.0");
                println!("Backend:    Native Rust");
                println!("Transport:  TCP/RDMA");
                println!();
                println!("Features:");
                println!("  - Point-to-point: Send, Recv, Isend, Irecv");
                println!("  - Collectives:    Bcast, Reduce, Allreduce");
                println!("  - Groups:         Scatter, Gather, Allgather");
                Ok(())
            }
            Self::Run { np, hostfile, program, args } => {
                println!("RMPI Run");
                println!("========");
                println!("Processes: {}", np);
                if let Some(hf) = hostfile {
                    println!("Hostfile:  {}", hf);
                }
                println!("Program:   {}", program);
                if !args.is_empty() {
                    println!("Args:      {}", args.join(" "));
                }
                println!();
                println!("Note: Full rmpi integration pending.");
                Ok(())
            }
            Self::Topology => {
                println!("RMPI Topology");
                println!("=============");
                println!("Local node only (no cluster connected).");
                Ok(())
            }
            Self::Bench { op, size } => {
                println!("RMPI Benchmark");
                println!("==============");
                println!("Operation: {}", op);
                println!("Size:      {}", size);
                println!();
                println!("Note: Full rmpi integration pending.");
                Ok(())
            }
        }
    }
}
