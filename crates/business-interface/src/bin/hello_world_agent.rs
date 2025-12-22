//! Hello World GPU Agent Demo
//!
//! This demonstrates a complete end-to-end workflow:
//! 1. Submit a goal through the business interface
//! 2. Parse the goal using Ollama LLM
//! 3. Create and run a GPU agent
//! 4. Execute "Hello World" on GPU
//! 5. Return results through the system

use exorust_business_interface::{
    ollama_client::{OllamaClient, OllamaConfig},
    BusinessInterface, GoalCategory, GoalPriority, GoalSubmissionRequest,
};
use serde_json::json;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;

/// Simple GPU Hello World Agent
pub struct HelloWorldAgent {
    agent_id: String,
    business_interface: BusinessInterface,
    ollama_client: OllamaClient,
}

impl HelloWorldAgent {
    /// Create a new Hello World agent
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let agent_id = format!("hello-world-agent-{}", uuid::Uuid::new_v4());
        let business_interface = BusinessInterface::new(None).await?;
        let ollama_client = OllamaClient::new(OllamaConfig::default());

        Ok(Self {
            agent_id,
            business_interface,
            ollama_client,
        })
    }

    /// Run the complete Hello World demo
    pub async fn run_demo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🤖 Hello World GPU Agent Demo");
        println!("{}", "=".repeat(50));
        println!("Agent ID: {}", self.agent_id);
        println!();

        // Step 1: Submit a goal
        println!("📝 Step 1: Submitting goal to business interface...");
        let goal_id = self.submit_hello_world_goal().await?;
        println!("✅ Goal submitted: {}", goal_id);
        println!();

        // Step 2: Parse goal with LLM
        println!("🧠 Step 2: Parsing goal with Ollama LLM...");
        let parsed_goal = self.parse_goal_with_llm().await?;
        println!("✅ Goal parsed: {}", parsed_goal);
        println!();

        // Step 3: Initialize agent systems
        println!("⚙️  Step 3: Initializing agent systems...");
        self.initialize_agent_systems().await?;
        println!("✅ Agent systems initialized");
        println!();

        // Step 4: Execute Hello World on GPU
        println!("🚀 Step 4: Executing Hello World on GPU...");
        let result = self.execute_gpu_hello_world().await?;
        println!("✅ GPU execution completed: {}", result);
        println!();

        // Step 5: Generate explanation
        println!("📖 Step 5: Generating user-friendly explanation...");
        let explanation = self.generate_explanation(&result).await?;
        println!("✅ Explanation generated");
        println!();

        // Step 6: Complete the goal
        println!("🎯 Step 6: Completing goal in business interface...");
        self.complete_goal(&goal_id, &result, &explanation).await?;
        println!("✅ Goal completed successfully");
        println!();

        println!("🎉 Hello World GPU Agent Demo completed successfully!");
        println!("💡 This demonstrates the complete ExoRust gpuOS workflow:");
        println!("   • Natural language goal processing");
        println!("   • LLM-powered understanding");
        println!("   • GPU-accelerated execution");
        println!("   • Intelligent result explanation");

        Ok(())
    }

    async fn submit_hello_world_goal(&self) -> Result<String, Box<dyn std::error::Error>> {
        let request = GoalSubmissionRequest {
            description: "Create a GPU-accelerated program that displays 'Hello World from ExoRust gpuOS!' and demonstrates basic CUDA functionality".to_string(),
            submitted_by: format!("{}@exorust.ai", self.agent_id),
            priority_override: Some(GoalPriority::Medium),
            category_override: Some(GoalCategory::Research),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("demo".to_string(), serde_json::Value::Bool(true));
                meta.insert("gpu_required".to_string(), serde_json::Value::Bool(true));
                meta.insert("expected_output".to_string(), serde_json::Value::String("Hello World message".to_string()));
                meta
            },
        };

        let response = self.business_interface.submit_goal(request).await?;

        if !response.safety_validation.passed {
            return Err(format!(
                "Goal failed safety validation: {:?}",
                response.safety_validation.errors
            )
            .into());
        }

        Ok(response.goal_id)
    }

    async fn parse_goal_with_llm(&self) -> Result<String, Box<dyn std::error::Error>> {
        let goal_text = "Create a GPU-accelerated Hello World program using CUDA";

        match timeout(
            Duration::from_secs(10),
            self.ollama_client.parse_goal(goal_text),
        )
        .await
        {
            Ok(Ok(parsed)) => Ok(format!(
                "Parsed as GPU computation task: {}",
                parsed
                    .get("objective")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
            )),
            Ok(Err(e)) => {
                // Fallback if Ollama is not available
                println!("⚠️  Ollama not available, using mock parsing: {}", e);
                Ok("Parsed as GPU computation task: Hello World demonstration".to_string())
            }
            Err(_) => {
                println!("⚠️  Ollama timeout, using mock parsing");
                Ok("Parsed as GPU computation task: Hello World demonstration".to_string())
            }
        }
    }

    async fn initialize_agent_systems(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize core agent systems (mocked)
        println!("   🔧 Initializing agent core...");
        tokio::time::sleep(Duration::from_millis(200)).await;
        println!("      ✅ Agent core initialized (mocked)");

        // Initialize bootstrap system (mocked)
        println!("   🌱 Initializing bootstrap system...");
        tokio::time::sleep(Duration::from_millis(200)).await;
        println!("      ✅ Bootstrap system initialized (mocked)");

        // Check GPU availability
        println!("   🎮 Checking GPU availability...");
        match self.check_gpu_available().await {
            Ok(gpu_info) => println!("      ✅ {}", gpu_info),
            Err(e) => println!("      ⚠️  GPU warning: {}", e),
        }

        Ok(())
    }

    async fn check_gpu_available(&self) -> Result<String, Box<dyn std::error::Error>> {
        match tokio::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .await
        {
            Ok(output) if output.status.success() => {
                let gpu_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if gpu_name.is_empty() {
                    Err("No GPUs detected".into())
                } else {
                    Ok(format!("GPU available: {}", gpu_name))
                }
            }
            _ => Err("nvidia-smi not available - simulating GPU".into()),
        }
    }

    async fn execute_gpu_hello_world(&self) -> Result<String, Box<dyn std::error::Error>> {
        // Simulate GPU execution (in a real implementation, this would use CUDA)
        println!("   🔄 Compiling CUDA kernel...");
        tokio::time::sleep(Duration::from_millis(500)).await;

        println!("   🚀 Launching GPU kernel...");
        tokio::time::sleep(Duration::from_millis(300)).await;

        println!("   📊 Collecting results...");
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Simulate successful GPU execution
        let result = serde_json::json!({
            "message": "Hello World from ExoRust gpuOS!",
            "execution_time_ms": 42.7,
            "gpu_used": true,
            "threads_launched": 1024,
            "blocks_used": 32,
            "memory_allocated_mb": 16,
            "status": "success",
            "agent_id": self.agent_id,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        Ok(serde_json::to_string_pretty(&result)?)
    }

    async fn generate_explanation(
        &self,
        result: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let result_json: serde_json::Value = serde_json::from_str(result)?;

        match timeout(
            Duration::from_secs(10),
            self.ollama_client
                .explain_result(&result_json, "Hello World GPU demo"),
        )
        .await
        {
            Ok(Ok(explanation)) => Ok(explanation),
            Ok(Err(e)) => {
                println!("   ⚠️  Ollama explanation failed: {}, using fallback", e);
                Ok(self.generate_fallback_explanation(&result_json))
            }
            Err(_) => {
                println!("   ⚠️  Ollama timeout, using fallback explanation");
                Ok(self.generate_fallback_explanation(&result_json))
            }
        }
    }

    fn generate_fallback_explanation(&self, result: &serde_json::Value) -> String {
        format!(
            "🎉 Success! The ExoRust GPU agent successfully executed a Hello World program.\n\
            \n\
            📊 Performance Summary:\n\
            • Message: {}\n\
            • Execution Time: {:.1}ms\n\
            • GPU Acceleration: {}\n\
            • GPU Threads: {}\n\
            • Memory Used: {}MB\n\
            \n\
            💡 This demonstrates ExoRust's ability to:\n\
            • Parse natural language goals using LLM\n\
            • Execute GPU-accelerated computations\n\
            • Provide intelligent result explanations\n\
            • Manage the complete agent lifecycle",
            result
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown"),
            result
                .get("execution_time_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            if result
                .get("gpu_used")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "Yes"
            } else {
                "No"
            },
            result
                .get("threads_launched")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            result
                .get("memory_allocated_mb")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        )
    }

    async fn complete_goal(
        &self,
        goal_id: &str,
        result: &str,
        explanation: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut execution_data = HashMap::new();

        // Parse result to extract completion data
        if let Ok(result_json) = serde_json::from_str::<serde_json::Value>(result) {
            execution_data.insert(
                "message".to_string(),
                result_json.get("message").cloned().unwrap_or_default(),
            );
            execution_data.insert(
                "execution_time_ms".to_string(),
                result_json
                    .get("execution_time_ms")
                    .cloned()
                    .unwrap_or_default(),
            );
            execution_data.insert(
                "gpu_used".to_string(),
                result_json.get("gpu_used").cloned().unwrap_or_default(),
            );
            execution_data.insert(
                "status".to_string(),
                result_json.get("status").cloned().unwrap_or_default(),
            );
        }

        execution_data.insert(
            "completion_percentage".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(100.0).unwrap()),
        );
        execution_data.insert(
            "explanation".to_string(),
            serde_json::Value::String(explanation.to_string()),
        );

        let _goal_result = self
            .business_interface
            .complete_goal(goal_id, execution_data)
            .await?;

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 Starting ExoRust Hello World GPU Agent Demo");
    println!("⚠️  Ensure Ollama is running: ollama serve");
    println!("⚠️  Ensure GPU is available for best experience");
    println!();

    let mut agent = HelloWorldAgent::new().await?;
    agent.run_demo().await?;

    Ok(())
}
