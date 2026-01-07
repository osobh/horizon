//! Bootstrap script templates
//!
//! Generates platform-specific bootstrap scripts for installing
//! the HPC agent on remote nodes.

use crate::core::inventory::{Architecture, NodeMode, OsType};

/// Bootstrap script output
#[derive(Debug, Clone)]
pub struct BootstrapScript {
    /// Script content
    pub content: String,
    /// Script file extension
    pub extension: &'static str,
    /// Interpreter command
    pub interpreter: &'static str,
}

/// Script generator for different platforms
#[derive(Debug, Default)]
pub struct ScriptGenerator {
    /// Agent download base URL
    agent_url: String,
}

impl ScriptGenerator {
    /// Create a new script generator
    pub fn new() -> Self {
        Self {
            agent_url: "https://releases.hpc-ai.io/agent".to_string(),
        }
    }

    /// Create with custom agent URL
    pub fn with_agent_url(mut self, url: String) -> Self {
        self.agent_url = url;
        self
    }

    /// Generate bootstrap script for the given platform
    pub fn generate(&self, os: &OsType, arch: &Architecture, mode: &NodeMode) -> String {
        match os {
            OsType::Linux => self.generate_linux(arch, mode),
            OsType::Darwin => self.generate_darwin(arch, mode),
            OsType::Windows => self.generate_windows(arch, mode),
        }
    }

    /// Generate full bootstrap script struct
    pub fn generate_script(&self, os: &OsType, arch: &Architecture, mode: &NodeMode) -> BootstrapScript {
        match os {
            OsType::Linux | OsType::Darwin => BootstrapScript {
                content: self.generate(os, arch, mode),
                extension: "sh",
                interpreter: "/bin/bash",
            },
            OsType::Windows => BootstrapScript {
                content: self.generate(os, arch, mode),
                extension: "ps1",
                interpreter: "powershell.exe",
            },
        }
    }

    /// Generate Linux bootstrap script
    fn generate_linux(&self, arch: &Architecture, mode: &NodeMode) -> String {
        let arch_str = match arch {
            Architecture::Amd64 => "amd64",
            Architecture::Arm64 => "arm64",
            Architecture::Unknown => "amd64", // Default
        };

        let install_section = match mode {
            NodeMode::Docker => self.docker_install_linux(),
            NodeMode::Binary => self.binary_install_linux(arch_str),
        };

        format!(
            r#"#!/bin/bash
set -e

# HPC-AI Bootstrap Script for Linux ({arch})
# Generated automatically - do not edit

AGENT_URL="{agent_url}"
ARCH="{arch}"
MODE="{mode}"

echo "=== HPC-AI Node Bootstrap ==="
echo "Platform: Linux/{arch}"
echo "Mode: {mode}"
echo ""

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        echo "Error: This script requires root privileges"
        exit 1
    fi
else
    SUDO=""
fi

# Check dependencies
check_command() {{
    if ! command -v "$1" &> /dev/null; then
        echo "Installing $1..."
        if command -v apt-get &> /dev/null; then
            $SUDO apt-get update && $SUDO apt-get install -y "$1"
        elif command -v yum &> /dev/null; then
            $SUDO yum install -y "$1"
        elif command -v dnf &> /dev/null; then
            $SUDO dnf install -y "$1"
        elif command -v pacman &> /dev/null; then
            $SUDO pacman -Sy --noconfirm "$1"
        else
            echo "Warning: Could not install $1"
        fi
    fi
}}

echo "[1/4] Checking dependencies..."
check_command curl
check_command jq

{install_section}

echo ""
echo "=== Bootstrap Complete ==="
echo "Agent is now running"
"#,
            agent_url = self.agent_url,
            arch = arch_str,
            mode = mode,
            install_section = install_section
        )
    }

    /// Docker installation for Linux
    fn docker_install_linux(&self) -> String {
        r#"
echo "[2/4] Setting up Docker..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | $SUDO sh
    $SUDO systemctl enable docker
    $SUDO systemctl start docker
fi

# Ensure docker is running
if ! $SUDO docker info &> /dev/null; then
    $SUDO systemctl start docker
fi

echo "[3/4] Pulling agent container..."
$SUDO docker pull hpc-ai/swarmlet:latest

echo "[4/4] Starting agent container..."
$SUDO docker run -d \
    --name swarmlet \
    --restart unless-stopped \
    --network host \
    --privileged \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /etc/swarmlet:/etc/swarmlet \
    hpc-ai/swarmlet:latest
"#.to_string()
    }

    /// Binary installation for Linux
    fn binary_install_linux(&self, arch: &str) -> String {
        format!(
            r#"
echo "[2/4] Downloading agent binary..."
DOWNLOAD_URL="${{AGENT_URL}}/linux/{arch}/swarmlet"
curl -fsSL "$DOWNLOAD_URL" -o /tmp/swarmlet
chmod +x /tmp/swarmlet

echo "[3/4] Installing agent..."
$SUDO mkdir -p /opt/hpc-ai/bin
$SUDO mv /tmp/swarmlet /opt/hpc-ai/bin/swarmlet
$SUDO mkdir -p /etc/swarmlet

# Create systemd service
$SUDO tee /etc/systemd/system/swarmlet.service > /dev/null << 'EOF'
[Unit]
Description=HPC-AI Swarmlet Agent
After=network.target

[Service]
Type=simple
ExecStart=/opt/hpc-ai/bin/swarmlet
Restart=always
RestartSec=5
Environment=HPC_CONFIG_DIR=/etc/swarmlet

[Install]
WantedBy=multi-user.target
EOF

echo "[4/4] Starting agent service..."
$SUDO systemctl daemon-reload
$SUDO systemctl enable swarmlet
$SUDO systemctl start swarmlet
"#,
            arch = arch
        )
    }

    /// Generate macOS bootstrap script
    fn generate_darwin(&self, arch: &Architecture, mode: &NodeMode) -> String {
        let arch_str = match arch {
            Architecture::Amd64 => "amd64",
            Architecture::Arm64 => "arm64",
            Architecture::Unknown => "arm64", // Default to arm64 for modern Macs
        };

        let install_section = match mode {
            NodeMode::Docker => self.docker_install_darwin(),
            NodeMode::Binary => self.binary_install_darwin(arch_str),
        };

        format!(
            r#"#!/bin/bash
set -e

# HPC-AI Bootstrap Script for macOS ({arch})
# Generated automatically - do not edit

AGENT_URL="{agent_url}"
ARCH="{arch}"
MODE="{mode}"

echo "=== HPC-AI Node Bootstrap ==="
echo "Platform: Darwin/{arch}"
echo "Mode: {mode}"
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "[1/4] Checking dependencies..."
brew install curl jq 2>/dev/null || true

{install_section}

echo ""
echo "=== Bootstrap Complete ==="
echo "Agent is now running"
"#,
            agent_url = self.agent_url,
            arch = arch_str,
            mode = mode,
            install_section = install_section
        )
    }

    /// Docker installation for macOS
    fn docker_install_darwin(&self) -> String {
        r#"
echo "[2/4] Checking Docker Desktop..."
if ! command -v docker &> /dev/null; then
    echo "Docker Desktop is required for macOS."
    echo "Please install Docker Desktop from: https://docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "[3/4] Pulling agent container..."
docker pull hpc-ai/swarmlet:latest

echo "[4/4] Starting agent container..."
docker run -d \
    --name swarmlet \
    --restart unless-stopped \
    -p 9000:9000 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.hpc/swarmlet:/etc/swarmlet \
    hpc-ai/swarmlet:latest
"#.to_string()
    }

    /// Binary installation for macOS
    fn binary_install_darwin(&self, arch: &str) -> String {
        format!(
            r#"
echo "[2/4] Downloading agent binary..."
DOWNLOAD_URL="${{AGENT_URL}}/darwin/{arch}/swarmlet"
curl -fsSL "$DOWNLOAD_URL" -o /tmp/swarmlet
chmod +x /tmp/swarmlet

echo "[3/4] Installing agent..."
sudo mkdir -p /opt/hpc-ai/bin
sudo mv /tmp/swarmlet /opt/hpc-ai/bin/swarmlet
mkdir -p ~/.hpc/swarmlet

# Create launchd plist
sudo tee /Library/LaunchDaemons/io.hpc-ai.swarmlet.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.hpc-ai.swarmlet</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/hpc-ai/bin/swarmlet</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/swarmlet.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/swarmlet.error.log</string>
</dict>
</plist>
EOF

echo "[4/4] Starting agent service..."
sudo launchctl load /Library/LaunchDaemons/io.hpc-ai.swarmlet.plist
"#,
            arch = arch
        )
    }

    /// Generate Windows bootstrap script (PowerShell)
    fn generate_windows(&self, arch: &Architecture, mode: &NodeMode) -> String {
        let arch_str = match arch {
            Architecture::Amd64 => "amd64",
            Architecture::Arm64 => "arm64",
            Architecture::Unknown => "amd64",
        };

        let install_section = match mode {
            NodeMode::Docker => self.docker_install_windows(),
            NodeMode::Binary => self.binary_install_windows(arch_str),
        };

        format!(
            r#"# HPC-AI Bootstrap Script for Windows ({arch})
# Generated automatically - do not edit
# Run as Administrator

$ErrorActionPreference = "Stop"

$AGENT_URL = "{agent_url}"
$ARCH = "{arch}"
$MODE = "{mode}"

Write-Host "=== HPC-AI Node Bootstrap ===" -ForegroundColor Cyan
Write-Host "Platform: Windows/{arch}"
Write-Host "Mode: {mode}"
Write-Host ""

# Check for admin rights
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {{
    Write-Host "Error: This script requires Administrator privileges" -ForegroundColor Red
    exit 1
}}

Write-Host "[1/4] Checking dependencies..."

{install_section}

Write-Host ""
Write-Host "=== Bootstrap Complete ===" -ForegroundColor Green
Write-Host "Agent is now running"
"#,
            agent_url = self.agent_url,
            arch = arch_str,
            mode = mode,
            install_section = install_section
        )
    }

    /// Docker installation for Windows
    fn docker_install_windows(&self) -> String {
        r#"
Write-Host "[2/4] Checking Docker Desktop..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker Desktop is required for Windows."
    Write-Host "Please install Docker Desktop from: https://docker.com/products/docker-desktop"
    exit 1
}

try {
    docker info | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "[3/4] Pulling agent container..."
docker pull hpc-ai/swarmlet:latest

Write-Host "[4/4] Starting agent container..."
docker run -d `
    --name swarmlet `
    --restart unless-stopped `
    -p 9000:9000 `
    -v //var/run/docker.sock:/var/run/docker.sock `
    -v $env:USERPROFILE\.hpc\swarmlet:/etc/swarmlet `
    hpc-ai/swarmlet:latest
"#.to_string()
    }

    /// Binary installation for Windows
    fn binary_install_windows(&self, arch: &str) -> String {
        format!(
            r#"
Write-Host "[2/4] Downloading agent binary..."
$DOWNLOAD_URL = "$AGENT_URL/windows/{arch}/swarmlet.exe"
$InstallDir = "C:\Program Files\HPC-AI"
$ConfigDir = "C:\ProgramData\HPC-AI\swarmlet"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

Invoke-WebRequest -Uri $DOWNLOAD_URL -OutFile "$InstallDir\swarmlet.exe"

Write-Host "[3/4] Installing agent as Windows service..."
# Create Windows service using sc.exe
$serviceName = "HPC-AI-Swarmlet"
$binaryPath = "$InstallDir\swarmlet.exe"

# Stop and remove existing service if present
if (Get-Service -Name $serviceName -ErrorAction SilentlyContinue) {{
    Stop-Service -Name $serviceName -Force
    sc.exe delete $serviceName
    Start-Sleep -Seconds 2
}}

# Create new service
sc.exe create $serviceName binPath= "$binaryPath" start= auto DisplayName= "HPC-AI Swarmlet Agent"
sc.exe description $serviceName "HPC-AI distributed computing agent"

Write-Host "[4/4] Starting agent service..."
Start-Service -Name $serviceName

# Verify service is running
$service = Get-Service -Name $serviceName
if ($service.Status -eq "Running") {{
    Write-Host "Service started successfully" -ForegroundColor Green
}} else {{
    Write-Host "Warning: Service may not have started correctly" -ForegroundColor Yellow
}}
"#,
            arch = arch
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linux_amd64_docker_script() {
        let gen = ScriptGenerator::new();
        let script = gen.generate(&OsType::Linux, &Architecture::Amd64, &NodeMode::Docker);

        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("Linux/amd64"));
        assert!(script.contains("docker"));
    }

    #[test]
    fn test_linux_arm64_binary_script() {
        let gen = ScriptGenerator::new();
        let script = gen.generate(&OsType::Linux, &Architecture::Arm64, &NodeMode::Binary);

        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("Linux/arm64"));
        assert!(script.contains("systemd"));
    }

    #[test]
    fn test_darwin_arm64_docker_script() {
        let gen = ScriptGenerator::new();
        let script = gen.generate(&OsType::Darwin, &Architecture::Arm64, &NodeMode::Docker);

        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("Darwin/arm64"));
        assert!(script.contains("Docker Desktop"));
    }

    #[test]
    fn test_windows_amd64_binary_script() {
        let gen = ScriptGenerator::new();
        let script = gen.generate(&OsType::Windows, &Architecture::Amd64, &NodeMode::Binary);

        // PowerShell script markers
        assert!(script.contains("$ErrorActionPreference"));
        assert!(script.contains("Write-Host"));
        assert!(script.contains("Windows/amd64"));
        assert!(script.contains("Windows service"));
    }

    #[test]
    fn test_script_struct() {
        let gen = ScriptGenerator::new();

        let linux_script = gen.generate_script(&OsType::Linux, &Architecture::Amd64, &NodeMode::Docker);
        assert_eq!(linux_script.extension, "sh");
        assert_eq!(linux_script.interpreter, "/bin/bash");

        let windows_script = gen.generate_script(&OsType::Windows, &Architecture::Amd64, &NodeMode::Docker);
        assert_eq!(windows_script.extension, "ps1");
        assert_eq!(windows_script.interpreter, "powershell.exe");
    }

    #[test]
    fn test_custom_agent_url() {
        let gen = ScriptGenerator::new().with_agent_url("https://custom.example.com".to_string());
        let script = gen.generate(&OsType::Linux, &Architecture::Amd64, &NodeMode::Docker);

        assert!(script.contains("https://custom.example.com"));
    }
}
