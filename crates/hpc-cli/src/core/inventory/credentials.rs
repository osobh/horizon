//! Credential management
//!
//! Handles SSH key storage and password management.

use anyhow::{Context, Result};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use super::store::InventoryStore;
use hpc_inventory::CredentialRef;

/// Credential store for managing SSH keys and passwords
pub struct CredentialStore {
    /// Base directory for keys (~/.hpc/inventory/keys/)
    keys_dir: PathBuf,
}

impl CredentialStore {
    /// Create a new credential store
    pub fn new() -> Result<Self> {
        let keys_dir = InventoryStore::keys_dir()?;
        fs::create_dir_all(&keys_dir)
            .with_context(|| format!("Failed to create keys directory: {:?}", keys_dir))?;

        // Set restrictive permissions on keys directory
        #[cfg(unix)]
        {
            let perms = fs::Permissions::from_mode(0o700);
            fs::set_permissions(&keys_dir, perms)?;
        }

        Ok(Self { keys_dir })
    }

    /// Get the path for a node's private key
    pub fn key_path(&self, node_id: &str) -> PathBuf {
        self.keys_dir.join(node_id)
    }

    /// Get the path for a node's public key
    pub fn pub_key_path(&self, node_id: &str) -> PathBuf {
        self.keys_dir.join(format!("{}.pub", node_id))
    }

    /// Generate a new SSH key pair for a node
    pub fn generate_ssh_key(&self, node_id: &str) -> Result<CredentialRef> {
        use russh_keys::key::KeyPair;

        let private_path = self.key_path(node_id);
        let public_path = self.pub_key_path(node_id);

        // Check if key already exists
        if private_path.exists() {
            anyhow::bail!(
                "SSH key already exists for node {}. Use --force to overwrite.",
                node_id
            );
        }

        // Generate Ed25519 key pair
        let keypair = KeyPair::generate_ed25519().context("Failed to generate SSH key")?;

        // Write private key using PKCS8 PEM format
        let mut private_key_content = Vec::new();
        russh_keys::encode_pkcs8_pem(&keypair, &mut private_key_content)
            .context("Failed to encode private key")?;

        fs::write(&private_path, &private_key_content)
            .with_context(|| format!("Failed to write private key to {:?}", private_path))?;

        // Set restrictive permissions on private key
        #[cfg(unix)]
        {
            let perms = fs::Permissions::from_mode(0o600);
            fs::set_permissions(&private_path, perms)?;
        }

        // Write public key
        let public_key = keypair.clone_public_key().context("Failed to get public key")?;
        let public_key_str = format!(
            "ssh-ed25519 {} hpc-{}",
            russh_keys::PublicKeyBase64::public_key_base64(&public_key),
            node_id
        );

        fs::write(&public_path, public_key_str)
            .with_context(|| format!("Failed to write public key to {:?}", public_path))?;

        Ok(CredentialRef::SshKey {
            path: private_path,
        })
    }

    /// Import an existing SSH key for a node
    pub fn import_ssh_key(&self, node_id: &str, source_path: &str) -> Result<CredentialRef> {
        let source = PathBuf::from(source_path);
        if !source.exists() {
            anyhow::bail!("SSH key file not found: {}", source_path);
        }

        // Validate the key can be read
        let key_content = fs::read_to_string(&source)
            .with_context(|| format!("Failed to read SSH key: {}", source_path))?;

        // Try to parse the key to validate it
        russh_keys::decode_secret_key(&key_content, None)
            .with_context(|| format!("Invalid SSH private key: {}", source_path))?;

        let dest_path = self.key_path(node_id);

        // Copy the key file
        fs::copy(&source, &dest_path)
            .with_context(|| format!("Failed to copy key to {:?}", dest_path))?;

        // Set restrictive permissions
        #[cfg(unix)]
        {
            let perms = fs::Permissions::from_mode(0o600);
            fs::set_permissions(&dest_path, perms)?;
        }

        // Try to copy public key if it exists
        let source_pub = PathBuf::from(format!("{}.pub", source_path));
        if source_pub.exists() {
            let dest_pub = self.pub_key_path(node_id);
            fs::copy(&source_pub, &dest_pub).ok(); // Ignore errors for public key
        }

        Ok(CredentialRef::SshKey { path: dest_path })
    }

    /// Store a password securely using the system keyring
    pub fn store_password(&self, node_id: &str, password: &str) -> Result<CredentialRef> {
        let key_id = format!("hpc-inventory-{}", node_id);

        let entry = keyring::Entry::new("hpc-cli", &key_id)
            .context("Failed to create keyring entry")?;

        entry
            .set_password(password)
            .context("Failed to store password in keyring")?;

        Ok(CredentialRef::Password { key_id })
    }

    /// Retrieve a password from the keyring
    pub fn get_password(&self, key_id: &str) -> Result<String> {
        let entry = keyring::Entry::new("hpc-cli", key_id)
            .context("Failed to access keyring entry")?;

        entry
            .get_password()
            .context("Failed to retrieve password from keyring")
    }

    /// Load an SSH key from a credential reference
    pub fn load_ssh_key(&self, cred_ref: &CredentialRef) -> Result<russh_keys::key::KeyPair> {
        match cred_ref {
            CredentialRef::SshKey { path } => {
                let key_content = fs::read_to_string(path)
                    .with_context(|| format!("Failed to read SSH key: {:?}", path))?;

                russh_keys::decode_secret_key(&key_content, None)
                    .with_context(|| format!("Failed to decode SSH key: {:?}", path))
            }
            CredentialRef::Password { .. } => {
                anyhow::bail!("Cannot load SSH key from password credential")
            }
            CredentialRef::SshAgent => {
                anyhow::bail!("SSH agent keys must be retrieved via SSH agent protocol")
            }
        }
    }

    /// Delete credentials for a node
    pub fn delete_credentials(&self, node_id: &str) -> Result<()> {
        let private_path = self.key_path(node_id);
        let public_path = self.pub_key_path(node_id);

        // Remove key files if they exist
        if private_path.exists() {
            fs::remove_file(&private_path)
                .with_context(|| format!("Failed to delete private key: {:?}", private_path))?;
        }

        if public_path.exists() {
            fs::remove_file(&public_path)
                .with_context(|| format!("Failed to delete public key: {:?}", public_path))?;
        }

        // Try to remove password from keyring
        let key_id = format!("hpc-inventory-{}", node_id);
        if let Ok(entry) = keyring::Entry::new("hpc-cli", &key_id) {
            let _ = entry.delete_password(); // Ignore errors
        }

        Ok(())
    }

    /// List all stored keys
    pub fn list_keys(&self) -> Result<Vec<KeyInfo>> {
        let mut keys = Vec::new();

        for entry in fs::read_dir(&self.keys_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip public keys and non-files
            if path.extension().is_some() || !path.is_file() {
                continue;
            }

            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let metadata = fs::metadata(&path)?;
            let has_public = self.pub_key_path(&name).exists();

            keys.push(KeyInfo {
                name,
                path: path.clone(),
                has_public_key: has_public,
                size_bytes: metadata.len(),
            });
        }

        keys.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(keys)
    }

    /// Get the public key content for a node
    pub fn get_public_key(&self, node_id: &str) -> Result<String> {
        let pub_path = self.pub_key_path(node_id);

        fs::read_to_string(&pub_path)
            .with_context(|| format!("Failed to read public key: {:?}", pub_path))
    }

    /// Check if a key exists for a node
    pub fn key_exists(&self, node_id: &str) -> bool {
        self.key_path(node_id).exists()
    }
}

/// Information about a stored SSH key
#[derive(Debug, Clone)]
pub struct KeyInfo {
    /// Key name/ID
    pub name: String,
    /// Path to private key
    pub path: PathBuf,
    /// Whether public key file exists
    pub has_public_key: bool,
    /// Size of private key file
    pub size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store(temp_dir: &TempDir) -> CredentialStore {
        let keys_dir = temp_dir.path().join("keys");
        fs::create_dir_all(&keys_dir).unwrap();
        CredentialStore { keys_dir }
    }

    #[test]
    fn test_key_paths() {
        let temp_dir = TempDir::new().unwrap();
        let store = create_test_store(&temp_dir);

        let private = store.key_path("test-node");
        let public = store.pub_key_path("test-node");

        assert!(private.ends_with("test-node"));
        assert!(public.ends_with("test-node.pub"));
    }

    #[test]
    fn test_generate_ssh_key() {
        let temp_dir = TempDir::new().unwrap();
        let store = create_test_store(&temp_dir);

        let cred_ref = store.generate_ssh_key("test-node").unwrap();

        match cred_ref {
            CredentialRef::SshKey { path } => {
                assert!(path.exists());
                assert!(store.pub_key_path("test-node").exists());
            }
            _ => panic!("Expected SshKey credential"),
        }
    }

    #[test]
    fn test_key_exists() {
        let temp_dir = TempDir::new().unwrap();
        let store = create_test_store(&temp_dir);

        assert!(!store.key_exists("nonexistent"));

        store.generate_ssh_key("new-node").unwrap();
        assert!(store.key_exists("new-node"));
    }

    #[test]
    fn test_delete_credentials() {
        let temp_dir = TempDir::new().unwrap();
        let store = create_test_store(&temp_dir);

        store.generate_ssh_key("to-delete").unwrap();
        assert!(store.key_exists("to-delete"));

        store.delete_credentials("to-delete").unwrap();
        assert!(!store.key_exists("to-delete"));
    }

    #[test]
    fn test_list_keys() {
        let temp_dir = TempDir::new().unwrap();
        let store = create_test_store(&temp_dir);

        store.generate_ssh_key("key1").unwrap();
        store.generate_ssh_key("key2").unwrap();

        let keys = store.list_keys().unwrap();
        assert_eq!(keys.len(), 2);

        let names: Vec<_> = keys.iter().map(|k| k.name.as_str()).collect();
        assert!(names.contains(&"key1"));
        assert!(names.contains(&"key2"));
    }
}
