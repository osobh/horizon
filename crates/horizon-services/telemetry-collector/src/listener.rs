use crate::config::ServerConfig as ServiceServerConfig;
use crate::handler::StreamHandler;
use anyhow::{Context, Result};
use quinn::{Endpoint, ServerConfig};
use rustls::ServerConfig as RustlsServerConfig;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct QuicListener {
    endpoint: Endpoint,
    handler: Arc<StreamHandler>,
    connection_limiter: Arc<Semaphore>,
}

impl QuicListener {
    pub async fn new(
        server_config: ServiceServerConfig,
        tls_cert_path: &Path,
        tls_key_path: &Path,
        handler: Arc<StreamHandler>,
    ) -> Result<Self> {
        // Load TLS certificates
        let (certs, key) = Self::load_tls_config(tls_cert_path, tls_key_path)?;

        // Create rustls config
        let mut rustls_config = RustlsServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .context("Failed to create TLS config")?;

        rustls_config.alpn_protocols = vec![b"hq-29".to_vec()];

        // Create quinn server config
        let quinn_server_config = ServerConfig::with_crypto(Arc::new(rustls_config));

        // Parse listen address
        let addr: SocketAddr = server_config
            .listen_addr
            .parse()
            .context("Failed to parse listen address")?;

        let max_connections = server_config.max_connections;

        // Create endpoint
        let endpoint = Endpoint::server(quinn_server_config, addr)
            .context("Failed to create QUIC endpoint")?;

        let connection_limiter = Arc::new(Semaphore::new(max_connections as usize));

        Ok(Self {
            endpoint,
            handler,
            connection_limiter,
        })
    }

    fn load_tls_config(
        cert_path: &Path,
        key_path: &Path,
    ) -> Result<(Vec<rustls::Certificate>, rustls::PrivateKey)> {
        use rustls_pemfile::{certs, pkcs8_private_keys};
        use std::fs::File;
        use std::io::BufReader;

        // Load certificates
        let cert_file = File::open(cert_path).context("Failed to open cert file")?;
        let mut cert_reader = BufReader::new(cert_file);
        let cert_ders = certs(&mut cert_reader).context("Failed to parse certificates")?;
        let certs: Vec<rustls::Certificate> =
            cert_ders.into_iter().map(rustls::Certificate).collect();

        // Load private key
        let key_file = File::open(key_path).context("Failed to open key file")?;
        let mut key_reader = BufReader::new(key_file);
        let key_ders =
            pkcs8_private_keys(&mut key_reader).context("Failed to parse private key")?;

        if key_ders.is_empty() {
            anyhow::bail!("No private key found");
        }

        let key = rustls::PrivateKey(key_ders.into_iter().next().unwrap());

        Ok((certs, key))
    }

    pub async fn serve(self) -> Result<()> {
        tracing::info!("QUIC listener started on {}", self.endpoint.local_addr()?);

        while let Some(conn) = self.endpoint.accept().await {
            let handler = self.handler.clone();
            let limiter = self.connection_limiter.clone();

            tokio::spawn(async move {
                // Acquire connection permit
                let _permit = limiter.acquire().await.ok();

                match conn.await {
                    Ok(connection) => {
                        if let Err(e) = Self::handle_connection(connection, handler).await {
                            tracing::error!("Connection error: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Connection failed: {}", e);
                    }
                }
            });
        }

        Ok(())
    }

    async fn handle_connection(conn: quinn::Connection, handler: Arc<StreamHandler>) -> Result<()> {
        tracing::debug!("New connection from {}", conn.remote_address());

        loop {
            match conn.accept_bi().await {
                Ok((_send, mut recv)) => {
                    let handler = handler.clone();
                    tokio::spawn(async move {
                        // Read data from stream
                        let mut buf = Vec::new();
                        while let Ok(Some(chunk)) = recv.read_chunk(1024, true).await {
                            buf.extend_from_slice(&chunk.bytes);
                        }

                        // Decode and handle batch
                        match StreamHandler::decode_length_prefixed(&buf) {
                            Ok(batch) => {
                                if let Err(e) = handler.handle_batch(batch).await {
                                    tracing::error!("Failed to handle batch: {}", e);
                                }
                            }
                            Err(e) => {
                                tracing::error!("Failed to decode batch: {}", e);
                            }
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                    tracing::debug!("Connection closed");
                    break;
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}
