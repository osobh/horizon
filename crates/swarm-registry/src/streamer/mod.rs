//! ImageStreamer - Progressive loading for instant container starts

use crate::{Result, SwarmImage, SwarmRegistryError};
use bytes::Bytes;
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::AsyncRead;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info};

/// Stream chunk size (64KB)
const CHUNK_SIZE: usize = 64 * 1024;

/// Prefetch window size (number of chunks to prefetch)
const PREFETCH_WINDOW: usize = 4;

/// Layer priority for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayerPriority {
    /// Critical layers needed for container start
    Critical = 0,
    /// High priority layers (runtime dependencies)
    High = 1,
    /// Normal priority layers
    Normal = 2,
    /// Low priority layers (optional components)
    Low = 3,
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chunk size for streaming
    pub chunk_size: usize,
    /// Number of chunks to prefetch
    pub prefetch_window: usize,
    /// Enable parallel layer streaming
    pub parallel_layers: bool,
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
    /// Buffer size for each stream
    pub buffer_size: usize,
    /// Enable compression
    pub enable_compression: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: CHUNK_SIZE,
            prefetch_window: PREFETCH_WINDOW,
            parallel_layers: true,
            max_concurrent_streams: 4,
            buffer_size: 1024 * 1024, // 1MB buffer
            enable_compression: true,
        }
    }
}

/// Stream statistics
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// Bytes streamed
    pub bytes_streamed: u64,
    /// Chunks streamed
    pub chunks_streamed: u64,
    /// Layers completed
    pub layers_completed: usize,
    /// Current throughput (bytes/sec)
    pub throughput: f64,
    /// Stream start time
    pub start_time: Option<std::time::Instant>,
}

/// Layer streaming state
#[derive(Debug)]
struct LayerStream {
    /// Layer hash
    layer_hash: String,
    /// Layer priority
    priority: LayerPriority,
    /// Total size
    total_size: u64,
    /// Bytes streamed
    bytes_streamed: u64,
    /// Chunk buffer
    buffer: VecDeque<Bytes>,
    /// Completion status
    completed: bool,
}

/// Image streamer for progressive loading
pub struct ImageStreamer {
    config: StreamConfig,
    /// Active streams
    active_streams: Arc<RwLock<HashMap<String, StreamHandle>>>,
    /// Stream statistics
    stats: Arc<RwLock<HashMap<String, StreamStats>>>,
}

/// Handle to an active stream
pub struct StreamHandle {
    /// Image being streamed
    image: SwarmImage,
    /// Layer streams
    layers: Vec<LayerStream>,
    /// Receiver for streamed chunks
    receiver: mpsc::Receiver<StreamChunk>,
    /// Control channel
    control_tx: mpsc::Sender<StreamControl>,
}

/// Streamed chunk of data
#[derive(Debug)]
pub struct StreamChunk {
    /// Layer this chunk belongs to
    pub layer_hash: String,
    /// Chunk data
    pub data: Bytes,
    /// Offset in the layer
    pub offset: u64,
    /// Whether this is the last chunk of the layer
    pub is_last: bool,
}

/// Stream control messages
#[derive(Debug)]
enum StreamControl {
    /// Pause streaming
    Pause,
    /// Resume streaming
    Resume,
    /// Change priority of a layer
    SetPriority(String, LayerPriority),
    /// Cancel streaming
    Cancel,
}

impl ImageStreamer {
    /// Create a new image streamer
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start streaming an image
    pub async fn stream_image(
        &self,
        image: SwarmImage,
        priorities: HashMap<String, LayerPriority>,
    ) -> Result<StreamHandle> {
        info!("Starting progressive stream for image {}", image.hash);

        // Create stream components
        let (chunk_tx, chunk_rx) = mpsc::channel(self.config.buffer_size);
        let (control_tx, control_rx) = mpsc::channel(10);

        // Initialize layer streams
        let mut layers = Vec::new();
        for layer_hash in &image.layers {
            let priority = priorities
                .get(layer_hash)
                .copied()
                .unwrap_or(LayerPriority::Normal);

            layers.push(LayerStream {
                layer_hash: layer_hash.clone(),
                priority,
                total_size: 0, // Will be set when streaming starts
                bytes_streamed: 0,
                buffer: VecDeque::new(),
                completed: false,
            });
        }

        // Sort layers by priority
        layers.sort_by_key(|l| l.priority);

        // Initialize stats
        {
            let mut stats = self.stats.write().await;
            stats.insert(
                image.hash.clone(),
                StreamStats {
                    start_time: Some(std::time::Instant::now()),
                    ..Default::default()
                },
            );
        }

        // Clone what we need for the task
        let image_for_task = image.clone();
        let priorities_for_task = priorities.clone();
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);

        // Start streaming task
        tokio::spawn(async move {
            if let Err(e) = Self::stream_task(
                image_for_task,
                priorities_for_task,
                chunk_tx,
                control_rx,
                config,
                stats,
            )
            .await
            {
                error!("Stream task failed: {}", e);
            }
        });

        // Create the handle to return
        let handle = StreamHandle {
            image: image.clone(),
            layers,
            receiver: chunk_rx,
            control_tx: control_tx.clone(),
        };

        // Register active stream (we need to clone for storage)
        {
            let mut streams = self.active_streams.write().await;
            let store_handle = StreamHandle {
                image: image.clone(),
                layers: Vec::new(),
                receiver: mpsc::channel(1).1, // Dummy receiver for storage
                control_tx: control_tx.clone(),
            };
            streams.insert(image.hash.clone(), store_handle);
        }

        Ok(handle)
    }

    /// Get streaming progress for an image
    pub async fn get_progress(&self, image_hash: &str) -> Option<f32> {
        let stats = self.stats.read().await;
        if let Some(stat) = stats.get(image_hash) {
            let streams = self.active_streams.read().await;
            if let Some(handle) = streams.get(image_hash) {
                let total_size: u64 = handle.layers.iter().map(|l| l.total_size).sum();

                if total_size > 0 {
                    return Some(stat.bytes_streamed as f32 / total_size as f32);
                }
            }
        }
        None
    }

    /// Get stream statistics
    pub async fn get_stats(&self, image_hash: &str) -> Option<StreamStats> {
        let stats = self.stats.read().await;
        stats.get(image_hash).cloned()
    }

    /// Cancel an active stream
    pub async fn cancel_stream(&self, image_hash: &str) -> Result<()> {
        let mut streams = self.active_streams.write().await;
        if let Some(handle) = streams.remove(image_hash) {
            handle
                .control_tx
                .send(StreamControl::Cancel)
                .await
                .map_err(|_| {
                    SwarmRegistryError::StreamingError("Failed to send cancel".to_string())
                })?;

            info!("Cancelled stream for image {}", image_hash);
            Ok(())
        } else {
            Err(SwarmRegistryError::StreamingError(
                "No active stream found".to_string(),
            ))
        }
    }

    async fn stream_task(
        image: SwarmImage,
        priorities: HashMap<String, LayerPriority>,
        chunk_tx: mpsc::Sender<StreamChunk>,
        mut control_rx: mpsc::Receiver<StreamControl>,
        config: StreamConfig,
        stats: Arc<RwLock<HashMap<String, StreamStats>>>,
    ) -> Result<()> {
        info!("Stream task started for image {}", image.hash);

        let mut paused = false;
        let layer_priorities = priorities;

        // Stream layers in priority order
        let mut sorted_layers: Vec<_> = image
            .layers
            .iter()
            .map(|hash| {
                let priority = layer_priorities
                    .get(hash)
                    .copied()
                    .unwrap_or(LayerPriority::Normal);
                (hash.clone(), priority)
            })
            .collect();

        sorted_layers.sort_by_key(|(_, p)| *p);

        for (layer_hash, _priority) in sorted_layers {
            if config.parallel_layers {
                // TODO: Implement parallel streaming
            } else {
                // Sequential streaming
                Self::stream_layer(
                    &layer_hash,
                    &chunk_tx,
                    &mut control_rx,
                    &config,
                    &stats,
                    &image.hash,
                    &mut paused,
                )
                .await?;
            }
        }

        info!("Stream completed for image {}", image.hash);
        Ok(())
    }

    async fn stream_layer(
        layer_hash: &str,
        chunk_tx: &mpsc::Sender<StreamChunk>,
        control_rx: &mut mpsc::Receiver<StreamControl>,
        config: &StreamConfig,
        stats: &Arc<RwLock<HashMap<String, StreamStats>>>,
        image_hash: &str,
        paused: &mut bool,
    ) -> Result<()> {
        debug!("Streaming layer {}", layer_hash);

        // Simulate reading layer data (in reality, would read from storage)
        let layer_data = Self::read_layer_data(layer_hash).await?;
        let total_size = layer_data.len();

        // Stream in chunks
        let mut offset = 0u64;
        let mut chunks_sent = 0u64;

        for chunk in layer_data.chunks(config.chunk_size) {
            // Check for control messages
            while let Ok(control) = control_rx.try_recv() {
                match control {
                    StreamControl::Pause => *paused = true,
                    StreamControl::Resume => *paused = false,
                    StreamControl::Cancel => {
                        return Err(SwarmRegistryError::StreamingError(
                            "Stream cancelled".to_string(),
                        ));
                    }
                    StreamControl::SetPriority(_, _) => {
                        // Priority changes handled at task level
                    }
                }
            }

            // Wait if paused
            while *paused {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Check for resume or cancel
                if let Ok(control) = control_rx.try_recv() {
                    match control {
                        StreamControl::Resume => *paused = false,
                        StreamControl::Cancel => {
                            return Err(SwarmRegistryError::StreamingError(
                                "Stream cancelled".to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
            }

            let is_last = offset + chunk.len() as u64 >= total_size as u64;

            let stream_chunk = StreamChunk {
                layer_hash: layer_hash.to_string(),
                data: Bytes::copy_from_slice(chunk),
                offset,
                is_last,
            };

            // Send chunk
            chunk_tx.send(stream_chunk).await.map_err(|_| {
                SwarmRegistryError::StreamingError("Failed to send chunk".to_string())
            })?;

            offset += chunk.len() as u64;
            chunks_sent += 1;

            // Update stats
            {
                let mut stats_map = stats.write().await;
                if let Some(stat) = stats_map.get_mut(image_hash) {
                    stat.bytes_streamed += chunk.len() as u64;
                    stat.chunks_streamed = chunks_sent;

                    if let Some(start_time) = stat.start_time {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        if elapsed > 0.0 {
                            stat.throughput = stat.bytes_streamed as f64 / elapsed;
                        }
                    }
                }
            }
        }

        // Update layer completion
        {
            let mut stats_map = stats.write().await;
            if let Some(stat) = stats_map.get_mut(image_hash) {
                stat.layers_completed += 1;
            }
        }

        Ok(())
    }

    async fn read_layer_data(_layer_hash: &str) -> Result<Vec<u8>> {
        // Simulate reading layer data
        // In reality, this would read from ContentAddressableStore
        Ok(vec![0u8; 1024 * 1024]) // 1MB dummy data
    }
}

impl StreamHandle {
    /// Receive the next chunk
    pub async fn recv(&mut self) -> Option<StreamChunk> {
        self.receiver.recv().await
    }

    /// Pause streaming
    pub async fn pause(&self) -> Result<()> {
        self.control_tx
            .send(StreamControl::Pause)
            .await
            .map_err(|_| SwarmRegistryError::StreamingError("Failed to send pause".to_string()))
    }

    /// Resume streaming
    pub async fn resume(&self) -> Result<()> {
        self.control_tx
            .send(StreamControl::Resume)
            .await
            .map_err(|_| SwarmRegistryError::StreamingError("Failed to send resume".to_string()))
    }

    /// Change layer priority
    pub async fn set_priority(&self, layer_hash: String, priority: LayerPriority) -> Result<()> {
        self.control_tx
            .send(StreamControl::SetPriority(layer_hash, priority))
            .await
            .map_err(|_| {
                SwarmRegistryError::StreamingError("Failed to send priority update".to_string())
            })
    }

    /// Cancel streaming
    pub async fn cancel(&self) -> Result<()> {
        self.control_tx
            .send(StreamControl::Cancel)
            .await
            .map_err(|_| SwarmRegistryError::StreamingError("Failed to send cancel".to_string()))
    }
}

/// Create a stream adapter for async reading
pub struct StreamReader {
    receiver: mpsc::Receiver<StreamChunk>,
    current_chunk: Option<Bytes>,
    position: usize,
}

impl StreamReader {
    /// Create a new stream reader from a handle
    pub fn new(handle: StreamHandle) -> Self {
        Self {
            receiver: handle.receiver,
            current_chunk: None,
            position: 0,
        }
    }
}

impl AsyncRead for StreamReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        loop {
            // If we have a current chunk, read from it
            if let Some(chunk) = &self.current_chunk {
                let remaining = chunk.len() - self.position;
                if remaining > 0 {
                    let to_read = remaining.min(buf.remaining());
                    buf.put_slice(&chunk[self.position..self.position + to_read]);
                    self.position += to_read;
                    return Poll::Ready(Ok(()));
                }
            }

            // Need new chunk
            match self.receiver.poll_recv(cx) {
                Poll::Ready(Some(chunk)) => {
                    self.current_chunk = Some(chunk.data);
                    self.position = 0;
                    // Continue loop to read from new chunk
                }
                Poll::Ready(None) => {
                    // Stream ended
                    return Poll::Ready(Ok(()));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image() -> SwarmImage {
        SwarmImage {
            hash: "sha256:test123".to_string(),
            metadata: crate::ImageMetadata {
                name: "test".to_string(),
                tag: "latest".to_string(),
                variant: crate::ImageVariant::Base,
                created: 12345,
                size: 3 * 1024 * 1024, // 3MB
                architecture: "amd64".to_string(),
                os: "linux".to_string(),
                env: vec![],
                entrypoint: None,
                cmd: None,
            },
            layers: vec![
                "layer1".to_string(),
                "layer2".to_string(),
                "layer3".to_string(),
            ],
            agent_config: None,
        }
    }

    #[tokio::test]
    async fn test_streamer_creation() {
        let streamer = ImageStreamer::new(StreamConfig::default());

        let streams = streamer.active_streams.read().await;
        assert!(streams.is_empty());
    }

    #[tokio::test]
    async fn test_stream_image() {
        let streamer = ImageStreamer::new(StreamConfig::default());
        let image = create_test_image();

        let mut priorities = HashMap::new();
        priorities.insert("layer1".to_string(), LayerPriority::Critical);
        priorities.insert("layer2".to_string(), LayerPriority::High);
        priorities.insert("layer3".to_string(), LayerPriority::Normal);

        let mut handle = streamer
            .stream_image(image.clone(), priorities)
            .await
            .unwrap();

        // Should receive chunks
        let chunk = handle.recv().await;
        assert!(chunk.is_some());
    }

    #[tokio::test]
    async fn test_stream_progress() {
        let streamer = ImageStreamer::new(StreamConfig::default());
        let image = create_test_image();

        let priorities = HashMap::new();
        let _handle = streamer
            .stream_image(image.clone(), priorities)
            .await
            .unwrap();

        // Give stream time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let progress = streamer.get_progress(&image.hash).await;
        assert!(progress.is_some());
    }

    #[tokio::test]
    async fn test_stream_stats() {
        let streamer = ImageStreamer::new(StreamConfig::default());
        let image = create_test_image();

        let priorities = HashMap::new();
        let _handle = streamer
            .stream_image(image.clone(), priorities)
            .await
            .unwrap();

        // Give stream time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let stats = streamer.get_stats(&image.hash).await;
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert!(stats.start_time.is_some());
    }

    #[tokio::test]
    async fn test_pause_resume() {
        let streamer = ImageStreamer::new(StreamConfig::default());
        let image = create_test_image();

        let priorities = HashMap::new();
        let handle = streamer
            .stream_image(image.clone(), priorities)
            .await
            .unwrap();

        // Test pause
        handle.pause().await.unwrap();

        // Test resume
        handle.resume().await.unwrap();
    }

    #[tokio::test]
    async fn test_cancel_stream() {
        let streamer = ImageStreamer::new(StreamConfig::default());
        let image = create_test_image();

        let priorities = HashMap::new();
        let _handle = streamer
            .stream_image(image.clone(), priorities)
            .await
            .unwrap();

        // Cancel stream
        streamer.cancel_stream(&image.hash).await.unwrap();

        // Should no longer be in active streams
        let streams = streamer.active_streams.read().await;
        assert!(!streams.contains_key(&image.hash));
    }

    #[tokio::test]
    async fn test_layer_priority_ordering() {
        let mut layers = vec![
            LayerStream {
                layer_hash: "layer1".to_string(),
                priority: LayerPriority::Normal,
                total_size: 1024,
                bytes_streamed: 0,
                buffer: VecDeque::new(),
                completed: false,
            },
            LayerStream {
                layer_hash: "layer2".to_string(),
                priority: LayerPriority::Critical,
                total_size: 1024,
                bytes_streamed: 0,
                buffer: VecDeque::new(),
                completed: false,
            },
            LayerStream {
                layer_hash: "layer3".to_string(),
                priority: LayerPriority::High,
                total_size: 1024,
                bytes_streamed: 0,
                buffer: VecDeque::new(),
                completed: false,
            },
        ];

        layers.sort_by_key(|l| l.priority);

        // Should be ordered: Critical, High, Normal
        assert_eq!(layers[0].priority, LayerPriority::Critical);
        assert_eq!(layers[1].priority, LayerPriority::High);
        assert_eq!(layers[2].priority, LayerPriority::Normal);
    }

    #[tokio::test]
    async fn test_stream_reader() {
        use tokio::io::AsyncReadExt;

        let streamer = ImageStreamer::new(StreamConfig::default());
        let image = create_test_image();

        let priorities = HashMap::new();
        let handle = streamer
            .stream_image(image.clone(), priorities)
            .await
            .unwrap();

        let mut reader = StreamReader::new(handle);
        let mut buffer = vec![0u8; 1024];

        // Should be able to read data
        let n = reader.read(&mut buffer).await.unwrap();
        assert!(n > 0);
    }

    #[test]
    fn test_chunk_size_calculation() {
        let config = StreamConfig::default();
        assert_eq!(config.chunk_size, CHUNK_SIZE);
        assert_eq!(config.prefetch_window, PREFETCH_WINDOW);
    }
}
