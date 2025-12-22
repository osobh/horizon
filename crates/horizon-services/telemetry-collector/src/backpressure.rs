/// Backpressure control using bounded channels
///
/// The backpressure mechanism is implemented using:
/// 1. Tokio bounded channels for write queues
/// 2. Semaphore for connection limits (in listener.rs)
/// 3. Queue depth monitoring
///
/// When backpressure threshold is exceeded, new data is rejected.
use tokio::sync::mpsc;

pub type BackpressureChannel<T> = mpsc::Sender<T>;
pub type BackpressureReceiver<T> = mpsc::Receiver<T>;

/// Create a backpressure channel with given capacity
pub fn create_channel<T>(capacity: usize) -> (BackpressureChannel<T>, BackpressureReceiver<T>) {
    mpsc::channel(capacity)
}
