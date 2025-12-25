//! Branch-prediction-friendly priority queue implementation.
//!
//! This module provides a segmented VecDeque-based priority queue that offers
//! O(1) enqueue and dequeue operations with predictable branch patterns.
//!
//! # Mechanical Sympathy
//!
//! Traditional `BinaryHeap` implementations suffer from poor branch prediction
//! because each heap operation (push/pop) involves O(log n) comparisons with
//! unpredictable outcomes based on data-dependent tree traversals.
//!
//! This implementation uses a fixed number of VecDeque segments (one per priority
//! level). The dequeue operation always checks the same 4 queues in the same order,
//! making it highly predictable for the CPU's branch predictor.
//!
//! # Performance Characteristics
//!
//! - **Enqueue**: O(1) - direct push to the appropriate segment
//! - **Dequeue**: O(1) - check 4 segments in fixed order (predictable branches)
//! - **Peek**: O(1) - check 4 segments in fixed order
//! - **Remove by ID**: O(n) - linear scan (unchanged from BinaryHeap)
//!
//! # Example
//!
//! ```rust
//! use stratoswarm_core::priority_queue::{PrioritySchedulerQueue, SchedulerPriority};
//!
//! let mut queue: PrioritySchedulerQueue<String> = PrioritySchedulerQueue::new();
//!
//! queue.enqueue("urgent task".to_string(), SchedulerPriority::Critical);
//! queue.enqueue("normal task".to_string(), SchedulerPriority::Normal);
//!
//! // Critical priority items are dequeued first
//! assert_eq!(queue.dequeue(), Some("urgent task".to_string()));
//! assert_eq!(queue.dequeue(), Some("normal task".to_string()));
//! ```

use std::collections::VecDeque;

/// Priority levels for the scheduler queue.
///
/// Items are dequeued in priority order: Critical > High > Normal > Low
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SchedulerPriority {
    /// Lowest priority - background tasks
    Low = 0,
    /// Normal priority - default for most tasks
    Normal = 1,
    /// High priority - time-sensitive tasks
    High = 2,
    /// Critical priority - must be processed immediately
    Critical = 3,
}

impl Default for SchedulerPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl SchedulerPriority {
    /// Returns all priority levels in descending order (Critical first)
    #[inline]
    pub const fn all_descending() -> [Self; 4] {
        [Self::Critical, Self::High, Self::Normal, Self::Low]
    }
}

/// A branch-prediction-friendly priority queue using segmented VecDeques.
///
/// Unlike `BinaryHeap`, this implementation provides O(1) operations with
/// predictable branch patterns that the CPU can efficiently predict.
///
/// # Type Parameters
///
/// - `T`: The item type stored in the queue
#[derive(Debug, Clone)]
pub struct PrioritySchedulerQueue<T> {
    /// Critical priority items
    critical: VecDeque<T>,
    /// High priority items
    high: VecDeque<T>,
    /// Normal priority items
    normal: VecDeque<T>,
    /// Low priority items
    low: VecDeque<T>,
}

impl<T> Default for PrioritySchedulerQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PrioritySchedulerQueue<T> {
    /// Creates a new empty priority queue.
    #[inline]
    pub fn new() -> Self {
        Self {
            critical: VecDeque::new(),
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
        }
    }

    /// Creates a new priority queue with pre-allocated capacity for each segment.
    ///
    /// This can reduce allocations if you know approximately how many items
    /// will be in each priority level.
    #[inline]
    pub fn with_capacity(capacity_per_level: usize) -> Self {
        Self {
            critical: VecDeque::with_capacity(capacity_per_level),
            high: VecDeque::with_capacity(capacity_per_level),
            normal: VecDeque::with_capacity(capacity_per_level),
            low: VecDeque::with_capacity(capacity_per_level),
        }
    }

    /// Enqueues an item with the given priority in O(1) time.
    ///
    /// Items with the same priority are processed in FIFO order.
    #[inline]
    pub fn enqueue(&mut self, item: T, priority: SchedulerPriority) {
        match priority {
            SchedulerPriority::Critical => self.critical.push_back(item),
            SchedulerPriority::High => self.high.push_back(item),
            SchedulerPriority::Normal => self.normal.push_back(item),
            SchedulerPriority::Low => self.low.push_back(item),
        }
    }

    /// Enqueues an item at the front of its priority level.
    ///
    /// This is useful for re-queuing items that failed processing.
    #[inline]
    pub fn enqueue_front(&mut self, item: T, priority: SchedulerPriority) {
        match priority {
            SchedulerPriority::Critical => self.critical.push_front(item),
            SchedulerPriority::High => self.high.push_front(item),
            SchedulerPriority::Normal => self.normal.push_front(item),
            SchedulerPriority::Low => self.low.push_front(item),
        }
    }

    /// Dequeues the highest priority item in O(1) time.
    ///
    /// Returns `None` if the queue is empty.
    ///
    /// # Branch Prediction
    ///
    /// This method always checks priorities in the same order (Critical, High,
    /// Normal, Low), making it highly predictable for the CPU's branch predictor.
    #[inline]
    pub fn dequeue(&mut self) -> Option<T> {
        self.critical
            .pop_front()
            .or_else(|| self.high.pop_front())
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }

    /// Peeks at the highest priority item without removing it.
    ///
    /// Returns `None` if the queue is empty.
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        self.critical
            .front()
            .or_else(|| self.high.front())
            .or_else(|| self.normal.front())
            .or_else(|| self.low.front())
    }

    /// Peeks at the highest priority item mutably without removing it.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut T> {
        if let Some(item) = self.critical.front_mut() {
            return Some(item);
        }
        if let Some(item) = self.high.front_mut() {
            return Some(item);
        }
        if let Some(item) = self.normal.front_mut() {
            return Some(item);
        }
        self.low.front_mut()
    }

    /// Returns the total number of items in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.critical.len() + self.high.len() + self.normal.len() + self.low.len()
    }

    /// Returns `true` if the queue contains no items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.critical.is_empty()
            && self.high.is_empty()
            && self.normal.is_empty()
            && self.low.is_empty()
    }

    /// Returns the number of items at the given priority level.
    #[inline]
    pub fn len_at_priority(&self, priority: SchedulerPriority) -> usize {
        match priority {
            SchedulerPriority::Critical => self.critical.len(),
            SchedulerPriority::High => self.high.len(),
            SchedulerPriority::Normal => self.normal.len(),
            SchedulerPriority::Low => self.low.len(),
        }
    }

    /// Clears all items from the queue.
    #[inline]
    pub fn clear(&mut self) {
        self.critical.clear();
        self.high.clear();
        self.normal.clear();
        self.low.clear();
    }

    /// Returns an iterator over all items in priority order.
    ///
    /// Items are yielded in the order they would be dequeued.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.critical
            .iter()
            .chain(self.high.iter())
            .chain(self.normal.iter())
            .chain(self.low.iter())
    }

    /// Drains all items from the queue in priority order.
    pub fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.critical
            .drain(..)
            .chain(self.high.drain(..))
            .chain(self.normal.drain(..))
            .chain(self.low.drain(..))
    }

    /// Retains only the items specified by the predicate.
    ///
    /// Items for which the predicate returns `false` are removed.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.critical.retain(|item| f(item));
        self.high.retain(|item| f(item));
        self.normal.retain(|item| f(item));
        self.low.retain(|item| f(item));
    }
}

impl<T: PartialEq> PrioritySchedulerQueue<T> {
    /// Removes and returns the first item that matches the given value.
    ///
    /// This is an O(n) operation.
    pub fn remove(&mut self, item: &T) -> Option<T> {
        if let Some(pos) = self.critical.iter().position(|x| x == item) {
            return self.critical.remove(pos);
        }
        if let Some(pos) = self.high.iter().position(|x| x == item) {
            return self.high.remove(pos);
        }
        if let Some(pos) = self.normal.iter().position(|x| x == item) {
            return self.normal.remove(pos);
        }
        if let Some(pos) = self.low.iter().position(|x| x == item) {
            return self.low.remove(pos);
        }
        None
    }

    /// Returns `true` if the queue contains the given item.
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        self.critical.contains(item)
            || self.high.contains(item)
            || self.normal.contains(item)
            || self.low.contains(item)
    }
}

impl<T> PrioritySchedulerQueue<T> {
    /// Removes and returns the first item that matches the predicate.
    ///
    /// This is an O(n) operation.
    pub fn remove_by<F>(&mut self, mut predicate: F) -> Option<T>
    where
        F: FnMut(&T) -> bool,
    {
        if let Some(pos) = self.critical.iter().position(|x| predicate(x)) {
            return self.critical.remove(pos);
        }
        if let Some(pos) = self.high.iter().position(|x| predicate(x)) {
            return self.high.remove(pos);
        }
        if let Some(pos) = self.normal.iter().position(|x| predicate(x)) {
            return self.normal.remove(pos);
        }
        if let Some(pos) = self.low.iter().position(|x| predicate(x)) {
            return self.low.remove(pos);
        }
        None
    }

    /// Finds the first item that matches the predicate.
    pub fn find<F>(&self, mut predicate: F) -> Option<&T>
    where
        F: FnMut(&T) -> bool,
    {
        self.critical
            .iter()
            .find(|x| predicate(x))
            .or_else(|| self.high.iter().find(|x| predicate(x)))
            .or_else(|| self.normal.iter().find(|x| predicate(x)))
            .or_else(|| self.low.iter().find(|x| predicate(x)))
    }

    /// Returns `true` if any item matches the predicate.
    pub fn any<F>(&self, mut predicate: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        self.critical.iter().any(|x| predicate(x))
            || self.high.iter().any(|x| predicate(x))
            || self.normal.iter().any(|x| predicate(x))
            || self.low.iter().any(|x| predicate(x))
    }
}

impl<T> Extend<(T, SchedulerPriority)> for PrioritySchedulerQueue<T> {
    fn extend<I: IntoIterator<Item = (T, SchedulerPriority)>>(&mut self, iter: I) {
        for (item, priority) in iter {
            self.enqueue(item, priority);
        }
    }
}

impl<T> FromIterator<(T, SchedulerPriority)> for PrioritySchedulerQueue<T> {
    fn from_iter<I: IntoIterator<Item = (T, SchedulerPriority)>>(iter: I) -> Self {
        let mut queue = Self::new();
        queue.extend(iter);
        queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_enqueue_dequeue() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.enqueue(1, SchedulerPriority::Normal);
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        assert_eq!(queue.dequeue(), Some(1));
        assert!(queue.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let mut queue: PrioritySchedulerQueue<&str> = PrioritySchedulerQueue::new();

        queue.enqueue("low", SchedulerPriority::Low);
        queue.enqueue("high", SchedulerPriority::High);
        queue.enqueue("normal", SchedulerPriority::Normal);
        queue.enqueue("critical", SchedulerPriority::Critical);

        // Should dequeue in priority order
        assert_eq!(queue.dequeue(), Some("critical"));
        assert_eq!(queue.dequeue(), Some("high"));
        assert_eq!(queue.dequeue(), Some("normal"));
        assert_eq!(queue.dequeue(), Some("low"));
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Normal);
        queue.enqueue(2, SchedulerPriority::Normal);
        queue.enqueue(3, SchedulerPriority::Normal);

        // Should maintain FIFO order within same priority
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
    }

    #[test]
    fn test_peek() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        assert!(queue.peek().is_none());

        queue.enqueue(1, SchedulerPriority::Low);
        queue.enqueue(2, SchedulerPriority::High);

        // Peek should return high priority item
        assert_eq!(queue.peek(), Some(&2));
        // Peek should not remove
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_len_at_priority() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Critical);
        queue.enqueue(2, SchedulerPriority::Critical);
        queue.enqueue(3, SchedulerPriority::Normal);

        assert_eq!(queue.len_at_priority(SchedulerPriority::Critical), 2);
        assert_eq!(queue.len_at_priority(SchedulerPriority::High), 0);
        assert_eq!(queue.len_at_priority(SchedulerPriority::Normal), 1);
        assert_eq!(queue.len_at_priority(SchedulerPriority::Low), 0);
    }

    #[test]
    fn test_remove_by() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::High);
        queue.enqueue(2, SchedulerPriority::Normal);
        queue.enqueue(3, SchedulerPriority::Normal);

        let removed = queue.remove_by(|&x| x == 2);
        assert_eq!(removed, Some(2));
        assert_eq!(queue.len(), 2);

        // Remaining items should be in order
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(3));
    }

    #[test]
    fn test_retain() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Critical);
        queue.enqueue(2, SchedulerPriority::High);
        queue.enqueue(3, SchedulerPriority::Normal);
        queue.enqueue(4, SchedulerPriority::Low);

        // Keep only even numbers
        queue.retain(|&x| x % 2 == 0);

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(4));
    }

    #[test]
    fn test_enqueue_front() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Normal);
        queue.enqueue(2, SchedulerPriority::Normal);
        queue.enqueue_front(0, SchedulerPriority::Normal);

        // Front-enqueued item should come first
        assert_eq!(queue.dequeue(), Some(0));
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
    }

    #[test]
    fn test_iter() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Low);
        queue.enqueue(2, SchedulerPriority::Critical);
        queue.enqueue(3, SchedulerPriority::Normal);

        let items: Vec<_> = queue.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 1]); // Priority order
    }

    #[test]
    fn test_from_iterator() {
        let items = vec![
            (1, SchedulerPriority::Low),
            (2, SchedulerPriority::High),
            (3, SchedulerPriority::Normal),
        ];

        let mut queue: PrioritySchedulerQueue<i32> = items.into_iter().collect();

        assert_eq!(queue.dequeue(), Some(2)); // High
        assert_eq!(queue.dequeue(), Some(3)); // Normal
        assert_eq!(queue.dequeue(), Some(1)); // Low
    }

    #[test]
    fn test_with_capacity() {
        let queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::with_capacity(100);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Critical);
        queue.enqueue(2, SchedulerPriority::Normal);
        queue.enqueue(3, SchedulerPriority::Low);

        queue.clear();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_contains() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Normal);
        queue.enqueue(2, SchedulerPriority::High);

        assert!(queue.contains(&1));
        assert!(queue.contains(&2));
        assert!(!queue.contains(&3));
    }

    #[test]
    fn test_find() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Low);
        queue.enqueue(2, SchedulerPriority::High);
        queue.enqueue(3, SchedulerPriority::Normal);

        // Should find highest priority match first
        let found = queue.find(|&x| x > 1);
        assert_eq!(found, Some(&2)); // High priority
    }

    #[test]
    fn test_any() {
        let mut queue: PrioritySchedulerQueue<i32> = PrioritySchedulerQueue::new();

        queue.enqueue(1, SchedulerPriority::Normal);
        queue.enqueue(2, SchedulerPriority::Normal);

        assert!(queue.any(|&x| x == 2));
        assert!(!queue.any(|&x| x == 3));
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(SchedulerPriority::default(), SchedulerPriority::Normal);
    }

    #[test]
    fn test_priority_ordering_enum() {
        assert!(SchedulerPriority::Critical > SchedulerPriority::High);
        assert!(SchedulerPriority::High > SchedulerPriority::Normal);
        assert!(SchedulerPriority::Normal > SchedulerPriority::Low);
    }
}
