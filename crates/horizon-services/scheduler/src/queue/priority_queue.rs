use std::collections::VecDeque;

use crate::models::{Job, Priority};

/// Four-level priority queue with FIFO ordering within each band
pub struct PriorityQueue {
    urgent: VecDeque<Job>,
    high: VecDeque<Job>,
    normal: VecDeque<Job>,
    low: VecDeque<Job>,
}

impl PriorityQueue {
    pub fn new() -> Self {
        Self {
            urgent: VecDeque::new(),
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
        }
    }

    /// Enqueue a job in O(1) time
    pub fn enqueue(&mut self, job: Job) {
        match job.priority {
            Priority::Urgent => self.urgent.push_back(job),
            Priority::High => self.high.push_back(job),
            Priority::Normal => self.normal.push_back(job),
            Priority::Low => self.low.push_back(job),
        }
    }

    /// Dequeue highest priority job in O(1) time
    pub fn dequeue(&mut self) -> Option<Job> {
        self.urgent
            .pop_front()
            .or_else(|| self.high.pop_front())
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }

    /// Peek at the highest priority job without removing it
    pub fn peek(&self) -> Option<&Job> {
        self.urgent
            .front()
            .or_else(|| self.high.front())
            .or_else(|| self.normal.front())
            .or_else(|| self.low.front())
    }

    /// Get total number of jobs in queue
    pub fn len(&self) -> usize {
        self.urgent.len() + self.high.len() + self.normal.len() + self.low.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get count of jobs by priority
    pub fn count_by_priority(&self, priority: Priority) -> usize {
        match priority {
            Priority::Urgent => self.urgent.len(),
            Priority::High => self.high.len(),
            Priority::Normal => self.normal.len(),
            Priority::Low => self.low.len(),
        }
    }

    /// Remove a specific job by ID (for cancellation)
    pub fn remove(&mut self, job_id: uuid::Uuid) -> Option<Job> {
        // Check urgent priority first
        if let Some(pos) = self.urgent.iter().position(|j| j.id == job_id) {
            return self.urgent.remove(pos);
        }

        // Check high priority
        if let Some(pos) = self.high.iter().position(|j| j.id == job_id) {
            return self.high.remove(pos);
        }

        // Check normal priority
        if let Some(pos) = self.normal.iter().position(|j| j.id == job_id) {
            return self.normal.remove(pos);
        }

        // Check low priority
        if let Some(pos) = self.low.iter().position(|j| j.id == job_id) {
            return self.low.remove(pos);
        }

        None
    }

    /// Clear all jobs from the queue
    pub fn clear(&mut self) {
        self.urgent.clear();
        self.high.clear();
        self.normal.clear();
        self.low.clear();
    }

    /// Get all jobs as a vector (for inspection/debugging)
    pub fn all_jobs(&self) -> Vec<Job> {
        let mut jobs = Vec::with_capacity(self.len());
        jobs.extend(self.urgent.iter().cloned());
        jobs.extend(self.high.iter().cloned());
        jobs.extend(self.normal.iter().cloned());
        jobs.extend(self.low.iter().cloned());
        jobs
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_job(user: &str, gpus: usize, priority: Priority) -> Job {
        Job::builder()
            .user_id(user)
            .gpu_count(gpus)
            .priority(priority)
            .build()
            .unwrap()
    }

    #[test]
    fn test_priority_ordering() {
        let mut queue = PriorityQueue::new();

        queue.enqueue(create_job("u1", 2, Priority::Low));
        queue.enqueue(create_job("u2", 2, Priority::High));
        queue.enqueue(create_job("u3", 2, Priority::Normal));

        assert_eq!(queue.dequeue().unwrap().priority, Priority::High);
        assert_eq!(queue.dequeue().unwrap().priority, Priority::Normal);
        assert_eq!(queue.dequeue().unwrap().priority, Priority::Low);
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut queue = PriorityQueue::new();

        let job1 = create_job("u1", 2, Priority::Normal);
        let job2 = create_job("u2", 2, Priority::Normal);
        let job3 = create_job("u3", 2, Priority::Normal);

        let id1 = job1.id;
        let id2 = job2.id;
        let id3 = job3.id;

        queue.enqueue(job1);
        queue.enqueue(job2);
        queue.enqueue(job3);

        assert_eq!(queue.dequeue().unwrap().id, id1);
        assert_eq!(queue.dequeue().unwrap().id, id2);
        assert_eq!(queue.dequeue().unwrap().id, id3);
    }

    #[test]
    fn test_enqueue_dequeue() {
        let mut queue = PriorityQueue::new();

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        queue.enqueue(create_job("u1", 4, Priority::Normal));
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let job = queue.dequeue().unwrap();
        assert_eq!(job.user_id, "u1");
        assert!(queue.is_empty());
    }

    #[test]
    fn test_peek() {
        let mut queue = PriorityQueue::new();

        assert!(queue.peek().is_none());

        let job = create_job("u1", 4, Priority::High);
        let job_id = job.id;
        queue.enqueue(job);

        assert_eq!(queue.peek().unwrap().id, job_id);
        assert_eq!(queue.len(), 1); // Peek doesn't remove

        queue.dequeue();
        assert!(queue.peek().is_none());
    }

    #[test]
    fn test_count_by_priority() {
        let mut queue = PriorityQueue::new();

        queue.enqueue(create_job("u1", 2, Priority::High));
        queue.enqueue(create_job("u2", 2, Priority::High));
        queue.enqueue(create_job("u3", 2, Priority::Normal));
        queue.enqueue(create_job("u4", 2, Priority::Low));

        assert_eq!(queue.count_by_priority(Priority::High), 2);
        assert_eq!(queue.count_by_priority(Priority::Normal), 1);
        assert_eq!(queue.count_by_priority(Priority::Low), 1);
    }

    #[test]
    fn test_remove() {
        let mut queue = PriorityQueue::new();

        let job1 = create_job("u1", 2, Priority::High);
        let job2 = create_job("u2", 2, Priority::Normal);
        let job3 = create_job("u3", 2, Priority::Low);

        let id2 = job2.id;

        queue.enqueue(job1);
        queue.enqueue(job2);
        queue.enqueue(job3);

        assert_eq!(queue.len(), 3);

        let removed = queue.remove(id2).unwrap();
        assert_eq!(removed.id, id2);
        assert_eq!(queue.len(), 2);

        // Should not affect ordering
        assert_eq!(queue.dequeue().unwrap().priority, Priority::High);
        assert_eq!(queue.dequeue().unwrap().priority, Priority::Low);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut queue = PriorityQueue::new();
        queue.enqueue(create_job("u1", 2, Priority::Normal));

        let fake_id = uuid::Uuid::new_v4();
        assert!(queue.remove(fake_id).is_none());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut queue = PriorityQueue::new();

        queue.enqueue(create_job("u1", 2, Priority::High));
        queue.enqueue(create_job("u2", 2, Priority::Normal));
        queue.enqueue(create_job("u3", 2, Priority::Low));

        assert_eq!(queue.len(), 3);

        queue.clear();

        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_all_jobs() {
        let mut queue = PriorityQueue::new();

        queue.enqueue(create_job("u1", 2, Priority::Low));
        queue.enqueue(create_job("u2", 2, Priority::High));
        queue.enqueue(create_job("u3", 2, Priority::Normal));

        let all = queue.all_jobs();
        assert_eq!(all.len(), 3);

        // Should be in priority order
        assert_eq!(all[0].priority, Priority::High);
        assert_eq!(all[1].priority, Priority::Normal);
        assert_eq!(all[2].priority, Priority::Low);
    }
}
