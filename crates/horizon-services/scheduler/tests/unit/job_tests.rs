use scheduler::models::{Job, JobState, Priority};
use scheduler::models::resource::ResourceRequest;
use uuid::Uuid;

#[test]
fn test_job_state_ordering() {
    assert!(JobState::Queued < JobState::Scheduled);
    assert!(JobState::Scheduled < JobState::Running);
}

#[test]
fn test_priority_ordering() {
    assert!(Priority::Low < Priority::Normal);
    assert!(Priority::Normal < Priority::High);
}

#[test]
fn test_job_builder_minimal() {
    let job = Job::builder()
        .user_id("user1")
        .gpu_count(4)
        .build()
        .unwrap();

    assert_eq!(job.user_id, "user1");
    assert_eq!(job.resources.gpu_count, 4);
    assert_eq!(job.state, JobState::Queued);
    assert_eq!(job.priority, Priority::Normal);
}

#[test]
fn test_job_builder_full() {
    let job = Job::builder()
        .user_id("user1")
        .job_name("training-job")
        .gpu_count(8)
        .gpu_type("H100")
        .cpu_cores(32)
        .memory_gb(256)
        .priority(Priority::High)
        .command("python train.py")
        .working_dir("/workspace")
        .build()
        .unwrap();

    assert_eq!(job.user_id, "user1");
    assert_eq!(job.job_name, Some("training-job".to_string()));
    assert_eq!(job.resources.gpu_count, 8);
    assert_eq!(job.resources.gpu_type, Some("H100".to_string()));
    assert_eq!(job.resources.cpu_cores, Some(32));
    assert_eq!(job.resources.memory_gb, Some(256));
    assert_eq!(job.priority, Priority::High);
    assert_eq!(job.command, Some("python train.py".to_string()));
    assert_eq!(job.working_dir, Some("/workspace".to_string()));
}

#[test]
fn test_job_builder_missing_user_id() {
    let result = Job::builder().gpu_count(4).build();
    assert!(result.is_err());
}

#[test]
fn test_job_builder_missing_gpu_count() {
    let result = Job::builder().user_id("user1").build();
    assert!(result.is_err());
}

#[test]
fn test_job_builder_zero_gpu_count() {
    let result = Job::builder().user_id("user1").gpu_count(0).build();
    assert!(result.is_err());
}

#[test]
fn test_job_state_transition_queued_to_scheduled() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    assert!(job.transition_to(JobState::Scheduled).is_ok());
    assert_eq!(job.state, JobState::Scheduled);
}

#[test]
fn test_job_state_transition_scheduled_to_running() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    job.transition_to(JobState::Scheduled).unwrap();
    assert!(job.transition_to(JobState::Running).is_ok());
    assert_eq!(job.state, JobState::Running);
}

#[test]
fn test_job_state_transition_running_to_completed() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    job.transition_to(JobState::Scheduled).unwrap();
    job.transition_to(JobState::Running).unwrap();
    assert!(job.transition_to(JobState::Completed).is_ok());
    assert_eq!(job.state, JobState::Completed);
}

#[test]
fn test_job_state_invalid_transition_queued_to_completed() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    let result = job.transition_to(JobState::Completed);
    assert!(result.is_err());
}

#[test]
fn test_job_state_invalid_transition_completed_to_queued() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    job.transition_to(JobState::Scheduled).unwrap();
    job.transition_to(JobState::Running).unwrap();
    job.transition_to(JobState::Completed).unwrap();

    let result = job.transition_to(JobState::Queued);
    assert!(result.is_err());
}

#[test]
fn test_job_state_transition_running_to_preempted() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    job.transition_to(JobState::Scheduled).unwrap();
    job.transition_to(JobState::Running).unwrap();
    assert!(job.transition_to(JobState::Preempted).is_ok());
    assert_eq!(job.state, JobState::Preempted);
}

#[test]
fn test_job_state_transition_preempted_to_queued() {
    let mut job = Job::builder()
        .user_id("user1")
        .gpu_count(2)
        .build()
        .unwrap();

    job.transition_to(JobState::Scheduled).unwrap();
    job.transition_to(JobState::Running).unwrap();
    job.transition_to(JobState::Preempted).unwrap();

    assert!(job.transition_to(JobState::Queued).is_ok());
    assert_eq!(job.state, JobState::Queued);
}

#[test]
fn test_job_can_be_cancelled_from_any_active_state() {
    let mut job1 = Job::builder().user_id("user1").gpu_count(2).build().unwrap();
    assert!(job1.transition_to(JobState::Cancelled).is_ok());

    let mut job2 = Job::builder().user_id("user1").gpu_count(2).build().unwrap();
    job2.transition_to(JobState::Scheduled).unwrap();
    assert!(job2.transition_to(JobState::Cancelled).is_ok());

    let mut job3 = Job::builder().user_id("user1").gpu_count(2).build().unwrap();
    job3.transition_to(JobState::Scheduled).unwrap();
    job3.transition_to(JobState::Running).unwrap();
    assert!(job3.transition_to(JobState::Cancelled).is_ok());
}

#[test]
fn test_job_serialization() {
    let job = Job::builder()
        .user_id("user1")
        .job_name("test-job")
        .gpu_count(4)
        .priority(Priority::High)
        .build()
        .unwrap();

    let json = serde_json::to_string(&job).unwrap();
    let deserialized: Job = serde_json::from_str(&json).unwrap();

    assert_eq!(job.id, deserialized.id);
    assert_eq!(job.user_id, deserialized.user_id);
    assert_eq!(job.job_name, deserialized.job_name);
    assert_eq!(job.resources.gpu_count, deserialized.resources.gpu_count);
    assert_eq!(job.priority, deserialized.priority);
}

#[test]
fn test_priority_boost_factor() {
    assert_eq!(Priority::High.boost_factor(), 4.0);
    assert_eq!(Priority::Normal.boost_factor(), 1.0);
    assert_eq!(Priority::Low.boost_factor(), 0.25);
}
