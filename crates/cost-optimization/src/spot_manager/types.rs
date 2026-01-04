pub enum FallbackStrategy {
    OnDemand,
    Wait,
    Cancel,
}
pub struct SpotInstance {
    pub id: String,
}
pub struct SpotInstanceRequest {
    pub instance_type: String,
}
pub enum SpotInstanceState {
    Pending,
    Running,
    Terminated,
}
