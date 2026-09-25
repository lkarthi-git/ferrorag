use async_trait::async_trait;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, info_span, warn, Instrument};

#[async_trait]
pub trait Topology: Clone + PartialEq + Eq + std::fmt::Debug + Send + Sync + Sized {
    type State;
    type Delta;
    type Error:std::fmt::Debug + Send + Sync;

    async fn execute_node(&self, state: &Self::State) -> Result<Self::Delta, Self::Error>;
    fn route(&self, state: &Self::State) -> Option<Self>;
}

pub trait Reducer<D, E> {
    fn reduce(&mut self, delta: D) -> Result<(), E>;
}

pub enum ExecutionResult<T: Topology> {
    Completed { state: T::State, ctx: ExecutionContext<T> },
    Suspended { state: T::State, pending_node: T , ctx: ExecutionContext<T> },
}

#[derive(Debug)]
pub enum GraphError<E> {
    NodeError(E),
    LogicalError(E),
    MaxStepsExceeded(usize),
    Cancelled,
    CheckpointError(String),
}

#[async_trait]
pub trait Checkpoint<T: Topology> {
    async fn save_checkpoint(&self, ctx: &ExecutionContext<T>, state: &T::State) -> Result<(), String>;
    async fn load_checkpoint(&self, thread_id: &str) -> Result<Option<(ExecutionContext<T>, T::State)>, String>;
}

pub enum CheckpointStrategy {
    OnSuspendOnly,            
    EveryNSteps(usize),       
    Always,                 
}

#[derive(Debug, Clone)]
pub struct ExecutionContext<K> {
    pub thread_id: String,
    pub next_node: Option<K>,
    pub step_count: usize,
    pub metadata: HashMap<String, String>,
    #[doc(hidden)] 
    pub is_resuming: bool
}

impl<K> ExecutionContext<K> {
    pub fn new(thread_id: String, next_node: K) -> Self {
        Self {
            thread_id,
            next_node: Some(next_node),
            step_count: 0,
            metadata: HashMap::new(),
            is_resuming: false,
        }
    }
    
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

pub struct GraphEngine<T: Topology> {
    max_steps: usize,
    interrupt_before: Vec<T>,
    cancel_token: Option<CancellationToken>,
    strategy: Option<CheckpointStrategy>,
    checkpoint: Option<Box<dyn Checkpoint<T> + Send + Sync>>,
}

impl<T: Topology> GraphEngine<T> {
    pub fn new() -> Self {
        Self {
            max_steps: usize::MAX,
            interrupt_before: Vec::new(),
            cancel_token: None,
            strategy: None,
            checkpoint: None,
        }
    }

    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn with_interrupt_before(mut self, node: T) -> Self {
        self.interrupt_before.push(node);
        self
    }

    pub fn with_cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }

    pub fn with_checkpoint(mut self, checkpoint: Box<dyn Checkpoint<T> + Send + Sync>, strategy: CheckpointStrategy) -> Self {
        self.checkpoint = Some(checkpoint);
        self.strategy = Some(strategy);
        self
    }

    pub async fn save(&self, ctx: &ExecutionContext<T>, state: &T::State) -> Result<(), GraphError<T::Error>> {
        if let Some(cp) = &self.checkpoint {
            let should_save = match &self.strategy {
                Some(CheckpointStrategy::Always) => true,
                Some(CheckpointStrategy::EveryNSteps(n)) => ctx.step_count > 0 && ctx.step_count % n == 0, 
                _ => false,
            };

            if should_save {
                let start = std::time::Instant::now();
                cp.save_checkpoint(ctx, state)
                    .await
                    .map_err(GraphError::CheckpointError)?;
                metrics::histogram!("graph_checkpoint_save_duration_seconds").record(start.elapsed().as_secs_f64());
                metrics::counter!("graph_checkpoints_saved_total", "reason" => "strategy").increment(1);
            }
        }
        Ok(())
    }

    pub async fn force_save(&self, ctx: &ExecutionContext<T>, state: &T::State) -> Result<(), GraphError<T::Error>> {
        if let Some(cp) = &self.checkpoint {
            let start = std::time::Instant::now();
            cp.save_checkpoint(ctx, state)
                .await
                .map_err(GraphError::CheckpointError)?;
            metrics::histogram!("graph_checkpoint_save_duration_seconds").record(start.elapsed().as_secs_f64());
            metrics::counter!("graph_checkpoints_saved_total", "reason" => "forced").increment(1);
        }
        Ok(())
    }

    pub async fn resume(&self, thread_id: &str) -> Result<ExecutionResult<T>, GraphError<T::Error>> 
    where 
        T::State: Reducer<T::Delta, T::Error>,
    {
        let cp = self.checkpoint.as_ref().ok_or_else(|| {
            GraphError::CheckpointError("Cannot resume: no checkpoint configured".to_string())
        })?;

        let (mut ctx, state) = cp.load_checkpoint(thread_id)
            .await
            .map_err(GraphError::CheckpointError)?
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("No checkpoint found for thread_id: {}", thread_id))
            })?;

        metrics::counter!("graph_checkpoints_loaded_total").increment(1);
        ctx.is_resuming = true;

        info!("Resuming thread {} from step {}", thread_id, ctx.step_count);
        self.execute(ctx, state).await
    }

    pub async fn execute(
        &self, 
        mut ctx: ExecutionContext<T>, 
        mut state: T::State
    ) -> Result<ExecutionResult<T>, GraphError<T::Error>> 
    where 
        T::State: Reducer<T::Delta, T::Error> 
    {
        let cancel_token = self.cancel_token.clone().unwrap_or_else(CancellationToken::new);
        let thread_id = ctx.thread_id.clone();
        // Wrap the entire execution in a tracing span tagged with the thread_id
        async {
            while let Some(node) = ctx.next_node.clone() {
                let node_name = format!("{:?}", node);

                if !ctx.is_resuming && self.interrupt_before.contains(&node) {
                    info!(node = %node_name, "Interrupting execution before node");
                    metrics::counter!("graph_suspensions_total", "node" => node_name.clone()).increment(1);
                    self.force_save(&ctx, &state).await?;
                    return Ok(ExecutionResult::Suspended { 
                        state, 
                        pending_node: node,
                        ctx 
                    });
                }

                ctx.is_resuming = false;

                if ctx.step_count >= self.max_steps {
                    error!("Max steps ({}) exceeded", self.max_steps);
                    metrics::counter!("graph_errors_total", "error_type" => "max_steps").increment(1);
                    return Err(GraphError::MaxStepsExceeded(ctx.step_count));
                }

                let start_time = std::time::Instant::now();

                let delta = tokio::select! {
                    _ = cancel_token.cancelled() => {
                        warn!("Execution cancelled at step {}", ctx.step_count);
                        metrics::counter!("graph_cancelled_total").increment(1);
                        
                        if let Err(e) = self.force_save(&ctx, &state).await {
                            error!("Failed to save checkpoint during cancellation: {:?}", e);
                        }
                        return Err(GraphError::Cancelled);
                    }
                    res = node.execute_node(&state) => {
                        res.map_err(|e| {
                            metrics::counter!("graph_errors_total", "error_type" => "node_execution").increment(1);
                            GraphError::NodeError(e)
                        })?
                    }
                };

                metrics::histogram!("graph_node_execution_duration_seconds", "node" => node_name.clone())
                    .record(start_time.elapsed().as_secs_f64());
                metrics::counter!("graph_nodes_executed_total", "node" => node_name.clone()).increment(1);

                state.reduce(delta).map_err(|e| {
                    metrics::counter!("graph_errors_total", "error_type" => "reduce").increment(1);
                    GraphError::LogicalError(e)
                })?;
                
                ctx.step_count += 1;
                ctx.next_node = node.route(&state);
                
                debug!(node = %node_name, step = ctx.step_count, "Node executed and state reduced");
                
                self.save(&ctx, &state).await?;
            }

            info!("Graph execution completed successfully");
            metrics::counter!("graph_completed_total").increment(1);
            Ok(ExecutionResult::Completed { state, ctx })
            
        }.instrument(info_span!("graph_execute", thread_id = %thread_id)).await
    }
}

// ==========================================
// UNIT TESTS
// ==========================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::{sleep, Duration};

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TestNode { Step1, Step2, Step3 }

    #[derive(Debug, Clone)]
    struct TestState { count: usize }

    enum TestDelta { Increment }

    #[async_trait]
    impl Topology for TestNode {
        type State = TestState;
        type Delta = TestDelta;
        type Error = String;

        async fn execute_node(&self, _state: &Self::State) -> Result<Self::Delta, Self::Error> {
            // Simulate minimal work
            sleep(Duration::from_millis(5)).await;
            Ok(TestDelta::Increment)
        }

        fn route(&self, _state: &Self::State) -> Option<Self> {
            match self {
                TestNode::Step1 => Some(TestNode::Step2),
                TestNode::Step2 => Some(TestNode::Step3),
                TestNode::Step3 => None, // End of graph
            }
        }
    }

    impl Reducer<TestDelta, String> for TestState {
        fn reduce(&mut self, delta: TestDelta) -> Result<(), String> {
            match delta {
                TestDelta::Increment => self.count += 1,
            }
            Ok(())
        }
    }

    // Mock Checkpoint Storage (In-Memory)
    #[derive(Clone)]
    struct MockCheckpoint {
        store: Arc<Mutex<HashMap<String, (ExecutionContext<TestNode>, TestState)>>>,
    }

    impl MockCheckpoint {
        fn new() -> Self {
            Self { store: Arc::new(Mutex::new(HashMap::new())) }
        }
    }

    #[async_trait]
    impl Checkpoint<TestNode> for MockCheckpoint {
        async fn save_checkpoint(&self, ctx: &ExecutionContext<TestNode>, state: &TestState) -> Result<(), String> {
            self.store.lock().unwrap().insert(ctx.thread_id.clone(), (ctx.clone(), state.clone()));
            Ok(())
        }

        async fn load_checkpoint(&self, thread_id: &str) -> Result<Option<(ExecutionContext<TestNode>, TestState)>, String> {
            Ok(self.store.lock().unwrap().get(thread_id).cloned())
        }
    }

    #[tokio::test]
    async fn test_graph_executes_to_completion() {
        let engine = GraphEngine::new();
        let ctx = ExecutionContext::new("t1".into(), TestNode::Step1);
        let state = TestState { count: 0 };

        let result = engine.execute(ctx, state).await.unwrap();

        match result {
            ExecutionResult::Completed { state, ctx } => {
                assert_eq!(state.count, 3); // 3 steps executed
                assert_eq!(ctx.step_count, 3);
            }
            _ => panic!("Expected Completed result"),
        }
    }

    #[tokio::test]
    async fn test_max_steps_exceeded() {
        let engine = GraphEngine::new().with_max_steps(2);
        let ctx = ExecutionContext::new("t2".into(), TestNode::Step1);
        let state = TestState { count: 0 };

        let result = engine.execute(ctx, state).await;
        assert!(matches!(result, Err(GraphError::MaxStepsExceeded(2))));
    }

    #[tokio::test]
    async fn test_suspend_and_resume() {
        let cp = MockCheckpoint::new();
        let engine = GraphEngine::new()
            .with_interrupt_before(TestNode::Step2)
            .with_checkpoint(Box::new(cp.clone()), CheckpointStrategy::OnSuspendOnly);

        let ctx = ExecutionContext::new("t3".into(), TestNode::Step1);
        let state = TestState { count: 0 };

        // 1. Initial run, should suspend before Step2
        let result1 = engine.execute(ctx, state).await.unwrap();
        match result1 {
            ExecutionResult::Suspended { state, pending_node, ctx } => {
                assert_eq!(state.count, 1); // Only Step1 ran
                assert_eq!(ctx.step_count, 1);
                assert_eq!(pending_node, TestNode::Step2);
            }
            _ => panic!("Expected Suspended result"),
        }

        // 2. Resume execution
        let result2 = engine.resume("t3").await.unwrap();
        match result2 {
            ExecutionResult::Completed { state, ctx } => {
                assert_eq!(state.count, 3); // Finished the remaining steps
                assert_eq!(ctx.step_count, 3);
            }
            _ => panic!("Expected Completed result on resume"),
        }
    }

    #[tokio::test]
    async fn test_cancellation_saves_state() {
        let cp = MockCheckpoint::new();
        let cancel_token = CancellationToken::new();
        
        let engine = GraphEngine::new()
            .with_cancel_token(cancel_token.clone())
            .with_checkpoint(Box::new(cp.clone()), CheckpointStrategy::OnSuspendOnly);

        let ctx = ExecutionContext::new("t4".into(), TestNode::Step1);
        let state = TestState { count: 0 };

        // Cancel it immediately before it executes
        cancel_token.cancel();

        let result = engine.execute(ctx, state).await;
        
        assert!(matches!(result, Err(GraphError::Cancelled)));

        // Verify the state was forcefully saved despite the cancellation
        let saved = cp.load_checkpoint("t4").await.unwrap();
        assert!(saved.is_some());
        
        let (saved_ctx, saved_state) = saved.unwrap();
        assert_eq!(saved_ctx.step_count, 0);
        assert_eq!(saved_state.count, 0);
    }
}