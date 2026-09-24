use async_trait::async_trait;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait Topology: Clone + PartialEq + Eq + std::fmt::Debug + Send + Sync + Sized {
    type State;
    type Delta;
    type Error;

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
pub trait Checkpoint<T: Topology>
{
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
}

impl<K> ExecutionContext<K> {
    pub fn new(thread_id: String, next_node: K) -> Self {
        Self {
            thread_id,
            next_node: Some(next_node),
            step_count: 0,
            metadata: HashMap::new(),
        }
    }
    
    /// Builder method for fluent initialization
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


pub struct GraphEngine<T: Topology> {
    max_steps: usize,
    interrupt_before:Vec<T>,
    cancel_token: Option<CancellationToken>,
    strategy: Option<CheckpointStrategy>,
    checkpoint: Option<Box<dyn Checkpoint<T> + Send + Sync>>,
}

impl<T: Topology> GraphEngine<T> {
    pub fn new() -> Self {
        Self {
            max_steps:usize::MAX,
            interrupt_before: Vec::new(),
            cancel_token: Some(CancellationToken::new()),
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


    pub fn with_checkpoint(mut self, checkpoint: Box<dyn Checkpoint<T> + Send + Sync>,strategy: CheckpointStrategy) -> Self {
        self.checkpoint = Some(checkpoint);
        self.strategy = Some(strategy);
        self
    }

    pub async fn save(&self, ctx: &ExecutionContext<T>, state: &T::State) -> Result<(), GraphError<T::Error>> {
        if let Some(cp) = &self.checkpoint {
            let should_save = match &self.strategy {
                Some(CheckpointStrategy::Always) => true,
                // Prevent saving on step 0 if EveryNSteps is used
                Some(CheckpointStrategy::EveryNSteps(n)) => ctx.step_count > 0 && ctx.step_count % n == 0, 
                _ => false,
            };

            if should_save {
                cp.save_checkpoint(ctx, state)
                    .await
                    .map_err(GraphError::CheckpointError)?;
            }
        }
        Ok(())
    }

    /// Unconditionally saves the state, ignoring the step-count strategy.
    /// Used when the graph is interrupted/suspended.
    pub async fn force_save(&self, ctx: &ExecutionContext<T>, state: &T::State) -> Result<(), GraphError<T::Error>> {
        if let Some(cp) = &self.checkpoint {
            cp.save_checkpoint(ctx, state)
                .await
                .map_err(GraphError::CheckpointError)?;
        }
        Ok(())
    }

    /// Fetches a saved checkpoint and immediately resumes execution.
    pub async fn resume(&self, thread_id: &str) -> Result<ExecutionResult<T>, GraphError<T::Error>> 
    where 
        T::State: Reducer<T::Delta, T::Error>,
    {
        let cp = self.checkpoint.as_ref().ok_or_else(|| {
            GraphError::CheckpointError("Cannot resume: no checkpoint configured".to_string())
        })?;

        let (ctx, state) = cp.load_checkpoint(thread_id)
            .await
            .map_err(GraphError::CheckpointError)?
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("No checkpoint found for thread_id: {}", thread_id))
            })?;

        tracing::info!("Resuming thread {} from step {}", thread_id, ctx.step_count);
        self.execute(ctx, state).await
    }

    pub async fn execute(&self, mut ctx: ExecutionContext<T>, mut state: T::State) -> Result<ExecutionResult<T>, GraphError<T::Error>> 
    where 
        T::State: Reducer<T::Delta, T::Error> 
    {
        let cancel_token = self.cancel_token.clone().unwrap_or_else(CancellationToken::new);
        
        while let Some(node) = ctx.next_node.clone() {

            if self.interrupt_before.contains(&node) {
                tracing::info!("Interrupting execution before {:?}", node);
                self.force_save(&ctx, &state).await?;
                return Ok(ExecutionResult::Suspended { 
                    state, 
                    pending_node: node,
                    ctx 
                });
            }

            if ctx.step_count >= self.max_steps {
                return Err(GraphError::MaxStepsExceeded(ctx.step_count));
            }

            let delta = tokio::select! {
                _ = cancel_token.cancelled() => {
                    tracing::info!("Execution cancelled at step {}", ctx.step_count);
                    return Err(GraphError::Cancelled);
                }
                res = node.execute_node(&state) => {
                    res.map_err(GraphError::NodeError)?
                }
            };

            state.reduce(delta).map_err(GraphError::LogicalError)?;
            
            ctx.step_count += 1;
            
            ctx.next_node = node.route(&state);
            
            self.save(&ctx, &state).await?;
        }

        Ok(ExecutionResult::Completed { state, ctx })
    }

}