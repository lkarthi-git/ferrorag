use std::collections::{HashMap,HashSet};
use async_trait::async_trait;
use futures::future::try_join_all;
use tokio_util::sync::CancellationToken;
use crate::node::Node;
use tokio::select;
use tracing::{debug, error, info, trace};

pub trait Reducer<D, E> {
    fn reduce(&mut self, delta: D) -> Result<(), E>;
}

#[async_trait]
pub trait GraphNode: Send + Sync {
    type State;
    type Delta;
    type Error;

    async fn execute(&self, state: &Self::State) -> Result<Self::Delta, Self::Error>;
}

pub enum Routing<S> {
    Direct(Vec<String>),
    Conditional(Box<dyn Fn(&S) -> Vec<String> + Send + Sync>),
}


pub struct NodeAdapter<N, S, D> 
where 
    N: Node,
{
    node: N,
    extractor: Box<dyn Fn(&S) -> N::Input + Send + Sync>,
    mapper: Box<dyn Fn(N::Output) -> D + Send + Sync>,
}

impl<N, S, D> NodeAdapter<N, S, D> 
where 
    N: Node,
{
    pub fn new<Ext, Map>(node: N, extractor: Ext, mapper: Map) -> Self 
    where 
        Ext: Fn(&S) -> N::Input + Send + Sync + 'static,
        Map: Fn(N::Output) -> D + Send + Sync + 'static,
    {
        Self {
            node,
            extractor: Box::new(extractor),
            mapper: Box::new(mapper),
        }
    }
}

#[async_trait]
impl<N, S, D> GraphNode for NodeAdapter<N, S, D>
where
    N: crate::node::Node + Send + Sync,
    S: Send + Sync,
    D: Send + Sync,
    N::Error: Send + Sync,
    N::Input: Send + Sync
{
    type State = S;
    type Delta = D;
    type Error = N::Error;

    async fn execute(&self, state: &Self::State) -> Result<Self::Delta, Self::Error> {
        let input = (self.extractor)(state);
        let output = self.node.execute(&input).await?;
        Ok((self.mapper)(output))
    }
}


#[derive(Debug)]
pub enum GraphError<E> {
    NodeError(E),
    MaxStepsExceeded(usize),
    NodeNotFound(String),
    Cancelled
}

pub struct Graph<S, D, E> {
    pub nodes: HashMap<String, Box<dyn GraphNode<State = S, Delta = D, Error = E> + Send + Sync>>,
    pub edges: HashMap<String, Routing<S>>,
    pub max_steps: usize,
}

impl<S, D, E> Graph<S, D, E> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            max_steps: 100,
        }
    }
    
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn add_node(&mut self, name: &str, node: Box<dyn GraphNode<State = S, Delta = D, Error = E> + Send + Sync>) {
        self.nodes.insert(name.to_string(), node);
    }

    pub fn add_edge(&mut self, from: &str, routing: Routing<S>) {
        self.edges.insert(from.to_string(), routing);
    }
}

impl<S, D, E> Graph<S, D, E> 
where 
    S: Reducer<D, E> + Send + Sync,
    D: Send + Sync,
    E: Send + Sync,
{
    pub async fn execute(&self, initial_state: S, start_node: &str,cancel_token: Option<CancellationToken>) -> Result<S, GraphError<E>> {
        let mut state = initial_state;
        let mut active_nodes = vec![start_node.to_string()];
        let mut step_count = 0;

        while !active_nodes.is_empty() {
            step_count += 1;
            if step_count > self.max_steps {
                return Err(GraphError::MaxStepsExceeded(step_count));
            }

            let mut futures = Vec::new();
            for node_name in &active_nodes {
                let node = self.nodes.get(node_name)
                    .ok_or_else(|| GraphError::NodeNotFound(node_name.clone()))?;
                futures.push(node.execute(&state));
            }
            let execution_future = try_join_all(futures);
            
            let deltas = if let Some(token) = &cancel_token {
                select! {
                    _ = token.cancelled() => {
                        info!("Graph execution cancelled at step {}", step_count);
                        return Err(GraphError::Cancelled);
                    }
                    result = execution_future => result.map_err(GraphError::NodeError)?,
                }
            } else {
                execution_future.await.map_err(GraphError::NodeError)?
            };

            for delta in deltas {
                state.reduce(delta).map_err(GraphError::NodeError)?;
            }
            
            let mut new_active_nodes: Vec<String> = Vec::new(); 
            
            for node_name in active_nodes.iter() {
                if let Some(edge) = self.edges.get(node_name) {
                    match edge {
                        Routing::Direct(next_nodes) => {
                            // Clone the strings into the new vector
                            new_active_nodes.extend(next_nodes.iter().cloned());
                        }
                        Routing::Conditional(routing_fn) => {
                            // Conditional functions generate fresh strings, so no clone needed
                            new_active_nodes.extend(routing_fn(&state));
                        }
                    }
                }
            }
            
            new_active_nodes.sort_unstable();
            new_active_nodes.dedup();
            
            active_nodes = new_active_nodes;
        }   
        
        Ok(state)
    }
}