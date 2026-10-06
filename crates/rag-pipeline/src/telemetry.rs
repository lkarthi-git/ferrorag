use metrics_exporter_prometheus::PrometheusBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_telemetry() {
    // ========================================================================
    // 1. Initialize Tracing Subscriber (Logs & Spans)
    // ========================================================================
    // EnvFilter allows you to control log levels via the RUST_LOG environment 
    // variable (e.g., `RUST_LOG=info` or `RUST_LOG=ferro_rag=debug`).
let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::new("error,ferro_rag=info,ferro_core=info")
        });
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    tracing::info!("Tracing initialized.");

    // ========================================================================
    // 2. Initialize Metrics Recorder (Counters, Gauges, Histograms)
    // ========================================================================
    // This spawns a background thread listening on port 9000.
    // When you visit http://localhost:9000/metrics, it will yield all your 
    // chunker gauges and throughput counters in Prometheus text format.
    let builder = PrometheusBuilder::new();
    
    builder
        .with_http_listener(([0, 0, 0, 0], 9000))
        .install()
        .expect("Failed to install Prometheus recorder");

    tracing::info!("Metrics exporter running at http://localhost:9000/metrics");
}