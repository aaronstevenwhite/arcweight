#![allow(dead_code)]
#![allow(clippy::module_inception)]
use criterion::{criterion_group, criterion_main, Criterion};

mod core {
    pub mod basic_operations;
    pub mod composition;
    pub mod creation;
    pub mod determinization;
    pub mod epsilon_removal;
    pub mod minimization;
    pub mod shortest_path;
}

mod memory {
    pub mod arc_storage;
    pub mod memory_operations;
    pub mod memory_usage;
    pub mod state_management;
}

mod io {
    pub mod deserialization;
    pub mod serialization;
}

mod optimization {
    pub mod epsilon_removal;
    pub mod minimization;
    pub mod optimization;
    pub mod weight_pushing;
}

mod parallel {
    pub mod arc_processing;
    pub mod parallel_operations;
    pub mod parallel_ops;
    pub mod parallel_processing;
    pub mod parallel_state_processing;
    pub mod state_processing;
    pub mod weight_processing;
}

fn run_all_benchmarks(c: &mut Criterion) {
    // Core operations
    core::creation::bench_linear_creation(c);
    core::creation::bench_branching_creation(c);
    core::composition::bench_linear_composition(c);
    core::composition::bench_mixed_composition(c);
    core::shortest_path::bench_linear_shortest_path(c);
    core::shortest_path::bench_branching_shortest_path(c);
    core::determinization::bench_linear_determinization(c);
    core::determinization::bench_branching_determinization(c);
    core::determinization::bench_non_deterministic_determinization(c);

    // Memory operations
    memory::arc_storage::bench_arc_count(c);
    memory::arc_storage::bench_arc_iteration(c);
    memory::arc_storage::bench_arc_lookup(c);
    memory::state_management::bench_state_creation(c);
    memory::state_management::bench_state_lookup(c);
    memory::state_management::bench_state_modification(c);
    memory::memory_usage::bench_large_fst_creation(c);
    memory::memory_usage::bench_fst_clone(c);
    memory::memory_usage::bench_fst_clear(c);

    // I/O operations
    io::serialization::bench_serialization(c);
    io::deserialization::bench_deserialization(c);

    // Optimization operations
    optimization::weight_pushing::bench_weight_pushing(c);
    optimization::weight_pushing::bench_weight_pushing_forward(c);
    optimization::weight_pushing::bench_weight_pushing_backward(c);
    optimization::minimization::bench_minimization(c);
    optimization::minimization::bench_minimization_with_weights(c);
    optimization::epsilon_removal::bench_epsilon_removal(c);
    optimization::epsilon_removal::bench_epsilon_removal_with_weights(c);

    // Parallel operations
    parallel::parallel_operations::bench_parallel_composition(c);
    parallel::parallel_operations::bench_parallel_shortest_path(c);
    parallel::parallel_operations::bench_parallel_arc_processing(c);
    parallel::parallel_state_processing::bench_parallel_state_processing(c);
    parallel::parallel_state_processing::bench_parallel_state_arc_count(c);
    parallel::parallel_state_processing::bench_parallel_state_weight_sum(c);
}

criterion_group!(benches, run_all_benchmarks);
criterion_main!(benches);
