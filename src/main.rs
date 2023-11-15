use faiss::{index_factory, Index, MetricType};
extern crate rand;
use std::time::Instant;

use rand::{distributions::Standard, prelude::*};

// https://github.com/facebookresearch/faiss/blob/0c2243c5b409fa1f9a7369e4b130cc5b046dd7c0/benchs/bench_ivf_fastscan.py#L57
//

const DIMENSIONS: u32 = 512;

// AKA num_clusters in orange terminology
const NLIST: usize = 1024;
const M: usize = 32;

const NUM_VECTORS_IN_INDEX: usize = 1_500_000;

const NUM_QUERIES: usize = 10000;
const K: usize = 1000;

// Build a faiss fastscan IVF PQ index with all random data and run some random
// searches on it as a benchmark.
fn main() {
    let start_time = Instant::now();
    let mut rng = thread_rng();
    let mut index =
        index_factory(DIMENSIONS, format!("IVF{NLIST},PQ{M}x4fs"), MetricType::L2).unwrap();

    let train_data: Vec<f32> = (0..(DIMENSIONS * NLIST as u32 * 100))
        .map(|_| rng.sample(Standard))
        .collect();

    println!("{:?} Training...", start_time.elapsed());
    index.train(&train_data).unwrap();

    println!("{:?} Building index...", start_time.elapsed());
    for _ in 0..NUM_VECTORS_IN_INDEX {
        let data: Vec<f32> = (0..DIMENSIONS).map(|_| rng.sample(Standard)).collect();
        index.add(&data).unwrap();
    }

    println!("{:?} Querying once...", start_time.elapsed());
    let query: Vec<f32> = (0..DIMENSIONS).map(|_| rng.sample(Standard)).collect();

    let result = index.search(&query, 5).unwrap();
    println!("{:?} Result: {:?}", start_time.elapsed(), result);
    for (i, (l, d)) in result
        .labels
        .iter()
        .zip(result.distances.iter())
        .enumerate()
    {
        println!("#{}: {} (D={})", i + 1, *l, *d);
    }

    let start_time = Instant::now();
    println!("Starting {} queries at k={}...", NUM_QUERIES, K);
    for _ in 0..NUM_QUERIES {
        let query: Vec<f32> = (0..DIMENSIONS).map(|_| rng.sample(Standard)).collect();
        let _result = index.search(&query, K).unwrap();
    }
    let elapsed = start_time.elapsed();
    println!(
        "Completed {} in {} seconds. That's an average of {} per sec.",
        NUM_QUERIES,
        elapsed.as_secs(),
        NUM_QUERIES / elapsed.as_secs() as usize,
    );
}
