mod common;

pub struct TrainingConfig<P: AsRef<std::path::Path>> {
    pub model_path: P,
    pub num_steps: usize,
    pub num_updates: usize,
    pub num_epochs: usize,
    pub num_batches: usize,
}

pub fn train<T, P: AsRef<std::path::Path>>(init_state: T, config: TrainingConfig<P>) {
    println!("Hello, world!");
}
