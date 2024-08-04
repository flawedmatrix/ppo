use ndarray::{prelude::*, RemoveAxis};

pub struct RunningMeanStd<D>
where
    D: Dimension,
{
    pub mean: Array<f32, D>,
    pub var: Array<f32, D>,
    count: f32,
}

impl<D> RunningMeanStd<D>
where
    D: Dimension,
{
    pub fn new<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::new_with_epsilon(shape, 1e-4)
    }

    pub fn new_with_epsilon<Sh>(shape: Sh, epsilon: f32) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let sh = shape.into_shape();
        Self {
            mean: Array::zeros(sh.clone()),
            var: Array::ones(sh.clone()),
            count: epsilon,
        }
    }

    pub fn update<E>(&mut self, x: &Array<f32, E>)
    where
        E: Dimension<Smaller = D> + RemoveAxis,
    {
        let batch_mean: Array<f32, D> = x.mean_axis(Axis(0)).unwrap();
        let batch_var: Array<f32, D> = x.var_axis(Axis(0), 0.);
        let batch_count = x.shape()[0];
        self.update_from_moments(batch_mean, batch_var, batch_count)
    }

    fn update_from_moments(
        &mut self,
        batch_mean: Array<f32, D>,
        batch_var: Array<f32, D>,
        batch_count: usize,
    ) {
        let b_count = batch_count as f32;

        let delta = batch_mean - &self.mean;
        let tot_count = self.count + b_count;

        let new_mean = &self.mean + &delta * b_count / tot_count;
        let m_a = &self.var * self.count;
        let m_b = batch_var * b_count;

        let m2 = m_a + m_b + (&delta * &delta) * self.count * b_count / tot_count;
        let new_var = m2 / tot_count;
        let new_count = tot_count;

        self.mean = new_mean;
        self.var = new_var;
        self.count = new_count;
    }
}
