pub fn sigmoid_scalar(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid(xs: Vec<f64>) -> Vec<f64> {
    xs.iter().map(|&x| sigmoid_scalar(x)).collect()
}

pub fn dot_product(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    assert_eq!(v1.len(), v2.len());
    v1.iter().zip(v2.iter()).map(|(v, u)| v * u).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_scalar_0_is_0_point_5() {
        assert_eq!(sigmoid_scalar(0.0), 0.5);
    }

    #[test]
    fn test_sigmoid_0_is_0_point_5() {
        assert_eq!(sigmoid(vec![0.0, 0.0]), vec![0.5, 0.5]);
    }
}
