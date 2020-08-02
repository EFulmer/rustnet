// TODO test using examples from Andrew Ng's course.
use super::utils;

// assume bias term already added to xs
pub fn mean_squared_error(m: i32, h_theta: &Vec<f64>, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> f64 {
    // TODO length checks
    let coef = 1.0 / (2.0 * (m as f64));
    let mut acc = 0.0;
    for (i, x_i) in x.iter().enumerate() {
        let pred = utils::dot_product(&h_theta, &x_i);
        let y_hat = pred - y.get(i).unwrap();
        acc += y_hat.powi(2);
    }
    coef * acc
}

/// Add a bias term to the front of xs.
pub fn add_bias_term(xs: &mut Vec<f64>) {
    xs.insert(0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_squared_error_cannot_be_less_than_zero() {
        let h = vec![1.0, 1.0, 1.0];
        let x = vec![vec![0.0, 0.0, 0.0]];
        let y = vec![0.0];
        let m = 1;
        let expected = 0.0;
        let actual = mean_squared_error(m, &h, &x, &y);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_mean_squared_error() {
        let h = vec![1.0, 2.0, 3.0];
        let x = vec![vec![4.0, 5.0, 6.0]];
        let y = vec![7.0];
        let m = 1;
        let expected = 313.0;
        let actual = mean_squared_error(m, &h, &x, &y);
        assert_eq!(expected, actual.round());
    }

    #[test]
    fn test_add_bias_term() {
        let v = vec![1.1, 3.7];
        let mut actual = v.clone();
        add_bias_term(&mut actual);
        assert_eq!(vec![1.0, 1.1, 3.7], actual);
    }
}
