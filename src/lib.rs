mod linear_regression;
mod utils;
mod ng;

fn binary_and_nn(x1: f64, x2: f64) -> f64 {
    let theta1 = vec![-30.0, 20.0, 20.0];
    let a1 = vec![1.0, x1, x2];
    let z2 = utils::dot_product(&a1, &theta1);
    let a2 = utils::sigmoid_scalar(z2);
    return a2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_and_nn_f_f() {
        assert_eq!(binary_and_nn(0.0, 0.0).round(), 0.0);
    }

    #[test]
    fn test_binary_and_nn_t_f() {
        assert_eq!(binary_and_nn(1.0, 0.0).round(), 0.0);
    }

    #[test]
    fn test_binary_and_nn_f_t() {
        assert_eq!(binary_and_nn(0.0, 1.0).round(), 0.0);
    }

    #[test]
    fn test_binary_and_nn_t_t() {
        assert_eq!(binary_and_nn(1.0, 1.0).round(), 1.0);
    }
}
