use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::linear_regression::{mean_squared_error};

/// Create an n-by-n identity matrix.
///
/// # Arguments
///
/// * `n` - the dimension of the matrix.
pub fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut mat = Vec::with_capacity(n);
    for row_ind in 0..n {
        let mut new_row = Vec::with_capacity(n);
        for col_ind in 0..n {
            if col_ind == row_ind {
                new_row.push(1.0);
            } else {
                new_row.push(0.0);
            }
        }
        mat.push(new_row);
    }
    mat
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix_2x2() {
        let expected = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let actual = identity_matrix(2);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_identity_matrix_3x3() {
        let expected = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let actual = identity_matrix(3);
        assert_eq!(expected, actual);
    }

    // TODO this code sucks (like the `if let` and excessive unwrapping and collecting),
    // let's clean it up later.
    fn load_testing_data() -> Vec<(f64, f64)> {
        let mut training_examples = Vec::new();
        // using `unwrap` here because it's test code.
        let file = File::open("src/ng/ex1data1.txt").unwrap();
        for line in BufReader::new(file).lines() {
            if let Ok(line) = line {
                let split: Vec<&str> = line.split(",").collect();
                let x = split.get(0).unwrap().parse::<f64>().unwrap();
                let y = split.get(1).unwrap().parse::<f64>().unwrap();
                training_examples.push((x, y));
            }
        }
        training_examples
    }

    // Cost function problem from Ng.
    #[test]
    fn test_cost_1() {
        // TODO switch to a column vector rather than a row vector?
        // can probably wait until I move to a real linear algebra library.
        let test_data = load_testing_data();

        let theta_1 = vec![0.0, 0.0];
        // TODO use `add_bias_term_for_all` once it's written instead of this.
        let mut x: Vec<Vec<f64>> = test_data
            .iter()
            .map(|&(x, y)| vec![1.0, x])
            .collect();
        let y: Vec<f64> = test_data
            .iter()
            .map(|&(x, y)| y)
            .collect();
        let m = x.len() as i32;

        let expected_1 = 32.07;
        let actual_1 = mean_squared_error(m, &theta_1, &x, &y);

        // TODO crappy assert_almost_equal, use a real function for it.
        assert!((actual_1 - expected_1).abs() < 0.1);

        let theta_2 = vec![-1.0, 2.0];
        let expected_2 = 54.24;
        let actual_2 = mean_squared_error(m, &theta_2, &x, &y);

        // TODO crappy assert_almost_equal, use a real function for it.
        assert!((actual_2 - expected_2).abs() < 0.01);
    }
}
