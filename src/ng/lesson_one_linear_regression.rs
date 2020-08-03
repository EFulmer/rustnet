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
}
