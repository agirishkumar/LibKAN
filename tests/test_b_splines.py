
import torch
from core.b_splines import b_spline_basis

def test_b_spline_basis():
    # Define test parameters
    x = torch.tensor([0.1, 0.4, 0.5, 0.9])
    knots = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    degree = 2
    
    # Compute B-spline basis functions
    basis = b_spline_basis(x, knots, degree)
    
    # Print the results
    print("B-spline basis functions:")
    print(basis)
    
    # Add more checks as needed
    assert basis.shape == (4, len(knots) - degree - 1), "Basis function shape mismatch"

if __name__ == "__main__":
    test_b_spline_basis()
