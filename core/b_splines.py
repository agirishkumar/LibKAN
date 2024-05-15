import torch

def b_spline_basis(x, knots, degree):
    """
    Compute B-spline basis functions for a given degree and knot vector.
    :param x: Input tensor of shape (batch_size,).
    :param knots: Knot vector tensor of shape (num_knots,).
    :param degree: Degree of the B-spline.
    :return: B-spline basis functions evaluated at x.
    """

    assert torch.all(knots[:-1] <= knots[1:]), "Knots must be non-decreasing."
    assert len(knots) >= 2 * (degree + 1), "Insufficient knots for given degree."

    # Number of basis functions
    n = len(knots) - 1 - degree
    # Initialize basis functions tensor
    basis = torch.zeros((x.size(0), n + degree + 1, degree + 1), dtype=torch.float32)
    
    # Initialize zero-degree basis functions
    for i in range(n + degree + 1):
        basis[:, i, 0] = ((knots[i] <= x) & (x < knots[i + 1])).float()
    
    # Compute higher-degree basis functions
    for d in range(1, degree + 1):
        for i in range(n + degree + 1 - d):
            left_term = (x - knots[i]) / (knots[i + d] - knots[i] + 1e-8) * basis[:, i, d - 1]
            right_term = (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1] + 1e-8) * basis[:, i + 1, d - 1]
            basis[:, i, d] = left_term + right_term
    
    return basis[:, :, degree]