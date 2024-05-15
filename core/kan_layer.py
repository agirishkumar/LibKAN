# core/kan_layer.py
import torch
import torch.nn as nn
from .b_splines import b_spline_basis

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3, num_knots=10):
        """
        Constructor for the KANLayer class.

        Parameters:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            degree (int): Degree of the B-spline.
            num_knots (int): Number of knots.

        Returns:
            None
        """
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.num_knots = num_knots
        
        # Define knot vector and control points for each edge
        self.knots = nn.Parameter(torch.linspace(0, 1, num_knots + degree + 1))
        self.control_points = nn.Parameter(torch.randn(out_features, in_features, num_knots + degree))

    def forward(self, x):
        batch_size, _ = x.size()
        outputs = torch.zeros(batch_size, self.out_features)
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                spline_basis = b_spline_basis(x[:, j], self.knots, self.degree)
                spline_value = torch.matmul(spline_basis, self.control_points[i, j, :])
                outputs[:, i] += spline_value
        
        return outputs
