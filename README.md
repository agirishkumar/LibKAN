# LibKAN
A comprehensive library for Kolmogorov-Arnold Networks (KANs) in PyTorch


Lib_KAN
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── b_splines.py         # Spline functions and utilities
│   ├── kan_layer.py         # KAN layer implementation
│   ├── kan_network.py       # KAN network implementation
│   ├── initialization.py    # Initialization methods
│   ├── regularization.py    # Regularization techniques
├── training/
│   ├── __init__.py
│   ├── train.py             # Training loop
│   ├── evaluation.py        # Evaluation metrics
│   ├── callbacks.py         # Callbacks for training
│   ├── optimizers.py        # Optimizers
├── utils/
│   ├── __init__.py
│   ├── visualization.py     # Visualization tools
│   ├── data_handling.py     # Data handling utilities
│   ├── save_load.py         # Save and load models
├── examples/
│   ├── __init__.py
│   ├── example_1.py         # Example usage script
│   ├── example_2.py         # Another example script
├── docs/
│   ├── index.md             # Documentation
│   ├── api_reference.md     # API reference
│   ├── user_guide.md        # User guide
└── tests/
    ├── __init__.py
    ├── test_splines.py      # Unit tests for splines
    ├── test_kan_layer.py    # Unit tests for KAN layer
    ├── test_kan_network.py  # Unit tests for KAN network
