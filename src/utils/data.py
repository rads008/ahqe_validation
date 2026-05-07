"""
Data loading and preprocessing utilities for quantum circuit experiments.

Provides simple functions to load and preprocess datasets with optional
scaling to [0, pi] range for quantum feature encoding.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings


def load_dataset(name: str, test_size=0.2, random_state=42, scale_to_pi=True):
    """
    Load and preprocess a dataset.
    
    Parameters
    ----------
    name : str
        Dataset name. Options: 'breast_cancer', 'ionosphere'
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    scale_to_pi : bool, default=True
        If True, scale features to [0, pi] range after standardization
        
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
        Split and preprocessed datasets
        
    Raises
    ------
    ValueError
        If dataset name is not recognized
    """
    name = name.lower().strip()
    
    # Load the dataset
    if name == 'breast_cancer':
        X, y = _load_breast_cancer()
    elif name == 'ionosphere':
        X, y = _load_ionosphere()
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                        f"Available: 'breast_cancer', 'ionosphere'")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Optionally scale to [0, pi] range
    if scale_to_pi:
        X_train = _scale_to_range(X_train, 0, np.pi)
        X_test = _scale_to_range(X_test, 0, np.pi)
    
    return X_train, X_test, y_train, y_test


def _load_breast_cancer():
    """Load breast cancer dataset from sklearn."""
    data = load_breast_cancer()
    return data.data, data.target


def _load_ionosphere():
    """
    Load ionosphere dataset from UCI repository.
    
    Returns
    -------
    X, y : numpy arrays
        Features and labels
    """
    try:
        import pandas as pd
        from urllib.request import urlretrieve
        import os
        
        # URL for ionosphere dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        
        # Try to download
        temp_file = "/tmp/ionosphere.data"
        try:
            urlretrieve(url, temp_file)
            df = pd.read_csv(temp_file, header=None)
            os.remove(temp_file)
        except Exception:
            # If download fails, use pandas to read directly
            df = pd.read_csv(url, header=None)
        
        # Last column is the label (g/b)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Convert labels to binary (g=1, b=0)
        y = (y == 'g').astype(int)
        
        return X, y
        
    except Exception as e:
        warnings.warn(f"Could not load ionosphere dataset: {e}")
        raise ValueError("Ionosphere dataset unavailable. Try 'breast_cancer' instead.")


def _scale_to_range(X, min_val=0, max_val=np.pi):
    """
    Scale features to a specified range.
    
    Applies min-max scaling to map features from their current range
    to [min_val, max_val].
    
    Parameters
    ----------
    X : numpy array
        Input features
    min_val : float, default=0
        Minimum value of output range
    max_val : float, default=pi
        Maximum value of output range
        
    Returns
    -------
    X_scaled : numpy array
        Scaled features in range [min_val, max_val]
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    
    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    
    # Scale to [0, 1] then to [min_val, max_val]
    X_scaled = (X - X_min) / X_range
    X_scaled = X_scaled * (max_val - min_val) + min_val
    
    return X_scaled


if __name__ == "__main__":
    print("=" * 60)
    print("Testing data loading module")
    print("=" * 60)
    
    # Test breast cancer dataset
    print("\n1. Loading Breast Cancer dataset...")
    X_train, X_test, y_train, y_test = load_dataset(
        'breast_cancer', 
        scale_to_pi=True
    )
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   Labels: {np.unique(y_train)}")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Test without pi scaling
    print("\n2. Loading Breast Cancer without pi scaling...")
    X_train2, X_test2, y_train2, y_test2 = load_dataset(
        'breast_cancer', 
        scale_to_pi=False
    )
    print(f"   Feature range: [{X_train2.min():.3f}, {X_train2.max():.3f}]")
    print(f"   Mean: {X_train2.mean():.3f}, Std: {X_train2.std():.3f}")
    
    # Test ionosphere dataset
    print("\n3. Loading Ionosphere dataset...")
    try:
        X_train3, X_test3, y_train3, y_test3 = load_dataset(
            'ionosphere',
            scale_to_pi=True
        )
        print(f"   Training set: {X_train3.shape}")
        print(f"   Test set: {X_test3.shape}")
        print(f"   Feature range: [{X_train3.min():.3f}, {X_train3.max():.3f}]")
        print(f"   Labels: {np.unique(y_train3)}")
        print(f"   Class distribution: {np.bincount(y_train3)}")
    except ValueError as e:
        print(f"   Warning: {e}")
    
    # Test error handling
    print("\n4. Testing error handling...")
    try:
        load_dataset('nonexistent')
    except ValueError as e:
        print(f"   Expected error caught: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)