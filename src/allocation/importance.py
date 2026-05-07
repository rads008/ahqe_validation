import numpy as np
from sklearn.feature_selection import mutual_info_classif


def estimate_importance(X, y, n_groups):
    """
    Estimate feature importance using mutual information and group averaging.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target labels
    n_groups : int
        Number of groups to partition features into
        
    Returns
    -------
    phi : ndarray, shape (n_groups,)
        Normalized importance scores, sum(phi) = 1
    features_per_group : list of int, length n_groups
        Number of features in each group
        
    Notes
    -----
    - Features are split into roughly equal groups
    - If n_features % n_groups != 0, earlier groups get one extra feature
    - Mutual information is computed per feature, then averaged within groups
    - Random state is fixed for determinism
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n_samples, n_features = X.shape
    
    if n_groups > n_features:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed n_features ({n_features})")
    
    # Compute mutual information for all features
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Partition features into groups
    features_per_group_base = n_features // n_groups
    remainder = n_features % n_groups
    
    group_importances = []
    features_per_group = []
    start_idx = 0
    
    for group_idx in range(n_groups):
        # First 'remainder' groups get one extra feature
        group_size = features_per_group_base + (1 if group_idx < remainder else 0)
        features_per_group.append(group_size)
        end_idx = start_idx + group_size
        
        # Average MI scores within this group
        group_mi = np.mean(mi_scores[start_idx:end_idx])
        group_importances.append(group_mi)
        
        start_idx = end_idx
    
    # Normalize to sum to 1
    phi = np.array(group_importances)
    phi_sum = phi.sum()
    
    if phi_sum > 0:
        phi = phi / phi_sum
    else:
        # If all MI scores are zero, use uniform distribution
        phi = np.ones(n_groups) / n_groups
    
    return phi, features_per_group