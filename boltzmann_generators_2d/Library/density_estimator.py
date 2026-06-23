import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def density_estimator(x, weights=None, k=10, optimize=False):
    """
    Fits a kernel density estimator to input data samples.

    Parameters
    ----------
    x : np.array
        Input data.
    weights : np.array
        Sample weights attached to x.
    k : int
        Number of folds for cross-validation bandwidth search (only used when optimize=True).
    optimize : bool
        If True, use k-fold cross-validation to select the optimal bandwidth.

    Returns
    -------
    estimator : KernelDensity
    """
    if optimize is True:
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=KFold(n_splits=k))
        grid.fit(x[:, None], sample_weight=weights)
        estimator = grid.best_estimator_
    else:
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde.fit(x[:, None], sample_weight=weights)
        estimator = kde

    return estimator
