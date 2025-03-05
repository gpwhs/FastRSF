import numpy as np
import numbers  # Add this import
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from sksurv.base import SurvivalAnalysisMixin
from sksurv.util import check_array_survival
from sksurv.functions import StepFunction


class FastRandomSurvivalForest(BaseEstimator, SurvivalAnalysisMixin):
    """Fast implementation of Random Survival Forest with optimized memory usage.

    This implementation avoids scikit-learn's tree structure entirely and uses a
    custom tree implementation specifically designed for survival analysis, with
    minimal memory overhead.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    max_depth : int or None, default=None
        Maximum depth of trees. If None, nodes are expanded until all leaves are pure
        or until all leaves contain less than min_samples_split samples.

    min_samples_split : int or float, default=6
        Minimum number of samples required to split a node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and
          ceil(min_samples_split * n_samples) are the minimum number of samples.

    min_samples_leaf : int or float, default=3
        Minimum number of samples required at a leaf node:
        - If int, then consider min_samples_leaf as the minimum number.
        - If float, then min_samples_leaf is a fraction and
          ceil(min_samples_leaf * n_samples) are the minimum number of samples.

    max_features : {'sqrt', 'log2', None} or int or float, default='sqrt'
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          max(1, int(max_features * n_features)) features are considered.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.

    bootstrap : bool, default=True
        Whether to use bootstrap sampling when building trees.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization performance.

    random_state : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the sampling of the features to consider when
        looking for the best split at each node.

    n_jobs : int, default=1
        The number of jobs to run in parallel. fit, predict, and predict_survival_function
        are parallelized over the trees.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimators_ : list
        Collection of fitted sub-estimators (SurvivalTree instances).

    unique_times_ : ndarray
        Unique time points in the training data.

    oob_score_ : float
        Score of the training dataset obtained using out-of-bag samples.
        Only available if oob_score is True.

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.
    """

    _parameter_constraints = {
        "n_estimators": [Interval(numbers.Integral, 1, None, closed="left")],
        "max_depth": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(numbers.Integral, 2, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
        ],
        "min_samples_leaf": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 0.5, closed="right"),
        ],
        "max_features": [
            Interval(numbers.Integral, 1, None, closed="left"),
            Interval(numbers.Real, 0, 1, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "bootstrap": ["boolean"],
        "oob_score": ["boolean"],
        "random_state": ["random_state"],
        "n_jobs": [Interval(numbers.Integral, 1, None, closed="left")],
        "verbose": [Interval(numbers.Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        oob_score=False,
        random_state=None,
        n_jobs=1,
        verbose=0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._predict_risk_score = True

    def _set_max_features(self, n_features):
        """Set self.max_features_ based on given n_features."""
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(n_features)))
        elif self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:  # float
            max_features = max(1, int(self.max_features * n_features))

        return max_features

    def fit(self, X, y):
        """Build a forest of survival trees from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : structured array
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        # Convert pandas DataFrame to numpy array if necessary
        if hasattr(X, "values"):
            X = X.values

        # Ensure X is a 2D array
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Check and extract event and time information
        event, time = check_array_survival(X, y)
        n_samples, n_features = X.shape

        # Get maximum number of features to consider for splits
        self.max_features_ = self._set_max_features(n_features)

        # Extract all unique event times for the entire dataset
        event_times = time[event]
        self.unique_times_ = np.unique(event_times)
        self.n_outputs_ = len(self.unique_times_)

        # Generate random states for each tree
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        # Collect bootstrapped indices for OOB score if required
        if self.bootstrap and self.oob_score:
            self.all_indices_ = []

        # Train trees in parallel
        trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_tree)(
                X,
                event,
                time,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                self.max_features_,
                self.bootstrap,
                seed,
                i,
            )
            for i, seed in enumerate(seeds)
        )

        self.estimators_ = [tree[0] for tree in trees]

        # Calculate OOB score if requested
        if self.bootstrap and self.oob_score:
            oob_indices = [tree[1] for tree in trees]
            self.oob_score_ = self._compute_oob_score(X, event, time, oob_indices)

        return self

    def _fit_tree(
        self,
        X,
        event,
        time,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap,
        seed,
        tree_idx,
    ):
        """Build a single tree for the forest.

        Returns
        -------
        tree : SurvivalTree
            The fitted tree
        oob_indices : ndarray or None
            Out-of-bag indices if bootstrap=True and oob_score=True
        """
        rng = check_random_state(seed)
        n_samples = X.shape[0]

        if bootstrap:
            indices = rng.randint(0, n_samples, n_samples)
            # Get OOB samples if requested
            if self.oob_score:
                unsampled_mask = np.ones(n_samples, dtype=bool)
                unsampled_mask[indices] = False
                oob_indices = np.arange(n_samples)[unsampled_mask]
            else:
                oob_indices = None

            sample_X = X[indices]
            sample_event = event[indices]
            sample_time = time[indices]
        else:
            sample_X = X
            sample_event = event
            sample_time = time
            oob_indices = None

        # Create and fit a tree
        tree = SurvivalTree(
            unique_times=self.unique_times_,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=rng,
        )
        tree.fit(sample_X, sample_event, sample_time)

        return tree, oob_indices

    def _compute_oob_score(self, X, event, time, oob_indices):
        """Compute out-of-bag concordance index score."""
        from sksurv.metrics import concordance_index_censored

        n_samples = X.shape[0]
        all_preds = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)

        # Get predictions from each tree for its OOB samples
        for i, (tree, oob_idx) in enumerate(zip(self.estimators_, oob_indices)):
            if len(oob_idx) > 0:
                tree_pred = tree.predict(X[oob_idx])
                all_preds[oob_idx] += tree_pred
                n_predictions[oob_idx] += 1

        # Apply correction for samples with no OOB predictions
        mask = n_predictions > 0
        all_preds[mask] /= n_predictions[mask]
        all_preds[~mask] = np.nan

        # Drop samples with no predictions
        good_idx = ~np.isnan(all_preds)
        result = concordance_index_censored(
            event[good_idx], time[good_idx], all_preds[good_idx]
        )

        return result[0]

    def predict(self, X):
        """Predict risk scores.

        Higher scores indicate shorter survival (higher risk).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores.
        """
        check_is_fitted(self, "estimators_")

        # Convert pandas DataFrame to numpy array if necessary
        if hasattr(X, "values"):
            X = X.values

        # Ensure X is a 2D array with the right data type
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Make predictions from all trees in parallel
        all_preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(tree.predict)(X) for tree in self.estimators_
        )

        # Average the predictions
        return np.mean(all_preds, axis=0)

    def predict_survival_function(self, X, return_array=False):
        """Predict survival function.

        For each tree in the ensemble, the survival function for an individual
        with feature vector x is computed from all samples in the leaf node
        where x falls. It is estimated by the Kaplan-Meier estimator.
        The ensemble survival function is the average of the individual tree
        predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        return_array : bool, default=False
            If True, return an array with the probability of survival for each
            `self.unique_times_`, otherwise an array of StepFunction.

        Returns
        -------
        survival : ndarray
            If `return_array` is True, an array with the probability of survival
            for each `self.unique_times_`. Otherwise, an array of length n_samples
            of StepFunction instances.
        """
        check_is_fitted(self, "estimators_")

        # Convert pandas DataFrame to numpy array if necessary
        if hasattr(X, "values"):
            X = X.values

        # Ensure X is a 2D array with the right data type
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get survival function predictions from all trees
        surv_preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(tree.predict_survival_function)(X) for tree in self.estimators_
        )

        # Average the predictions
        n_samples = X.shape[0]
        avg_surv = np.zeros((n_samples, len(self.unique_times_)))

        for i in range(n_samples):
            for tree_preds in surv_preds:
                avg_surv[i] += tree_preds[i]

        avg_surv /= len(self.estimators_)

        if return_array:
            return avg_surv

        # Convert to step functions
        return self._array_to_step_function(avg_surv)

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazard function.

        The cumulative hazard function for an individual with feature vector x
        is computed from all samples in the leaf node where x falls.
        It is estimated by the Nelson–Aalen estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        return_array : bool, default=False
            If True, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of StepFunction.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is True, an array with the cumulative hazard rate
            for each `self.unique_times_`. Otherwise, an array of length n_samples
            of StepFunction instances.
        """
        check_is_fitted(self, "estimators_")

        # Get survival function and convert to cumulative hazard
        surv = self.predict_survival_function(X, return_array=True)
        chf = -np.log(np.clip(surv, 1e-16, 1.0))  # Protect against log(0)

        if return_array:
            return chf

        return self._array_to_step_function(chf)

    def _array_to_step_function(self, array):
        """Convert array of values to array of step functions."""
        funcs = np.empty(array.shape[0], dtype=object)
        for i in range(array.shape[0]):
            funcs[i] = StepFunction(x=self.unique_times_, y=array[i])
        return funcs


class SurvivalTree:
    """Custom implementation of a survival tree.

    This is a custom implementation designed for memory efficiency and
    performance, optimized specifically for survival analysis.

    Parameters
    ----------
    unique_times : ndarray
        Unique event times in the dataset.

    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.

    min_samples_split : int or float, default=6
        Minimum number of samples required to split a node.

    min_samples_leaf : int or float, default=3
        Minimum number of samples required at a leaf node.

    max_features : int, default=None
        Number of features to consider for best split. If None,
        use all features.

    random_state : int or RandomState, default=None
        Controls the randomness in the feature selection process.
    """

    class Node:
        """Tree node data structure."""

        def __init__(self, n_samples=0):
            self.left = None
            self.right = None
            self.feature = None
            self.threshold = None
            self.n_samples = n_samples
            self.is_leaf = False
            self.survival_curves = (
                None  # Will hold KM estimates for samples in this node
            )
            self.depth = 0

    def __init__(
        self,
        unique_times,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features=None,
        random_state=None,
    ):
        self.unique_times = unique_times
        self.max_depth = max_depth if max_depth is not None else float("inf")
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = check_random_state(random_state)
        self.n_outputs_ = len(unique_times)
        self.root = None

    def fit(self, X, event, time):
        """Build the tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        event : array-like of shape (n_samples,)
            Binary event indicator.

        time : array-like of shape (n_samples,)
            Time of event or censoring.

        Returns
        -------
        self : object
            Fitted tree.
        """
        self.n_features_in_ = X.shape[1]

        # Convert min_samples_split and min_samples_leaf to absolute counts
        n_samples = X.shape[0]
        if isinstance(self.min_samples_split, float):
            self.min_samples_split_ = max(
                2, int(np.ceil(self.min_samples_split * n_samples))
            )
        else:
            self.min_samples_split_ = self.min_samples_split

        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf_ = max(
                1, int(np.ceil(self.min_samples_leaf * n_samples))
            )
        else:
            self.min_samples_leaf_ = self.min_samples_leaf

        # Create the root node and recursively build the tree
        self.root = self.Node(n_samples=n_samples)
        indices = np.arange(n_samples)

        # Build the tree recursively
        self._build_tree(self.root, X, event, time, indices, depth=0)

        return self

    def _build_tree(self, node, X, event, time, indices, depth):
        """Recursively build the tree."""
        n_samples = len(indices)
        node.depth = depth

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split_
            or n_samples < 2 * self.min_samples_leaf_
            or np.all(event[indices] == 0)
        ):  # All censored

            # Make this a leaf node
            node.is_leaf = True

            # Compute Kaplan-Meier estimate for this node
            node.survival_curves = self._compute_survival_curves(
                event[indices], time[indices]
            )

            return

        # Find the best split
        feature, threshold, improvement = self._find_best_split(X, event, time, indices)

        # If no good split was found, make this a leaf
        if feature is None or improvement <= 0:
            node.is_leaf = True
            node.survival_curves = self._compute_survival_curves(
                event[indices], time[indices]
            )
            return

        # Apply the split
        node.feature = feature
        node.threshold = threshold

        # Split the data
        left_indices = indices[X[indices, feature] <= threshold]
        right_indices = indices[X[indices, feature] > threshold]

        # Check if split is valid (enough samples in each child)
        if (
            len(left_indices) < self.min_samples_leaf_
            or len(right_indices) < self.min_samples_leaf_
        ):
            node.is_leaf = True
            node.survival_curves = self._compute_survival_curves(
                event[indices], time[indices]
            )
            return

        # Create child nodes
        node.left = self.Node(n_samples=len(left_indices))
        node.right = self.Node(n_samples=len(right_indices))

        # Recursively build the tree
        self._build_tree(node.left, X, event, time, left_indices, depth + 1)
        self._build_tree(node.right, X, event, time, right_indices, depth + 1)

    def _find_best_split(self, X, event, time, indices):
        """Find the best split for a node."""
        n_samples = len(indices)

        # Randomly select a subset of features to consider
        n_features = X.shape[1]
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = self.random_state.choice(
                n_features, self.max_features, replace=False
            )
        else:
            feature_indices = np.arange(n_features)

        # Initialize
        best_feature = None
        best_threshold = None
        best_improvement = -float("inf")

        # Compute parent node score
        parent_score = self._compute_node_score(event[indices], time[indices])

        # Try all features
        for feature in feature_indices:
            feature_values = X[indices, feature]

            # Get unique feature values as potential thresholds
            # For efficiency, we use a subset of thresholds for features with many unique values
            unique_values = np.unique(feature_values)
            if len(unique_values) > 10:
                potential_thresholds = np.percentile(
                    unique_values, np.linspace(0, 100, 10)
                )
            else:
                potential_thresholds = unique_values

            # Try all thresholds
            for threshold in potential_thresholds:
                # Split the node
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Check if split is valid
                if (
                    np.sum(left_mask) < self.min_samples_leaf_
                    or np.sum(right_mask) < self.min_samples_leaf_
                ):
                    continue

                # Compute improvement
                left_score = self._compute_node_score(
                    event[indices][left_mask], time[indices][left_mask]
                )
                right_score = self._compute_node_score(
                    event[indices][right_mask], time[indices][right_mask]
                )

                # Weighted sum of child scores
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                improvement = parent_score - (
                    n_left / n_samples * left_score + n_right / n_samples * right_score
                )

                # Update best if improvement is better
                if improvement > best_improvement:
                    best_feature = feature
                    best_threshold = threshold
                    best_improvement = improvement

        return best_feature, best_threshold, best_improvement

    def _compute_node_score(self, event, time):
        """Compute log-rank statistic for a node.

        This is a simplified version of the log-rank test that
        serves as a splitting criterion.
        """
        if len(event) <= 1 or np.sum(event) == 0:
            return 0

        # Get risk groups for each time point
        sorted_idx = np.argsort(time)
        sorted_time = time[sorted_idx]
        sorted_event = event[sorted_idx]

        # Count events and at-risk for each time point
        unique_times = np.unique(sorted_time[sorted_event == 1])
        n_events = np.zeros_like(unique_times, dtype=float)
        n_at_risk = np.zeros_like(unique_times, dtype=float)

        for i, t in enumerate(unique_times):
            # Count events at this time
            n_events[i] = np.sum(sorted_event[sorted_time == t])
            # Count at risk at this time (including this time)
            n_at_risk[i] = np.sum(sorted_time >= t)

        # Simplified log-rank statistic
        if np.any(n_at_risk == 0):
            return 0

        hazard = n_events / n_at_risk
        risk_score = -np.sum(np.log(1 - hazard))

        return risk_score

    def _compute_survival_curves(self, event, time):
        """Compute Kaplan-Meier survival curves for samples in a node."""
        # If all samples are censored, return all 1's
        if np.sum(event) == 0:
            return np.ones(len(self.unique_times))

        # Sort by time
        sorted_idx = np.argsort(time)
        sorted_time = time[sorted_idx]
        sorted_event = event[sorted_idx]

        # Compute survival function using Kaplan-Meier
        survival = np.ones(len(self.unique_times))
        at_risk = len(sorted_time)

        last_i = 0
        for i, t in enumerate(self.unique_times):
            # Find events that occurred at or before this time
            while last_i < len(sorted_time) and sorted_time[last_i] <= t:
                if sorted_event[last_i]:
                    # Event occurred at this time
                    survival[i:] *= 1 - 1 / at_risk
                at_risk -= 1
                last_i += 1

        return survival

    def predict(self, X):
        """Predict risk scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores.
        """
        # Apply tree to get leaf nodes for each sample
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            # Find the leaf node for this sample
            node = self._apply_tree(self.root, X[i])

            # The risk score is the average cumulative hazard
            # (higher score = higher risk of event)
            survival = node.survival_curves
            cum_hazard = -np.log(np.clip(survival, 1e-16, 1.0))
            predictions[i] = np.mean(cum_hazard)

        return predictions

    def predict_survival_function(self, X):
        """Predict survival function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        survival_funcs : ndarray of shape (n_samples, n_times)
            Predicted survival functions.
        """
        n_samples = X.shape[0]
        survival = np.zeros((n_samples, len(self.unique_times)))

        for i in range(n_samples):
            # Find the leaf node for this sample
            node = self._apply_tree(self.root, X[i])
            survival[i] = node.survival_curves

        return survival

    def _apply_tree(self, node, sample):
        """Apply the tree to a single sample, returning the leaf node."""
        if node.is_leaf:
            return node

        if sample[node.feature] <= node.threshold:
            return self._apply_tree(node.left, sample)
        else:
            return self._apply_tree(node.right, sample)
