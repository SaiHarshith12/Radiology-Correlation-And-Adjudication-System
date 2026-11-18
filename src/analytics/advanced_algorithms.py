"""Small NumPy implementations used by the app's exploratory algorithm lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REPORT_STATES = ("PRESENT", "ABSENT", "NOT_MENTIONED")


def build_report_feature_matrix(
    labels: pd.DataFrame,
    conditions: list[str],
) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    """Create one one-hot report feature row per report ID."""
    if isinstance(conditions, str):
        conditions = [conditions]
    else:
        conditions = list(conditions)

    if not conditions:
        raise ValueError("At least one condition is required.")

    deduplicated = list(dict.fromkeys(conditions))
    if len(deduplicated) != len(conditions):
        raise ValueError("Condition names must be unique.")

    required = {"report_id", *deduplicated}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(
            "The labeled report dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    reports = labels.loc[:, ["report_id", *deduplicated]].copy()
    reports = reports.dropna(subset=["report_id"])
    if reports.empty:
        raise ValueError("The labeled report dataset has no report IDs.")

    for condition in deduplicated:
        reports[condition] = (
            reports[condition]
            .fillna("NOT_MENTIONED")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        invalid = sorted(
            set(reports[condition].unique()).difference(REPORT_STATES)
        )
        if invalid:
            raise ValueError(
                f"Unexpected labels for {condition}: {', '.join(invalid)}"
            )

    reports = reports.groupby("report_id", sort=True)[deduplicated].agg(
        lambda values: values.mode().iloc[0]
    )

    feature_columns: dict[str, pd.Series] = {}
    for condition in conditions:
        for state in REPORT_STATES:
            feature_columns[f"{condition} — {state.lower()}"] = (
                reports[condition] == state
            ).astype(float)

    features = pd.DataFrame(feature_columns, index=reports.index)
    return reports.index.astype(str).tolist(), features, features.to_numpy()


def kmeans(
    values: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 100,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster finite, two-dimensional numeric observations with k-means++."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("K-means expects a non-empty 2D feature matrix.")
    if not np.isfinite(data).all():
        raise ValueError("K-means features must contain only finite values.")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("The number of clusters must be a positive integer.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")

    unique_rows = np.unique(data, axis=0)
    if n_clusters > len(unique_rows):
        raise ValueError(
            "The number of clusters cannot exceed the number of unique "
            "feature profiles."
        )

    rng = np.random.default_rng(random_state)
    centers = [unique_rows[rng.integers(len(unique_rows))]]
    while len(centers) < n_clusters:
        distances = np.min(
            np.sum((unique_rows[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total_distance = distances.sum()
        if total_distance == 0:
            remaining = [
                row for row in unique_rows
                if not any(np.array_equal(row, center) for center in centers)
            ]
            centers.append(remaining[0])
        else:
            centers.append(
                unique_rows[rng.choice(len(unique_rows), p=distances / total_distance)]
            )
    centers_array = np.asarray(centers, dtype=float)

    labels = np.zeros(data.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = np.sum((data[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers_array.copy()
        nearest_distances = distances[np.arange(len(data)), new_labels]
        counts = np.bincount(new_labels, minlength=n_clusters)

        for cluster in range(n_clusters):
            if counts[cluster]:
                new_centers[cluster] = data[new_labels == cluster].mean(axis=0)
            else:
                farthest = int(np.argmax(nearest_distances))
                new_centers[cluster] = data[farthest]
                new_labels[farthest] = cluster
                nearest_distances[farthest] = -1

        converged = np.array_equal(new_labels, labels) and np.allclose(
            new_centers, centers_array, atol=tolerance, rtol=0
        )
        labels = new_labels
        centers_array = new_centers
        if converged:
            break

    counts = np.bincount(labels, minlength=n_clusters)
    for cluster in np.flatnonzero(counts == 0):
        distances = np.sum((data - centers_array[labels]) ** 2, axis=1)
        movable = counts[labels] > 1
        distances[~movable] = -1
        point = int(np.argmax(distances))
        counts[labels[point]] -= 1
        labels[point] = cluster
        counts[cluster] += 1

    for cluster in range(n_clusters):
        centers_array[cluster] = data[labels == cluster].mean(axis=0)
    inertia = float(np.sum((data - centers_array[labels]) ** 2))
    return labels, centers_array, inertia


def pca(
    values: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project finite numeric observations onto their principal components."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] == 0:
        raise ValueError("PCA expects at least two rows and one feature.")
    if not np.isfinite(data).all():
        raise ValueError("PCA features must contain only finite values.")
    max_components = min(data.shape)
    if (
        not isinstance(n_components, (int, np.integer))
        or n_components < 1
        or n_components > max_components
    ):
        raise ValueError(
            f"n_components must be between 1 and {max_components}."
        )

    centered = data - data.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    variance = singular_values**2
    total_variance = variance.sum()
    if total_variance == 0:
        raise ValueError("PCA cannot project a cohort with no feature variance.")

    components = right_vectors[:n_components]
    scores = centered @ components.T
    explained_variance_ratio = variance[:n_components] / total_variance
    return scores, explained_variance_ratio


def hmm_forward_backward(
    observations: list[int],
    initial_probabilities: np.ndarray,
    transition_probabilities: np.ndarray,
    emission_probabilities: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return per-step hidden-state posteriors and sequence log likelihood."""
    initial = np.asarray(initial_probabilities, dtype=float)
    transitions = np.asarray(transition_probabilities, dtype=float)
    emissions = np.asarray(emission_probabilities, dtype=float)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("Initial probabilities must be a non-empty vector.")
    n_states = initial.size
    if transitions.shape != (n_states, n_states):
        raise ValueError("Transition probabilities must be a square state matrix.")
    if (
        emissions.ndim != 2
        or emissions.shape[0] != n_states
        or emissions.shape[1] == 0
    ):
        raise ValueError("Emission probabilities must have one row per state.")

    if np.isscalar(observations):
        observations = [observations]
    else:
        observations = list(observations)

    if len(observations) == 0:
        raise ValueError("At least one observation is required.")
    if any(
        not isinstance(value, (int, np.integer))
        or value < 0
        or value >= emissions.shape[1]
        for value in observations
    ):
        raise ValueError("An observation index is outside the emission matrix.")

    for name, probabilities, axis in (
        ("initial", initial, None),
        ("transition", transitions, 1),
        ("emission", emissions, 1),
    ):
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(f"{name.capitalize()} probabilities must be finite and non-negative.")
        sums = probabilities.sum() if axis is None else probabilities.sum(axis=axis)
        if not np.allclose(sums, 1.0):
            raise ValueError(f"{name.capitalize()} probabilities must sum to 1.")

    length = len(observations)
    forward = np.zeros((length, n_states), dtype=float)
    scales = np.zeros(length, dtype=float)
    forward[0] = initial * emissions[:, observations[0]]
    scales[0] = forward[0].sum()
    if scales[0] == 0:
        raise ValueError("The observation sequence has zero probability.")
    forward[0] /= scales[0]

    for time in range(1, length):
        forward[time] = (
            forward[time - 1] @ transitions
        ) * emissions[:, observations[time]]
        scales[time] = forward[time].sum()
        if scales[time] == 0:
            raise ValueError("The observation sequence has zero probability.")
        forward[time] /= scales[time]

    backward = np.ones((length, n_states), dtype=float)
    for time in range(length - 2, -1, -1):
        backward[time] = transitions @ (
            emissions[:, observations[time + 1]] * backward[time + 1]
        )
        backward[time] /= scales[time + 1]

    posteriors = forward * backward
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    return posteriors, float(np.log(scales).sum())


def binary_sum_product(
    prior_probability: float,
    likelihood_factors: list[np.ndarray],
) -> np.ndarray:
    """Combine likelihood messages for a binary variable and normalize."""
    if not np.isfinite(prior_probability) or not 0 <= prior_probability <= 1:
        raise ValueError("The prior probability must be between 0 and 1.")

    messages = [np.array([1 - prior_probability, prior_probability])]
    for factor in likelihood_factors:
        likelihood = np.asarray(factor, dtype=float)
        if (
            likelihood.shape != (2,)
            or not np.isfinite(likelihood).all()
            or np.any(likelihood < 0)
        ):
            raise ValueError(
                "Each likelihood factor must be a finite, non-negative "
                "two-value vector."
            )
        messages.append(likelihood)

    unnormalized = np.prod(np.vstack(messages), axis=0)
    normalizer = unnormalized.sum()
    if normalizer == 0:
        raise ValueError("The evidence has zero probability under the model.")
    return unnormalized / normalizer


def build_supervised_report_dataset(
    labels: pd.DataFrame,
    target_condition: str,
    feature_conditions: list[str],
) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    """Build report-level predictors and binary labels for explicit findings."""
    features_requested = list(feature_conditions)
    if not features_requested:
        raise ValueError("At least one predictor condition is required.")
    if target_condition in features_requested:
        raise ValueError("The target condition cannot also be a predictor.")
    if len(set(features_requested)) != len(features_requested):
        raise ValueError("Predictor condition names must be unique.")

    required = {"report_id", target_condition, *features_requested}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(
            "The labeled report dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    reports = labels.loc[:, ["report_id", target_condition, *features_requested]]
    reports = reports.dropna(subset=["report_id"]).copy()
    if reports.empty:
        raise ValueError("The labeled report dataset has no report IDs.")

    for condition in [target_condition, *features_requested]:
        reports[condition] = (
            reports[condition]
            .fillna("NOT_MENTIONED")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        invalid = sorted(
            set(reports[condition].unique()).difference(REPORT_STATES)
        )
        if invalid:
            raise ValueError(
                f"Unexpected labels for {condition}: {', '.join(invalid)}"
            )

    target_by_report = reports.groupby("report_id", sort=True)[
        target_condition
    ].agg(lambda values: values.mode().iloc[0])
    target_by_report.index = target_by_report.index.astype(str)
    explicit_targets = target_by_report[
        target_by_report.isin(("PRESENT", "ABSENT"))
    ]
    if explicit_targets.empty:
        raise ValueError(
            f"No reports have an explicit PRESENT or ABSENT label for "
            f"{target_condition}."
        )

    _, feature_frame, _ = build_report_feature_matrix(
        reports,
        features_requested,
    )
    feature_frame.index = feature_frame.index.astype(str)
    report_ids = explicit_targets.index.tolist()
    feature_frame = feature_frame.reindex(report_ids)
    if feature_frame.isna().any().any():
        raise ValueError("Could not align report labels with predictor features.")

    targets = (explicit_targets.to_numpy() == "PRESENT").astype(int)
    return report_ids, feature_frame, targets


@dataclass
class _TreeNode:
    prediction: int
    feature: int = -1
    threshold: float = 0.0
    left: _TreeNode | None = None
    right: _TreeNode | None = None


class _CARTClassifier:
    def __init__(self, root: _TreeNode, feature_importances: np.ndarray):
        self.root = root
        self.feature_importances = feature_importances

    def predict(self, values: np.ndarray) -> np.ndarray:
        predictions = []
        for row in values:
            node = self.root
            while node.feature >= 0:
                node = (
                    node.left
                    if row[node.feature] <= node.threshold
                    else node.right
                )
                if node is None:
                    raise RuntimeError("The fitted decision tree is incomplete.")
            predictions.append(node.prediction)
        return np.asarray(predictions, dtype=int)


def _weighted_gini(
    targets: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    total_weight = sample_weights.sum()
    if total_weight <= 0:
        return 0.0
    positive_weight = sample_weights[targets == 1].sum()
    positive_rate = positive_weight / total_weight
    return 2.0 * positive_rate * (1.0 - positive_rate)


def _fit_cart_classifier(
    values: np.ndarray,
    targets: np.ndarray,
    max_depth: int,
    sample_weights: np.ndarray | None = None,
    max_features: int | None = None,
    random_state: int = 42,
) -> _CARTClassifier:
    data = np.asarray(values, dtype=float)
    labels = np.asarray(targets, dtype=int)
    weights = (
        np.ones(len(labels), dtype=float)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    rng = np.random.default_rng(random_state)
    importances = np.zeros(data.shape[1], dtype=float)
    root_weight = weights.sum()

    def grow(indices: np.ndarray, depth: int) -> _TreeNode:
        node_targets = labels[indices]
        node_weights = weights[indices]
        total_weight = node_weights.sum()
        positive_weight = node_weights[node_targets == 1].sum()
        prediction = int(positive_weight >= total_weight - positive_weight)
        node = _TreeNode(prediction=prediction)

        if (
            depth >= max_depth
            or len(indices) < 2
            or np.unique(node_targets).size < 2
            or total_weight <= 0
        ):
            return node

        if max_features is None or max_features >= data.shape[1]:
            candidate_features = np.arange(data.shape[1])
        else:
            candidate_features = np.sort(
                rng.choice(
                    data.shape[1],
                    size=max(1, max_features),
                    replace=False,
                )
            )

        parent_impurity = _weighted_gini(node_targets, node_weights)
        best_gain = 0.0
        best_split = None
        for feature in candidate_features:
            unique_values = np.unique(data[indices, feature])
            if len(unique_values) < 2:
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = data[indices, feature] <= threshold
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]
                if not len(left_indices) or not len(right_indices):
                    continue
                left_weights = weights[left_indices]
                right_weights = weights[right_indices]
                split_impurity = (
                    left_weights.sum()
                    * _weighted_gini(labels[left_indices], left_weights)
                    + right_weights.sum()
                    * _weighted_gini(labels[right_indices], right_weights)
                ) / total_weight
                gain = parent_impurity - split_impurity
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_split = (feature, float(threshold), left_indices, right_indices)

        if best_split is None:
            return node

        feature, threshold, left_indices, right_indices = best_split
        node.feature = int(feature)
        node.threshold = threshold
        node.left = grow(left_indices, depth + 1)
        node.right = grow(right_indices, depth + 1)
        importances[feature] += best_gain * total_weight / root_weight
        return node

    root = grow(np.arange(len(labels)), 0)
    total_importance = importances.sum()
    if total_importance > 0:
        importances /= total_importance
    return _CARTClassifier(root, importances)


def _format_tree_rules(
    node: _TreeNode,
    feature_names: list[str],
    depth: int = 0,
) -> str:
    indent = "  " * depth
    if node.feature < 0:
        label = "PRESENT" if node.prediction else "ABSENT"
        return f"{indent}Predict {label}"
    left_rules = _format_tree_rules(node.left, feature_names, depth + 1)
    right_rules = _format_tree_rules(node.right, feature_names, depth + 1)
    return (
        f"{indent}if {feature_names[node.feature]} <= {node.threshold:.3g}:\n"
        f"{left_rules}\n"
        f"{indent}else:\n"
        f"{right_rules}"
    )


def _format_tree_graph(
    node: _TreeNode,
    feature_names: list[str],
) -> str:
    def quote(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "digraph CART {",
        '  graph [rankdir=TB, bgcolor="transparent"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", color="#526581"];',
        '  edge [fontname="Arial", color="#718096"];',
    ]
    next_id = 0

    def add_node(current: _TreeNode) -> int:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        if current.feature < 0:
            label = "Predict PRESENT" if current.prediction else "Predict ABSENT"
            color = "#d9f2e6" if current.prediction else "#fff0d6"
            lines.append(
                f'  n{node_id} [label="{quote(label)}", fillcolor="{color}"];'
            )
            return node_id

        feature_name = feature_names[current.feature]
        label = f"{feature_name} <= {current.threshold:.3g}?"
        lines.append(
            f'  n{node_id} [label="{quote(label)}", fillcolor="#e7efff"];'
        )
        left_id = add_node(current.left)
        right_id = add_node(current.right)
        lines.append(f'  n{node_id} -> n{left_id} [label="yes"];')
        lines.append(f'  n{node_id} -> n{right_id} [label="no"];')
        return node_id

    add_node(node)
    lines.append("}")
    return "\n".join(lines)


def evaluate_binary_classifier(
    values: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    model_name: str,
    test_fraction: float = 0.2,
    max_depth: int = 4,
    n_estimators: int = 25,
    random_state: int = 42,
) -> dict[str, object]:
    """Train and evaluate CART, bagged trees, or AdaBoost on a stratified holdout."""
    data = np.asarray(values, dtype=float)
    labels = np.asarray(targets)
    names = list(feature_names)
    if data.ndim != 2 or not data.shape[0] or not data.shape[1]:
        raise ValueError("Supervised learning expects a non-empty 2D feature matrix.")
    if labels.ndim != 1 or len(labels) != len(data):
        raise ValueError("The target vector must have one label per feature row.")
    if not np.isfinite(data).all():
        raise ValueError("Supervised learning features must be finite.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Binary classifier targets must be encoded as 0 or 1.")
    labels = labels.astype(int)
    if len(names) != data.shape[1]:
        raise ValueError("Feature names must match the feature matrix columns.")
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    if not np.isfinite(test_fraction) or not 0.1 <= test_fraction <= 0.5:
        raise ValueError("The test fraction must be between 0.1 and 0.5.")
    if not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer.")
    if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")
    if model_name not in ("CART", "Random forest", "AdaBoost"):
        raise ValueError("Choose CART, Random forest, or AdaBoost.")

    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []
    for class_label in (0, 1):
        class_indices = rng.permutation(np.flatnonzero(labels == class_label))
        if len(class_indices) < 2:
            raise ValueError(
                "At least two PRESENT and two ABSENT reports are required."
            )
        test_count = max(
            1,
            min(len(class_indices) - 1, int(round(len(class_indices) * test_fraction))),
        )
        test_parts.append(class_indices[:test_count])
        train_parts.append(class_indices[test_count:])
    train_indices = rng.permutation(np.concatenate(train_parts))
    test_indices = rng.permutation(np.concatenate(test_parts))
    train_values = data[train_indices]
    train_targets = labels[train_indices]

    estimator_count = 1
    tree_rules = None
    tree_graph = None
    if model_name == "CART":
        model = _fit_cart_classifier(
            train_values,
            train_targets,
            max_depth=max_depth,
            random_state=random_state,
        )
        predictions = model.predict(data[test_indices])
        feature_importances = model.feature_importances
        tree_rules = _format_tree_rules(model.root, names)
        tree_graph = _format_tree_graph(model.root, names)
    elif model_name == "Random forest":
        estimator_count = n_estimators
        max_features = max(1, int(np.sqrt(data.shape[1])))
        models = []
        for estimator in range(n_estimators):
            bootstrap_indices = rng.integers(0, len(train_values), len(train_values))
            models.append(
                _fit_cart_classifier(
                    train_values[bootstrap_indices],
                    train_targets[bootstrap_indices],
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=random_state + estimator + 1,
                )
            )
        votes = np.vstack(
            [estimator.predict(data[test_indices]) for estimator in models]
        )
        predictions = (votes.mean(axis=0) >= 0.5).astype(int)
        feature_importances = np.mean(
            [estimator.feature_importances for estimator in models],
            axis=0,
        )
    else:
        signed_targets = np.where(train_targets == 1, 1.0, -1.0)
        sample_weights = np.full(
            len(train_targets),
            1.0 / len(train_targets),
            dtype=float,
        )
        estimators = []
        alphas = []
        for estimator in range(n_estimators):
            weak_learner = _fit_cart_classifier(
                train_values,
                train_targets,
                max_depth=1,
                sample_weights=sample_weights,
                random_state=random_state + estimator,
            )
            weak_predictions = np.where(
                weak_learner.predict(train_values) == 1,
                1.0,
                -1.0,
            )
            error = float(
                sample_weights[weak_predictions != signed_targets].sum()
            )
            if error >= 0.5 - 1e-12:
                break
            error = max(error, 1e-12)
            alpha = 0.5 * np.log((1.0 - error) / error)
            estimators.append(weak_learner)
            alphas.append(alpha)
            if error <= 1e-12:
                break
            sample_weights *= np.exp(
                -alpha * signed_targets * weak_predictions
            )
            sample_weights /= sample_weights.sum()

        estimator_count = len(estimators)
        if estimators:
            scores = sum(
                alpha
                * np.where(estimator.predict(data[test_indices]) == 1, 1.0, -1.0)
                for estimator, alpha in zip(estimators, alphas)
            )
            predictions = (scores >= 0).astype(int)
            feature_importances = np.average(
                [estimator.feature_importances for estimator in estimators],
                axis=0,
                weights=alphas,
            )
        else:
            majority_class = int(np.mean(train_targets) >= 0.5)
            predictions = np.full(len(test_indices), majority_class, dtype=int)
            feature_importances = np.zeros(data.shape[1], dtype=float)
            tree_rules = (
                "No decision stump performed better than chance on the "
                "training split; predictions use the training majority class."
            )

    actual = labels[test_indices]
    true_positive = int(np.sum((actual == 1) & (predictions == 1)))
    true_negative = int(np.sum((actual == 0) & (predictions == 0)))
    false_positive = int(np.sum((actual == 0) & (predictions == 1)))
    false_negative = int(np.sum((actual == 1) & (predictions == 0)))
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "metrics": {
            "Accuracy": float(np.mean(predictions == actual)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Balanced accuracy": float((recall + specificity) / 2.0),
        },
        "confusion_matrix": np.array(
            [[true_negative, false_positive], [false_negative, true_positive]]
        ),
        "y_test": actual,
        "y_pred": predictions,
        "test_indices": test_indices,
        "feature_importances": feature_importances,
        "tree_rules": tree_rules,
        "tree_graph": tree_graph,
        "estimators_used": estimator_count,
    }
