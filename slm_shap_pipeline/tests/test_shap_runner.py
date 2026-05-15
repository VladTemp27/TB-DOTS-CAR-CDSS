import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from slm_shap_pipeline.shap_runner import compute_shap_for_row

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c"]


def test_compute_shap_returns_dict_keyed_by_semantic_names():
    mock_booster = MagicMock()
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3]])

    with patch("slm_shap_pipeline.shap_runner.shap.TreeExplainer", return_value=mock_explainer):
        result = compute_shap_for_row(mock_booster, np.array([[1.0, 2.0, 3.0]]), FEATURE_NAMES)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"feat_a", "feat_b", "feat_c"}
    assert abs(result["feat_a"] - 0.1) < 1e-6
    assert abs(result["feat_b"] - (-0.2)) < 1e-6


def test_compute_shap_handles_binary_output():
    """shap.TreeExplainer may return list of arrays for binary classifiers; take index [1]."""
    mock_booster = MagicMock()
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [
        np.array([[-0.1, 0.2, -0.3]]),
        np.array([[0.1, -0.2, 0.3]]),
    ]

    with patch("slm_shap_pipeline.shap_runner.shap.TreeExplainer", return_value=mock_explainer):
        result = compute_shap_for_row(mock_booster, np.array([[1.0, 2.0, 3.0]]), FEATURE_NAMES)

    assert abs(result["feat_a"] - 0.1) < 1e-6
    assert abs(result["feat_b"] - (-0.2)) < 1e-6
