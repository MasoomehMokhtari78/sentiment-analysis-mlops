from src.monitoring.metrics import calculate_accuracy


def test_calculate_accuracy_returns_dict():

    result = calculate_accuracy()

    assert isinstance(result, dict)
    assert "accuracy" in result


def test_accuracy_range():

    result = calculate_accuracy()

    if result["accuracy"] is None:
        assert result["message"] == "no labeled data available"
    else:
        assert 0.0 <= result["accuracy"] <= 1.0