import io
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

from asreviewcontrib.dory.entrypoint import DoryEntryPoint


def test_execute_cache_calls_cache():
    dory = DoryEntryPoint()
    with patch.object(dory, "cache") as mock_cache:
        dory.execute(["cache", "xgboost", "sbert"])
        mock_cache.assert_called_once_with(["xgboost", "sbert"])


@patch("asreviewcontrib.dory.entrypoint.load_extension")
def test_cache_loads_fe_model(mock_load):
    mock_fe_model = MagicMock()
    mock_fe_model().named_steps = {"sentence_transformer": MagicMock(_model="dummy")}
    mock_load.return_value = mock_fe_model

    dory = DoryEntryPoint()

    captured_output = io.StringIO()
    sys.stdout = captured_output

    dory.cache(["sbert"])

    sys.stdout = sys.__stdout__
    mock_load.assert_called_with("models.feature_extractors", "sbert")
    assert "Loaded FE sbert" in captured_output.getvalue()


@patch("asreviewcontrib.dory.entrypoint.load_extension")
def test_cache_fallback_to_classifier(mock_load):
    mock_load.side_effect = [ValueError("not a FE"), MagicMock()]

    captured_output = io.StringIO()
    sys.stdout = captured_output

    dory = DoryEntryPoint()
    dory.cache(["xgboost"])

    sys.stdout = sys.__stdout__

    assert mock_load.call_args_list == [
        (("models.feature_extractors", "xgboost"),),
        (("models.classifiers", "xgboost"),),
    ]
    assert "Loaded CLS xgboost" in captured_output.getvalue()


@patch("asreviewcontrib.dory.entrypoint.load_extension")
def test_cache_model_not_found(mock_load):
    mock_load.side_effect = [ValueError("not FE"), ValueError("not CLS")]

    dory = DoryEntryPoint()
    with patch("builtins.print") as mock_print:
        dory.cache(["unknown_model"])
        mock_print.assert_any_call("Error: Model 'unknown_model' not found.")


@patch("asreviewcontrib.dory.entrypoint.load_extension")
@patch("builtins.print")
def test_cache_handles_keyerror(mock_print, mock_load_extension):
    # Mock an FE model that lacks 'sentence_transformer' in named_steps
    mock_fe_model_instance = MagicMock()
    mock_fe_model_instance.named_steps = {}  # Will raise KeyError
    mock_fe_model = MagicMock(return_value=mock_fe_model_instance)
    mock_load_extension.return_value = mock_fe_model

    dory = DoryEntryPoint()
    dory.cache(["sbert"])

    # Ensure it still prints that the FE was loaded even if KeyError occurred
    mock_print.assert_called_once_with("Loaded FE sbert")
    mock_load_extension.assert_called_once_with("models.feature_extractors", "sbert")


@patch.object(DoryEntryPoint, "_get_all_models")
@patch.object(DoryEntryPoint, "cache")
def test_execute_cache_all(mock_cache, mock_get_all):
    xgboost_mock = MagicMock()
    sbert_mock = MagicMock()
    xgboost_mock.name = "xgboost"
    sbert_mock.name = "sbert"

    mock_get_all.return_value = [xgboost_mock, sbert_mock]

    dory = DoryEntryPoint()
    dory.execute(["cache-all"])

    mock_cache.assert_called_once_with(["xgboost", "sbert"])


@patch.object(DoryEntryPoint, "_get_all_models")
@patch("builtins.print")
def test_execute_list(mock_print, mock_get_all):
    xgboost_mock = MagicMock()
    sbert_mock = MagicMock()
    xgboost_mock.name = "xgboost"
    sbert_mock.name = "sbert"
    mock_get_all.return_value = [xgboost_mock, sbert_mock]

    dory = DoryEntryPoint()
    dory.execute(["list"])

    mock_print.assert_called_once_with(["xgboost", "sbert"])


def test_execute_invalid_command_prints_help():
    with patch("argparse.ArgumentParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser
        mock_parser.parse_args.return_value = MagicMock(command="invalid")

        dory = DoryEntryPoint()
        dory.execute(["invalid"])

        mock_parser.print_help.assert_called_once()


def test_version_output(capsys):
    dory = DoryEntryPoint()

    with pytest.raises(SystemExit):
        dory.execute(["--version"])

    captured = capsys.readouterr()

    assert re.search(r"asreview dory \d+", captured.out)


@patch("asreviewcontrib.dory.entrypoint.extensions")
def test_get_all_models(mock_extensions):
    mock_fe_1 = MagicMock()
    mock_fe_2 = MagicMock()
    mock_cls_1 = MagicMock()
    mock_cls_2 = MagicMock()

    # Mock __str__ return values used in filtering logic
    mock_fe_1.__str__.return_value = "asreviewcontrib.dory.sbert"
    mock_fe_2.__str__.return_value = "external.sbert"
    mock_cls_1.__str__.return_value = "asreviewcontrib.dory.xgboost"
    mock_cls_2.__str__.return_value = "external.xgboost"

    # Add a `.name` attribute for readability
    mock_fe_1.name = "sbert"
    mock_cls_1.name = "xgboost"

    mock_extensions.side_effect = [
        [mock_fe_1, mock_fe_2],
        [mock_cls_1, mock_cls_2],
    ]

    dory = DoryEntryPoint()
    models = dory._get_all_models()

    assert len(models) == 2
    assert {m.name for m in models} == {"sbert", "xgboost"}
