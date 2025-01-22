import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from transformers import AutoTokenizer, AutoModelForCausalLM

from continuity_of_features.config import ModelConfig, ExperimentConfig
from continuity_of_features.continuity_experiments import (
    load_model,
    get_prompts_and_labels,
    get_hidden_states,
    perform_pca,
    run_experiment,
)

# Fixtures
@pytest.fixture
def model_config():
    return ModelConfig(
        name="test-model",
        short_name="test",
        hf_name="test/model",
        device="cpu",
        cache_dir="test/cache",
        layers_to_plot=[0, 1]
    )

@pytest.fixture
def experiment_config():
    return ExperimentConfig(
        base_items=["Monday", "Tuesday"],
        display_labels=["Mon", "Tue"],
        string_modifiers=["early", "late"],
        optional_prefix="It is ",
        pca_components=2,
        pca_n_base_samples=2
    )

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=AutoTokenizer)
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    return tokenizer

@pytest.fixture
def mock_model():
    model = Mock(spec=AutoModelForCausalLM)
    # Create mock hidden states
    hidden_states = [torch.tensor([[[1.0, 2.0, 3.0]]]) for _ in range(2)]
    model.return_value = Mock(hidden_states=hidden_states)
    return model

# Tests for load_model
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_load_model_success(mock_model_load, mock_tokenizer_load, model_config):
    mock_tokenizer_load.return_value = Mock()
    mock_model_load.return_value = Mock()
    
    tokenizer, model = load_model(model_config)
    
    assert tokenizer is not None
    assert model is not None
    mock_tokenizer_load.assert_called_once_with(model_config.hf_name)
    mock_model_load.assert_called_once_with(
        model_config.hf_name,
        device_map=model_config.device,
        cache_dir=model_config.cache_dir
    )

def test_load_model_failure(model_config):
    with pytest.raises(RuntimeError):
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Test error")
            load_model(model_config)

# Tests for get_prompts_and_labels
def test_get_prompts_and_labels(experiment_config):
    prompts, labels = get_prompts_and_labels(experiment_config)
    
    expected_prompts = [
        "It is Monday",
        "It is Tuesday",
        "It is early on Monday",
        "It is late on Monday",
        "It is early on Tuesday",
        "It is late on Tuesday"
    ]
    
    expected_labels = [
        "Mon",
        "Tue",
        "Early Mon",
        "Late Mon",
        "Early Tue",
        "Late Tue"
    ]
    
    assert prompts == expected_prompts
    assert labels == expected_labels

def test_get_prompts_and_labels_empty_modifiers(experiment_config):
    experiment_config.time_modifiers = []
    prompts, labels = get_prompts_and_labels(experiment_config)
    
    assert len(prompts) == len(experiment_config.input_strings)
    assert len(labels) == len(experiment_config.display_labels)

# Tests for get_hidden_states
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_get_hidden_states_shape(n_layers, experiment_config, mock_tokenizer, mock_model):
    prompts = ["Test prompt 1", "Test prompt 2"]
    
    # Mock model output
    hidden_states = [torch.randn(1, 1, 768) for _ in range(n_layers)]
    mock_output = Mock(hidden_states=hidden_states)
    mock_model.return_value = mock_output
    
    with patch("torch.no_grad"):
        result = get_hidden_states(prompts, mock_tokenizer, mock_model)
    
    assert result.shape == (len(prompts), n_layers, 768)

# Tests for perform_pca
def test_perform_pca():
    # Create synthetic hidden states
    n_samples = 10
    n_features = 100
    hidden_states = np.random.randn(n_samples, 2, n_features)  # 2 layers
    
    experiment_config = Mock(
        pca_components=2,
        pca_n_base_samples=5
    )
    
    result = perform_pca(hidden_states, layer=1, config=experiment_config)
    
    assert result.shape == (n_samples, 2)  # Should reduce to 2D
    assert np.all(np.isfinite(result))  # No NaN or inf values

# Integration test
@patch("torch.cuda.empty_cache")
def test_run_experiment_integration(
    mock_cuda_empty,
    model_config,
    experiment_config,
    mock_tokenizer,
    mock_model
):
    with patch("your_module.load_model") as mock_load_model:
        mock_load_model.return_value = (mock_tokenizer, mock_model)
        with patch("your_module.create_plot") as mock_create_plot:
            run_experiment(experiment_config, model_config)
            
            # Verify plot was created for each layer
            assert mock_create_plot.call_count == len(model_config.layers_to_plot)
            
            # Verify cleanup
            mock_cuda_empty.assert_called_once()
