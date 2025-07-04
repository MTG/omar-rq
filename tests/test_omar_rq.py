import pytest
import torch
import numpy as np
from omar_rq import get_model

# List of all available model_ids from README.md
MODEL_IDS = [
    "mtg-upf/omar-rq-base",
    "mtg-upf/omar-rq-multicodebook",
    "mtg-upf/omar-rq-multifeature",
    "mtg-upf/omar-rq-multifeature-25hz",
    "mtg-upf/omar-rq-multifeature-25hz-fsq",
]


@pytest.fixture(params=MODEL_IDS)
def model(request):
    return get_model(model_id=request.param, device="cpu")


@pytest.fixture(params=[MODEL_IDS[0]])
def base_model(request):
    return get_model(model_id=request.param, device="cpu")


def test_load_model(model):
    assert hasattr(model, "extract_embeddings")
    assert hasattr(model, "eps")


def test_inference_with_dummy_data(base_model):
    x = torch.randn(1, 16000)
    out = base_model.extract_embeddings(x)
    print(out.shape)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, 15, 1024)  # batch size


def test_exception_on_empty_input(base_model):
    x = torch.empty(1, 0)
    with pytest.raises(Exception):
        base_model.extract_embeddings(x)


def test_exception_on_nan_input(base_model):
    x = torch.full((1, 16000), float("nan"))
    with pytest.raises(Exception):
        base_model.extract_embeddings(x)


def test_exception_on_numpy_input(base_model):
    x = np.random.randn(1, 16000)
    with pytest.raises(Exception):
        base_model.extract_embeddings(x)
