import numpy as np
import torch
from PIL import Image

from mednet.inference import predict, preprocess
from mednet.model import MedNet


def test_preprocess_rgb_non_square():
    img = Image.fromarray(np.random.randint(0, 255, (120, 200, 3), dtype=np.uint8))
    pre = preprocess(img, "dermamnist")
    assert pre.tensor.shape == (3, 28, 28) and pre.display.shape == (28, 28, 3)
    assert -1.0 <= pre.tensor.min() and pre.tensor.max() <= 1.0


def test_preprocess_gray_is_repeated_to_three_channels():
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    pre = preprocess(img, "octmnist")
    assert torch.equal(pre.tensor[0], pre.tensor[1]) and torch.equal(pre.tensor[1], pre.tensor[2])


def test_preprocess_handles_rgba_and_palette():
    assert preprocess(Image.new("RGBA", (50, 40), (255, 0, 0, 128)), "bloodmnist").tensor.shape == (3, 28, 28)
    assert preprocess(Image.new("P", (50, 40)), "bloodmnist").tensor.shape == (3, 28, 28)


def test_predict_returns_probability_vector():
    probs = predict(MedNet(8).eval(), torch.randn(3, 28, 28))
    assert probs.shape == (8,) and abs(probs.sum() - 1) < 1e-5
