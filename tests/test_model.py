import torch

from mednet.config import DATASETS
from mednet.model import MedNet, count_parameters

# Parameter counts printed by the original notebook (Cell 7) - proves the architecture was ported faithfully.
NOTEBOOK_PARAMS = {"bloodmnist": 3_269_838, "octmnist": 3_268_810, "dermamnist": 3_269_581}


def test_parameter_counts_match_notebook():
    for name, expected in NOTEBOOK_PARAMS.items():
        assert count_parameters(MedNet(DATASETS[name].n_classes, True)) == expected


def test_forward_shape_for_every_dataset():
    for spec in DATASETS.values():
        out = MedNet(spec.n_classes).eval()(torch.zeros(2, 3, 28, 28))
        assert out.shape == (2, spec.n_classes)


def test_cbam_adds_parameters():
    assert count_parameters(MedNet(7, True)) > count_parameters(MedNet(7, False))


def test_state_dict_roundtrip(tmp_path):
    a = MedNet(7).eval()
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b = MedNet(7).eval()
    b.load_state_dict(torch.load(tmp_path / "m.pt", map_location="cpu", weights_only=True))
    x = torch.randn(1, 3, 28, 28)
    assert torch.allclose(a(x), b(x))
