from chgnet.model import CHGNet
from pytest import fixture

from cascade.learning.chgnet import CHGNetInterface


@fixture()
def chgnet() -> CHGNet:
    return CHGNet.load(use_device='cpu')


def test_inference(chgnet, example_data):
    cg = CHGNetInterface()
    energy, forces, stresses = cg.evaluate(chgnet, example_data)

    assert energy.shape == (2,)
    for atoms, f in zip(example_data, forces):
        assert f.shape == (len(atoms), 3)
    assert stresses.shape == (2, 3, 3)
