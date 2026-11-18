
from aiexperiments.datasets.shd_pooled import SHD


def test_shd_pooled():
    assert SHD.sensor_size == (350, 1, 2)
    shd = SHD(save_to="./data", train=True)
    assert SHD.sensor_size == (350, 1, 2)
    assert shd.sensor_size == (350, 1, 2)
    events, target = shd[0]
    for event in events:
        assert event[1] < SHD.pool_size
