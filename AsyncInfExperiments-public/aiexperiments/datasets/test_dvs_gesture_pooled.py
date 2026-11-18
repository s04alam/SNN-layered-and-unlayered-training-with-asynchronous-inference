from aiexperiments.datasets.dvs_gesture_pooled import DVSGesture
import tonic


def test_dvs_gesture_pooled():
    assert DVSGesture.sensor_size == (32, 32, 2)
    dvsgesture = DVSGesture(save_to="./data", train=True)
    assert DVSGesture.sensor_size == (32, 32, 2)
    assert dvsgesture.sensor_size == (32, 32, 2)
    events, target = dvsgesture[0]
    transform = tonic.transforms.ToFrame(
        sensor_size=(32, 32, 2),
        time_window=50000, 
    )
    frames = transform(events) # (129, 2, 32, 32)
    for frame in frames:
        assert frame.shape == (2, 32, 32)
