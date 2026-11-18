import numpy as np

from aiexperiments.datasets.circle_dataset import CircleDataset


"""
Generates the circle dataset. This dataset features three kinds of noise:
- Path noise: a wavy noise on the circular path. Changes the definition of the path.
- Point noise: a pertubation for each point on the path. Does not affect the definition of the path.
- Background noise: random noise in the grid. Completely separate from the path.

The procedure is as follows:
1. Generate `max_sequence_length` evenly separated angles between `offset` and `offset` + 2pi * `label`, where
    `offset` is in the range [0, 2pi] and `label` = 1 for clockwise and -1 for counter-clockwise.
2. Sample from a normal distribution N(0, `path_noise_level`) `max_sequence_length`-1 times.
3. Take the cumulative sum of the sampled noise, and append a 0 to the start of the resulting array.
    The array should now have a length equal to `max_sequence_length`. The purpose of this is to make the noise
    to the path random but smooth.
4. Add the cumulative noise from step 3 to the generated angles from step 1.
5. Generate start_x and start_y, representing a random point in the grid
6. Generate (x,y) coordinates by repeating for each angle from step 4: 
   6.1. x = start_x + ROUND(COS(angle) + N(0, path_noise_level))
   6.2. y = start_y + ROUND(SIN(angle) + N(0, path_noise_level))
   6.3. Clip x and y such that they are in range [0,2]
7. Generate an event by sampling an x where all posibilities in the grid have p=`background_noise_level` except
    that of the true x from step 6, which has p=1-`width`*`background_noise_level`. Repeat for y.
8. Remove all events from 7 which do not cause a change in x or y.


Parameters:
    - path_noise_level: 
"""
def generate_data(samples, path_noise_level, point_noise_level, background_noise_level, max_sequence_length, width, height):
    data = []
    labels = []
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])

    # Function to generate events from coordinate sequence
    def generate_events(coords):
        events = np.zeros(coords.shape[0], dtype=dtype)
        events[0] = (
            coords[0][0],
            coords[0][1],
            0,
            0,
        )  # First event with timestamp 0 and p=1
        for i in range(1, len(coords)):
            if not np.array_equal(coords[i][:2], coords[i - 1][:2]):
                x = np.random.choice(
                    np.arange(width), 
                    p=[background_noise_level if j != coords[i][0] else 1 - (width-1) * background_noise_level for j in range(width)]
                )
                y = np.random.choice(
                    np.arange(height), 
                    p=[background_noise_level if j != coords[i][1] else 1 - (height-1) * background_noise_level for j in range(height)]
                )
                events[i] = (x, y, i, 0)
        return events[events["t"] != 0]  # Remove unused events

    for i in range(samples):
        # Generate random starting point
        start_x, start_y = np.random.randint(0, width), np.random.randint(0, height)

        for label in range(2):
            offset = np.random.rand() * 2 * np.pi
            theta_start = offset
            theta_end = (offset + 2 * np.pi) if label == 0 else (offset - 2 * np.pi)
            theta_increments = np.random.normal(0, path_noise_level, max_sequence_length - 1)
            theta = np.linspace(
                theta_start, theta_end, max_sequence_length
            ) + np.concatenate(([0], np.cumsum(theta_increments)))

            x = np.clip(
                start_x
                + np.round(
                    2 * np.cos(theta)
                    + np.random.normal(0, point_noise_level, max_sequence_length)
                ),
                0,
                width - 1,
            ).astype(int)
            y = np.clip(
                start_y
                + np.round(
                    2 * np.sin(theta)
                    + np.random.normal(0, point_noise_level, max_sequence_length)
                ),
                0,
                height - 1,
            ).astype(int)

            data.append(
                generate_events(
                    np.stack((x, y, np.arange(max_sequence_length)), axis=-1)
                )
            )
            labels.append(label)

    return data, labels


def save_data(samples, labels, to_file):
    sample_dict = {str(i): sample for i, sample in enumerate(samples)}
    np.savez(to_file, **sample_dict, labels=np.array(labels))


if __name__ == "__main__":
    train_samples = 5000
    test_samples = 5000
    path_noise_level = 0.1
    point_noise_level = 0.1
    background_noise_level = 0.02
    sequence_length = CircleDataset.sample_time
    width = CircleDataset.sensor_size[0]
    height = CircleDataset.sensor_size[1]
    train_data, train_labels = generate_data(
        train_samples, path_noise_level, point_noise_level, background_noise_level, sequence_length, width, height
    )
    save_data(
        train_data,
        train_labels,
        "./aiexperiments/datasets/circle_dataset_data/train_data.npz",
    )
    test_data, test_labels = generate_data(
        test_samples, path_noise_level, point_noise_level, background_noise_level, sequence_length, width, height
    )
    save_data(
        test_data,
        test_labels,
        "./aiexperiments/datasets/circle_dataset_data/test_data.npz",
    )
