import numpy as np
import tqdm

from visual_odometer import VisualOdometer
from post_processing.utils.ensaio import EnsaioReader


def compute_displacements(path):
    ensaio = EnsaioReader(path)

    odometer = VisualOdometer(
        ensaio.get_img(0)[1].shape,
        frequency_window_params={"factor": 0.1},
        async_mode=True,
    )
    displacements, quaternions, timestamps = [], [], []

    imgs = ensaio.get_all_imgs()

    for i, (timestamp, img) in tqdm.tqdm(
        enumerate(imgs), desc=f"{path.stem}", total=len(imgs)
    ):
        odometer.feed_image(img)

        dx, dy = odometer.get_displacement()
        displacements.append([dx, dy])
        quaternions.append([1, 0, 0, 0])
        timestamps.append(timestamp)

    displacements = np.array(displacements)
    quaternions = np.array(quaternions)
    timestamps = np.array(timestamps)
    trajectory = np.cumsum(displacements, axis=0)

    return trajectory, displacements, quaternions, timestamps
