import numpy as np
import tqdm

from visual_odometer import VisualOdometer
from post_processing.utils.ensaio import EnsaioReader

def _lower_bound(arr, val):
    for i in range(len(arr)):
        if arr[i] > val:
            return max(0, i-1)

    return len(arr)-1


def _interpolate_quaternion(timestamp, quaternions):
    i = _lower_bound(quaternions[:,0], timestamp)
    return quaternions[i,1:]
    

def compute_displacements(ensaio: EnsaioReader):
    odometer = VisualOdometer(
        ensaio.get_img(0)[1].shape,
        frequency_window_params={"factor": 0.1},
        async_mode=True,
    )

    imgs = ensaio.get_all_imgs()

    unmatched_quaternions = np.array([[data["timestamp"], data["qw"], data["qx"], data["qy"], data["qz"]] for data in ensaio.get_imu_data()])
    quaternions = []
    timestamps = []

    displacements = []
    for i, (timestamp, img) in tqdm.tqdm(
        enumerate(imgs), desc=f"{ensaio.get_name()}", total=len(imgs)
    ):
        odometer.feed_image(img)

        dx, dy = odometer.get_displacement()
        displacements.append([dx, dy])
        timestamps.append(timestamp)

        quaternions.append(_interpolate_quaternion(timestamp, unmatched_quaternions))
        

    displacements = np.array(displacements)
    quaternions = np.array(quaternions)
    trajectory = np.cumsum(displacements, axis=0)

    return trajectory, displacements, quaternions, timestamps
