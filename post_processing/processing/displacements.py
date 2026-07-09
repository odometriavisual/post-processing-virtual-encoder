from queue import Queue, ShutDown as QueueShutdown
from threading import Thread

import numpy as np
from tqdm import tqdm

from visual_odometer import VisualOdometer
from post_processing.utils.ensaio import EnsaioReader


def _upper_bound(arr, val):
    # finds first element in an ordered array such that arr[i] > val
    def _helper(first, last):
        if first == last:
            return first

        mid = (first + last) // 2

        if arr[mid] > val:
            return _helper(first, mid)

        return _helper(mid + 1, last)

    return _helper(0, len(arr) - 1)


def _interpolate_quaternion(timestamp, quaternions):
    i = _upper_bound(quaternions[:, 0], timestamp)

    t1, q1 = quaternions[i, 0], quaternions[i, 1:]

    if i >= 1:
        t0, q0 = quaternions[i - 1, 0], quaternions[i - 1, 1:]
        return (timestamp - t0) * (q1 - q0) / (t1 - t0) + q0

    return q1


def compute_displacements(ensaio: EnsaioReader):
    odometer = VisualOdometer(
        ensaio.get_img(0)[1].shape,
        frequency_window_params={"factor": 0.1},
        async_mode=True,
    )

    imgs_queue = Queue()

    def _pub_imgs():
        for i in range(ensaio.get_img_count()):
            imgs_queue.put(ensaio.get_img(i))

        imgs_queue.shutdown(False)

    Thread(target=_pub_imgs, daemon=True).start()

    unmatched_quaternions = np.array(
        [
            [data["timestamp"], data["qw"], data["qx"], data["qy"], data["qz"]]
            for data in ensaio.get_imu_data()
        ]
    )
    quaternions = []
    timestamps = []

    displacements = []
    for i in tqdm(range(ensaio.get_img_count())):
        try:
            timestamp, img = imgs_queue.get()
        except QueueShutdown:
            break

        odometer.feed_image(img)

        dx, dy = odometer.get_displacement()
        displacements.append([dx, dy])
        timestamps.append(timestamp)

        quaternions.append(_interpolate_quaternion(timestamp, unmatched_quaternions))

    displacements = np.array(displacements)
    quaternions = np.array(quaternions)
    trajectory = np.cumsum(displacements, axis=0)

    return trajectory, displacements, quaternions, timestamps
