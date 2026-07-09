from queue import Queue, ShutDown as QueueShutdown
from threading import Thread

from scipy.spatial.transform import Slerp, Rotation
import numpy as np
from tqdm import tqdm

from visual_odometer import VisualOdometer
from post_processing.utils.ensaio import EnsaioReader


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

    imu_data = ensaio.get_imu_data()
    interpolate_quaternion = Slerp(
        np.array([d["timestamp"] for d in imu_data]),
        Rotation.from_quat([[d["qx"], d["qy"], d["qz"], d["qw"]] for d in imu_data])
    )
    
    timestamps = []
    displacements = []

    for i in tqdm(range(ensaio.get_img_count())):
        try:
            timestamp, img = imgs_queue.get()
        except QueueShutdown:
            break

        # dx, dy = 0, 0

        odometer.feed_image(img)
        dx, dy = odometer.get_displacement()

        displacements.append([dx, dy])
        timestamps.append(timestamp)

    quaternions = interpolate_quaternion(timestamps[:-1])

    displacements = np.array(displacements)
    trajectory = np.cumsum(displacements, axis=0)

    return trajectory, displacements, quaternions, timestamps
