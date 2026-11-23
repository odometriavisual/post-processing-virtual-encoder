from queue import Queue
# from queue import ShutDown as QueueShutdown
from threading import Thread

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from visual_odometer import VisualOdometer

from post_processing.utils.ensaio import EnsaioReader


def compute_displacements(ensaio: EnsaioReader):
    odometer = VisualOdometer(
        ensaio.get_img(0)[1].shape,
        frequency_window_params={"factor": 0.1},
        displacement_estimation_method="orb",
        async_mode=False,
    )

    imgs_queue = Queue()

    # def _pub_imgs():
    for i in range(ensaio.get_img_count()):
        imgs_queue.put(ensaio.get_img(i))

        # imgs_queue.shutdown(False)

    # Thread(target=_pub_imgs, daemon=True).start()

    imu_data = ensaio.get_imu_data()
    
    timestamps = []
    displacements = []

    theta = 0
    for i in tqdm(range(ensaio.get_img_count())):
        # try:
        timestamp, img = imgs_queue.get()
        # except QueueShutdown:
        #     break

        odometer.feed_image(img)
        disp = odometer.get_displacement()

        theta += -disp[2]
        rot = np.array([[np.cos(theta), -np.sin(theta)],  [np.sin(theta),  np.cos(theta)]])

        d = rot @ [-disp[0], disp[1]]

        displacements.append(d)
        timestamps.append(timestamp)

    try:
        interpolate_quaternion = Slerp(
            np.array([d["timestamp"] for d in imu_data]),
            Rotation.from_quat([[d["qx"], d["qy"], d["qz"], d["qw"]] for d in imu_data])
        )
        quaternions = interpolate_quaternion(timestamps[:-1])

    except Exception:
        quaternions = []

    displacements = np.array(displacements)
    trajectory = np.cumsum(displacements, axis=0)
    acc = np.array([[d["acc_x"], d["acc_y"], d["acc_z"]] for d in imu_data])

    return trajectory, displacements, quaternions, acc, timestamps
