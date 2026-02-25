import numpy as np
import cv2
import tqdm
from post_processing.utils.ensaio import EnsaioReader


def find_circle_and_bbox(frame, min_radius=0, max_radius=0):
    gray = cv2.medianBlur(frame, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1,
        minDist=20,
        param1=300,
        param2=0.90,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        x, y, r = circles[0][0]
        d = 2 * r

        # bounding box
        top_left = (x - r, y - r)
        bottom_right = (x + r, y + r)

        cv2.circle(frame, (int(round(x)), int(round(y))), int(round(r)), (0, 255, 0), 1)
        cv2.rectangle(
            frame,
            np.int32(np.around(top_left)),
            np.int32(np.around(bottom_right)),
            (0, 0, 255),
            1,
        )

        return float(d), float(d), float(r), frame
    else:
        return None, None, None, frame


def calibrate_spatial_resolution(path):
    ensaio = EnsaioReader(path)
    imgs = ensaio.get_all_imgs()
    avg_img = np.zeros_like(ensaio.get_img(0)[1], dtype=np.float32)
    avg_size = 0

    for i, (timestamp, img) in tqdm.tqdm(
        enumerate(imgs), desc=f"{path.stem}", total=len(imgs)
    ):
        width, height, radius, output_img = find_circle_and_bbox(
            ensaio.get_img(0)[1], min_radius=12, max_radius=600
        )

        avg_img += output_img

        if width and height:
            avg_size += 2 * radius

    avg_img /= len(imgs)
    avg_size /= len(imgs)

    return avg_size, avg_img
