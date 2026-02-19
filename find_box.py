# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "opencv-python",
#     "matplotlib",
#     "numpy",
# ]
# ///
import cv2
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from zipfile import ZipFile


def find_circle_and_bbox(frame, min_radius=0, max_radius=0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

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


base_path = Path("/tmp/")
ensaios = [("91_20260120T155442 38mm_4mm.zip", 25, 4, 0)]

data = []

for ensaio_name, dist, diameter, frame_index in ensaios:
    with ZipFile(base_path / ensaio_name, "r") as zip:
        img_names = sorted(
            [filename for filename in zip.namelist() if ".jpg" in filename]
        )

        frame = cv2.imdecode(
            np.frombuffer(zip.read(img_names[frame_index]), dtype=np.uint8), cv2.IMREAD_COLOR
        )

        data.append([ensaio_name, dist, diameter, frame])


for name, dist, diameter, frames in data:
    width, height, radius, output_img = find_circle_and_bbox(
        frame, min_radius=12, max_radius=600
    )


    if width and height:
        GSD = diameter / (2 * radius)
        print(f"{name}: {dist = :.2f}, {diameter = :.2f} {radius = :.2f} {width = :.2f} {GSD = :.3f}")
        plt.axis("off")
        plt.title(f"Estimated {GSD = :.6f} mm/px")
    else:
        print(f"{name}: No circles found in the given radius range.")

    plt.imshow(output_img)
    plt.show()
