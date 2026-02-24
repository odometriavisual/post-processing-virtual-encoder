import traceback
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import cv2
from visual_odometer import VisualOdometer

from post_processing.utils.ensaio import EnsaioReader


def plot_displacements_2d(args, path, trajectory, displacements):
    plt.figure(figsize=(10, 6))
    axis0 = plt.subplot2grid((2, 3), (0, 0), 2, 2)
    axis1 = plt.subplot2grid((2, 3), (0, 2), 1, 1)
    axis2 = plt.subplot2grid((2, 3), (1, 2), 1, 1)

    if args.px:
        axis0.set_xlabel("Deslocamento X / (px)")
        axis0.set_ylabel("Deslocamento Y / (px)")
    else:
        axis0.set_xlabel("Deslocamento X / (mm)")
        axis0.set_ylabel("Deslocamento Y / (mm)")
        trajectory /= 20.601

    axis0.plot(trajectory[:, 0], trajectory[:, 1], label="Trajetória")

    if args.reference_trajectory is not None:
        axis0.plot(
            args.reference_trajectory[:, 1],
            args.reference_trajectory[:, 2],
            label="Referência",
            color="hotpink",
        )

    axis0.set_title(f"{path.stem}")
    axis0.grid(True)
    axis0.legend()
    axis0.axis("equal")

    axis1.set_title("Deslocamentos absolutos (px)")
    absolute_displacements = np.linalg.norm(displacements, axis=1)
    axis1.plot(absolute_displacements)

    axis2.set_title("Fase dos deslocamentos (rad)")
    phase = np.arctan2(displacements[:, 1], displacements[:, 0])
    axis2.plot(phase)

    if args.draw_vertices:
        t = np.array(trajectory.reshape((-1, 1, 2)), dtype=np.float32)
        epsilon = np.max(trajectory.reshape(-1)) * 0.1
        polygonal_trajectory = cv2.approxPolyDP(t, epsilon, False)

        for x, y in polygonal_trajectory.reshape((-1, 2)):
            if abs(x) + abs(y) > 0.0001:
                label_text = f"({x:.2f}, {y:.2f})"
                axis0.plot(
                    x, y, "o", markersize=4, color="red"
                )  # 'o' creates a circle marker
                axis0.text(
                    x,
                    y + 1.0,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="blue",
                )

    plt.savefig(path.with_suffix("." + args.reconstruction_format))
    plt.close()


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

    plt.axis("off")
    plt.title(f"Estimated circle diameter = {avg_size:.3f} px")
    plt.imshow(avg_img)
    plt.savefig(path.with_suffix(".jpg"))
    plt.close()


def try_load(path):
    if path.is_file():
        data = np.load(path, allow_pickle=True)
        return (
            data["trajectory"],
            data["displacements"],
            data["quaternions"],
            data["timestamps"],
        )

    return False


def save(path, trajectory, displacements, quaternions, timestamps):
    np.savez(
        path,
        trajectory=trajectory,
        displacements=displacements,
        quaternions=quaternions,
        timestamps=timestamps,
    )


def process_ensaio(args, path):
    try:
        if args.calibration:
            calibrate_spatial_resolution(path)

        else:
            data = try_load(path.with_suffix(".npz"))

            if not data or args.force_processing:
                trajectory, displacements, quaternions, timestamps = (
                    compute_displacements(path)
                )
                save(
                    path.with_suffix(".npz"),
                    trajectory,
                    displacements,
                    quaternions,
                    timestamps,
                )
            else:
                print(f"Found existing cache for {path.stem}, using it...")
                trajectory, displacements, quaternions, timestamps = data

            plot_displacements_2d(args, path, trajectory, abs(displacements))

    except Exception as e:
        print(f"Error processing {path.stem}: {e}")


def main(args):
    if args.reference_trajectory is not None:
        args.reference_trajectory = np.load(args.reference_trajectory)["arr_0"]
        theta = np.radians(float(args.rotate_reference))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        args.reference_trajectory[:, 1:3] -= args.reference_trajectory[0, 1:3]
        args.reference_trajectory[:, 1:3] = (R @ args.reference_trajectory[:, 1:3].T).T

    if args.recursive:
        for root, dirs, files in pathlib.Path(args.path).walk():
            for file in files:
                if file[-4:] != ".zip":
                    continue

                path = root / pathlib.Path(file)
                process_ensaio(args, path)
    else:
        process_ensaio(args, pathlib.Path(args.path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Virtual Encoder: post processing",
        description="Post processing application for the virtual encoder\n"
        + "for recontruction and calibration purposes.",
    )

    parser.add_argument(
        "--recursive", "-r", help="recurses over directories", action="store_true"
    )
    parser.add_argument(
        "--calibration",
        "-c",
        help="calibration routine using bouding boxes",
        action="store_true",
    )
    parser.add_argument(
        "--reference-trajectory",
        "--rt",
        help="provide a npz file containing the reference trajectory",
        action="store",
    )
    parser.add_argument(
        "--rotate-reference",
        "--rotref",
        help="correction rotation for the reference trajectory in degrees",
        action="store",
        default=0,
    )
    parser.add_argument(
        "--px",
        help="ignore spatial resolution and generate results in pixels",
        action="store_true",
    )
    parser.add_argument(
        "--force-processing",
        "-f",
        help="ignore existing caches and (re)process displacements",
        action="store_true",
    )
    parser.add_argument(
        "--draw-vertices",
        help="draw vertices on the curve",
        action="store_true",
    )

    parser.add_argument(
        "--reconstruction-format",
        help="format to save the reconstructions",
        action="store",
        default="jpg",
    )

    parser.add_argument("path", nargs="?", default=False)

    args = parser.parse_args()

    if args.path:
        main(args)
    else:
        try:
            from post_processing.ui.menu_interface import show_main_menu

            show_main_menu()
        except Exception as e:
            print(e)
            traceback.print_exc()
