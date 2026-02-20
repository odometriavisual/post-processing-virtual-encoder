import traceback
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from visual_odometer import VisualOdometer

from post_processing.utils.ensaio import EnsaioReader


def plot_displacements_2d(path, trajectory, displacements):
    plt.figure(figsize=(10, 6))
    axis0 = plt.subplot2grid((2, 3), (0, 0), 2, 2)
    axis1 = plt.subplot2grid((2, 3), (0, 2), 1, 1)
    axis2 = plt.subplot2grid((2, 3), (1, 2), 1, 1)

    axis0.plot(trajectory[:, 0], trajectory[:, 1], label="Trajetória")
    axis0.set_xlabel("Deslocamento X")
    axis0.set_ylabel("Deslocamento Y")
    axis0.set_title(f"{path.stem}")
    axis0.grid(True)
    axis0.legend()
    axis0.axis("equal")

    x, y = trajectory[-1, 0], trajectory[-1, 1]
    label_text = f"({x:.2f}, {y:.2f})"
    axis0.plot(x, y, "o", markersize=8, color="red")  # 'o' creates a circle marker
    axis0.text(
        x, y + 0.05, label_text, ha="center", va="bottom", fontsize=10, color="blue"
    )

    axis1.set_title("Deslocamentos absolutos (px)")
    absolute_displacements = np.linalg.norm(displacements, axis=1)
    axis1.plot(absolute_displacements)

    axis2.set_title("Fase dos deslocamentos (rad)")
    phase = np.arctan2(displacements[:, 1], displacements[:, 0])
    axis2.plot(phase)

    plt.savefig(path.with_suffix(".jpg"))
    plt.close()


def process_ensaio(path):
    ensaio = EnsaioReader(path)

    odometer = VisualOdometer(
        (480, 640), frequency_window_params={"factor": 1.0}, async_mode=True
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


def main(args):
    if args.recursive:
        for root, dirs, files in pathlib.Path(args.path).walk():
            for file in files:
                if file[-4:] != ".zip":
                    continue

                path = root / pathlib.Path(file)

                try:
                    data = try_load(path.with_suffix(".npz"))

                    if not data or args.force_processing:
                        trajectory, displacements, quaternions, timestamps = (
                            process_ensaio(path)
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

                    plot_displacements_2d(path, trajectory, abs(displacements))

                except Exception as e:
                    print(f"Error processing {path.stem}: {e}")

    else:
        process_ensaio(pathlib.Path(args.path))


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
        "--force-processing",
        "-f",
        help="ignore existing caches and (re)process displacements",
        action="store_true",
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
