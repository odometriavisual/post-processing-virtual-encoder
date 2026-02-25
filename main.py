import traceback
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

from post_processing.plot import plot_2d
from post_processing.processing import compute_displacements, calibrate_spatial_resolution


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
            avg_size, avg_img = calibrate_spatial_resolution(path)
            plt.axis("off")
            plt.title(f"Estimated circle diameter = {avg_size:.3f} px")
            plt.imshow(avg_img)
            plt.savefig(path.with_suffix(".jpg"))
            plt.close()

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

            plot_2d(args, path, trajectory, abs(displacements))

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
            for file in sorted(files):
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
