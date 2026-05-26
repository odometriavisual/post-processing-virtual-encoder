import traceback
import pathlib
import argparse
import csv

import numpy as np

from post_processing.plot import plot_2d, plot_circle_and_bb_box
from post_processing.processing import (
    compute_displacements,
    calibrate_spatial_resolution,
)
from post_processing.utils.cache import (
    try_load_displacement_cache,
    save_displacement_cache,
)


def load_override_data(path):
    override = {}
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            override[row["name"]] = {"px_p_mm": float(row["px_p_mm"])}

    return override


def process_ensaio(args, path):
    try:
        if args.calibration:
            avg_size, avg_img = calibrate_spatial_resolution(path)
            plot_circle_and_bb_box(path, avg_size, avg_img)
        else:
            data = try_load_displacement_cache(path.with_suffix(".npz"))

            if not data or args.force_processing:
                trajectory, displacements, quaternions, timestamps = (
                    compute_displacements(path)
                )
                save_displacement_cache(
                    path.with_suffix(".npz"),
                    trajectory,
                    displacements,
                    quaternions,
                    timestamps,
                )
            else:
                print(f"Found existing cache for {path.stem}, using it...")
                trajectory, displacements, quaternions, timestamps = data

            px_p_mm = args.override.get(path.stem, {"px_p_mm": 20.601})["px_p_mm"]
            plot_2d(args, path, trajectory, displacements, px_p_mm)

    except KeyboardInterrupt as e:
        raise e

    except Exception as e:
        print(f"Error processing {path.stem}: {e}")
        traceback.print_exc()


def main(args):
    if args.reference_trajectory is not None:
        args.reference_trajectory = np.load(args.reference_trajectory)["arr_0"]
        theta = np.radians(float(args.rotate_reference))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        args.reference_trajectory[:, 1:3] -= args.reference_trajectory[0, 1:3]
        args.reference_trajectory[:, 1:3] = (R @ args.reference_trajectory[:, 1:3].T).T

    if args.override:
        args.override = load_override_data(args.override)
    else:
        args.override = dict()

    path = pathlib.Path(args.path)
    if path.is_dir():
        for root, dirs, files in path.walk():
            for file in sorted(files):
                if file[-4:] != ".zip":
                    continue

                path = root / pathlib.Path(file)
                process_ensaio(args, path)
    else:
        process_ensaio(args, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Virtual Encoder: post processing",
        description="Post processing application for the virtual encoder\n"
        + "for recontruction and calibration purposes.",
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
        "--override",
        help="provide a csv data containing parameter overrides for individual ensaios",
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
        try:
            main(args)
        except KeyboardInterrupt:
            pass
    else:
        try:
            from post_processing.ui.menu_interface import show_main_menu

            show_main_menu()
        except Exception as e:
            print(e)
            traceback.print_exc()
