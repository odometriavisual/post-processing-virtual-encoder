import numpy as np


def try_load_displacement_cache(path):
    try:
        if path.is_file():
            data = np.load(path, allow_pickle=True)
            return (
                data["trajectory"],
                data["displacements"],
                data["quaternions"],
                data["acc"],
                data["timestamps"],
                data["px_p_mm"]
            )

    except:
        pass

    return False


def save_displacement_cache(path, trajectory, displacements, quaternions, acc, timestamps, px_p_mm):
    np.savez(
        path,
        trajectory=trajectory,
        displacements=displacements,
        quaternions=quaternions,
        acc=acc,
        timestamps=timestamps,
        px_p_mm=px_p_mm,
    )
