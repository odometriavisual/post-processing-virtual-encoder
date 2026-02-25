import numpy as np


def try_load_displacement_cache(path):
    if path.is_file():
        data = np.load(path, allow_pickle=True)
        return (
            data["trajectory"],
            data["displacements"],
            data["quaternions"],
            data["timestamps"],
        )

    return False


def save_displacement_cache(path, trajectory, displacements, quaternions, timestamps):
    np.savez(
        path,
        trajectory=trajectory,
        displacements=displacements,
        quaternions=quaternions,
        timestamps=timestamps,
    )
