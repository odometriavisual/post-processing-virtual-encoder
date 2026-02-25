import numpy as np
import cv2
import matplotlib.pyplot as plt

def __approximate_polygonal_trajectory(trajectory):
    epsilon = np.max(trajectory.reshape(-1)) * 0.1
    reshaped_trajectory = np.array(trajectory.reshape((-1, 1, 2)), dtype=np.float32)

    polygonal_trajectory = cv2.approxPolyDP(reshaped_trajectory, epsilon, False)
    return polygonal_trajectory.reshape((-1, 2))

def plot_2d(args, path, trajectory, displacements, spatial_resolution):
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
        trajectory /= spatial_resolution

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
        for x, y in __approximate_polygonal_trajectory(trajectory):
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

