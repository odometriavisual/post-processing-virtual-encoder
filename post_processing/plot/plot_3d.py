import matplotlib.pyplot as plt
import numpy as np


def plot(args, trajectory, title):
    # Configurar o gráfico 3D com subplots
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")

    # Plotar a trajetória 3D
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=title)

    # Limites automáticos com a escala preservada
    x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
    y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])
    z_min, z_max = np.min(trajectory[:, 2]), np.max(trajectory[:, 2])

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    # Calculando a extensão dos eixos
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Ajustando os limites para que todos os eixos tenham a mesma escala
    ax.set_xlim([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim([z_mid - max_range / 2, z_mid + max_range / 2])

    unit = "px" if args.px else "mm"
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.legend()

    final_position = trajectory[-1]
    final_displacement = np.linalg.norm(final_position)
    ax.text(
        final_position[0],
        final_position[1],
        final_position[2],
        f"{final_displacement:.1f} {unit}",
        color="red",
    )

    # Calcular o erro normalizado
    # error = np.linalg.norm(final_position) / 100
    # ax.text(
    #     -50, -500, 0.0,  # Posição do texto abaixo do título
    #     f'Erro normalizado: {error:.2f}%',  # Texto com o erro normalizado, agora com 2 casas decimais
    #     color='blue', ha='center', va='center', transform=ax.transAxes
    # )


def compute_trajectory(odometry, rotations, cam2imu):
    trajectory = []
    position = np.array([0.0, 0.0, 0.0])

    for displacement, r in zip(odometry, rotations):
        position += r.apply(cam2imu @ displacement)
        trajectory.append(position.copy())

    return np.array(trajectory)


def plot_3d(args, path, odometry, rotations, spatial_resolution):
    # How the imu axis align with the camera axis, camex version
    cam2imu = np.array([[0, 0], [-1, 0], [0, 1]])

    trajectory = compute_trajectory(odometry, rotations, cam2imu)

    if args.px:
        plot(args, trajectory, path.stem)
    else:
        plot(args, trajectory / spatial_resolution, path.stem)

    # Ajustar layout e mostrar o gráfico
    plt.tight_layout()
    plt.show()
