import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from post_processing.utils.imu_tools import fast_rot, rotate_vector

def plot_3d(args, path, list_displacements, list_quaternions, spatial_resolution):
    # Lista com as variações de deslocamento
    variations = [
        # [1, 2],
        # [1, 3],
        # [2, 1],
        [2, 3],
        # [3, 2],
        # [3, 1]
    ]

    # Mapeamento para exibir dx, dy e 0 nos títulos
    value_map = {
        0: '0',
        1: 'dx',
        2: 'dy'
    }

    # Configurar o gráfico 3D com subplots
    fig = plt.figure(figsize=(18, 12))


    # Função para calcular e plotar a trajetória
    def calculate_and_plot(rotation_base_1, rotation_base_2, ax, title, dx_dy_multiplyer):
        trajectory = []
        position = np.array([0.0, 0.0, 0.0])

        # print(rotation_base_1)
        # print(rotation_base_2)
        # print("")

        # Iterar sobre os deslocamentos e quaternions
        for displacement, r in zip(list_displacements, list_quaternions):
            dx, dy = displacement * dx_dy_multiplyer

            quiver_position_1 = r.apply([rotation_base_1[1], rotation_base_1[2], rotation_base_1[3]])
            quiver_position_2 = r.apply([rotation_base_2[1], rotation_base_2[2], rotation_base_2[3]])
            dx_influence = quiver_position_1 * dx
            dy_influence = quiver_position_2 * dy
            displacement_3d = dx_influence + dy_influence

            # Acumular o deslocamento
            position += displacement_3d / spatial_resolution

            # Armazenar a posição acumulada
            trajectory.append(position.copy())

        trajectory = np.array(trajectory)

        # Plotar a trajetória 3D
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=title)

        # Limites automáticos com a escala preservada
        x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
        y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])
        z_min, z_max = np.min(trajectory[:, 2]), np.max(trajectory[:, 2])

        # Calculando a extensão dos eixos
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Ajustando os limites para que todos os eixos tenham a mesma escala
        ax.set_xlim([x_min - max_range * 0.1, x_max + max_range * 0.1])
        ax.set_ylim([y_min - max_range * 0.1, y_max + max_range * 0.1])
        ax.set_zlim([z_min - max_range * 0.1, z_max + max_range * 0.1])

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend()

        # Mostrar o deslocamento total (última posição) em mm
        final_position = position
        ax.text(
            final_position[0], final_position[1], final_position[2],
            f'({final_position[0]:.1f} mm, {final_position[1]:.1f} mm, {final_position[2]:.1f} mm)',
            color='red'
        )

        # Calcular o erro normalizado
        # error = np.linalg.norm(final_position) / 100
        # ax.text(
        #     -50, -500, 0.0,  # Posição do texto abaixo do título
        #     f'Erro normalizado: {error:.2f}%',  # Texto com o erro normalizado, agora com 2 casas decimais
        #     color='blue', ha='center', va='center', transform=ax.transAxes
        # )

    num_variations = len(variations)
    cols = math.ceil(math.sqrt(num_variations))  # Define o número de colunas como a raiz arredondada para cima
    rows = math.ceil(num_variations / cols)  # Calcula o número de linhas necessário

    for i, order in enumerate(variations):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')

        # 1 é o azul
        # 2 é o vermelho

        # A ordem é [0, rosa, azul, vermelho]

        rotation_base_1 = [0, 0, 0, 0]
        rotation_base_1[order[0]] = 1
        rotation_base_2 = [0, 0, 0, 0]
        rotation_base_2[order[1]] = 1

        #title_values = [value_map[order[1]], value_map[order[2]], value_map[order[3]]]
        title = ", ".join(map(str, order))
        calculate_and_plot(rotation_base_1, rotation_base_2, ax, title, dx_dy_multiplyer = [1, 1])

    # Ajustar layout e mostrar o gráfico
    plt.tight_layout()
    plt.show()

    # plt.ioff()
