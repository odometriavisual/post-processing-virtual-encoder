from ensaio import Ensaio
from visual_odometer import VisualOdometer

import matplotlib.pyplot as plt
import numpy as np

# reference = -(np.load("trajectory.npz")["arr_0"][:, 1:] - np.array([300., 100.]))
root = "/home/fernando/Documents/encoder/ensaios/2026.02.09 testes LEDs submerso camex"
ensaios = [
    root + "/2 asas curvas/1770666681723291244 mov_asa_curva_2_rente_45",
    root + "/2 asas curvas/1770666714564114832 mov_asa_curva_2_rente_90",
    root + "/2 asas curvas/1770666822253653559 mov_asa_curva_2_avançado_0",
    root + "/2 asas curvas/1770666877678230850 mov_asa_curva_2_avançado_45",
    root + "/2 asas curvas/1770666921078629657 mov_asa_curva_2_avançado_90",
    root + "/2 asas retas/1770663980757436042 mov_asa_reta_2_rente_0",
    root + "/2 asas retas/1770664020828088354 mov_asa_reta_2_rente_45",
    root + "/2 asas retas/1770664054178679203 mov_asa_reta_2_rente_90",
    root + "/2 asas retas/1770664110393871415 mov_asa_reta_2_avancado_0",
    root + "/2 asas retas/1770664175175599738 mov_asa_reta_2_avancado_90",
    root + "/4 asas curvas/1770666000323863469 mov_asa_curva_4_rente_0",
    root + "/4 asas curvas/1770666084884390828 mov_asa_curva_4_rente_90",
    root + "/4 asas curvas/1770666133014342703 mov_asa_curva_4_avancado_0",
    root + "/4 asas curvas/1770666169069273024 mov_asa_curva_4_avancado_45",
    root + "/4 asas curvas/1770666224862954684 mov_asa_curva_4_avancado_90",
    root + "/4 asas retas/1770662850635328939 mov_asa_reta_rente_45",
    root + "/4 asas retas/1770662893956187516 mov_asa_reta_rente_0",
    root + "/4 asas retas/1770662941526634562 mov_asa_reta_rente_90",
    root + "/4 asas retas/1770663094073906580 mov_asa_reta_avancado_0",
]


def plot_displacements(name, trajectory, absolute_displacements, phase):
    # Plotar o gráfico 2D
    plt.figure(name)
    axis0 = plt.subplot2grid((2, 3), (0, 0), 2, 2)
    axis1 = plt.subplot2grid((2, 3), (0, 2), 1, 1)
    axis2 = plt.subplot2grid((2, 3), (1, 2), 1, 1)

    axis0.plot(trajectory[:, 0], trajectory[:, 1], label="Trajetória")
    axis0.set_xlabel("Deslocamento X (mm)")
    axis0.set_ylabel("Deslocamento Y (mm)")
    axis0.set_title("Trajetória 2D (em mm)")
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
    axis1.plot(absolute_displacements)

    axis2.set_title("Fase dos deslocamentos (rad)")
    axis2.plot(phase)

    plt.savefig(f"{root}/{name}.png")



for name in ensaios:
    print(name)
    ensaio = Ensaio(name)
    name = name.split(' ')[-1]

    if True or not ensaio.has_displacements():
        try:
            odometer = VisualOdometer((480, 640), frequency_window_params={"factor": 1.0}, async_mode=True)
            displacements, quaternions, timestamps = [], [], []

            imgs = ensaio.get_all_imgs()

            for i, (timestamp, img) in enumerate(imgs):
                odometer.feed_image(img)

                dx, dy = odometer.get_displacement()
                displacements.append([dx, dy])
                quaternions.append([1, 0, 0, 0])
                timestamps.append(timestamp)
                print(f"{name}: {i}/{ensaio.get_img_count()}")

            displacements = np.array(displacements)
            quaternions = np.array(quaternions)
            timestamps = np.array(timestamps)

            # Salvar dados
            ensaio.set_displacements(
                displacements=displacements, quaternions=quaternions, timestamps=timestamps
            )

            displacements = ensaio.get_displacements()["displacements"]

            phase = np.arctan2(displacements[:, 1], displacements[:, 0])

            absolute_displacements = np.linalg.norm(displacements, axis=1)
            phase = np.arctan2(displacements[:, 1], displacements[:, 0])
            trajectory = np.cumsum(displacements, axis=0)
            plot_displacements(f"{name}", trajectory, absolute_displacements, phase)

        except:
            pass

    
plt.show()
