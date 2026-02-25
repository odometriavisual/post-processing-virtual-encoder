import matplotlib.pyplot as plt

def plot_circle_and_bb_box(path, avg_size, avg_img):
    plt.axis("off")
    plt.title(f"Estimated circle diameter = {avg_size:.3f} px")
    plt.imshow(avg_img)
    plt.savefig(path.with_suffix(".jpg"))
    plt.close()
