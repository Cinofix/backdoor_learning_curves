import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_context("paper")
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def remove_ticks(ax):
    ax.set_xticks([], [])
    ax.set_yticks([], [])


def plot_mnist_image(x, name):
    x = x.tondarray()
    x = x.reshape(28, 28)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    img = ax.imshow(x, cmap="gray")
    remove_ticks(ax)
    fig.tight_layout()
    plt.savefig(name, bbox_inches="tight")
    plt.show()


def plot_mnist_image_grad(x, name):
    x = x.tondarray()
    x = x.reshape(28, 28)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    img = ax.imshow(x, cmap="Oranges")
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04, format="%.1f")
    cbar.ax.tick_params(labelsize=21)
    remove_ticks(ax)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=10)
    fig.tight_layout()
    remove_ticks(ax)
    fig.tight_layout()
    plt.savefig(name, bbox_inches="tight")
    plt.show()


def to_img(x, w=32, h=32):
    x = x.tondarray()
    x = x.reshape((3, w, h)).transpose(1, 2, 0)
    return x


def plot_rgb_image_grad(x, name, w=32, h=32, vmin=-0.25, vmax=1.75):
    x = to_img(x, w, h)
    x = x.max(axis=2)
    print(x.shape)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    img = ax.imshow(x, vmin=vmin, vmax=vmax, cmap="Oranges")
    cbar = fig.colorbar(img, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    remove_ticks(ax)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=10)
    fig.tight_layout()
    plt.savefig(name, bbox_inches="tight")
    plt.show()


def plot_rgb_image(x, name, w=32, h=32):
    x = to_img(x, w, h)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    img = ax.imshow(x)
    remove_ticks(ax)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=10)
    fig.tight_layout()
    plt.savefig(name, bbox_inches="tight")
    plt.show()
