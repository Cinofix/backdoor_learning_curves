import pickle
import numpy as np
from src.utilities.plot.settings import *


def load_stats(name):
    with open(name + ".pickle", "rb") as handle:
        stats = pickle.load(handle)
    return stats


seeds = [100, 101, 110]  # , 111, 1000)

net = "resnet50"
stats = load_stats("nn/" + net)

epochs = stats[seeds[0]]["epochs"]
p_poison = stats[seeds[0]]["p_poison"]
print(epochs, p_poison)
# plt.plot(stats[seeds[0]]["betas"], stats[seeds[0]]["backdoor_loss_beta"][0])
# plt.show()


colors_palette = [
    sns.color_palette("mako_r"),
    sns.color_palette("gist_heat_r"),
    sns.color_palette("summer_r"),
    sns.color_palette("RdPu"),
    sns.color_palette("RdPu"),
]

backdoor_loss = {epoch: [[] for _ in p_poison] for epoch in epochs}
backdoor_accuracy = {epoch: [[] for _ in p_poison] for epoch in epochs}
clean_accuracy = {epoch: [[] for _ in p_poison] for epoch in epochs}

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax = ax.flatten()

for e, epoch in enumerate(epochs):  #
    for p, percentage in enumerate(p_poison):

        backdoor_test_loss = np.array(stats[seeds[0]]["backdoor_loss_beta"][e][p])
        clean_test_loss = np.array(stats[seeds[0]]["clean_loss_beta"][e][p])
        backdoor_test_accuracy = np.array(
            stats[seeds[0]]["backdoor_accuracy_beta"][e][p]
        )
        clean_test_accuracy = np.array(stats[seeds[0]]["clean_accuracy_beta"][e][p])

        for seed in seeds[1:]:
            print(seed)
            backdoor_test_loss += np.array(stats[seed]["backdoor_loss_beta"][e][p])
            clean_test_loss += np.array(stats[seed]["clean_loss_beta"][e][p])
            backdoor_test_accuracy += np.array(
                stats[seed]["backdoor_accuracy_beta"][e][p]
            )
            clean_test_accuracy += np.array(stats[seed]["clean_accuracy_beta"][e][p])

        backdoor_test_loss /= len(seeds)
        clean_test_loss /= len(seeds)

        backdoor_test_accuracy /= len(seeds)
        clean_test_accuracy /= len(seeds)

        backdoor_loss[epoch][p] = backdoor_test_loss
        backdoor_accuracy[epoch][p] = backdoor_test_accuracy
        clean_accuracy[epoch][p] = clean_test_accuracy

        ax[e].set_xlabel("$\\beta$")
        ax[e].set_ylabel("Test Loss")
        ax[e].set_title(
            "Resnet50 epochs={}".format(epoch), fontsize=18
        )
        ax[e].tick_params(axis="y", labelsize=14)
        ax[e].tick_params(axis="x", labelsize=14)
        alphas = [0.5, 0.7, 1]

        legend = "TS+BT p=%.3f" % percentage if e != 0 else None
        sns.lineplot(
            stats[seeds[0]]["betas"],
            backdoor_test_loss,
            ax=ax[e],  # [e],
            linewidth=3,
            color=colors_palette[0][1 + 2 * p],
            alpha=alphas[p],
            label=legend,
        )
        legend = "TS p=%.3f" % percentage if e != 0 else None

        sns.lineplot(
            stats[seeds[0]]["betas"],
            clean_test_loss,
            ax=ax[e],  # [e],
            linewidth=3,
            color=colors_palette[0][1 + 2 * p],
            alpha=0.5,
            label=legend,
            linestyle="dotted",
        )
plt.legend(
    markerscale=0.5,
    framealpha=0.2,
    handletextpad=0.2,
    labelspacing=0.25,
    fontsize=15,
    loc=1,
)

plt.tight_layout()
plt.savefig(
    "nn/" + net + "-epochs_FINAL.pdf".format(epoch), bbox_inches="tight", pad_inches=0,
)
plt.show()


# compute the slope


def slope(curve, betas, delta=7):
    h = betas[delta] - betas[0]
    # print(betas[delta])
    tangent = (curve[delta] - curve[0]) / h
    slope = -2 / np.pi * np.arctan(tangent)
    return slope


betas = stats[100]["betas"]
for p, poison in enumerate(p_poison):
    for e, epoch in enumerate(epochs):
        print(
            "Epoch: {} Poison: {} Slope: {} backdoor eff=: {} clean acc=: {}".format(
                epoch,
                poison,
                slope(backdoor_loss[epoch][p], betas, 7),
                backdoor_accuracy[epoch][p][-1],
                clean_accuracy[epoch][p][-1],
            )
        )

"""
stats = load_stats("nn/" + net + "_learning_slope_imagenette2-320_ECML_final_epoch50")
stats2 = load_stats("nn/" + net + "_learning_slope_imagenette2-320_ECML_final")
for key in seeds:
    stats2[key]['backdoor_accuracy_beta'] += stats[key]['backdoor_accuracy_beta'] 
    stats2[key]['clean_accuracy_beta'] += stats[key]['clean_accuracy_beta'] 
    stats2[key]['backdoor_loss_beta'] += stats[key]['backdoor_loss_beta'] 
    stats2[key]['clean_loss_beta'] += stats[key]['clean_loss_beta'] 
stats2[key]['epochs'] += stats[key]['epochs']
"""