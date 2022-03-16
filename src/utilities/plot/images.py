import matplotlib.pyplot as plt


def plot_triggered_samples(clf_p, ds, backdoor_attack, shape, c_map=None):

    k, w, h = shape
    fig, axs = plt.subplots(3, 3, figsize=(10, 8))
    axs = axs.flatten()
    for i in range(9):
        x = ds.X[ds.Y == i, :][0, :]
        p = backdoor_attack.trigger_input(x, i)

        p_view = p.tondarray().reshape(k, w, h).transpose(1, 2, 0)
        if k == 1:
            p_view = p_view[:, :, 0]
        axs[i].imshow(p_view, cmap=c_map)
        axs[i].set_title(
            "Original label: %d, i = %d predicted label %d"
            % (clf_p.predict(x).item(), i, clf_p.predict(p.flatten()).item()),
            fontsize=8,
        )
    plt.show()
    plt.close()
