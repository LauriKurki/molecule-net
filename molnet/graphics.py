import jax.numpy as jnp
import matplotlib.pyplot as plt


def save_predictions(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    losses: jnp.ndarray,
    outdir: str,
):

    titles = ['H', 'C', 'N', 'O', 'F']

    n_samples = inputs.shape[0]

    for sample in range(n_samples):
        inp = inputs[sample]
        target = targets[sample]
        pred = preds[sample]
        loss = losses[sample]

        fig = plt.figure(figsize=(18, 10), layout='constrained')
        subfigs = fig.subfigures(1, 5, wspace=0.07, width_ratios=[1, 2, 2, 1, 1])

        fig.suptitle(f'mse: {loss:.4f}', fontsize=16)
        subfigs[0].suptitle(f'Input')
        subfigs[1].suptitle(f'Prediction')
        subfigs[2].suptitle(f'Target')
        subfigs[3].suptitle(f'Prediction (sum over species)')
        subfigs[4].suptitle(f'Target (sum over species)')

        axs_input = subfigs[0].subplots(5, 1)
        axs_pred = subfigs[1].subplots(5, 5)
        axs_target = subfigs[2].subplots(5, 5)
        axs_pred_sum = subfigs[3].subplots(5, 1)
        axs_target_sum = subfigs[4].subplots(5, 1)

        for i in range(5):
            for j in range(5):
                axs_pred[i, j].imshow(pred[..., i*2, j], cmap='gray')
                axs_pred[i, j].set_xticks([])
                axs_pred[i, j].set_yticks([])
                axs_target[i, j].imshow(target[..., i*2, j], cmap='gray')
                axs_target[i, j].set_xticks([])
                axs_target[i, j].set_yticks([])

        axs_input[0].set_ylabel('Far')
        axs_input[-1].set_ylabel('Close')
        for i in range(5):
            height = i*2
            axs_input[i].imshow(inp[..., height, 0], cmap='gray')
            ps = axs_pred_sum[i].imshow(jnp.sum(pred[..., height, :], axis=-1), cmap='gray')
            ts = axs_target_sum[i].imshow(jnp.sum(target[..., height, :], axis=-1), cmap='gray')
        
            for ax in [axs_input[i], axs_pred_sum[i], axs_target_sum[i]]:
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

            for ax in [axs_input, axs_pred_sum, axs_target_sum]:
                ax[0].set_ylabel('Far')
                ax[-1].set_ylabel('Close')

        subfigs[3].colorbar(ps, ax=axs_pred_sum, location='right', shrink=0.5)
        subfigs[4].colorbar(ts, ax=axs_target_sum, location='right', shrink=0.5)

        for i, title in enumerate(titles):
            axs_pred[0, i].set_title(title)
            axs_target[0, i].set_title(title)
            axs_pred[0, 0].set_ylabel('Far')
            axs_pred[-1, 0].set_ylabel('Close')
            axs_target[0, 0].set_ylabel('Far')
            axs_target[-1, 0].set_ylabel('Close')

        plt.savefig(f'{outdir}/{sample:02}_prediction.png')
        plt.close()


def save_simple_predictions(
    inputs: jnp.ndarray,
    preds: jnp.ndarray,
    targets: jnp.ndarray,
    outdir: str,
):
    n_samples = inputs.shape[0]

    for sample in range(n_samples):
        inp = inputs[sample]
        pred = preds[sample]
        target = targets[sample]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(inp, cmap='gray')
        axs[0].set_title('Input')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        im = axs[1].imshow(pred, cmap='gray')
        axs[1].set_title('Prediction')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        plt.colorbar(im, ax=axs[1], location='right')

        im = axs[2].imshow(target, cmap='gray')
        axs[2].set_title('Target')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        plt.colorbar(im, ax=axs[2], location='right')

        plt.savefig(f'{outdir}/{sample:02}_total.png')
        plt.close()
