import numpy as np
import jax.numpy as jnp
from skimage import feature

import ase
import ase.data
from ase import io
import matplotlib.pyplot as plt

from typing import List

INDEX_TO_ELEM = {
    0: 'H',
    1: 'C', 2: 'N', 3: 'O', 4: 'F',
    5: 'Si', 6: 'P', 7: 'S', 8: 'Cl',
    9: 'Br'}

ELEM_TO_COLOR = {
    "H": 'white',
    "C": 'gray',
    "N": 'blue',
    "O": 'red',
    "F": 'green',
    "Si": 'orange',
    "P": 'purple',
    "S": 'yellow',
    "Cl": 'lime',
    "Br": 'brown'
}

INDEX_TO_COLOR = {i: ELEM_TO_COLOR[elem] for i, elem in INDEX_TO_ELEM.items()}
NUMBER_TO_COLOR = {
    1: 'white',
    6: 'gray',
    7: 'blue',
    8: 'red',
    9: 'green'
}


def save_predictions(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    losses: jnp.ndarray,
    outdir: str,
    start_save_idx: int = 0,
):

    n_samples = inputs.shape[0]
    n_species = targets.shape[-1]

    for sample in range(n_samples):
        inp = inputs[sample]
        target = targets[sample]
        pred = preds[sample]
        loss = losses[sample]

        total_columns = n_species*2 + 3
        total_rows = 5

        fig = plt.figure(figsize=(total_columns*3, total_rows*3), layout='constrained')
        subfigs = fig.subfigures(1, 5, wspace=0.07, width_ratios=[1, 2, 2, 1, 1])

        fig.suptitle(f'mse: {loss:.3e}', fontsize=16)
        subfigs[0].suptitle(f'Input')
        subfigs[1].suptitle(f'Prediction')
        subfigs[2].suptitle(f'Target')
        subfigs[3].suptitle(f'Prediction (sum over species)')
        subfigs[4].suptitle(f'Target (sum over species)')

        n_heights = inp.shape[-2]
        axs_input = subfigs[0].subplots(5, 1)
        axs_pred = subfigs[1].subplots(5, n_species)
        axs_target = subfigs[2].subplots(5, n_species)
        axs_pred_sum = subfigs[3].subplots(5, 1)
        axs_target_sum = subfigs[4].subplots(5, 1)

        vmax = target.max()

        for i in range(5):
            height = n_heights // 5 * i
            for j in range(n_species):
                axs_pred[i, j].imshow(pred[..., height, j], cmap='gray', vmin=0, vmax=vmax, origin='lower')
                axs_pred[i, j].set_xticks([])
                axs_pred[i, j].set_yticks([])
                axs_target[i, j].imshow(target[..., height, j], cmap='gray', vmin=0, vmax=vmax, origin='lower')
                axs_target[i, j].set_xticks([])
                axs_target[i, j].set_yticks([])

        axs_input[0].set_ylabel('Far')
        axs_input[-1].set_ylabel('Close')
        for i in range(5):
            height = n_heights // 5 * i
            axs_input[i].imshow(inp[..., height, 0], cmap='gray', origin='lower')
            ps = axs_pred_sum[i].imshow(jnp.sum(pred[..., height, :], axis=-1), cmap='gray', origin='lower')
            ts = axs_target_sum[i].imshow(jnp.sum(target[..., height, :], axis=-1), cmap='gray', origin='lower')
        
            for ax in [axs_input[i], axs_pred_sum[i], axs_target_sum[i]]:
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])

            for ax in [axs_input, axs_pred_sum, axs_target_sum]:
                ax[0].set_ylabel('Far')
                ax[-1].set_ylabel('Close')

        subfigs[3].colorbar(ps, ax=axs_pred_sum, location='right', shrink=0.5)
        subfigs[4].colorbar(ts, ax=axs_target_sum, location='right', shrink=0.5)

        for i in range(n_species):
            axs_pred[0, i].set_title(INDEX_TO_ELEM[i])
            axs_target[0, i].set_title(INDEX_TO_ELEM[i])
            axs_pred[0, 0].set_ylabel('Far')
            axs_pred[-1, 0].set_ylabel('Close')
            axs_target[0, 0].set_ylabel('Far')
            axs_target[-1, 0].set_ylabel('Close')

        save_idx = start_save_idx + sample
        plt.savefig(f'{outdir}/{save_idx:02}_prediction.png')
        plt.close()


def save_simple_predictions(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    outdir: str,
    start_save_idx: int = 0,
):
    n_samples = inputs.shape[0]

    for sample in range(n_samples):
        inp = inputs[sample]
        pred = preds[sample]
        target = targets[sample]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(inp, cmap='gray', origin='lower')
        axs[0].set_title('Input')
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        im = axs[1].imshow(pred, cmap='gray', origin='lower')
        axs[1].set_title('Prediction')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        plt.colorbar(im, ax=axs[1], location='right')

        im = axs[2].imshow(target, cmap='gray', origin='lower')
        axs[2].set_title('Target')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        plt.colorbar(im, ax=axs[2], location='right')

        save_index = start_save_idx + sample
        plt.savefig(f'{outdir}/{save_index:02}_simple_prediction.png')
        plt.close()


def save_attention_maps(
    inputs: jnp.ndarray,
    attention_maps: List[jnp.ndarray],
    outdir: str,
    start_save_idx: int = 0,
):
    n_samples = inputs.shape[0]
    n_heights = inputs.shape[-2]
    n_maps = len(attention_maps)

    for sample in range(n_samples):

        fig = plt.figure()
        subfigs = fig.subfigures(n_maps+1, 1, wspace=0.07)
        subfigs[0].suptitle('Input')

        axs_input = subfigs[0].subplots(1, n_heights)
        for height in range(n_heights):
            axs_input[height].imshow(inputs[sample, ..., height, 0], cmap='gray')
            axs_input[height].set_xticks([])
            axs_input[height].set_yticks([])

        for i, attention_map in enumerate(attention_maps):
            subfigs[i+1].suptitle(f'Attention map {i}')

            axs = subfigs[i+1].subplots(1, n_heights)
            for height in range(n_heights):
                axs[height].imshow(attention_map[sample, ..., height, :].mean(axis=-1), cmap='gray', origin='lower')
                axs[height].set_xticks([])
                axs[height].set_yticks([])

        save_index = start_save_idx + sample
        plt.savefig(f'{outdir}/{save_index:02}_attention.png')
        plt.close()


def save_predictions_as_molecules(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    preds: jnp.ndarray,
    xyzs: jnp.ndarray,
    outdir: str,
    scan_dim: np.ndarray = np.array([16, 16, 1]),
    z_cutoff: float = 1.0,
    peak_threshold: float = 0.5,
    start_save_idx: int = 0,
    preds_are_logits: bool = False
) -> None:

    n_samples = inputs.shape[0]

    for sample in range(n_samples):
        inp = inputs[sample]
        target = targets[sample]
        pred = preds[sample]
        xyz = xyzs[sample]

        # Flip pred and target z axes
        target = target[..., ::-1, :]
        pred = pred[..., ::-1, :]

        if preds_are_logits:
            pred = jnp.exp(pred)

        fig = plt.figure(figsize=(18, 6), layout='constrained')
        subfig1, subfig2 = fig.subfigures(1, 2, wspace=0.07)

        for i in range(10):
            ax = subfig1.add_subplot(2, 5, i+1)
            if i==0: ax.set_title('Input, far')
            if i==9: ax.set_title('Close')
            ax.imshow(inp[..., -10+i, 0], cmap='gray', origin='lower')
            ax.axis('off')

        axes = subfig2.subplots(2, 3)
        for i, (grid, name) in enumerate(zip([pred, target], ['pred', 'target'])):
            peaks = feature.peak_local_max(
                grid,
                min_distance=5,
                exclude_border=0,
                threshold_rel=peak_threshold
            )

            xyz_from_peaks = peaks[:, [1, 0, 2]] * scan_dim / target.shape[:3]
            elem_from_peaks = peaks[:, 3]

            mol = ase.Atoms(
                positions=xyz_from_peaks,
                symbols=[INDEX_TO_ELEM[elem] for elem in elem_from_peaks],
                cell=scan_dim
            )

            # Top to z_cutoff
            mol.positions[:, 2] -= mol.get_positions()[:, 2].max() - z_cutoff
            elements = mol.get_atomic_numbers()

            # for upper row, plot from above
            plot_molecule(
                ax=axes[0, i],
                x=xyz_from_peaks[:, 0],
                y=xyz_from_peaks[:, 1],
                z=xyz_from_peaks[:, 2],
                elements=elements,
                xlim=(0, scan_dim[0]),
                ylim=(0, scan_dim[1]),
            )
            axes[0, i].set_title(f"{name} from HeightMultiMap")

            # for lower row, plot from the side
            plot_molecule(
                ax=axes[1, i],
                x=xyz_from_peaks[:, 0],
                y=xyz_from_peaks[:, 2],
                z=xyz_from_peaks[:, 1],
                elements=elements,
                xlim=(0, scan_dim[0]),
                ylim=(-3, scan_dim[2]+1),
            )
            axes[1, i].set_title("sideview")

            # Save the molecule
            save_index = start_save_idx + sample
            io.write(f'{outdir}/{save_index:02}_{name}.xyz', mol)


        # Save the entire molecule
        xyz = xyz[xyz[:, -1] != 0]
        true_mol = ase.Atoms(
            positions=xyz[:, :3],
            numbers=xyz[:, -1].astype(int),
            cell=scan_dim
        )
        true_mol.center(axis=(0, 1))
        true_mol.positions[:, 2] -= true_mol.get_positions()[:, 2].max() - z_cutoff
        save_index = start_save_idx + sample
        io.write(f'{outdir}/{save_index:02}_true.xyz', true_mol)

        xyz = true_mol.get_positions()
        elements = true_mol.get_atomic_numbers()
        plot_molecule(
            ax=axes[0, 2],
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            elements=elements,
            xlim=(0, scan_dim[0]),
            ylim=(0, scan_dim[1]),
        )
        axes[0, 2].set_title(f"Entire molecule")
        
        plot_molecule(
            ax=axes[1, 2],
            x=xyz[:, 0],
            y=xyz[:, 2],
            z=xyz[:, 1],
            elements=elements,
            xlim=(0, scan_dim[0]),
            ylim=(xyz[:, 2].min()-2, scan_dim[2]+1),
            z_cutoff=z_cutoff
        )
        axes[1, 2].set_title(f"side view")

        save_index = start_save_idx + sample
        plt.savefig(f'{outdir}/{save_index:02}_molecules.png')
        plt.close()


def plot_molecule(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    elements: np.ndarray, # atomic numbers
    show_bonds: bool = True,
    xlim = (0, 16),
    ylim = (0, 16),
    z_cutoff: float = None
) -> None:
    
    # Convert atomic numbers to colors
    colors = [NUMBER_TO_COLOR[elem] for elem in elements]

    # Compute sizes based on z
    sizes = np.clip(z, .5, 1) * 40

    ax.scatter(x, y, s=sizes, c=colors, edgecolors='black')

    if z_cutoff is not None:
        ax.axhline(y.max() - z_cutoff, color='black', linestyle='--')

    if show_bonds:
        pos = np.stack([x, y, z], axis=-1)
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                bond_length = ase.data.covalent_radii[elements[i]] + ase.data.covalent_radii[elements[j]]
                if np.linalg.norm(pos[i] - pos[j]) < bond_length * 1.1:                
                    ax.plot([x[i], x[j]], [y[i], y[j]], color='black', linewidth=0.5, zorder=-1)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect('equal')
