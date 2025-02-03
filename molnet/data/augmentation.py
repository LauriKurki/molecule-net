import math
import random

import tensorflow as tf
from .rotate import rotate

from typing import Tuple


def normalize_images(images):
    """
    Normalizes a stack of 2D images to zero mean and unit variance.

    Args:
        images (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).

    Returns:
        tf.Tensor: The normalized images.
    """

    # Compute the mean and standard deviation of each slice
    xmean = tf.reduce_mean(images, axis=(0, 1), keepdims=True)
    xstd = tf.math.reduce_std(images, axis=(0, 1), keepdims=True)

    # Normalize each slice separately
    return (images - xmean) / xstd

def add_noise(
    x: tf.Tensor,
    noise_amp: float = 0.1,
    random_amplitude: bool = True
):
    """
    Add random noise to a stack of 2D images.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        noise_amp (float): The amplitude of the noise to be added.
        random_amplitude (bool): If True, the noise amplitude is randomly chosen between 0 and noise_amp.
    """

    # Choose amplitude
    if random_amplitude:
        noise_amp = tf.random.uniform([], minval=0, maxval=noise_amp)

    noise = tf.random.uniform(tf.shape(x), minval=-0.5, maxval=0.5) * noise_amp

    # scale noise by (max-min) of x
    xmin = tf.reduce_min(x)
    xmax = tf.reduce_max(x)

    x = x + noise * (xmax - xmin)

    return x

def center_crop(
    x: tf.Tensor,
    y: tf.Tensor,
    size: int,
    shift: int = 0):
    """
    Center-crops a stack of 2D images to the specified size.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        y (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        size (int): The size of the crop.
        shift (int): max-shift of the crop.

    Returns:
        tf.Tensor: The center-cropped images.
    """

    # Compute the crop size
    crop_size = tf.constant([size, size])

    # Compute the starting point of the crop
    start = (tf.shape(x)[:2] - crop_size) // 2
    shift = tf.random.uniform([2], minval=-shift, maxval=shift, dtype=tf.int32)
    start += shift

    # Crop the images
    x = x[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1]]
    y = y[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1]]

    return x, y

def random_crop(x: tf.Tensor, y: tf.Tensor, size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly crops a stack of 2D images and corresponding atom maps to the specified size.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        y (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        size (int): The size of the crop.

    Returns:
        (tf.Tensor, tf.Tensor): The cropped images and atom maps.
    """

    # Define the excluded border size
    border = 0.1
    border_px = tf.cast(size * border, tf.int32)

    # Compute the starting points of the crop
    start_x = tf.random.uniform([], minval=border_px, maxval=x.shape[0] - size - border_px, dtype=tf.int32)
    start_y = tf.random.uniform([], minval=border_px, maxval=x.shape[1] - size - border_px, dtype=tf.int32)

    # Crop the images
    x = x[start_x:start_x + size, start_y:start_y + size]
    y = y[start_x:start_x + size, start_y:start_y + size]

    return x, y


def random_rotate(
    x: tf.Tensor,
    xyz: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Rotates the input tensors by a random angle in the xy-plane.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        xyz (tf.Tensor): A 2D tensor of shape (N, 5) representing coordinates.

    Returns:
        (tf.Tensor, tf.Tensor): The rotated tensors.
    """

    # Rotate the tensors in the xy-plane by a random angle
    angle = random.uniform(0, 360)
    radians = angle * math.pi / 180

    rotation_matrix = tf.constant(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)]
        ]
    )

    # Apply rotation to xyz coordinates
    # First, we need to translate the coordinates to the origin
    xy_mean = tf.reduce_mean(xyz[:, :2], axis=0, keepdims=True)
    xy_centered = xyz[:, :2] - xy_mean

    # Rotate the centered coordinates and translate back
    xyz_rot = tf.matmul(xy_centered, rotation_matrix) + xy_mean

    # Concatanate the rotated coordinates with the original z-coordinate
    xyz_rot = tf.concat([xyz_rot, xyz[:, 2:]], axis=1)

    # Transpose tensor [X, Y, Z, C] -> [Z, X, Y, C] temporarily for rotation
    x = tf.transpose(x, [2, 0, 1, 3])
    x_rot = rotate(
        x,
        angle,
        fill_mode='nearest',
        interpolation='bilinear'
    )

    # Transpose back to original shape
    x_rot = tf.transpose(x_rot, [1, 2, 0, 3])

    return x_rot, xyz_rot


def random_rotate_image_and_atom_map(
    x: tf.Tensor,
    y: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply same random rotation to images and atom maps.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        y (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).

    Returns:
        (tf.Tensor, tf.Tensor): The rotated tensors.
    """

    angle = random.uniform(0, 360)

    # Transpose tensors [X, Y, Z, C] -> [Z, X, Y, C] temporarily for rotation
    x = tf.transpose(x, [2, 0, 1, 3])
    y = tf.transpose(y, [2, 0, 1, 3])
    x_rot = rotate(
        x,
        angle,
        fill_mode='nearest',
        interpolation='bilinear'
    )
    y_rot = rotate(
        y,
        angle,
        fill_mode='nearest',
        interpolation='bilinear'
    )

    # Transpose back to original shape
    x_rot = tf.transpose(x_rot, [1, 2, 0, 3])
    y_rot = tf.transpose(y_rot, [1, 2, 0, 3])

    return x_rot, y_rot


def add_random_cutouts(images, cutout_probs, cutout_size_range, image_size):
    """
    Adds random black patches to a 3D stack of 2D images.

    Args:
        images (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        cutout_probs (list): A list of probabilities for the number of patches to add.
        cutout_size_range (tuple): A tuple (min_size, max_size) defining the range of patch sizes.

    Returns:
        tf.Tensor: The tensor with black patches added, same shape as the input.
    """
    def add_patches_to_slice(slice_2d):
        # Sample the number of patches using weighted probabilities
        num_patches = tf.random.categorical(
            logits=tf.math.log([cutout_probs]), num_samples=1
        )[0, 0]
        for _ in range(num_patches):
            patch_x_size = tf.random.uniform([], minval=cutout_size_range[0], maxval=cutout_size_range[1] + 1, dtype=tf.int32)
            patch_y_size = tf.random.uniform([], minval=cutout_size_range[0], maxval=cutout_size_range[1] + 1, dtype=tf.int32)
            x = tf.random.uniform([], minval=0, maxval=image_size - patch_x_size, dtype=tf.int32)
            y = tf.random.uniform([], minval=0, maxval=image_size - patch_y_size, dtype=tf.int32)

            # Create a patch mask and apply it
            slice_2d = tf.tensor_scatter_nd_update(
                slice_2d,
                indices=tf.reshape(
                    tf.stack(tf.meshgrid(tf.range(x, x + patch_x_size), tf.range(y, y + patch_y_size), indexing="ij"), axis=-1),
                    [-1, 2]
                ),
                updates=tf.zeros([patch_x_size * patch_y_size, slice_2d.shape[-1]], dtype=slice_2d.dtype)
            )
        return slice_2d

    # Transpose to bring Z (third dimension) to the first dimension
    transposed_images = tf.transpose(images, perm=[2, 0, 1, 3])
    
    # Apply the function to each slice along the new first dimension (original Z)
    updated_images = tf.map_fn(lambda slice_2d: add_patches_to_slice(slice_2d), transposed_images)
    
    # Transpose back to the original shape
    return tf.transpose(updated_images, perm=[1, 2, 0, 3])


def random_slice_shift(x, max_shift_per_slice: float = 0.02, max_total_shift: float = 0.05):
    """
    Randomly and independently shift each slice of the 3D stack with respect to the previous slice.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        max_shift_per_slice (float): The maximum fraction of image size to shift.
    """

    shifts = tf.random.uniform((x.shape[-2] - 1, 2), -max_shift_per_slice, max_shift_per_slice)

    # Translate into cumulative shift, first slice remains unchanged so we prepend a zero shift
    cumulative_shifts = tf.concat([[[0, 0]], tf.cumsum(shifts, axis=0)], axis=0)

    # Clip values to ensure shifts are within the allowed range
    cumulative_shifts = tf.clip_by_value(cumulative_shifts, -max_total_shift, max_total_shift)

    # Reverse so that the last slice stays unchanged
    cumulative_shifts = tf.reverse(cumulative_shifts, axis=[0])

    # Translate fractional shift into pixel shift
    xy_shape = tf.cast(tf.shape(x)[:2], tf.float32)
    pixel_shifts = tf.round(cumulative_shifts * xy_shape)
    pixel_shifts = tf.cast(pixel_shifts, tf.int32)

    # Apply shifts to each slice by rolling the array
    shifted_x = tf.stack(
        [
            tf.roll(
                x[..., i, :], shift=pixel_shifts[i], axis=(0, 1)
            ) for i in range(x.shape[-2])
        ], axis=-2
    )
    return shifted_x
