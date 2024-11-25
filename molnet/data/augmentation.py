import tensorflow as tf


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
    return (images - xmean) / (xstd + 1e-9)


def random_rotate_3d_stacks(x, y):
    """
    Randomly rotates two 3D stacks of 2D images in the XY plane by 0, 90, 180, or 270 degrees.
    The same rotation is applied to both stacks.

    Args:
        x (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        y (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).

    Returns:
        tuple: Rotated stacks (x_rotated, y_rotated) with the same shapes as inputs.
    """
    # Choose a random rotation (0, 1, 2, or 3 corresponding to 0°, 90°, 180°, 270°)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

    # Define rotation operations
    def rotate_90(tensor):
        return tf.transpose(tf.reverse(tensor, axis=[1]), perm=[1, 0, 2, 3])

    def rotate_180(tensor):
        return tf.reverse(tensor, axis=[0, 1])

    def rotate_270(tensor):
        return tf.transpose(tf.reverse(tensor, axis=[0]), perm=[1, 0, 2, 3])

    # Apply the same rotation to both stacks
    x_rotated = tf.switch_case(
        branch_index=k,
        branch_fns=[
            lambda: x,          # 0°: No rotation
            lambda: rotate_90(x),  # 90°
            lambda: rotate_180(x), # 180°
            lambda: rotate_270(x)  # 270°
        ]
    )

    y_rotated = tf.switch_case(
        branch_index=k,
        branch_fns=[
            lambda: y,          # 0°: No rotation
            lambda: rotate_90(y),  # 90°
            lambda: rotate_180(y), # 180°
            lambda: rotate_270(y)  # 270°
        ]
    )

    return x_rotated, y_rotated


def random_flip_3d_stacks(images, targets):
    """
    Randomly flips two 3D stacks of 2D images along the X or Y axis.
    The same flip is applied to both stacks.

    Args:
        images (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).
        targets (tf.Tensor): A 4D tensor of shape (X, Y, Z, channels).

    Returns:
        tuple: Flipped stacks (images_flipped, targets_flipped) with the same shapes as inputs.
    """

    # Choose a random flip (0 or 1 corresponding to no flip or flip along the X or Y axis)
    k = tf.random.uniform([], minval=0, maxval=3, dtype=tf.int32)

    # Define flip operations
    def flip_x(tensor):
        return tf.reverse(tensor, axis=[0])

    def flip_y(tensor):
        return tf.reverse(tensor, axis=[1])

    # Apply the same flip to both stacks
    images_flipped = tf.switch_case(
        branch_index=k,
        branch_fns=[
            lambda: images,        # No flip
            lambda: flip_x(images),  # Flip along the X axis
            lambda: flip_y(images)  # Flip along the Y axis
        ]
    )

    targets_flipped = tf.switch_case(
        branch_index=k,
        branch_fns=[
            lambda: targets,          # No flip
            lambda: flip_x(targets),  # Flip along the X axis
            lambda: flip_y(targets)  # Flip along the Y axis
        ]
    )

    return images_flipped, targets_flipped


def add_random_cutouts(images, cutout_probs, cutout_size_range):
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
            x = tf.random.uniform([], minval=0, maxval=slice_2d.shape[0] - patch_x_size, dtype=tf.int32)
            y = tf.random.uniform([], minval=0, maxval=slice_2d.shape[1] - patch_y_size, dtype=tf.int32)

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
