import math
import tensorflow as tf

from typing import Any


def rotate(
    image: tf.Tensor,
    degrees: float,
    interpolation: str = 'nearest',
    fill_mode: str = 'reflect',
    fill_value: float = 0.0,
) -> tf.Tensor:
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.

  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = tf.cast(degrees * degrees_to_radians, tf.float32)

  image_height = tf.cast(tf.shape(image)[1], tf.float32)
  image_width = tf.cast(tf.shape(image)[2], tf.float32)
  transforms = _convert_angles_to_transform(
      angles=radians, image_width=image_width, image_height=image_height)
  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = transform(
    image,
    transforms=transforms,
    interpolation=interpolation,
    fill_mode=fill_mode,
    fill_value=fill_value,
  )
  return image


def to_4d(image: tf.Tensor) -> tf.Tensor:
  """Converts an input Tensor to 4 dimensions.

  4D image => [N, H, W, C] or [N, C, H, W]
  3D image => [1, H, W, C] or [1, C, H, W]
  2D image => [1, H, W, 1]

  Args:
    image: The 2/3/4D input tensor.

  Returns:
    A 4D image tensor.

  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.

  """
  shape = tf.shape(image)
  original_rank = tf.rank(image)
  left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
  right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
  new_shape = tf.concat(
      [
          tf.ones(shape=left_pad, dtype=tf.int32),
          shape,
          tf.ones(shape=right_pad, dtype=tf.int32),
      ],
      axis=0,
  )
  return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
  """Converts a 4D image back to `ndims` rank."""
  shape = tf.shape(image)
  begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
  end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
  new_shape = shape[begin:end]
  return tf.reshape(image, new_shape)

def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
                                 image_height: tf.Tensor) -> tf.Tensor:
  """Converts an angle or angles to a projective transform.

  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8).

  Raises:
    `TypeError` if `angles` is not rank 0 or 1.

  """
  angles = tf.convert_to_tensor(angles, dtype=tf.float32)
  if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
    angles = angles[None]
  elif len(angles.get_shape()) != 1:
    raise TypeError('Angles should have a rank 0 or 1.')
  x_offset = ((image_width - 1) -
              (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
               (image_height - 1))) / 2.0
  y_offset = ((image_height - 1) -
              (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
               (image_height - 1))) / 2.0
  num_angles = tf.shape(angles)[0]
  return tf.concat(
      values=[
          tf.math.cos(angles)[:, None],
          -tf.math.sin(angles)[:, None],
          x_offset[:, None],
          tf.math.sin(angles)[:, None],
          tf.math.cos(angles)[:, None],
          y_offset[:, None],
          tf.zeros((num_angles, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def _apply_transform_to_images(
    images,
    transforms,
    fill_mode='reflect',
    fill_value=0.0,
    interpolation='bilinear',
    output_shape=None,
    name=None,
):
  """Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape `(num_images, num_rows, num_columns,
      num_channels)` (NHWC). The rank must be statically known (the shape is
      not `TensorShape(None)`).
    transforms: Projective transform matrix/matrices. A vector of length 8 or
      tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1,
      b2, c0, c1], then it maps the *output* point `(x, y)` to a transformed
      *input* point `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) /
      k)`, where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared
      to the transform mapping input points to output points. Note that
      gradients are not backpropagated into transformation parameters.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
    fill_value: a float represents the value to be filled outside the
      boundaries when `fill_mode="constant"`.
    interpolation: Interpolation mode. Supported values: `"nearest"`,
      `"bilinear"`.
    output_shape: Output dimension after the transform, `[height, width]`. If
      `None`, output is the same size as input image.
    name: The name of the op.  Fill mode behavior for each valid value is as
      follows
      - `"reflect"`: `(d c b a | a b c d | d c b a)` The input is extended by
      reflecting about the edge of the last pixel.
      - `"constant"`: `(k k k k | a b c d | k k k k)` The input is extended by
      filling all values beyond the edge with the same constant value k = 0.
      - `"wrap"`: `(a b c d | a b c d | a b c d)` The input is extended by
      wrapping around to the opposite edge.
      - `"nearest"`: `(a a a a | a b c d | d d d d)` The input is extended by
      the nearest pixel.  Input shape: 4D tensor with shape:
      `(samples, height, width, channels)`, in `"channels_last"` format.
      Output shape: 4D tensor with shape: `(samples, height, width, channels)`,
      in `"channels_last"` format.

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.
  """
  with tf.name_scope(name or 'transform'):
    if output_shape is None:
      output_shape = tf.shape(images)[1:3]
      if not tf.executing_eagerly():
        output_shape_value = tf.get_static_value(output_shape)
        if output_shape_value is not None:
          output_shape = output_shape_value

    output_shape = tf.convert_to_tensor(
        output_shape, tf.int32, name='output_shape'
    )

    if not output_shape.get_shape().is_compatible_with([2]):
      raise ValueError(
          'output_shape must be a 1-D Tensor of 2 elements: '
          'new_height, new_width, instead got '
          f'output_shape={output_shape}'
      )

    fill_value = tf.convert_to_tensor(fill_value, tf.float32, name='fill_value')

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        output_shape=output_shape,
        fill_value=fill_value,
        transforms=transforms,
        fill_mode=fill_mode.upper(),
        interpolation=interpolation.upper(),
    )


def transform(
    image: tf.Tensor,
    transforms: Any,
    interpolation: str = 'nearest',
    output_shape=None,
    fill_mode: str = 'reflect',
    fill_value: float = 0.0,
) -> tf.Tensor:
  """Transforms an image."""
  original_ndims = tf.rank(image)
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  if transforms.shape.rank == 1:
    transforms = transforms[None]
  image = to_4d(image)
  image = _apply_transform_to_images(
      images=image,
      transforms=transforms,
      interpolation=interpolation,
      fill_mode=fill_mode,
      fill_value=fill_value,
      output_shape=output_shape,
  )
  return from_4d(image, original_ndims)