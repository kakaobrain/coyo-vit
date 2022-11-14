import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp


def padded_center_crop(image, size=224, CROP_PADDING=32):
    """Crops to center of image with padding."""
    original_shape = tf.shape(image)
    image_height = original_shape[0]
    image_width = original_shape[1]

    padded_center_crop_size = tf.cast(
        ((size / (size + CROP_PADDING)) * tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    # crop_window = tf.stack([offset_height, offset_width, padded_center_crop_size, padded_center_crop_size])
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, padded_center_crop_size,
                                          padded_center_crop_size)
    return image


def distorted_bounding_box_crop(image,
                                bbox=None,
                                min_object_covered=0.1,
                                aspect_ratio_range=(3. / 4., 4. / 3.),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
      image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged
        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
        image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding box
        supplied.
      aspect_ratio_range: An optional list of `floats`. The cropped area of the
        image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `floats`. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
      scope: Optional scope for name_scope.
    Returns:
      A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    with tf1.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image  # , distort_bbox


def cutmix_mask(alpha, h, w):
    """Returns image mask for CutMix."""
    r_x = tf.random.uniform([], 0, w, tf.int32)
    r_y = tf.random.uniform([], 0, h, tf.int32)

    area = tfp.distributions.Beta(alpha, alpha).sample()
    patch_ratio = tf.cast(tf.math.sqrt(1 - area), tf.float32)
    r_w = tf.cast(patch_ratio * tf.cast(w, tf.float32), tf.int32)
    r_h = tf.cast(patch_ratio * tf.cast(h, tf.float32), tf.int32)
    bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
    bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
    bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
    bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

    # Create the binary mask.
    pad_left = bbx1
    pad_top = bby1
    pad_right = tf.maximum(w - bbx2, 0)
    pad_bottom = tf.maximum(h - bby2, 0)
    r_h = bby2 - bby1
    r_w = bbx2 - bbx1

    mask = tf.pad(
        tf.ones((r_h, r_w)),
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
        mode='CONSTANT',
        constant_values=0)
    mask.set_shape((h, w))
    return mask[..., None]  # Add channel dim.


def cutmix(image, label, mask):
    """Applies CutMix regularization to a batch of images and labels.
    Reference: https://arxiv.org/pdf/1905.04899.pdf
    Arguments:
      image: a Tensor of batched images.
      label: a Tensor of batched labels.
      mask: a Tensor of batched masks.
    Returns:
      A new dict of features with updated images and labels with the same
      dimensions as the input with CutMix regularization applied.
    """
    # actual area of cut & mix pixels
    mix_area = tf.reduce_sum(mask) / tf.cast(tf.size(mask), mask.dtype)
    mask = tf.cast(mask, image.dtype)
    mixed_image = (1. - mask) * image + mask * image[::-1]
    mix_area = tf.cast(mix_area, label.dtype)
    mixed_label = (1. - mix_area) * label + mix_area * label[::-1]

    return mixed_image, mixed_label


def mixup(batch_size, alpha, image, label):
    """Applies Mixup regularization to a batch of images and labels.
    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412
    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      image: a Tensor of batched images.
      label: a Tensor of batch labels.
    Returns:
      A new dict of features with updated images and labels with the same
      dimensions as the input with Mixup regularization applied.
    """
    mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    img_weight = tf.cast(
        tf.reshape(mix_weight, [batch_size, 1, 1, 1]), image.dtype)
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    image = image * img_weight + image[::-1] * (1. - img_weight)
    label_weight = tf.cast(mix_weight, label.dtype)
    label = label * label_weight + label[::-1] * (1 - label_weight)
    return image, label


def mixing(batch_size, mixup_alpha, cutmix_alpha, features, labels):
    """Applies mixing regularization to a batch of images and labels.
    Arguments:
      batch_size: The input batch size for images and labels.
      mixup_alpha: Float that controls the strength of Mixup regularization.
      cutmix_alpha: FLoat that controls the strenght of Cutmix regularization.
      features: a dict of batched images.
      labels: a dict of batched labels.
    Returns:
      A new dict of features with updated images and labels with the same
      dimensions as the input.
    """
    image, label = features['image'], labels['label']
    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        # split the batch half-half, and aplly mixup and cutmix for each half.
        bs = batch_size // 2
        img1, lab1 = mixup(bs, mixup_alpha, image[:bs], label[:bs])
        img2, lab2 = cutmix(image[bs:], label[bs:], features['cutmix_mask'][bs:])
        features['image'] = tf.concat([img1, img2], axis=0)
        labels['label'] = tf.concat([lab1, lab2], axis=0)
    elif mixup_alpha > 0.0:
        features['image'], labels['label'] = mixup(batch_size, mixup_alpha, image, label)
    elif cutmix_alpha > 0.0:
        features['image'], labels['label'] = cutmix(image, label, features['cutmix_mask'])
    return features, labels


def resize_smaller_aspect_ratio(image, size=256):
    shape = tf.shape(image)
    height, width = shape[-3], shape[-2]
    if height > width:
        ratio = height / width
        resize_shape = (int(ratio * size), size)
    else:
        ratio = width / height
        resize_shape = (size, int(ratio * size))

    return tf.image.resize(
        image, **{
            'method': 'bicubic',
            'size': resize_shape
        })


def vit_inception_crop(
        image,
        size=224,
        area_range=(0.05, 1.0),
        min_object_covered=0.0,
        scope=None,
):
    original_shape = tf.shape(image)
    channels = image.shape[-1]
    bbox = tf.zeros([0, 0, 4], tf.float32)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        original_shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        area_range=area_range,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    image = tf.slice(image, bbox_begin, bbox_size)
    image.set_shape([None, None, channels])

    image = tf.image.resize(image, [size, size])
    return image


def inception_center_crop(image, size=224):
    CROP_PADDING = 32
    original_shape = tf.shape(image)
    image_height = original_shape[0]
    image_width = original_shape[1]

    padded_center_crop_size = tf.cast(
        (
                (size / (size + CROP_PADDING))
                * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    # crop_window = tf.stack([offset_height, offset_width, padded_center_crop_size, padded_center_crop_size])
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        padded_center_crop_size,
        padded_center_crop_size,
    )
    return image


def InceptionCrop(image, size=224):
    shape = tf.shape(image)
    box_begin, box_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.05, 1.),
        min_object_covered=0,
        use_image_if_no_bounding_boxes=True
    )
    image = tf.slice(image, box_begin, box_size)
    image.set_shape([None, None, 3])

    image = tf.cast(tf.image.resize(image, [size, size]), tf.float32)

    return image


def ResizeSmall(image, size):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    ratio = (
            tf.cast(size, tf.float32) /
            tf.cast(tf.minimum(h, w), tf.float32)
    )
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
    resized_image = tf.image.resize(image, [h, w], method="area")

    return resized_image


def CentralCrop(image, size):
    crop_size = [size, size]
    h, w = crop_size[0], crop_size[1]
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    cropped_image = tf.image.crop_to_bounding_box(image, dy, dx, h, w)

    cropped_image = tf.cast(cropped_image, tf.float32)
    return cropped_image
