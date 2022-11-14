"""
Mixup: Mixup Augmentation
"""
import tensorflow as tf

AUTO = tf.data.AUTOTUNE


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    """
    sample beta distribution
    Parameter:
        size: data_size
    return:
        gamma_sample: gamma with same size
    """
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    gamma_sample = gamma_1_sample / (gamma_1_sample + gamma_2_sample)
    return gamma_sample


def mixup(ds_one, ds_two, alpha=0.2):
    """
    mixup
    Parameter:
        ds_one: data 1
        ds_two: data 2
        alpha: hyper-parameter
    return:
        images: mixup_ed images
        labels: mixup_ed labels
    """
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)
