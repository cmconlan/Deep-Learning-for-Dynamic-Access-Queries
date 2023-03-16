import tensorflow as tf


def eye(N, name=None):
    """
    Returns NxN identity matrix.
    """
    return tf.diag(tf.ones(tf.stack([N, ]), dtype=tf.float32), name=name)
