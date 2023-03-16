import os
import math

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import methods.tf_utils as tf_utils

# Custom pseudoinverse op
curr_dir = os.path.dirname(os.path.realpath(__file__))
#pinv_module = tf.load_op_library(os.path.join(curr_dir, 'pinv_svd_log.so'))


# class GaussianProcess_WithLocs():
#     """
#     Builds computation graph for vanilla GP.
#     """
#     def __init__(self, sigma_l, sigma_f, sigma_n=None, train_features=None,
#                  test_features=None, train_locs=None, test_locs=None,
#                  test=False, method='pseudo', reuse=False, alpha=1,
#                  constant_mean=False, kernel='squared_exponential'):
#         self.sigma_l = sigma_l
#         self.sigma_f = sigma_f
#         self.sigma_n = sigma_n

#         self.sigma_l_loc = sigma_l
#         self.sigma_f_loc = sigma_f
#         self.method = method
#         self.alpha = alpha
#         self.constant_mean = constant_mean
#         self.kernel = kernel
#         self.set_up_graph(train_features, test_features, train_locs, test_locs, test, reuse)

#     def set_up_graph(self, train_features, test_features, train_locs, test_locs, test, reuse):
#         """
#         Builds computation graph.
#         """
#         with tf.device('/cpu:0'):
#             self._create_training_data(train_features, train_locs)
#             self.set_sigmas(reuse)
#             self._compute_lml_components()
#             self._compute_terms()
#             self.lml = tf.add_n(
#                 [self.data, self.det, self.constant], name='lml')
#             if test_features is not None or test:
#                 self.add_test_graph(test_features, test_locs)

#     def set_sigmas(self, reuse):
#         """
#         Sets hyperparameters of kernel function.
#         """
#         with tf.variable_scope('sigmas', reuse=reuse):
#             self.sigma_l_tf = tf.get_variable(
#                 name='sigma_l', shape=[1], dtype=tf.float32,
#                 initializer=tf.constant_initializer(self.sigma_l),
#                 trainable=True)
#             train_sigma_f = (self.kernel == 'squared_exponential')
#             self.sigma_f_tf = tf.get_variable(
#                 name='sigma_f', shape=[1], dtype=tf.float32,
#                 initializer=tf.constant_initializer(self.sigma_f),
#                 trainable=train_sigma_f)

#             if self.sigma_n is not None:
#                 self.log_sigma_n_tf = tf.get_variable(
#                     name='log_sigma_n', shape=[1], dtype=tf.float32,
#                     initializer=tf.constant_initializer(
#                         np.log(self.sigma_n)), trainable=True)
#                 self.sigma_n_tf = tf.exp(self.log_sigma_n_tf, name='sigma_n')

#         with tf.variable_scope('sigmas_loc', reuse=reuse):
#             self.sigma_l_tf_loc = tf.get_variable(
#                 name='sigma_l_loc', shape=[1], dtype=tf.float32,
#                 initializer=tf.constant_initializer(self.sigma_l_loc),
#                 trainable=True)
#             train_sigma_f_loc = (self.kernel == 'squared_exponential')
#             self.sigma_f_tf_loc = tf.get_variable(
#                 name='sigma_f_loc', shape=[1], dtype=tf.float32,
#                 initializer=tf.constant_initializer(self.sigma_f_loc),
#                 trainable=train_sigma_f_loc)

#     def add_test_graph(self, test_features=None, test_locs=None):
#         """
#         Add test examples to TF GP graph.
#         """
#         self._create_test_data(test_features, test_locs)
#         self._compute_test_posterior()
#         self._compute_compound_loss()
#         self._compute_test_accuracy()

#     def _compute_test_accuracy(self):
#         """
#         Computes some metrics to assess the accuracy of test predictions.
#         """
#         # Compute sum of squared residuals
#         self.residuals = tf.subtract(
#             self.y_test_mean, self.y_test, name='residuals')
#         self.ss_res = tf.reduce_sum(tf.square(self.residuals), name='ss_res')
#         # Compute total sum of squares
#         self.ss_tot = tf.reduce_sum(
#             tf.square(self.y_test - tf.reduce_mean(self.y_test)),
#             name='ss_tot')
#         # Average total sum of squares
#         self.ss_tot_mean = tf.div(
#             self.ss_tot, tf.to_float(tf.shape(self.y_test)[0]),
#             name='ss_tot_mean')
#         # Compute MSE
#         self.mean_squared_error = tf.div(
#             tf.reduce_sum(tf.square(self.residuals)),
#             tf.to_float(tf.shape(self.y_test)[0]), name='mean_squared_error')
#         # Compute R^2
#         self.R_squared = tf.subtract(
#             1.0, tf.div(self.ss_res, self.ss_tot), name='R_squared')

#     def _create_test_data(self, test_features, test_locs):
#         """
#         Adds test data.
#         """
#         if test_features is not None:
#             self.X_test = test_features
#         else:
#             self.X_test = tf.placeholder(
#                 dtype=tf.float32, shape=[None, None], name='X_test')
#         if test_locs is not None:
#             self.locs_test = test_locs
#         else:
#             self.locs_test = tf.placeholder(
#                 dtype=tf.float32, shape=[None, 2], name='locs_test')
#         self.y_test = tf.placeholder(
#             dtype=tf.float32, shape=[None, None], name='y_test')

#     def _compute_test_posterior(self):
#         """
#         Computes posterior of test data.
#         """
#         # Compute covariance matrix blocks
#         self._compute_covariance_blocks()
#         # Compute test means and covariances from only spatial correlations
#         # (no features)
#         self.y_test_mean = tf.matmul(
#             self.K_21, self.Ky_inv_y, name='y_test_mean')
#         if self.constant_mean:
#             self.y_test_mean += tf.reduce_mean(self.y_train_original)
#         self.y_test_cov = tf.subtract(
#             self.K_22, tf.matmul(self.K_21, self.K_11_inv_K_12),
#             name='y_test_cov')

#     def _compute_compound_loss(self):
#         """
#         Computes semi-supervised loss that is a weighted sum of the negative
#         log marginal likelihood and the sum of test variances.
#         """
#         # Compute sum of test variances
#         self.sum_test_variances = tf.reduce_sum(tf.multiply(
#             tf_utils.eye(tf.shape(self.y_test_cov)[0]),
#             self.y_test_cov), name='sum_test_variances')
#         # Compute weighted compound loss for semi-supervised training
#         self.lml_component = tf.div(
#             tf.negative(self.lml), tf.to_float(tf.shape(self.X_train)[0]),
#             name='lml_component')
#         self.sum_test_variances_component = tf.multiply(
#             (self.alpha / (tf.to_float(tf.shape(self.X_test)[0]))),
#             self.sum_test_variances,
#             name='sum_test_variances_component')
#         self.semisup_loss = tf.add(
#             self.lml_component, self.sum_test_variances_component,
#             name='semisup_loss')

#     def _compute_covariance_blocks(self):
#         """
#         Computes covariance blocks used for computing GP posterior.
#         """
#         # TODO
#         self.K_11 = tf.identity(self.Ky, name='K_11')
#         self.K_12 = self._compute_covariance_matrix(
#             self.X_train, self.X_test, self.sigma_l_tf, self.sigma_f_tf, name='K_12')
#         self.K_12 += self._compute_covariance_matrix(
#             self.locs_train, self.locs_test, self.sigma_l_tf_loc, self.sigma_f_tf_loc, name='K_12_loc')
#         self.K_21 = self._compute_covariance_matrix(
#             self.X_test, self.X_train, self.sigma_l_tf, self.sigma_f_tf)
#         self.K_21 += self._compute_covariance_matrix(
#             self.locs_test, self.locs_train, self.sigma_l_tf_loc, self.sigma_f_tf_loc)
#         self.K_22 = self._compute_covariance_matrix(
#             self.X_test, self.X_test, self.sigma_l_tf, self.sigma_f_tf)
#         self.K_22 += self._compute_covariance_matrix(
#             self.locs_test, self.locs_test, self.sigma_l_tf_loc, self.sigma_f_tf_loc)
#         if self.method == 'pseudo':
#             self.K_11_inv = pinv_module.pseudo_inverse_no_det(
#                 self.K_11, name='K_11_inv')
#             self.K_11_inv_K_12 = tf.matmul(
#                 self.K_11_inv, self.K_12, name='K_11_inv_K_12')
#         elif self.method == 'inverse':
#             self.K_11_inv = tf.matrix_inverse(
#                 self.K_11, name='K_11_inv')
#             self.K_11_inv_K_12 = tf.matmul(
#                 self.K_11_inv, self.K_12, name='K_11_inv_K_12')

#     def _create_training_data(self, train_features, train_locs):
#         """
#         Creates placeholders for training data.
#         """
#         if train_features is not None:
#             self.X_train = train_features
#         else:
#             self.X_train = tf.placeholder(
#                 dtype=tf.float32, shape=[None, None], name='X_train')
#         if train_locs is not None:
#             self.locs_train = train_locs
#         else:
#             self.locs_train = tf.placeholder(
#                 dtype=tf.float32, shape=[None, 2], name='locs_train')
#         self.y_train_original = tf.placeholder(
#             dtype=tf.float32, shape=[None, None], name='y_train_original')
#         if self.constant_mean:
#             self.y_train = self.y_train_original - tf.reduce_mean(
#                 self.y_train_original)
#         else:
#             self.y_train = self.y_train_original

#     def _compute_lml_components(self):
#         """
#         Computes intermediate values needed for lml computation.
#         """
#         self.n_tf = tf.to_int32(tf.shape(self.y_train)[0], name='n_tf')
#         self._compute_Ky()
#         self._compute_Ky_inv()

#     def _compute_Ky(self):
#         """
#         Computes the train covariance matrix.
#         """
#         self.K = self._compute_covariance_matrix(self.X_train, self.X_train, self.sigma_l_tf, self.sigma_f_tf, name='K')\
#             + self._compute_covariance_matrix(self.locs_train, self.locs_train, self.sigma_l_tf_loc, self.sigma_f_tf_loc, name='K_loc')
#         # Add diagonal for noisy observations
#         if self.sigma_n is not None:
#             self.Ky = tf.add(
#                 self.K, tf.multiply(
#                     tf.square(self.sigma_n_tf), tf_utils.eye(self.n_tf)),
#                 name='Ky')
#         else:
#             self.Ky = tf.identity(self.K, name='Ky')

#     def _compute_covariance_matrix(self, S1, S2, sigma_l, sigma_f, name='cov_matrix'):
#         """
#         Computes the squared exponential covariance matrix.
#         """
#         if self.kernel == 'square_polynomial':
#             return tf.square(tf.square(sigma_l) +\
#                 tf.matmul(S1, tf.transpose(S2)))
#         elif self.kernel == 'squared_exponential':
#             distance_matrix = self._compute_dist_matrix(
#                 S1, S2, name=name+'_dist_matrix')
#             denom = tf.multiply(
#                 tf.constant(2.0, dtype=tf.float32), tf.square(sigma_l),
#                 name='cov_matrix_denom')
#             # Squared exponential covariance
#             exponent = tf.div(-distance_matrix, denom)
#             return tf.multiply(tf.square(sigma_f), tf.exp(exponent), name=name)

#     def _compute_dist_matrix(self, S1, S2, name='distance_matrix'):
#         """
#         Computes the Euclidean distance matrix.
#         """
#         # Compute s1 squared matrix
#         s1_squared_norm = tf.reduce_sum(
#             tf.square(S1), reduction_indices=1, keep_dims=True)
#         # Compute s2 squared matrix
#         s2_squared_norm = tf.reduce_sum(
#             tf.square(tf.transpose(S2)), reduction_indices=0, keep_dims=True)
#         term1 = tf.matmul(s1_squared_norm, tf.ones_like(s2_squared_norm))
#         term2 = tf.matmul(tf.ones_like(s1_squared_norm), s2_squared_norm)
#         # Compute cross term
#         term3 = tf.matmul(S1, S2, transpose_b=True)
#         # Compute distance matrix
#         term4 = tf.multiply(tf.constant(-2.0, dtype=tf.float32), term3)
#         distance_matrix = tf.add_n([term1, term2, term4])
#         return distance_matrix

#     def _compute_Ky_inv(self):
#         """
#         Computes intermediate value Ky_inv to be used in lml computation.
#         """
#         if self.method == 'pseudo':
#             self.Ky_inv, self.Ky_logdet = pinv_module.pseudo_inverse(self.Ky)
#             # Compute Ky_inv_y
#             self.Ky_inv_y = tf.matmul(
#                 self.Ky_inv, self.y_train, name='Ky_inv_y')

#         elif self.method == 'inverse':
#             self.Ky_inv = tf.matrix_inverse(self.Ky, name='Ky_inv')
#             # Compute Ky_inv_y
#             self.Ky_inv_y = tf.matmul(
#                 self.Ky_inv, self.y_train, name='Ky_inv_y')

#         elif self.method == 'constant':
#             # Multiply by a constant, then invert
#             self.Ky_constant = tf.placeholder(
#                 dtype=tf.float32, shape=[1, 1], name='Ky_constant')
#             self.Ky_scaled = tf.multiply(
#                 self.Ky_constant, self.Ky, name='Ky_scaled')
#             self.Ky_scaled_inv = tf.matrix_inverse(
#                 self.Ky_scaled, name='Ky_scaled_inv')
#             # Re-multiply by the same constant
#             self.Ky_inv = tf.multiply(
#                 self.Ky_constant, self.Ky_scaled_inv, name='Ky_inv')
#             # Compute Ky_inv_y
#             self.Ky_inv_y = tf.matmul(
#                 self.Ky_inv, self.y_train, name='Ky_inv_y')

#         elif self.method == 'cholesky':
#             # Use cholesky decomposition to invert
#             self.L_Ky = tf.cholesky(self.Ky, name='L_Ky')
#             self.Ky_inv_y = tf.matrix_solve(
#                 tf.transpose(self.L_Ky),
#                 tf.matrix_solve(self.L_Ky, self.y_train), name='Ky_inv_y')

#         elif self.method == 'constant_cholesky':
#             # Multiply by constant, then use cholesky decomposition to invert
#             self.Ky_constant = tf.placeholder(
#                 dtype=tf.float32, shape=[1, 1], name='Ky_constant')
#             self.Ky_scaled = tf.multiply(
#                 self.Ky_constant, self.Ky, name='Ky_scaled')
#             self.L_Ky_scaled = tf.cholesky(self.Ky_scaled, name='L_Ky_scaled')
#             self.Ky_inv_y_scaled = tf.matrix_solve(
#                 tf.transpose(self.L_Ky_scaled),
#                 tf.matrix_solve(self.L_Ky_scaled, self.y_train),
#                 name='Ky_inv_y_scaled')
#             self.Ky_inv_y = tf.multiply(
#                 self.Ky_constant, self.Ky_inv_y_scaled, name='Ky_inv_y')

#     def _compute_constant(self):
#         """
#         Computes constant term in lml.
#         """
#         self.constant = tf.reshape(tf.negative(
#             (tf.cast(self.n_tf, dtype=tf.float32) / 2.0) *
#             tf.log(2 * math.pi)), shape=[1, 1], name='constant')

#     def _compute_data(self):
#         """
#         Computes first data term in lml.
#         """
#         self.data = tf.reshape(
#             tf.multiply(-0.5, tf.matmul(
#                 tf.transpose(self.y_train), self.Ky_inv_y)), shape=[1, 1],
#             name='data')

#     def _compute_det(self):
#         """
#         Computes determinant terms in lml.
#         """
#         if self.method == 'pseudo':
#             self.det = tf.multiply(-0.5, self.Ky_logdet, name='det')
#         elif self.method == 'inverse':
#             self.det = tf.multiply(
#                 -0.5, tf.log(tf.matrix_determinant(self.Ky)), name='det')
#         else:
#             # Compute log determinant using previous determinant constant and
#             # Cholesky decomposition
#             self.mask_Ky = tf_utils.eye(tf.shape(self.Ky)[0], name='mask_Ky')
#             self.L_Ky = tf.cholesky(self.Ky)
#             self.det = tf.negative(tf.reduce_sum(
#                 self.mask_Ky * tf.log(tf.abs(
#                     0.5 * (self.L_Ky + tf.transpose(self.L_Ky)))),
#                 keep_dims=True), name='det')

#     def _compute_terms(self):
#         """
#         Computes all terms of lml.
#         """
#         self._compute_data()
#         self._compute_det()
#         self._compute_constant()


#%%

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import check_ops

svd = linalg_ops.svd

#%%
def _maybe_validate_matrix(a, validate_args):
  """Checks that input is a `float` matrix."""
  assertions = []
  if not a.dtype.is_floating:
    raise TypeError('Input `a` must have `float`-like `dtype` '
                    '(saw {}).'.format(a.dtype.name))
  if a.shape is not None and a.shape.rank is not None:
    if a.shape.rank < 2:
      raise ValueError('Input `a` must have at least 2 dimensions '
                       '(saw: {}).'.format(a.shape.rank))
  elif validate_args:
    assertions.append(
        check_ops.assert_rank_at_least(
            a, rank=2, message='Input `a` must have at least 2 dimensions.'))
  return assertions

def pinv(a, rcond=None, validate_args=False, name=None):
  """Compute the Moore-Penrose pseudo-inverse of one or more matrices.
  Calculate the [generalized inverse of a matrix](
  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
  singular-value decomposition (SVD) and including all large singular values.
  The pseudo-inverse of a matrix `A`, is defined as: 'the matrix that 'solves'
  [the least-squares problem] `A @ x = b`,' i.e., if `x_hat` is a solution, then
  `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
  `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
  `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]
  This function is analogous to [`numpy.linalg.pinv`](
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
  It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
  default `rcond` is `1e-15`. Here the default is
  `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.
  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
      (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
      set to zero. Must broadcast against `tf.shape(a)[:-2]`.
      Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: 'pinv'.
  Returns:
    a_pinv: (Batch of) pseudo-inverse of input `a`. Has same shape as `a` except
      rightmost two dimensions are transposed.
  Raises:
    TypeError: if input `a` does not have `float`-like `dtype`.
    ValueError: if input `a` has fewer than 2 dimensions.
  #### Examples
  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  a = tf.constant([[1.,  0.4,  0.5],
                   [0.4, 0.2,  0.25],
                   [0.5, 0.25, 0.35]])
  tf.matmul(tf.linalg.pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)
  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tf.linalg.pinv(a), a)
  # ==> array([[ 0.76,  0.37,  0.21, -0.02],
               [ 0.37,  0.43, -0.33,  0.02],
               [ 0.21, -0.33,  0.81,  0.01],
               [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
  ```
  #### References
  [1]: G. Strang. 'Linear Algebra and Its Applications, 2nd Ed.' Academic Press,
       Inc., 1980, pp. 139-142.
  """
  with ops.name_scope(name or 'pinv'):
    a = ops.convert_to_tensor(a, name='a')

    assertions = _maybe_validate_matrix(a, validate_args)
    if assertions:
      with ops.control_dependencies(assertions):
        a = array_ops.identity(a)

    dtype = a.dtype.as_numpy_dtype

    if rcond is None:

      def get_dim_size(dim):
        dim_val = tensor_shape.dimension_value(a.shape[dim])
        if dim_val is not None:
          return dim_val
        return array_ops.shape(a)[dim]

      num_rows = get_dim_size(-2)
      num_cols = get_dim_size(-1)
      if isinstance(num_rows, int) and isinstance(num_cols, int):
        max_rows_cols = float(max(num_rows, num_cols))
      else:
        max_rows_cols = math_ops.cast(
            math_ops.maximum(num_rows, num_cols), dtype)
      rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = ops.convert_to_tensor(rcond, dtype=dtype, name='rcond')

    # Calculate pseudo inverse via SVD.
    # Note: if a is Hermitian then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,  # Sigma
        left_singular_vectors,  # U
        right_singular_vectors,  # V
    ] = svd(
        a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = rcond * math_ops.reduce_max(singular_values, axis=-1)
    singular_values = array_ops.where_v2(
        singular_values > array_ops.expand_dims_v2(cutoff, -1), singular_values,
        np.array(np.inf, dtype))

    # By the definition of the SVD, `a == u @ s @ v^H`, and the pseudo-inverse
    # is defined as `pinv(a) == v @ inv(s) @ u^H`.
    a_pinv = math_ops.matmul(
        right_singular_vectors / array_ops.expand_dims_v2(singular_values, -2),
        left_singular_vectors,
        adjoint_b=True)

    if a.shape is not None and a.shape.rank is not None:
      a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv

#%%

class GaussianProcess():
    """
    Builds computation graph for vanilla GP.
    """
    def __init__(
        self, sigma_l, sigma_f, sigma_n=None, train_features=None,
        test_features=None, test=False, method='inverse',
            reuse=False, alpha=1, constant_mean=False, kernel='squared_exponential'):
        self.sigma_l = sigma_l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.method = method
        self.alpha = alpha
        self.constant_mean = constant_mean
        self.kernel = kernel
        self.set_up_graph(train_features, test_features, test, reuse)

    def set_up_graph(self, train_features, test_features, test, reuse):
        """
        Builds computation graph.
        """
        with tf.device('/cpu:0'):
            self._create_training_data(train_features)
            self.set_sigmas(reuse)
            self._compute_lml_components()
            self._compute_terms()
            self.lml = tf.add_n(
                [self.data, self.det, self.constant], name='lml')
            if test_features is not None or test:
                self.add_test_graph(test_features)

    def set_sigmas(self, reuse):
        """
        Sets hyperparameters of kernel function.
        """
        with tf.variable_scope('sigmas', reuse=reuse):
            self.sigma_l_tf = tf.get_variable(
                name='sigma_l', shape=[1], dtype=tf.float32,
                initializer=tf.constant_initializer(self.sigma_l),
                trainable=True)

            train_sigma_f = (self.kernel == 'squared_exponential')
            self.sigma_f_tf = tf.get_variable(
                name='sigma_f', shape=[1], dtype=tf.float32,
                initializer=tf.constant_initializer(self.sigma_f),
                trainable=train_sigma_f)

            if self.sigma_n is not None:
                #self.sigma_n_tf = tf.get_variable(
                #    name='sigma_n', shape=[1], dtype=tf.float32,
                #    initializer=tf.constant_initializer(self.sigma_n),
                #    trainable=True)
                self.log_sigma_n_tf = tf.get_variable(
                    name='log_sigma_n', shape=[1], dtype=tf.float32,
                    initializer=tf.constant_initializer(
                        np.log(self.sigma_n)), trainable=True)
                self.sigma_n_tf = tf.exp(self.log_sigma_n_tf, name='sigma_n')

    def add_test_graph(self, test_features=None):
        """
        Add test examples to TF GP graph.
        """
        self._create_test_data(test_features)
        self._compute_test_posterior()
        self._compute_compound_loss()
        self._compute_test_accuracy()

    def _compute_test_accuracy(self):
        """
        Computes some metrics to assess the accuracy of test predictions.
        """
        # Compute sum of squared residuals
        self.residuals = tf.subtract(
            self.y_test_mean, self.y_test, name='residuals')
        self.ss_res = tf.reduce_sum(tf.square(self.residuals), name='ss_res')
        # Compute total sum of squares
        self.ss_tot = tf.reduce_sum(
            tf.square(self.y_test - tf.reduce_mean(self.y_test)),
            name='ss_tot')
        # Average total sum of squares
        self.ss_tot_mean = tf.div(
            self.ss_tot, tf.to_float(tf.shape(self.y_test)[0]),
            name='ss_tot_mean')
        # Compute MSE
        self.mean_squared_error = tf.div(
            tf.reduce_sum(tf.square(self.residuals)),
            tf.to_float(tf.shape(self.y_test)[0]), name='mean_squared_error')
        # Compute R^2
        self.R_squared = tf.subtract(
            1.0, tf.div(self.ss_res, self.ss_tot), name='R_squared')

    def _create_test_data(self, test_features):
        """
        Adds test data.
        """
        if test_features is not None:
            self.X_test = test_features
        else:
            self.X_test = tf.placeholder(
                dtype=tf.float32, shape=[None, None], name='X_test')
        self.y_test = tf.placeholder(
            dtype=tf.float32, shape=[None, None], name='y_test')

    def _compute_test_posterior(self):
        """
        Computes posterior of test data.
        """
        # Compute covariance matrix blocks
        self._compute_covariance_blocks()
        # Compute test means and covariances from only spatial correlations
        # (no features)
        self.y_test_mean = tf.matmul(
            self.K_21, self.Ky_inv_y, name='y_test_mean')
        if self.constant_mean:
            self.y_test_mean += tf.reduce_mean(self.y_train_original)
        self.y_test_cov = tf.subtract(
            self.K_22, tf.matmul(self.K_21, self.K_11_inv_K_12),
            name='y_test_cov')

    def _compute_compound_loss(self):
        """
        Computes semi-supervised loss that is a weighted sum of the negative
        log marginal likelihood and the sum of test variances.
        """
        # Compute sum of test variances
        self.sum_test_variances = tf.reduce_sum(tf.multiply(
            tf_utils.eye(tf.shape(self.y_test_cov)[0]),
            self.y_test_cov), name='sum_test_variances')
        # Compute weighted compound loss for semi-supervised training
        self.lml_component = tf.div(
            tf.negative(self.lml), tf.to_float(tf.shape(self.X_train)[0]),
            name='lml_component')
        self.sum_test_variances_component = tf.multiply(
            (self.alpha / (tf.to_float(tf.shape(self.X_test)[0]))),
            self.sum_test_variances,
            name='sum_test_variances_component')
        self.semisup_loss = tf.add(
            self.lml_component, self.sum_test_variances_component,
            name='semisup_loss')

    def _compute_covariance_blocks(self):
        """
        Computes covariance blocks used for computing GP posterior.
        """
        self.K_11 = tf.identity(self.Ky, name='K_11')
        print('Comput Cov Blocks')
        print(self.K_11)
        self.K_12 = self._compute_covariance_matrix(
            self.X_train, self.X_test, name='K_12')
        self.K_21 = self._compute_covariance_matrix(
            self.X_test, self.X_train)
        self.K_22 = self._compute_covariance_matrix(
            self.X_test, self.X_test)
        if self.method == 'pseudo':
            self.K_11_inv = pinv(self.K_11)
            #self.K_11_inv = pinv_module.pseudo_inverse_no_det(self.K_11, name='K_11_inv')
            self.K_11_inv_K_12 = tf.matmul(
                self.K_11_inv, self.K_12, name='K_11_inv_K_12')
        elif self.method == 'inverse':
            try:
                self.K_11_inv = tf.linalg.inv(self.K_11, name='K_11_inv')
            except:
                self.K_11_inv = pinv(self.K_11, name='K_11_inv')
            self.K_11_inv_K_12 = tf.matmul(
                self.K_11_inv, self.K_12, name='K_11_inv_K_12')

    def _create_training_data(self, train_features):
        """
        Creates placeholders for training data.
        """
        if train_features is not None:
            self.X_train = train_features
        else:
            self.X_train = tf.placeholder(
                dtype=tf.float32, shape=[None, None], name='X_train')
        self.y_train_original = tf.placeholder(
            dtype=tf.float32, shape=[None, None], name='y_train_original')
        if self.constant_mean:
            self.y_train = self.y_train_original - tf.reduce_mean(
                self.y_train_original)
        else:
            self.y_train = self.y_train_original

    def _compute_lml_components(self):
        """
        Computes intermediate values needed for lml computation.
        """
        self.n_tf = tf.to_int32(tf.shape(self.y_train)[0], name='n_tf')
        self._compute_Ky()
        self._compute_Ky_inv()

    def _compute_Ky(self):
        """
        Computes the train covariance matrix.
        """
        self.K = self._compute_covariance_matrix(self.X_train, self.X_train, name='K')
        
        # Add diagonal for noisy observations
        if self.sigma_n is not None:
            print('Sigma None')
            self.Ky = tf.add(self.K, tf.multiply(tf.square(self.sigma_n_tf), tf_utils.eye(self.n_tf)),name='Ky')
            print(self.Ky)
        else:
            print('Sigma')
            self.Ky = tf.identity(self.K, name='Ky')

    def _compute_covariance_matrix(self, S1, S2, name='cov_matrix'):
        """
        Computes the squared exponential covariance matrix.
        """
        if self.kernel == 'square_polynomial':
            return tf.square(tf.square(self.sigma_l_tf) +\
                tf.matmul(S1, tf.transpose(S2)))
        elif self.kernel == 'squared_exponential':

            distance_matrix = self._compute_dist_matrix(
                S1, S2, name=name+'_dist_matrix')
            denom = tf.multiply(
                tf.constant(2.0, dtype=tf.float32), tf.square(self.sigma_l_tf),
                name='cov_matrix_denom')

            # Squared exponential covariance
            exponent = tf.div(-distance_matrix, denom)
            return tf.multiply(tf.square(self.sigma_f_tf), tf.exp(exponent), name=name)

    def _compute_dist_matrix(self, S1, S2, name='distance_matrix'):
        """
        Computes the Euclidean distance matrix.
        """
        # Compute s1 squared matrix
        s1_squared_norm = tf.reduce_sum(
            tf.square(S1), reduction_indices=1, keep_dims=True)
        # Compute s2 squared matrix
        s2_squared_norm = tf.reduce_sum(
            tf.square(tf.transpose(S2)), reduction_indices=0, keep_dims=True)
        term1 = tf.matmul(s1_squared_norm, tf.ones_like(s2_squared_norm))
        term2 = tf.matmul(tf.ones_like(s1_squared_norm), s2_squared_norm)
        # Compute cross term
        term3 = tf.matmul(S1, S2, transpose_b=True)
        # Compute distance matrix
        term4 = tf.multiply(tf.constant(-2.0, dtype=tf.float32), term3)
        distance_matrix = tf.add_n([term1, term2, term4])
        return distance_matrix

    def _compute_Ky_inv(self):
        """
        Computes intermediate value Ky_inv to be used in lml computation.
        """
        if self.method == 'pseudo':
            self.Ky_inv = pinv(self.Ky)
            #self.Ky_inv, self.Ky_logdet = pinv_module.pseudo_inverse(self.Ky)
            # Compute Ky_inv_y
            self.Ky_inv_y = tf.matmul(
                self.Ky_inv, self.y_train, name='Ky_inv_y')

        elif self.method == 'inverse':
            print('Inverse Selection Triggered compute ky inv')
            print(self.Ky)
            #self.Ky_inv = tf.matrix_inverse(self.Ky, name='Ky_inv')
            try:
                self.Ky_inv = tf.linalg.inv(self.Ky, name='Ky_inv')
            except:
                self.Ky_inv = pinv(self.Ky, name='Ky_inv')
            #print(self.Ky_inv_y)
            # Compute Ky_inv_y
            self.Ky_inv_y = tf.matmul(
                self.Ky_inv, self.y_train, name='Ky_inv_y')

        elif self.method == 'constant':
            # Multiply by a constant, then invert
            self.Ky_constant = tf.placeholder(
                dtype=tf.float32, shape=[1, 1], name='Ky_constant')
            self.Ky_scaled = tf.multiply(
                self.Ky_constant, self.Ky, name='Ky_scaled')
            self.Ky_scaled_inv = tf.matrix_inverse(
                self.Ky_scaled, name='Ky_scaled_inv')
            # Re-multiply by the same constant
            self.Ky_inv = tf.multiply(
                self.Ky_constant, self.Ky_scaled_inv, name='Ky_inv')
            # Compute Ky_inv_y
            self.Ky_inv_y = tf.matmul(
                self.Ky_inv, self.y_train, name='Ky_inv_y')

        elif self.method == 'cholesky':
            # Use cholesky decomposition to invert
            self.L_Ky = tf.cholesky(self.Ky, name='L_Ky')
            self.Ky_inv_y = tf.matrix_solve(
                tf.transpose(self.L_Ky),
                tf.matrix_solve(self.L_Ky, self.y_train), name='Ky_inv_y')

        elif self.method == 'constant_cholesky':
            # Multiply by constant, then use cholesky decomposition to invert
            self.Ky_constant = tf.placeholder(
                dtype=tf.float32, shape=[1, 1], name='Ky_constant')
            self.Ky_scaled = tf.multiply(
                self.Ky_constant, self.Ky, name='Ky_scaled')
            self.L_Ky_scaled = tf.cholesky(self.Ky_scaled, name='L_Ky_scaled')
            self.Ky_inv_y_scaled = tf.matrix_solve(
                tf.transpose(self.L_Ky_scaled),
                tf.matrix_solve(self.L_Ky_scaled, self.y_train),
                name='Ky_inv_y_scaled')
            self.Ky_inv_y = tf.multiply(
                self.Ky_constant, self.Ky_inv_y_scaled, name='Ky_inv_y')
        
    def _compute_constant(self):
        """
        Computes constant term in lml.
        """
        self.constant = tf.reshape(tf.negative(
            (tf.cast(self.n_tf, dtype=tf.float32) / 2.0) *
            tf.log(2 * math.pi)), shape=[1, 1], name='constant')

    def _compute_data(self):
        """
        Computes first data term in lml.
        """
        self.data = tf.reshape(
            tf.multiply(-0.5, tf.matmul(
                tf.transpose(self.y_train), self.Ky_inv_y)), shape=[1, 1],
            name='data')

    def _compute_det(self):
        """
        Computes determinant terms in lml.
        """
        self.mask_Ky = tf_utils.eye(tf.shape(self.Ky)[0], name='mask_Ky')
        self.L_Ky = tf.cholesky(self.Ky)
        self.det = tf.negative(tf.reduce_sum(self.mask_Ky * tf.log(tf.abs(0.5 * (self.L_Ky + tf.transpose(self.L_Ky)))),keep_dims=True), name='det')

    def _compute_terms(self):
        """
        Computes all terms of lml.
        """
        self._compute_data()
        self._compute_det()
        self._compute_constant()

######################################################################
######################################################################
######################################################################


# @ops.RegisterGradient("PseudoInverse")
# def _pseudo_inverse_grad(op, grad_wrt_inv, grad_wrt_det):
#         """The gradients for `pseudo_inverse`.
#         Args:
#         op: The `pseudo_inverse` `Operation` that we are differentiating,
#         which we can use to find the inputs and outputs of the original op.
#         Computes the pseudo-inverse and the log determinant.
#         grad: Gradient with respect to the output of the `pseudo_inverse` op.
#         Returns:
#         Gradients with respect to the input of `pseudo_inverse`.
#         """
#         mat_inv = op.outputs[0]
#         mat_det = op.outputs[1]
#         # Matrix pseudo-inverse gradient
#         mat_grad = -math_ops.matmul(
#             array_ops.transpose(mat_inv),
#             math_ops.matmul(grad_wrt_inv, array_ops.transpose(mat_inv)))
#         # Matrix determinant gradient
#         det_grad = mat_inv * grad_wrt_det
#         # Return gradients
#         return [mat_grad + det_grad]


# @ops.RegisterGradient("PseudoInverseNoDet")
# def _pseudo_inverse_no_det_grad(op, grad_wrt_inv):
#         """The gradients for `pseudo_inverse_no_det`.
#         Args:
#         op: The `pseudo_inverse_no_det` `Operation` that we are
#         differentiating, which we can use
#         to find the inputs and outputs of the original op.
#         grad: Gradient with respect to the output of the
#         `pseudo_inverse_no_det` op.
#         Returns:
#         Gradients with respect to the input of `pseudo_inverse_no_det`.
#         """
#         mat_inv = op.outputs[0]
#         mat_grad = -math_ops.matmul(
#             array_ops.transpose(mat_inv),
#             math_ops.matmul(grad_wrt_inv, array_ops.transpose(mat_inv)))
#         # Return gradient
#         return [mat_grad]

######################################################################
