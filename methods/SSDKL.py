import shutil

from sklearn.utils import shuffle
import numpy as np
import os
import tensorflow as tf
import methods.gp as gp
import importlib
import time

def load_dynamic(model):
    """
    Load a class dynamically from the classname.
    """
    return getattr(importlib.import_module('methods.nn'), model)


class ModelTrainer:
    """
    A class that can train a model on some data and save the results.
    """
    def __init__(self, x, y, trainMask, testMask, valMask, labeledMask, unlabeledMask, results_dir, max_iters,
                 num_trials, optimizer_type='adam', lr_decay=0.9,
                 lr_decay_every=50, momentum=0.9, beta1=0.9, beta2=0.999,
                 epsilon=1e-3, use_gpu=False, save_results=True,
                 max_models_to_keep=5, save_every=20, verbose=False):
        """
        Sets up ModelTrainer object.
        """

        self.dim = x.shape[1]
        self.X_labeled = x[labeledMask]
        self.X_val = x[valMask]
        self.X_test = x[testMask]
        self.X_unlabeled = x[testMask]

        self.y_labeled = y[labeledMask]
        self.y_val = y[valMask]
        self.y_test = y[testMask]
        self.y_unlabeled = y[testMask]

        self.y_labeled = np.reshape(
            self.y_labeled, newshape=(self.y_labeled.size, 1))
        self.y_val = np.reshape(
            self.y_val, newshape=(self.y_val.size, 1))
        self.y_unlabeled = np.reshape(
            self.y_unlabeled, newshape=(self.y_unlabeled.size, 1))
        self.y_test = np.reshape(
            self.y_test, newshape=(self.y_test.size, 1))

        self.results_dir = results_dir
        self.max_iters = max_iters
        self.num_trials = num_trials
        self.max_models_to_keep = max_models_to_keep
        self.save_every = save_every
        self.config_opt(optimizer_type, lr_decay, lr_decay_every, momentum,
                        beta1, beta2, epsilon)
        self.use_gpu = use_gpu
        self._make_dirs(save_results)
        self.verbose = verbose
        self.transductive = False

    def config_opt(self, optimizer_type, lr_decay, lr_decay_every,
                   momentum, beta1, beta2, epsilon):
        """
        Configures optimizer for training.
        """
        self.optimizer_type = optimizer_type
        self.lr_decay = lr_decay
        self.lr_decay_every = lr_decay_every
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def run_trials(self):
        """
        Runs multiple trials of training.
        """
        self._initialize_metrics()
        self.trial = 0
        self.check_learning_counter = 0
        while self.trial < self.num_trials:
            t0 = time.time()
            print('Starting trial {}:'.format(self.trial + 1))
            self.best_saved_metric = None
            print('Start training')
            self._train()
            print('End training')
            print('Check Learning Conter : ' + str(self.check_learning_counter))
            t1 = time.time()
            print('Finished trial {}: {} seconds elapsed\n'.format(
                self.trial + 1, t1 - t0))
            if self.check_learning_counter < 3:
                self._check_learning()
            else:
                self.trial += 1
                self.check_learning_counter = 0
        if self.save_results:
            self._write_results()

    def _clear_metrics(self):
        """
        Clear metrics for trial.
        """
        self.lmls_labeled[self.trial, :] = np.inf
        self.vars_unlabeled[self.trial, :] = np.inf
        self.losses_semisup[self.trial, :] = np.inf
        self.mses_val[self.trial, :] = np.inf
        self.mses_test[self.trial, :] = np.inf
        self.sigma_ls[self.trial, :] = np.inf
        self.sigma_fs[self.trial, :] = np.inf
        self.sigma_ns[self.trial, :] = np.inf
        try:
            self.sigma_ls_loc[self.trial, :] = np.inf
            self.sigma_fs_loc[self.trial, :] = np.inf
        except:
            pass

        self.counter = 0

    def _create_train_op(self, lr, loss, var_list, cap=False):
        """
        Returns a training op that minimizes the specified loss.
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=lr, global_step=self.global_step,
            decay_steps=self.lr_decay_every, decay_rate=self.lr_decay,
            staircase=True)
        if self.optimizer_type == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=self.beta1,
                beta2=self.beta2, epsilon=self.epsilon)

        # Create training op w/ gradient clipping option
        if cap:
            grads_and_vars = self.optimizer.compute_gradients(
                loss, var_list=var_list)
            capped_grads_and_vars = \
                [(tf.clip_by_value(grad, -5.0, 5.0), var)
                 for grad, var in grads_and_vars if grad is not None]
            train_op = self.optimizer.apply_gradients(
                capped_grads_and_vars, global_step=self.global_step)
        else:
            train_op = self.optimizer.minimize(
                loss, global_step=self.global_step, var_list=var_list)
        return train_op

    def _increment_counter(self):
        """
        Increments the counter for deciding when to stop training.
        """
        if self.mse_val > min(self.mses_val[self.trial, :self.step]):
            self.counter += 1
        else:
            self.counter = 0

    def _initialize_metrics(self):
        """
        Sets up metrics to be stored.
        """
        initial_metrics = np.full((self.num_trials, self.max_iters), np.inf)
        self.lmls_labeled = np.copy(initial_metrics)
        self.vars_unlabeled = np.copy(initial_metrics)
        self.losses_semisup = np.copy(initial_metrics)
        self.mses_labeled = np.copy(initial_metrics)
        self.mses_val = np.copy(initial_metrics)
        self.mses_test = np.copy(initial_metrics)
        self.sigma_ls = np.copy(initial_metrics)
        self.sigma_fs = np.copy(initial_metrics)
        self.sigma_ns = np.copy(initial_metrics)
        self.sigma_ls_loc = np.copy(initial_metrics)
        self.sigma_fs_loc = np.copy(initial_metrics)

    def _make_dirs(self, save_results):
        """
        Check to see if log and model directories exist and create them if
        needed.
        """
        self.save_results = save_results
        if self.save_results:
            self.log_dir = os.path.join(self.results_dir, 'logs')
            self._remake_dir(self.log_dir)
            self._remake_dir(os.path.join(self.results_dir, 'results'))

    def _remake_dir(self, directory):
        """
        If directory exists, delete it and remake it, if it doesn't exist, make
        it.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    def _run_training(self, sess):
        """
        Trains the model.
        """
        self._initialize_training(sess)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            sess=sess, coord=self.coord)
        self._evaluate_initial(sess)
        while not self.coord.should_stop():
            self._take_gradient_step(sess)
        self.coord.request_stop()
        self.coord.join(self.threads)

    def _train(self):
        """
        Trains the parameters of the GP on regression task.
        """
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(device_count = {'GPU': 0})
        if self.use_gpu:
            with tf.Graph().as_default():
                with tf.device(self.device):
                    with tf.Session(config=tf.ConfigProto(
                        allow_soft_placement=True, log_device_placement=False,
                            gpu_options=gpu_options)) as sess:
                        self._run_training(sess)
        else:
            with tf.Graph().as_default():
                with tf.device('/cpu:0'):
                    with tf.Session(config=config) as sess:
                        self._run_training(sess)

    def _write_results(self):
        """
        Writes results produced by experiment.
        """
        np.save(os.path.join(self.results_dir, 'results/lmls_labeled'),
                self.lmls_labeled)
        np.save(os.path.join(self.results_dir, 'results/vars_unlabeled'),
                self.vars_unlabeled)
        np.save(os.path.join(self.results_dir, 'results/losses_semisup'),
                self.losses_semisup)
        np.save(os.path.join(self.results_dir, 'results/mses_val'),
                self.mses_val)
        np.save(os.path.join(self.results_dir, 'results/mses_test'),
                self.mses_test)
        np.save(os.path.join(self.results_dir, 'results/sigma_ls'),
                self.sigma_ls)
        np.save(os.path.join(self.results_dir, 'results/sigma_fs'),
                self.sigma_fs)
        np.save(os.path.join(self.results_dir, 'results/sigma_ns'),
                self.sigma_ns)
        np.save(os.path.join(self.results_dir, 'results/sigma_ls'),
                self.sigma_ls_loc)
        np.save(os.path.join(self.results_dir, 'results/sigma_fs'),
                self.sigma_fs_loc)

    def _write_summary(self):
        """
        Writes summary to log.
        """
        self.writer.add_summary(self.summary, self.step)

    def _create_feed_dict(self):
        """
        Creates feed_dict for training.
        """
        raise NotImplementedError

    def _evaluate_step(self):
        """
        Evaluates gradient step.
        """
        raise NotImplementedError

    def _evaluate_initial(self):
        """
        Evaluate initial data values and losses.
        """
        raise NotImplementedError

    def _initialize_training(self):
        """
        Initializes training.
        """
        raise NotImplementedError

    def _print_iteration_summary(self):
        """
        Prints a summary of training iteration.
        """
        raise NotImplementedError

    def _setup_logging(self):
        """
        Sets up writer for logging training.
        """
        raise NotImplementedError

    def _setup_saver(self):
        """
        Sets up saver to save and load trained models.
        """
        raise NotImplementedError

    def _take_gradient_step(self):
        """
        Takes a gradient step and computes new losses.
        """
        raise NotImplementedError

class SemisupDKLTrainer(ModelTrainer):
    """
    Extends the ModelTrainer class for semi-supervised training of NN + GP.
    """
    def __init__(self,  x, y, trainMask, testMask, valMask, labeledMask, unlabeledMask, results_dir, max_iters,
                 num_trials, sigma_l=1.0, sigma_f=1.0, sigma_n=1.5, alpha=1.0,
                 batch_unlabeled=0, lr_nn=1e-4, lr_gp=1e-1,
                 optimizer_type='adam', lr_decay=0.9, lr_decay_every=50,
                 momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-3,
                 use_gpu=False, gpu_fraction=0.333, save_results=False,
                 max_models_to_keep=5, save_every=5, load_nn=False,
                 load_gp=False, kernel='squared_exponential', verbose=False,
                 use_cnn=0):
        """
        Sets up SemisupNNLinearGPTrainer object.
        """
        ModelTrainer.__init__(
            self,  x, y, trainMask, testMask, valMask, labeledMask, unlabeledMask, results_dir, max_iters, num_trials,
            optimizer_type, lr_decay, lr_decay_every, momentum, beta1, beta2,
            epsilon, use_gpu, save_results, max_models_to_keep, save_every,
            verbose)
        self.sigma_l_init = sigma_l
        self.sigma_f_init = sigma_f
        self.sigma_n_init = sigma_n
        self.kernel = kernel
        self.lr_gp = lr_gp
        self.lr_nn = lr_nn
        self.alpha = alpha
        self.batch_unlabeled = batch_unlabeled
        self.gpu_fraction = gpu_fraction
        self.load_nn = load_nn
        self.load_gp = load_gp
        if self.use_gpu:
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'
        self.use_cnn = use_cnn

    def _check_learning(self):
        """
        Check to make sure the model improved. Same as GP for now.
        """
        if ((self.losses_semisup[self.trial, 0] -
             min(self.losses_semisup[self.trial, :self.step]))) > 0:
            self.trial += 1
            self.check_learning_counter = 0
        else:
            self.check_learning_counter += 1

    def _compute_embeddings(self, sess):
        """
        Computes the current embeddings using DKL and semisup parameters.
        """
        self.feed_dict_embeddings = {self.nn_labeled.X: self.X_labeled,
                                     self.nn_labeled.y: self.y_labeled,
                                     self.nn_val.X: self.X_val,
                                     self.nn_val.y: self.y_val,
                                     self.nn_test.X: self.X_test,
                                     self.nn_test.y: self.y_test,
                                     self.nn_unlabeled.X: self.X_unlabeled,
                                     self.nn_unlabeled.y: self.y_unlabeled}
        self.embeddings_labeled, self.embeddings_val, self.embeddings_test, \
                self.embeddings_unlabeled = \
            sess.run(
                [self.nn_labeled.embeddings, self.nn_val.embeddings,
                 self.nn_test.embeddings, self.nn_unlabeled.embeddings],
                feed_dict=self.feed_dict_embeddings)

    def _create_feed_dict(self):
        """
        Creates feed_dict for training.
        """
        # Get a sample of unlabeled data
        X_unlabeled_batch, y_unlabeled_batch = shuffle(
            self.X_unlabeled, self.y_unlabeled)
        X_unlabeled_batch = X_unlabeled_batch[:self.batch_unlabeled]
        y_unlabeled_batch = y_unlabeled_batch[:self.batch_unlabeled]
        self.feed_dict = {
            self.gp_val.y_train_original: self.y_labeled,
            self.gp_val.y_test: self.y_val,
            self.gp_test.y_train_original: self.y_labeled,
            self.gp_test.y_test: self.y_test,
            self.gp_unlabeled.y_train_original: self.y_labeled,
            self.gp_unlabeled.y_test: y_unlabeled_batch,#self.y_unlabeled,
            self.nn_labeled.X: self.X_labeled,
            self.nn_val.X: self.X_val,
            self.nn_test.X: self.X_test,
            self.nn_unlabeled.X: X_unlabeled_batch}#self.X_unlabeled}


    def _evaluate_initial(self, sess):
        """
        Evaluate initial data values and losses.
        """
        self._create_feed_dict()
        self.sigma_l, self.sigma_f, self.sigma_n, lml_labeled, \
            self.var_unlabeled, self.loss_semisup, self.mse_val, \
            self.mse_test = sess.run(
                [self.gp_val.sigma_l_tf,
                 self.gp_val.sigma_f_tf,
                 self.gp_val.sigma_n_tf,
                 self.gp_val.lml,
                 self.gp_unlabeled.sum_test_variances,
                 self.gp_unlabeled.semisup_loss,
                 self.gp_val.mean_squared_error,
                 self.gp_test.mean_squared_error],
                feed_dict=self.feed_dict)
        self.lml_labeled = lml_labeled[0, 0]
        self._print_initial_summary()
        self._record_metrics()

    def _evaluate_step(self, sess):
        """
        Evaluates gradient step.
        """

        self._create_feed_dict()
        self.summary, _ = sess.run([self.merged_summaries, self.train_op],
                                   feed_dict=self.feed_dict)
        self.sigma_l, self.sigma_f, self.sigma_n, lml_labeled, \
            self.var_unlabeled, self.loss_semisup, self.mse_val, \
            self.mse_test = \
            sess.run(
                [self.gp_val.sigma_l_tf,
                 self.gp_val.sigma_f_tf,
                 self.gp_val.sigma_n_tf,
                 self.gp_val.lml,
                 self.gp_unlabeled.sum_test_variances,
                 self.gp_unlabeled.semisup_loss,
                 self.gp_val.mean_squared_error,
                 self.gp_test.mean_squared_error],
                feed_dict=self.feed_dict)
        # Save predictions
        if False:
            predictions, truth = sess.run(
                [self.gp_test.y_test_mean, self.gp_test.y_test], feed_dict=self.feed_dict)
            np.save(os.path.join(self.results_dir, 'results/{}predictions{}'.format(self.trial, self.step)), predictions)
            np.save(os.path.join(self.results_dir, 'results/{}truth{}'.format(self.trial, self.step)), truth)
        self.lml_labeled = lml_labeled[0, 0]

    def _initialize_training(self, sess):
        """
        Initializes NN and GP training.
        """
        self.sigma_l = self.sigma_l_init
        self.sigma_f = self.sigma_f_init
        self.sigma_n = self.sigma_n_init
        self._clear_metrics()
        self._setup_nns_gps(sess)
        # Create training op on semisup objective
        self.trainable_vars_gp = [v for v in tf.trainable_variables()
                                  if 'sigma' in v.name]
        self.trainable_vars_nn = [v for v in tf.trainable_variables() if
                                  'nn' in v.name]
        self.train_op_gp = self._create_train_op(
            lr=self.lr_gp, loss=self.gp_val.semisup_loss,
            var_list=self.trainable_vars_gp, cap=True)
        self.train_op_nn = self._create_train_op(
            lr=self.lr_nn, loss=self.gp_val.semisup_loss,
            var_list=self.trainable_vars_nn, cap=True)
        self.train_op = tf.group(self.train_op_gp, self.train_op_nn)
        self.init = tf.global_variables_initializer()
        sess.run(self.init)
        self.step = 0
        self.counter = 0

    def _print_initial_summary(self):
        """
        Prints a summary of initial values.
        """
        if self.verbose:
            print('Features:')
            print(' Labeled: {}'.format(self.X_labeled.shape))
            print(' Val: {}'.format(self.X_val.shape))
            print(' Test: {}'.format(self.X_test.shape))
            print('Labels:')
            print(' Labeled: {}'.format(self.y_labeled.shape))
            print(' Val: {}'.format(self.y_val.shape))
            print(' Test: {}'.format(self.y_test.shape))
            print('Log marginal likelihood:')
            print(' Labeled: {}'.format(self.lml_labeled))
            print('Unlabeled:')
            print(' Sum variances: {}'.format(self.var_unlabeled))
            print(' Semisup loss: {}'.format(self.loss_semisup))
            print('MSE:')
            print(' Val: {}'.format(self.mse_val))
            print(' Test: {}'.format(self.mse_test))
            print('GP parameters:')
            print(' Sigma_l: {}'.format(self.sigma_l))
            print(' Sigma_f: {}'.format(self.sigma_f))
            print(' Sigma_n: {}'.format(self.sigma_n))

    def _print_iteration_summary(self, duration=None):
        """
        Prints a summary of training iteration.
        """
        if self.verbose:
            print('Iteration {} complete:'.format(self.step))
            print(' GP parameters:')
            print('  Sigma_l: {}'.format(self.sigma_l))
            print('  Sigma_f: {}'.format(self.sigma_f))
            print('  Sigma_n: {}'.format(self.sigma_n))
            print('Log marginal likelihood:')
            print(' Labeled: {}'.format(self.lml_labeled))
            print('Unlabeled:')
            print(' Sum variances: {}'.format(self.var_unlabeled))
            print(' Semisup loss: {}'.format(self.loss_semisup))
            print('MSE:')
            print(' Val: {}'.format(self.mse_val))
            print(' Test: {}'.format(self.mse_test))
            if duration is not None:
                print('Iteration {} finished in {} seconds\n'.format(
                    self.step, duration))

    def _record_metrics(self):
        """
        Records current metrics.
        """
        self.lmls_labeled[self.trial, self.step] = self.lml_labeled
        self.vars_unlabeled[self.trial, self.step] = self.var_unlabeled
        self.losses_semisup[self.trial, self.step] = self.loss_semisup
        self.mses_val[self.trial, self.step] = self.mse_val
        self.mses_test[self.trial, self.step] = self.mse_test
        self.sigma_ls[self.trial, self.step] = self.sigma_l
        self.sigma_fs[self.trial, self.step] = self.sigma_f
        self.sigma_ns[self.trial, self.step] = self.sigma_n

    def _save_embeddings(self, model):
        """
        Saves the embeddings and labels.
        """
        self.embedding_dir = os.path.join(
            self.results_dir, 'embeddings')
        #self._remake_dir(self.embedding_dir)
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
        np.save(os.path.join(self.embedding_dir, 'X_train_' + model),
                self.embeddings_labeled)
        np.save(os.path.join(self.embedding_dir, 'X_val_' + model),
                self.embeddings_val)
        np.save(os.path.join(self.embedding_dir, 'X_test_' + model),
                self.embeddings_test)
        np.save(os.path.join(self.embedding_dir, 'X_unlabeled_' + model),
                self.embeddings_unlabeled)
        np.save(os.path.join(self.embedding_dir, 'y_train_' + model),
                self.y_labeled)
        np.save(os.path.join(self.embedding_dir, 'y_val_' + model),
                self.y_val)
        np.save(os.path.join(self.embedding_dir, 'y_test_' + model),
                self.y_test)
        np.save(os.path.join(self.embedding_dir, 'y_unlabeled_' + model),
                self.y_unlabeled)

    def _setup_gps(self):
        """
        Sets up GPs.
        """
        self.gp_val = gp.GaussianProcess(
            self.sigma_l, self.sigma_f, self.sigma_n,
            train_features=self.nn_labeled.embeddings,
            test_features=self.nn_val.embeddings,
            kernel=self.kernel,
            test=True, reuse=False, alpha=self.alpha)
        self.gp_test = gp.GaussianProcess(
            self.sigma_l, self.sigma_f, self.sigma_n,
            train_features=self.nn_labeled.embeddings,
            test_features=self.nn_test.embeddings,
            kernel=self.kernel,
            test=True, reuse=True, alpha=self.alpha)
        self.gp_unlabeled = gp.GaussianProcess(
            self.sigma_l, self.sigma_f, self.sigma_n,
            train_features=self.nn_labeled.embeddings,
            test_features=self.nn_unlabeled.embeddings,
            kernel=self.kernel,
            test=True, reuse=True, alpha=self.alpha)

    def _setup_logging(self, sess):
        """
        Sets up writer for logging training.
        """
        self.writer = tf.summary.FileWriter(
            logdir=os.path.join(self.log_dir, str(self.trial + 1)),
            graph=sess.graph)
        tf.summary.scalar(self.gp_val.sigma_l_tf.name,
                          self.gp_val.sigma_l_tf[0])
        tf.summary.scalar(self.gp_val.sigma_f_tf.name,
                          self.gp_val.sigma_f_tf[0])
        tf.summary.scalar(self.gp_val.sigma_n_tf.name,
                          self.gp_val.sigma_n_tf[0])
        tf.summary.scalar(self.gp_val.lml.name,
                          self.gp_val.lml[0,0])
        tf.summary.scalar(self.gp_val.data.name, self.gp_val.data[0,0])
        tf.summary.scalar(self.gp_val.det.name, self.gp_val.det[0,0])
        tf.summary.scalar(self.gp_val.constant.name, self.gp_val.constant[0,0])
        try:
            tf.summary.scalar(self.gp_unlabeled.sum_test_variances.name,
                              self.gp_unlabeled.sum_test_variances)
            tf.summary.scalar(self.gp_unlabeled.semisup_loss.name,
                              self.gp_unlabeled.semisup_loss[0,0])
            tf.summary.scalar(
                self.gp_unlabeled.sum_test_variances_component.name,
                self.gp_unlabeled.sum_test_variances_component)
            tf.summary.scalar(self.gp_unlabeled.lml_component.name,
                              self.gp_unlabeled.lml_component[0,0])
        except:
            pass
        tf.summary.scalar(self.gp_val.mean_squared_error.name,
                          self.gp_val.mean_squared_error)
        tf.summary.scalar(self.gp_test.mean_squared_error.name,
                          self.gp_test.mean_squared_error)
        tf.summary.scalar(self.gp_unlabeled.mean_squared_error.name,
                          self.gp_unlabeled.mean_squared_error)
        self.merged_summaries = tf.summary.merge_all()

    def _setup_nns(self):
        """
        Sets up NNs.
        """
        if self.use_cnn == 1:
            NN = load_dynamic('CNN_MNIST')
        else:
            NN = load_dynamic('NN_UCI')
        self.nn_labeled = NN(
            dim=self.dim, train_phase=True, name='nn_labeled')
        self.nn_val = NN(
            dim=self.dim, train_phase=False, name='nn_val')
        self.nn_test = NN(
            dim=self.dim, train_phase=False, name='nn_test')
        self.nn_unlabeled = NN(
            dim=self.dim, train_phase=False, name='nn_unlabeled')

    def _setup_nns_gps(self, sess):
        """
        Sets up NNs and GPs for NN and GP experiment.
        """
        self._setup_nns()
        self._setup_gps()
        self._setup_logging(sess)

    def _take_gradient_step(self, sess):
        """
        Takes a gradient step on the semi-supervised loss and computes new
        losses.
        """
        t0 = time.time()
        self._evaluate_step(sess)
        self._write_summary()
        self.step += 1
        t1 = time.time()
        self._print_iteration_summary(t1 - t0)
        
        if self.step == 1:
            self.best_saved_metric = self.mse_val +1

        if ((self.step % self.save_every == 0) and (self.mse_val < self.best_saved_metric)):
            # Saving trained embeddings after semisup DKL
            self._compute_embeddings(sess)
            self._save_embeddings('semisup')
        self._record_metrics()
        self._increment_counter()
        if (self.step >= self.max_iters - 1) or (self.counter >= 50):
            self.coord.request_stop()

#%%

results_dir = 'training/ssdkl'
max_iters = 200
num_trials = 2

SSDKLModel = SemisupDKLTrainer(x, y, trainMask, testMask, valMask, labeledMask, unlabeledMask,results_dir,max_iters,num_trials,save_results=True)
SSDKLModel.run_trials()

#%%

from tensorflow.linalg import pinv


#%%

tf.linalg.pinv

#%%





#%%