import os
import tensorflow as tf
import numpy as np
import pandas as pd
import click
from itertools import zip_longest


#%%

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def mlp(x, layer_sizes, scope='mlp'):
    with tf.variable_scope(scope):
        for i, h in enumerate(layer_sizes):
            activation = None if i == (len(layer_sizes) - 1) else tf.nn.relu
            x = tf.contrib.layers.fully_connected(x, h, activation_fn=activation, scope=str(i))
    return x

def mlp_placeholders(layer_sizes, n_input):
    """
    Create placeholders for weights for a MLP
    """
    placeholders = []
    n_and_layer_sizes = [n_input] + layer_sizes
    for h_i, h_o in zip(n_and_layer_sizes[:-1], n_and_layer_sizes[1:]):
        W = tf.placeholder(tf.float32, shape=[h_i, h_o])
        b = tf.placeholder(tf.float32, shape=[h_o])
        placeholders.append(W)
        placeholders.append(b)
    return placeholders


def evaluate_mlp_placeholders(x, placeholders):
    for i, tup in enumerate(grouper(placeholders, 2)):
        W, b = tup
        activation = tf.identity if i == (len(placeholders) - 1) else tf.nn.relu
        x = activation(tf.matmul(x, W) + b)
    return x


def shuffle_idxs(X):
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    return idxs


def create_feed_dicts(X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_val, y_val,
                      X_test, y_test, teacher_placeholders, teacher_ewas,
                      X_pl, y_pl, labeled_mask_pl, student_noise_pl, teacher_noise_pl):
    # feed teacher values
    teacher_feed_dict = {teacher_var: teacher_val
                         for teacher_var, teacher_val in zip(teacher_placeholders, teacher_ewas)}

    # get half labeled, half unlabeled
    l_idxs = shuffle_idxs(X_labeled)
    u_idxs = shuffle_idxs(X_unlabeled)
    X_l, y_l = X_labeled[l_idxs], y_labeled[l_idxs]
    y_l = y_l[:, np.newaxis]
    X_u = X_unlabeled[u_idxs][:X_labeled.shape[0]]
    X_batch = np.vstack([X_l, X_u])
    half_batch_zeros = np.zeros((X_labeled.shape[0], 1), np.float32)
    y_batch = np.vstack([y_l, half_batch_zeros])
    l_mask = np.vstack([np.ones((X_labeled.shape[0], 1), np.float32),
                        half_batch_zeros])
    feed_dict = {X_pl: X_batch, y_pl: y_batch, labeled_mask_pl: l_mask,
                 student_noise_pl: 0.4 * np.random.randn(*X_batch.shape),
                 teacher_noise_pl: 0.4 * np.random.randn(*X_batch.shape)}
    feed_dict.update(teacher_feed_dict)

    # make the validation feed dict
    # validation size always less than 64
    validation_feed_dict = {X_pl: X_val, y_pl: y_val[:, np.newaxis], labeled_mask_pl: np.ones((y_val.size, 1)),
                            student_noise_pl: np.zeros(X_val.shape, np.float32),
                            teacher_noise_pl: np.zeros(X_val.shape, np.float32)}
    validation_feed_dict.update(teacher_feed_dict)

    # make the test feed dict
    test_feed_dict = {X_pl: X_test, y_pl: y_test[:, np.newaxis], labeled_mask_pl: np.ones((y_test.size, 1)),
                      student_noise_pl: np.zeros(X_test.shape, np.float32),
                      teacher_noise_pl: np.zeros(X_test.shape, np.float32)}
    test_feed_dict.update(teacher_feed_dict)
    
    pred_dict = {X_pl: X_unlabeled, y_pl: y_unlabeled[:, np.newaxis], labeled_mask_pl: np.ones((y_unlabeled.size, 1)),
                  student_noise_pl: np.zeros(X_unlabeled.shape, np.float32),
                  teacher_noise_pl: np.zeros(X_unlabeled.shape, np.float32)}
    pred_dict.update(teacher_feed_dict)
    
    return feed_dict, validation_feed_dict, test_feed_dict, pred_dict


def optimize_mean_teacher(x,y,testMask,valMask,labeledMask,unlabeledMask):
    #Get data
    tf.reset_default_graph()
    X_labeled = x[labeledMask]
    y_labeled = y[labeledMask]
    
    X_unlabeled = x[unlabeledMask]
    y_unlabeled = y[unlabeledMask]
    
    X_test = x[testMask]
    y_test = y[testMask]
    
    X_val = x[valMask]
    y_val = y[valMask]
    
    # Parameters
    learning_rate = 1e-3
    training_epochs = 1000

    # moving average for teacher
    alpha = 0.999

    # Network Parameters
    n_input = X_labeled.shape[1]
    layer_sizes = [100, 50, 50, 2, 1]

    # input
    X = tf.placeholder(tf.float32, [None, n_input], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    labeled_mask = tf.placeholder(tf.float32, [None, 1], name='mask')
    student_input_noise = tf.placeholder(tf.float32, [None, n_input], name='s_noise')
    teacher_input_noise = tf.placeholder(tf.float32, [None, n_input], name='t_noise')

    # give student and teacher different perturbed versions of input
    student = mlp(X+student_input_noise, layer_sizes, scope='student')
    teacher_placeholders = mlp_placeholders(layer_sizes, n_input)
    teacher = evaluate_mlp_placeholders(X+teacher_input_noise, teacher_placeholders)

    student_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='student')

    # label and consistency costs
    labeled_cost = tf.reduce_mean(tf.square((student - y) * labeled_mask))
    consistency_cost = tf.reduce_mean(tf.square(teacher - student))
    cost = labeled_cost + consistency_cost

    # Define loss and optimizer, minimize the squared error
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # initialize teacher ewas
        teacher_ewas = sess.run(student_variables)

        train_fd, val_fd, test_fd, pred_dict = create_feed_dicts(
            X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_val, y_val,
            X_test, y_test, teacher_placeholders, teacher_ewas,
            X, y, labeled_mask, student_input_noise, teacher_input_noise)

        # Training cycle
        try:
            # counter for validation getting worse
            val_counter = 0
            last_val = np.inf
            val_mses, test_mses = [], []
            for epoch in range(training_epochs):
                # Loop over all batches
                train_fd, val_fd, test_fd, pred_dict = create_feed_dicts(
                    X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_val, y_val,
                    X_test, y_test, teacher_placeholders, teacher_ewas,
                    X, y, labeled_mask, student_input_noise, teacher_input_noise)
                _, c, train_mse = sess.run([optimizer, cost, labeled_cost],
                                           feed_dict=train_fd)
                val_mse = sess.run(labeled_cost, feed_dict=val_fd)
                test_mse = sess.run(labeled_cost, feed_dict=test_fd)
    
                # update teacher
                student_var_vals = sess.run(student_variables)
                for j, val in enumerate(student_var_vals):
                    teacher_ewas[j] = alpha * teacher_ewas[j] + (1-alpha) * val
    
                #print("    Epoch: {}, training_loss={:.2f}, train_mse={:.2f}".format(epoch+1, c, train_mse))
                #print("                      val_mse={:.2f}, test_mse={:.2f}".format(val_mse, test_mse))
                val_mses.append(val_mse)
                test_mses.append(test_mse)
    
                if val_mse >= last_val:
                    val_counter += 1
                last_val = val_mse

                if val_counter > 10:
                    break
        except Exception:
            #print("Stopping")
            pass

        predictionStudent = sess.run(student, feed_dict=test_fd)
        predictionTeacher = sess.run(teacher, feed_dict=test_fd)

        best_idx = np.argmin(val_mses)
        chosen_test_mse = test_mses[best_idx]
        best_val_mse = val_mses[best_idx]
    tf.reset_default_graph()
    return np.sqrt(chosen_test_mse), np.sqrt(best_val_mse), predictionStudent