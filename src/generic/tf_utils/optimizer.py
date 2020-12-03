import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
import collections

AccOptimizer = collections.namedtuple("AccOptimizer", ["zero", "accumulate", "update"])


def create_optimizer(network, config, finetune=list(),
                     optim_cst=tf.train.AdamOptimizer,
                     var_list=None,
                     accumulate_gradient=False,
                     apply_update_ops=True,
                     loss=None):

    # Retrieve conf
    lrt = config['learning_rate']
    clip_val = config.get('clip_val', 0.)
    weight_decay = config['weight_decay']
    weight_decay_add = config['weight_decay_add']
    weight_decay_remove = config.get('weight_decay_remove', [])
    gradient_noise_std = config.get('gradient_noise_std', 0)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    if config['lr_decay']:
        lr_decay = tf.train.exponential_decay(lrt, global_step, 1492, 0.9, staircase=False)
    else:
        lr_decay = lrt
    # create optimizer
    optimizer = optim_cst(learning_rate=tf.maximum(lr_decay, 1e-4))

    # Extract trainable variables if not provided
    if var_list is None:
        var_list = network.get_parameters(finetune=finetune)

    # Apply weight decay
    if loss is None:
        loss = network.get_loss()

    # Apply weight decay
    training_loss = loss
    if weight_decay > 0:
        training_loss = loss + l2_regularization(var_list, weight_decay=weight_decay,
                                                 weight_decay_add=weight_decay_add,
                                                 weight_decay_remove=weight_decay_remove)

    # compute gradient
    grad = optimizer.compute_gradients(training_loss, var_list=var_list)

    # apply gradient clipping
    if clip_val > 0:
        grad = clip_gradient(grad, clip_val=clip_val)

    if gradient_noise_std > 0:
        grad = gradient_noise(grad, step=global_step, std=gradient_noise_std)

    update_ops = []
    if apply_update_ops:
        update_ops = [ops for ops in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if ops.name in network.scope_name]

    # Compute accumulated gradient
    if accumulate_gradient:
        zero, acc, update = get_accumulate_gradient_ops(grad)

        # update ops after each accumulation step
        with tf.control_dependencies(acc + update_ops):
            acc = tf.no_op()

        optimize = AccOptimizer(zero=zero,
                                accumulate=acc,
                                update=optimizer.apply_gradients(update))

    # Compute classic gradient
    else:
        with tf.control_dependencies(update_ops):
            optimize = optimizer.apply_gradients(grad, global_step=global_step)

    accuracy = network.get_accuracy()

    return optimize, [loss, accuracy]


def create_multi_gpu_optimizer(networks, config, finetune=list(), accumulate_gradient=False, optim_cst=tf.train.AdamOptimizer):
#TODO implement accumulated gradient

    # Retrieve conf
    lrt = config['learning_rate']
    clip_val = config.get('clip_val', 0.)
    weight_decay = config['weight_decay']
    weight_decay_add = config['weight_decay_add']
    weight_decay_remove = config.get('weight_decay_remove', [])

    # Create optimizer
    optimizer = optim_cst(learning_rate=lrt)

    gradients, losses, accuracies = [], [], []
    for i, network in enumerate(networks):
        with tf.device('gpu:{}'.format(i)):

            # Retrieve trainable variables from network
            train_vars = network.get_parameters(finetune=finetune)

            # Apply weight decay
            loss = network.get_loss()

            training_loss = loss
            if weight_decay > 0:
                training_loss += l2_regularization(train_vars,
                                                   weight_decay=weight_decay,
                                                   weight_decay_add=weight_decay_add,
                                                   weight_decay_remove=weight_decay_remove)
            # compute gradient
            grads = optimizer.compute_gradients(training_loss, train_vars)
            gradients.append(grads)

            # Retrieve training loss
            losses.append(network.get_loss())

            # Retrieve evaluation loss
            accuracies.append(network.get_accuracy())

    # Synchronize and average gradient/loss/accuracy
    avg_grad = average_gradient(gradients)
    avg_loss = tf.reduce_mean(tf.stack(losses))
    avg_accuracy = tf.reduce_mean(tf.stack(accuracies))

    # Clip gradient
    if clip_val > 0:
        avg_grad = clip_gradient(avg_grad, clip_val=clip_val)

    if accumulate_gradient:
        zero, acc, update = get_accumulate_gradient_ops(avg_grad)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(acc + update_ops):
            acc = tf.no_op()

        optimize = (zero, acc, optimizer.apply_gradients(update))

        assert False, "Not (yet) tested..."

    else:
        # Apply gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = optimizer.apply_gradients(avg_grad)

    return optimize, [avg_loss, avg_accuracy]


#https://stackoverflow.com/questions/46772685/how-to-accumulate-gradients-in-tensorflow
def get_accumulate_gradient_ops(gvs):

    # zero initializer
    var_list = [gv[1] for gv in gvs]
    accum_vars = [tf.Variable(tf.zeros_like(v.initialized_value()), trainable=False) for v in var_list]
    zero_ops = [v.assign(tf.zeros_like(v)) for v in accum_vars]

    # create acc/update operations 
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    train_ops = [(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)]

    return zero_ops, accum_ops, train_ops


def clip_gradient(gvs, clip_val):
    clipped_gvs = [(tf.clip_by_norm(grad, clip_val), var) for grad, var in gvs]
    return clipped_gvs


# https://openreview.net/pdf?id=rkjZ2Pcxe Decay magic number comes from the paper
def gradient_noise(gvs, step, std, decay=0.55):
    stddev = 1. * std / tf.pow(tf.to_float(step + 1), decay)
    gvs = [(grad + tf.random_normal(shape=tf.shape(grad), mean=0., stddev=stddev, dtype=tf.float32),
            var)
           for grad, var in gvs]
    return gvs


def l2_regularization(params, weight_decay, weight_decay_add=list(), weight_decay_remove=list()):
    with tf.variable_scope("l2_normalization"):

        params = [v for v in params if
                          any([(needle in v.name) for needle in weight_decay_add]) and
                      not any([(needle in v.name) for needle in weight_decay_remove])]

        if params:
            regularizer = tfc_layers.l2_regularizer(scale=weight_decay)
            weight_decay = tfc_layers.apply_regularization(regularizer, weights_list=params)
        else:
            weight_decay = 0

        return weight_decay


def average_gradient(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

