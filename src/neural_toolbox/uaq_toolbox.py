import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers


def compute_q_att(context, keep_dropout, no_glimpse=2, glimpse_embedding_size=300, reuse=False):
    with tf.variable_scope("glimpse"):

        glimpses = []
        with tf.variable_scope("glimpse"):
            g_feature_maps = tf.nn.dropout(context, keep_dropout)  # B*L*C
            g_feature_maps = tfc_layers.fully_connected(g_feature_maps,
                                                        num_outputs=glimpse_embedding_size,
                                                        biases_initializer=None,
                                                        activation_fn=tf.nn.relu,
                                                        scope='q_projection1',
                                                        reuse=reuse)

            e = tfc_layers.fully_connected(g_feature_maps,
                                           num_outputs=no_glimpse,
                                           biases_initializer=None,
                                           activation_fn=None,
                                           scope='q_projection2',
                                           reuse=reuse)  # B*L*G

            # e = tf.reshape(e, shape=[-1, h * w, no_glimpse])

            for i in range(no_glimpse):
                ev = e[:, :, i]
                alpha = tf.nn.softmax(ev)
                # apply soft attention
                soft_glimpses = context * tf.expand_dims(alpha, -1)
                soft_glimpses = tf.reduce_sum(soft_glimpses, axis=1)

                glimpses.append(soft_glimpses)

        full_glimpse = tf.concat(glimpses, axis=1)

    return full_glimpse


def compute_current_att(feature_maps, context, config, is_training, reuse=False):
    with tf.variable_scope("current_attention"):
        glimpse_embedding_size = config['no_attention_mlp']
        keep_dropout = config['drop_out_keep']
        dropout_keep_ratio = tf.cond(is_training,
                               lambda: tf.constant(keep_dropout),
                               lambda: tf.constant(1.0))
        h = int(feature_maps.get_shape()[1])
        # w = int(feature_maps.get_shape()[2])
        c = int(feature_maps.get_shape()[2])

        # reshape state to perform batch operation
        context = tf.nn.dropout(context, dropout_keep_ratio)
        projected_context = tfc_layers.fully_connected(context,
                                                       num_outputs=glimpse_embedding_size,
                                                       biases_initializer=None,
                                                       activation_fn=tf.nn.relu,
                                                       scope='hidden_layer',
                                                       reuse=reuse)

        projected_context = tf.expand_dims(projected_context, axis=1)
        projected_context = tf.tile(projected_context, [1, h, 1])
        projected_context = tf.reshape(projected_context, [-1, glimpse_embedding_size])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h, c])

        g_feature_maps = tf.reshape(feature_maps, shape=[-1, c])  # linearise the feature map as as single batch
        g_feature_maps = tf.nn.dropout(g_feature_maps, dropout_keep_ratio)
        g_feature_maps = tfc_layers.fully_connected(g_feature_maps,
                                                    num_outputs=glimpse_embedding_size,
                                                    biases_initializer=None,
                                                    activation_fn=tf.nn.relu,
                                                    scope='image_projection',
                                                    reuse=reuse)

        hadamard = g_feature_maps * projected_context
        hadamard = tf.nn.dropout(hadamard, dropout_keep_ratio)

        e = tfc_layers.fully_connected(hadamard,
                                       num_outputs=1,
                                       biases_initializer=None,
                                       activation_fn=None,
                                       scope='hadamard_projection',
                                       reuse=reuse)

        e = tf.reshape(e, shape=[-1, h])

        # alpha = tf.nn.softmax(e)
        # apply soft attention
        # soft_glimpses = feature_maps * tf.expand_dims(alpha, -1)
        # soft_glimpses = tf.reduce_sum(soft_glimpses, axis=1)

    return e


def maskedSoftmax(logits, mask):
    """
    Masked softmax over dim 1
    :param logits: (N, L)
    :param mask: (N, L)
    :return: probabilities (N, L)
    from: https://github.com/tensorflow/tensorflow/issues/11756
    """
    indices = tf.where(mask)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
    result = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
    result.set_shape(logits.shape)
    return result


def OD_compute(matrix):
    # matrix: b*n*d
    # vector: b*d
    # b = int(matrix.get_shape()[0])
    n = 36
    d = 2048

    m1 = tf.tile(tf.expand_dims(matrix, 2), [1, 1, n, 1])  # b*n*n*d
    m2 = tf.tile(tf.expand_dims(matrix, 1), [1, n, 1, 1])  # b*n*n*d
    output = tf.reshape((m1 - m2), shape=[-1, n, n * d])

    return output
