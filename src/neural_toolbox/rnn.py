import tensorflow as tf
import tensorflow.contrib.rnn as tfc_rnn


def create_cell(num_units, reuse=False, layer_norm=False, cell="gru", scope="rnn"):

    with tf.variable_scope(scope):

        if cell == "gru":

            if layer_norm:

                 from neural_toolbox.LayerNormBasicGRUCell import LayerNormBasicGRUCell

                 rnn_cell = LayerNormBasicGRUCell(
                     num_units=num_units,
                     layer_norm=layer_norm,
                     activation=tf.nn.tanh,
                     reuse=reuse)

            else:
                rnn_cell = tfc_rnn.GRUCell(
                    num_units=num_units,
                    activation=tf.nn.tanh,
                    # activation=tf.nn.tanh,
                    reuse=reuse)

        elif cell == "lstm":
            rnn_cell = tfc_rnn.LayerNormBasicLSTMCell(
                num_units=num_units,
                layer_norm=layer_norm,
                activation=tf.nn.tanh,
                reuse=reuse)

        else:
            assert False, "Invalid RNN cell"

    return rnn_cell


def rnn_factory(inputs, num_hidden, seq_length,
                cell="gru",
                bidirectional=False,
                max_pool=False,
                layer_norm=False,
                initial_state_fw=None, initial_state_bw=None,
                reuse=False):

    if bidirectional:

        num_hidden = num_hidden / 2

        rnn_cell_forward = create_cell(num_hidden, layer_norm=layer_norm, reuse=reuse, cell=cell, scope="forward")
        rnn_cell_backward = create_cell(num_hidden, layer_norm=layer_norm, reuse=reuse, cell=cell,  scope="backward")

        rnn_states, last_rnn_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_forward,
            cell_bw=rnn_cell_backward,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            inputs=inputs,
            sequence_length=seq_length,
            dtype=tf.float32)

        if cell == "lstm":
            last_rnn_state = tuple(last_state.h for last_state in last_rnn_state)

        # Concat forward/backward
        rnn_states = tf.concat(rnn_states, axis=2)
        last_rnn_state = tf.concat(last_rnn_state, axis=1)

    else:

        rnn_cell_forward = create_cell(num_hidden, layer_norm=layer_norm, reuse=reuse, cell=cell,  scope="forward")

        rnn_states, last_rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell_forward,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=seq_length)

        if cell == "lstm":
            last_rnn_state = last_rnn_state.h

    if max_pool:
        last_rnn_state = tf.reduce_max(rnn_states, axis=1)

    return rnn_states, last_rnn_state


def get_shape(tensor):
    """Returns static shape if available and dynamic shape otherwise."""
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
    return dims


def batch_gather(tensor, indices):
    """Gather in batch from a tensor of arbitrary size.

    In pseudocode this module will produce the following:
    output[i] = tf.gather(tensor[i], indices[i])

    Args:
    tensor: Tensor of arbitrary size.
    indices: Vector of indices.
    Returns:
    output: A tensor of gathered values.
    """
    shape = get_shape(tensor)
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)
    return output


def rnn_beam_search(batch_size, update_fn, initial_state, sequence_length, beam_width,
                    begin_token_id, end_token_id, name="rnn"):
    """Beam-search decoder for recurrent models.

    Args:
    update_fn: Function to compute the next state and logits given the current
               state and ids.
    initial_state: Recurrent model states.
    sequence_length: Length of the generated sequence.
    beam_width: Beam width.
    begin_token_id: Begin token id.
    end_token_id: End token id.
    name: Scope of the variables.
    Returns:
    ids: Output indices.
    logprobs: Output log probabilities probabilities.
    """
    # batch_size = initial_state.shape.as_list()[0]

    state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_width, 1])

    sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])

    ids = tf.tile([[begin_token_id]], [batch_size, beam_width])
    sel_ids = tf.zeros([batch_size, beam_width, 0], dtype=ids.dtype)
    sel_masks = tf.zeros([batch_size, beam_width, 0], dtype=ids.dtype)
    # seq_length = tf.zeros([batch_size, beam_width], dtype=ids.dtype)
    mask = tf.ones([batch_size, beam_width], dtype=tf.float32)

    for i in range(sequence_length):
        with tf.variable_scope(name, reuse=True if i > 0 else None):

            state, logits = update_fn(state, ids)
            logits = tf.nn.log_softmax(logits)

            sum_logprobs = (
            tf.expand_dims(sel_sum_logprobs, axis=2) +
            (logits * tf.expand_dims(mask, axis=2)))

            num_classes = logits.shape.as_list()[-1]

            sel_sum_logprobs, indices = tf.nn.top_k(
            tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]), k=beam_width)

            ids = indices % num_classes

            beam_ids = indices // num_classes

            state = batch_gather(state, beam_ids)

            sel_ids = tf.concat([batch_gather(sel_ids, beam_ids),
                           tf.expand_dims(ids, axis=2)], axis=2)
            # end_tokens = tf.tile(tf.expand_dims(tf.expand_dims(end_token_id, 0), 1), [batch_size, beam_width])
            # seq_length = tf.where(tf.not_equal(ids, end_token_id), seq_length+1, seq_length)

            mask = (batch_gather(mask, beam_ids) * tf.to_float(tf.not_equal(ids, end_token_id)))
            sel_masks = tf.concat([batch_gather(sel_masks, beam_ids),
                           tf.expand_dims(tf.cast(mask, dtype=sel_masks.dtype), axis=2)], axis=2)
        seq_length = tf.reduce_sum(sel_masks, axis=2)+1
    return sel_ids, seq_length



