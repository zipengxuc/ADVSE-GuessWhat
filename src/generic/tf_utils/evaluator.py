from tqdm import tqdm
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np


# TODO check if optimizers are always ops? Maybe there is a better check
def is_optimizer(x):
    return hasattr(x, 'op_def')


def is_summary(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.string


def is_float(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32


def is_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0


def make_as_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


class Evaluator(object):
    def __init__(self, provided_sources, scope="", network=None, tokenizer=None):  # debug purpose only, do not use in the code

        self.provided_sources = provided_sources
        self.scope = scope
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network = network
        self.tokenizer = tokenizer

    def process(self, sess, iterator, outputs, listener=None, show_progress=True):

        assert isinstance(outputs, list), "outputs must be a list"

        is_training = any([is_optimizer(x) for x in outputs])

        original_outputs = list(outputs)
        if listener is not None:
            outputs += make_as_list(listener.require())  # add require outputs
            outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
            listener.before_epoch(is_training)

        n_iter, mean_ratio = 1., 0.
        aggregated_outputs = [[] for v in outputs if is_scalar(v) and v in original_outputs]
        # aggregated_outputs = [[],[]]

        # Showing progress is optional
        progress = tqdm if show_progress else lambda x: x
        for batch in progress(iterator):

            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training
            batch["is_dynamic"] = True

            # evaluate the network on the batch
            results = self.execute(sess, outputs, batch)

            # process the results
            i = 0
            for var, result in zip(outputs, results):

                if is_scalar(var) and var in original_outputs:
                    aggregated_outputs[i].append(result*len(batch["raw"]))
                    i += 1

                if listener is not None and listener.valid(var):
                    listener.after_batch(result, batch, is_training)

            mean_ratio += len(batch["raw"])
            n_iter += 1

        if listener is not None:
            listener.after_epoch(is_training)

        aggregated_outputs = [sum(out) / mean_ratio for out in aggregated_outputs]

        return aggregated_outputs
        # return results

    def execute(self, sess, output, batch):
        feed_dict = {self.scope + key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        return sess.run(output, feed_dict=feed_dict)


class EvaluatorC(object):
    def __init__(self, provided_sources, scope="", network=None,
                 tokenizer=None):  # debug purpose only, do not use in the code

        self.provided_sources = provided_sources
        self.scope = scope
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network = network
        self.tokenizer = tokenizer

    def process(self, sess, iterator, distri, outputs, listener=None, show_progress=True):

        assert isinstance(outputs, list), "outputs must be a list"

        is_training = any([is_optimizer(x) for x in outputs])

        original_outputs = list(outputs)
        if listener is not None:
            outputs += make_as_list(listener.require())  # add require outputs
            outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
            listener.before_epoch(is_training)

        n_iter, mean_ratio = 1., 0.
        aggregated_outputs = [[] for v in outputs if is_scalar(v) and v in original_outputs]
        # aggregated_outputs = [[],[]]

        # compute skewness
        skewness = compute_skewness(distri)

        # Showing progress is optional
        progress = tqdm if show_progress else lambda x: x
        for batch in progress(iterator):

            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training
            batch["is_dynamic"] = True
            batch["softmax"] = distri
            batch["skewness"] = skewness

            # evaluate the network on the batch
            results = self.execute(sess, outputs, batch)

            # process the results
            i = 0
            for var, result in zip(outputs, results):

                if is_scalar(var) and var in original_outputs:
                    aggregated_outputs[i].append(result * len(batch["raw"]))
                    i += 1

                if listener is not None and listener.valid(var):
                    listener.after_batch(result, batch, is_training)

            mean_ratio += len(batch["raw"])
            n_iter += 1

        if listener is not None:
            listener.after_epoch(is_training)

        aggregated_outputs = [sum(out) / mean_ratio for out in aggregated_outputs]

        return aggregated_outputs
        # return results

    def execute(self, sess, output, batch):
        feed_dict = {self.scope + key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        return sess.run(output, feed_dict=feed_dict)


class EvaluatorCR(object):
    def __init__(self, provided_sources, scope="", network=None,
                 tokenizer=None):  # debug purpose only, do not use in the code

        self.provided_sources = provided_sources
        self.scope = scope
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network = network
        self.tokenizer = tokenizer

    def process(self, sess, iterator, distri, q_flag, outputs, listener=None, show_progress=True):

        assert isinstance(outputs, list), "outputs must be a list"

        is_training = any([is_optimizer(x) for x in outputs])

        original_outputs = list(outputs)
        if listener is not None:
            outputs += make_as_list(listener.require())  # add require outputs
            outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
            listener.before_epoch(is_training)

        n_iter, mean_ratio = 1., 0.
        aggregated_outputs = [[] for v in outputs if is_scalar(v) and v in original_outputs]
        # aggregated_outputs = [[],[]]

        # compute skewness
        skewness = compute_skewness(distri)

        # Showing progress is optional
        progress = tqdm if show_progress else lambda x: x
        for batch in progress(iterator):

            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training
            batch["is_dynamic"] = True
            batch["softmax"] = distri
            batch["skewness"] = skewness
            batch["q_flag"] = q_flag

            # evaluate the network on the batch
            results = self.execute(sess, outputs, batch)

            # process the results
            i = 0
            for var, result in zip(outputs, results):

                if is_scalar(var) and var in original_outputs:
                    aggregated_outputs[i].append(result * len(batch["raw"]))
                    i += 1

                if listener is not None and listener.valid(var):
                    listener.after_batch(result, batch, is_training)

            mean_ratio += len(batch["raw"])
            n_iter += 1

        if listener is not None:
            listener.after_epoch(is_training)

        aggregated_outputs = [sum(out) / mean_ratio for out in aggregated_outputs]

        return aggregated_outputs
        # return results

    def execute(self, sess, output, batch):
        feed_dict = {self.scope + key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        return sess.run(output, feed_dict=feed_dict)


class EvaluatorHRED(object):
    def __init__(self, provided_sources, scope="", network=None,
                 tokenizer=None):  # debug purpose only, do not use in the code

        self.provided_sources = provided_sources
        self.scope = scope
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network = network
        self.tokenizer = tokenizer
        self.dialog_flow = network.dialog_flow()

    def process(self, sess, iterator, is_training=False, optimizer=None, show_progress=True):

        n_iter, mean_ratio = 1., 0.
        total_loss = 0.
        # Showing progress is optional
        progress = tqdm if show_progress else lambda x: x
        for batch in progress(iterator):

            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training
            q_his = batch["q_his"]
            q_his_lengths = batch["q_his_lengths"]
            q_mask = batch['q_his_mask']
            a_his = batch["a_his"]
            turn = batch["q_turn"]
            batch_size = batch["q_his"].shape[0]
            rnn_dialog_state = np.zeros([batch_size, 1200])
            loss = tf.Variable(0.)
            flag = None

            for i in range(turn):
                if i > 0:
                    flag = 1.0
                question = q_his[:, i, :]
                # print(question.shape)
                mask = q_mask[:, i, :]
                # print(mask.shape)
                answer = a_his[:, i]
                seq_length_question = q_his_lengths[:, i]

                # dialog flow
                loss_crt, rnn_dialog_state = self.execute(sess, self.dialog_flow, flag, batch, question, mask, answer,
                                                      seq_length_question, rnn_dialog_state)
                loss = tf.assign_add(loss, loss_crt)

            # optimize while training
            print(loss)
            gradients = optimizer.compute_gradients(loss)
            print(gradients)

            # no gradient clipping
            # Clip gradient by L2 norm
            # gradients = gradients_part1+gradients_part2
            gradients = [(tf.clip_by_norm(g, 5), v)
                         for g, v in gradients]
            solver_op = optimizer.apply_gradients(gradients)

            # Training operation
            # Partial-run can't fetch training operations
            # some workaround to make partial-run work
            # with tf.control_dependencies([solver_op]):
            #     train_step = tf.constant(0)
            sess.run(solver_op)

            # process the results
            total_loss += loss*len(batch["raw"])
            mean_ratio += len(batch["raw"])
            n_iter += 1

        average_loss = total_loss / mean_ratio

        return average_loss

    def execute(self, sess, output, flag, batch, question, mask, answer, seq_length_question, rnn_dialog_state):
        feed_dict = {self.scope + key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        feed_dict[self.scope+"question"+":0"] = question
        feed_dict[self.scope+"mask"+":0"] = mask
        feed_dict[self.scope+"answer"+":0"] = answer
        feed_dict[self.scope+"q_length"+":0"] = seq_length_question
        feed_dict[self.scope+"rnn_dialog_state"+":0"] = rnn_dialog_state
        feed_dict[self.scope+"flag"+":0"] = flag
        return sess.run(output, feed_dict=feed_dict)


class MultiGPUEvaluator(object):
    """Wrapper for evaluating on multiple GPUOptions

    parameters
    ----------
        provided_sources: list of sources
            Each source has num_gpus placeholders with name:
            name_scope[gpu_index]/network_scope/source
        network_scope: str
            Variable scope of the model
        name_scopes: list of str
            List that defines name_scope for each GPU
    """

    def __init__(self, provided_sources, name_scopes, writer=None,
                 networks=None, tokenizer=None):  # Debug purpose only, do not use here

        # Dispatch sources
        self.provided_sources = provided_sources
        self.name_scopes = name_scopes
        self.writer = writer

        self.multi_gpu_sources = []
        for source in self.provided_sources:
            for name_scope in name_scopes:
                self.multi_gpu_sources.append(os.path.join(name_scope, source))

        # Debug tools, do not use in the code!
        self.networks = networks
        self.tokenizer = tokenizer

    def process(self, sess, iterator, outputs, listener=None):

        assert listener is None, "Listener are not yet supported with multi-gpu evaluator"
        assert isinstance(outputs, list), "outputs must be a list"

        # check for optimizer to define training/eval mode
        is_training = any([is_optimizer(x) for x in outputs])

        # Prepare epoch
        n_iter, mean_ratio = 1., 0.
        aggregated_outputs = [[] for v in outputs if is_scalar(v)]

        scope_to_do = list(self.name_scopes)
        multi_gpu_batch = dict()
        for batch in tqdm(iterator):

            assert len(scope_to_do) > 0

            # apply training mode
            batch['is_training'] = is_training

            # update multi-gpu batch
            name_scope = scope_to_do.pop()
            for source, v in batch.items():
                multi_gpu_batch[os.path.join(name_scope, source)] = v

            if not scope_to_do:  # empty list -> multi_gpu_batch is ready!

                n_iter += 1

                # Execute the batch
                results = self.execute(sess, outputs, multi_gpu_batch)

                # reset mini-batch
                scope_to_do = list(self.name_scopes)
                multi_gpu_batch = dict()

                # process the results
                i = 0
                for var, result in zip(outputs, results):
                    if is_scalar(var) and var in outputs:
                        aggregated_outputs[i].append(var * len(batch["raw"]))
                        i += 1

                    elif is_summary(var):  # move into listener?
                        self.writer.add_summary(result)

                    # No listener as "results" may arrive in different orders... need to find a way to unshuffle them

        aggregated_outputs = [sum(out) / mean_ratio for out in aggregated_outputs]

        return aggregated_outputs

    def execute(self, sess, output, batch):
        feed_dict = {key + ":0": value for key, value in batch.items() if key in self.multi_gpu_sources}
        return sess.run(output, feed_dict=feed_dict)


def compute_skewness(distri):
    mean = np.mean(distri, axis=2)
    median = np.median(distri, axis=2)
    std = np.std(distri, axis=2, ddof=1)
    return 3*(mean-median)/std
