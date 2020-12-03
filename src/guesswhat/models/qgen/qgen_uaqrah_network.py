import tensorflow as tf

from neural_toolbox import rnn
from neural_toolbox.gsf import *
from neural_toolbox.uaq_toolbox import compute_q_att, compute_current_att, maskedSoftmax, OD_compute

from generic.tf_utils.abstract_network import AbstractNetwork
from guesswhat.models.qgen.qgen_utils import *

from neural_toolbox.reading_unit import create_reading_unit, create_film_layer_with_reading_unit
from neural_toolbox.film_stack import FiLM_Stack

import tensorflow.contrib.layers as tfc_layers


class QGenNetworkHREDDecoderUAQRAH(AbstractNetwork):

    def __init__(self, config, num_words, rl_module, device='', reuse=tf.AUTO_REUSE):
        AbstractNetwork.__init__(self, "qgen", device=device)

        self.rl_module = rl_module

        # Create the scope for this graph
        with tf.variable_scope(self.scope_name, reuse=reuse):

            # Misc
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            self._is_dynamic = tf.placeholder(tf.bool, name='is_dynamic')
            batch_size = None

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))
            # dropout_keep = tf.constant(1.0)

            #####################
            #   WORD EMBEDDING
            #####################

            with tf.variable_scope('word_embedding', reuse=reuse):
                self.dialogue_emb_weights = tf.get_variable("dialogue_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]])

            #####################
            #   DIALOGUE
            #####################

            if self.rl_module is not None:
                self._q_flag = tf.placeholder(tf.float32, shape=[None, batch_size], name='q_flag')
                self._cum_rewards = tf.placeholder(tf.float32, shape=[batch_size, None, None], name='cum_reward')
                # self._skewness = tf.placeholder(tf.float32, shape=[None, batch_size], name='skewness')
                # self._softmax = tf.placeholder(tf.float32, shape=[None, batch_size, None], name='softmax')
                # self._guess_softmax = tf.transpose(self._softmax, [1, 0, 2])  # batch, turn, o_n
            self._q_his = tf.placeholder(tf.int32, [batch_size, None, None], name='q_his')
            # self._q_his_mask = tf.placeholder(tf.float32, [batch_size, None, None], name='q_his_mask')
            self._a_his = tf.placeholder(tf.int32, [batch_size, None, None], name='a_his')
            self._q_his_lengths = tf.placeholder(tf.int32, [batch_size, None], name='q_his_lengths')
            self._q_turn = tf.placeholder(tf.int32, None, name='q_turn')

            bs = tf.shape(self._q_his)[0]

            self.rnn_cell_word = rnn.create_cell(config['dialogue']["rnn_word_units"],
                                               layer_norm=config["dialogue"]["layer_norm"], reuse=tf.AUTO_REUSE,
                                               cell=config['dialogue']["cell"], scope="forward_word")
            # self.word_init_states = rnn_cell_word.zero_state(batch_size, dtype=tf.float32)

            self.rnn_cell_context = rnn.create_cell(config['dialogue']["rnn_context_units"],
                                               layer_norm=config["dialogue"]["layer_norm"], reuse=tf.AUTO_REUSE,
                                               cell=config['dialogue']["cell"], scope="forward_context")
            self.rnn_cell_pair = rnn.create_cell(config['dialogue']["rnn_context_units"],
                                               layer_norm=config["dialogue"]["layer_norm"], reuse=tf.AUTO_REUSE,
                                               cell=config['dialogue']["cell"], scope="forward_pair")

            # ini
            dialogue_last_states_ini = self.rnn_cell_context.zero_state(bs, dtype=tf.float32)
            pair_states_ini = self.rnn_cell_pair.zero_state(bs, dtype=tf.float32)
            q_ini = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=tf.fill([bs], num_words))
            outputs_w_ini, _ = tf.nn.dynamic_rnn(cell=self.rnn_cell_word, inputs=tf.expand_dims(q_ini, axis=1),
                                                 dtype=tf.float32, scope="forward_word")
            qatt_ini = compute_q_att(outputs_w_ini, dropout_keep)
            answer_ini = self._a_his[:, 0]

            assert config['decoder']["cell"] != "lstm", "LSTM are not yet supported for the decoder"
            self.decoder_cell = rnn.create_cell(cell=config['decoder']["cell"],
                                                num_units=config['fusion']['projection_size'],
                                                layer_norm=config["decoder"]["layer_norm"],
                                                reuse=reuse)

            self.decoder_projection_layer = tf.layers.Dense(num_words-1)

            loss_ini = 0.

            self.q_turn_loop = tf.cond(self._is_dynamic,
                                   lambda: self._q_turn,
                                   lambda: tf.constant(5, dtype=tf.int32))
            self.turn_0 = tf.constant(0, dtype=tf.int32)
            self.m_num = tf.Variable(1.)
            self.a_yes_token = tf.constant(0, dtype=tf.int32)
            self.a_no_token = tf.constant(1, dtype=tf.int32)
            self.a_na_token = tf.constant(2, dtype=tf.int32)

            #####################
            #   IMAGES
            #####################

            self._image = tf.placeholder(tf.float32, [batch_size] + config['image']["dim"], name='image')  # B, 36, 2048
            if config['image'].get('normalize', False):
                self.img_feature = tf.nn.l2_normalize(self._image, dim=-1, name="fc_normalization")
            self.vis_dif = OD_compute(self._image)

            # Pool Image Features
            with tf.variable_scope("q_guide_image_pooling"):
                att_ini_q = compute_current_att(self.img_feature, qatt_ini,
                                              config["pooling"], self._is_training, reuse=reuse)
                att_ini_q = tf.nn.softmax(att_ini_q, axis=-1)

            with tf.variable_scope("h_guide_image_pooling"):
                att_ini_h = compute_current_att(self.img_feature, dialogue_last_states_ini,
                                              config["pooling"], self._is_training, reuse=reuse)
                att_ini_h = tf.nn.softmax(att_ini_h, axis=-1)

            self.att_ini = att_ini_h + att_ini_q
            visual_features_ini = tf.reduce_sum(self.img_feature * tf.expand_dims(self.att_ini, -1), axis=1)
            visual_features_ini = tf.nn.l2_normalize(visual_features_ini, dim=-1, name="v_fea_normalization")

            def compute_v_dif(att):
                max_obj = tf.argmax(att, axis=1)
                obj_oneh = tf.one_hot(max_obj, depth=36)  # 64,36
                vis_dif_select = tf.boolean_mask(self.vis_dif, obj_oneh)  # 64,36,36*2048 to 64,36*2048
                vis_dif_select = tf.reshape(vis_dif_select, [bs, 36, 2048])
                vis_dif_weighted = tf.reduce_sum(vis_dif_select * tf.expand_dims(att, -1), axis=1)
                vis_dif_weighted = tf.nn.l2_normalize(vis_dif_weighted, dim=-1, name="v_dif_normalization")
                return vis_dif_weighted

            vis_dif_weighted_ini = compute_v_dif(self.att_ini)

            with tf.variable_scope("compute_beta"):
                pair_states_ini = tf.nn.dropout(pair_states_ini, dropout_keep)
                beta_ini = tfc_layers.fully_connected(
                            pair_states_ini,
                            num_outputs=2,
                            activation_fn=tf.nn.softmax,
                            reuse=reuse,
                            scope="beta_computation")
            beta_0_ini = tf.tile(tf.expand_dims(tf.gather(beta_ini, 0, axis=1), 1), [1, 2048])
            beta_1_ini = tf.tile(tf.expand_dims(tf.gather(beta_ini, 1, axis=1), 1), [1, 2048])

            v_final_ini = beta_0_ini * visual_features_ini +beta_1_ini * vis_dif_weighted_ini

            turn_i = 0
            totals_ini = 0.

            with tf.variable_scope("multimodal_fusion"):

                visdiag_embedding_ini = tfc_layers.fully_connected(
                            tf.concat([dialogue_last_states_ini, v_final_ini], axis=-1),
                            num_outputs=config['fusion']['projection_size'],
                            activation_fn=tf.nn.tanh,
                            reuse=reuse,
                            scope="visdiag_projection")

            def uaqra_att(vis_fea, q_fea, dialogue_state, h_pair, att_prev, config, answer, m_num, is_training):
                with tf.variable_scope("q_guide_image_pooling"):
                    att_q = compute_current_att(vis_fea, q_fea, config, is_training, reuse=True)
                f_att_q = cond_gumbel_softmax(is_training, att_q)
                a_list = tf.reshape(answer, shape=[-1, 1])  # is it ok?
                a_list = a_list - 5
                att_na = att_prev
                att_yes = f_att_q * att_prev
                att_no = (1 - f_att_q) * att_prev
                att_select_na = tf.where(a_list == self.a_na_token, att_na, att_no)
                att_select_yes = tf.where(a_list == self.a_yes_token, att_yes, att_select_na)
                att_select = att_select_yes
                att_norm = tf.nn.l2_normalize(att_select, dim=-1, name="att_normalization")
                att_enlarged = att_norm * m_num
                att_mask = tf.greater(att_enlarged, 0.)
                att_new = maskedSoftmax(att_enlarged, att_mask)
                with tf.variable_scope("h_guide_image_pooling"):
                    att_h = compute_current_att(self.img_feature, dialogue_state, config, is_training, reuse=True)
                    att_h = tf.nn.softmax(att_h, axis=-1)
                att = att_new + att_h

                visual_features = tf.reduce_sum(self.img_feature * tf.expand_dims(att, -1), axis=1)
                visual_features = tf.nn.l2_normalize(visual_features, dim=-1, name="v_fea_normalization")
                vis_dif_weighted = compute_v_dif(att)

                with tf.variable_scope("compute_beta"):
                    h_pair = tf.nn.dropout(h_pair, dropout_keep)  # considering about the tanh activation
                    beta = tfc_layers.fully_connected(
                        h_pair,
                        num_outputs=2,
                        activation_fn=tf.nn.softmax,
                        reuse=True,
                        scope="beta_computation")

                beta_0 = tf.tile(tf.expand_dims(tf.gather(beta, 0, axis=1), 1), [1, 2048])
                beta_1 = tf.tile(tf.expand_dims(tf.gather(beta, 1, axis=1), 1), [1, 2048])
                v_final = beta_0 * visual_features + beta_1 * vis_dif_weighted

                return att, v_final, beta

            def cond_loop(cur_turn, loss, totals, q_att, h_pair, answer, att_prev, dialogue_state, visual_features, visdiag_embedding, beta):
                # print_info = tf.Print(cur_turn, [cur_turn], "x:")
                # cur_turn = cur_turn + print_info
                return tf.less(cur_turn, self.q_turn_loop)

            def dialog_flow(cur_turn, loss, totals, q_att, h_pair, answer, att_prev, dialogue_state, visual_features, visdiag_embedding, beta):

                att, v_final, beta = tf.cond(tf.equal(cur_turn, self.turn_0),
                                          lambda: (self.att_ini, visual_features, beta_ini),
                                          lambda: uaqra_att(self.img_feature, q_att, dialogue_state, h_pair, att_prev,
                                                            config["pooling"],
                                                            answer, self.m_num, self._is_training))
                repeat_v_final = tf.tile(tf.expand_dims(v_final, axis=1), [1, 12, 1])

                with tf.variable_scope("multimodal_fusion"):
                    # concat
                    visdiag_embedding = tfc_layers.fully_connected(
                        tf.concat([dialogue_state, v_final], axis=-1),
                        num_outputs=config['fusion']['projection_size'],
                        activation_fn=tf.nn.tanh,
                        reuse=True,
                        scope="visdiag_projection")

                #####################
                #   TARGET QUESTION
                #####################

                self._question = self._q_his[:, cur_turn, :]
                self._answer = self._a_his[:, cur_turn]
                self._seq_length_question = self._q_his_lengths[:, cur_turn]
                self._mask = tf.sequence_mask(lengths=self._seq_length_question-1, maxlen=12, dtype=tf.float32)

                self.word_emb_question = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=self._question)
                self.word_emb_question = tf.nn.dropout(self.word_emb_question, dropout_keep)

                self.word_emb_question_input = self.word_emb_question[:, :-1, :]
                self.word_emb_question_encode = self.word_emb_question[:, 1:, :]

                self.word_emb_answer = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=self._answer)
                self.word_emb_answer = tf.nn.dropout(self.word_emb_answer, dropout_keep)

                #####################
                #   DECODER
                #####################

                self.decoder_states, _ = tf.nn.dynamic_rnn(
                    cell=self.decoder_cell,
                    inputs=tf.concat([self.word_emb_question_input, repeat_v_final], axis=-1),
                    dtype=tf.float32,
                    initial_state=visdiag_embedding,
                    sequence_length=self._seq_length_question-1,
                    scope="decoder")

                self.decoder_outputs = self.decoder_projection_layer(self.decoder_states)

                #####################
                #   LOSS
                #####################

                # compute the softmax for evaluation

                ''' Compute policy gradient '''
                if self.rl_module is not None:

                    # Step 1: compute the state-value function
                    self._cum_rewards_current = self._cum_rewards[:, cur_turn, 1:]
                    self._cum_rewards_current *= self._mask

                    # q cost
                    # self._q_flag_cur = self._q_flag[cur_turn, :]

                    # guess softmax
                    # self._guess_cur = self._guess_softmax[:, cur_turn+1, :]
                    # self._guess_pre = self._guess_softmax[:, cur_turn, :]
                    # skewness
                    # self._skewness_cur = self._skewness[cur_turn, :]
                    # Step 2: compute the state-value function
                    value_state = self.decoder_states
                    if self.rl_module.stop_gradient:
                        value_state = tf.stop_gradient(self.decoder_states)
                    # value_state = tf.nn.dropout(value_state, dropout_keep)
                    v_num_hidden_units = int(int(value_state.get_shape()[-1]) / 4)

                    with tf.variable_scope('value_function'):
                        self.value_function = tf.keras.models.Sequential()
                        # self.value_function.add(tf.layers.Dropout(rate=dropout_keep))
                        self.value_function.add(tf.layers.Dense(units=v_num_hidden_units,
                                                                activation=tf.nn.relu,
                                                                input_shape=(int(value_state.get_shape()[-1]),),
                                                                name="value_function_hidden"))
                        # self.value_function.add(tf.layers.Dropout(rate=dropout_keep))
                        self.value_function.add(tf.layers.Dense(units=1,
                                                                activation=None,
                                                                name="value_function"))
                        self.value_function.add(tf.keras.layers.Reshape((-1,)))

                    # Step 3: compute the RL loss (reinforce, A3C, PPO etc.)
                    loss_i = rl_module(cum_rewards=self._cum_rewards_current,
                                       value_function=self.value_function(value_state),
                                       policy_state=self.decoder_outputs,
                                       actions=self._question[:, 1:],
                                       action_mask=self._mask)

                                       # q_flag=self._q_flag_cur,
                                       # pre_g=self._guess_pre,
                                       # cur_g=self._guess_cur,
                                       # skew=self._skewness_cur)

                    loss += loss_i
                    totals = 1.0

                else:
                    '''supervised loss'''
                    with tf.variable_scope('ml_loss'):
                        cr_loss = tfc_seq.sequence_loss(logits=self.decoder_outputs,
                                                             targets=self._question[:, 1:],
                                                             weights=self._mask,
                                                             average_across_timesteps=False,
                                                             average_across_batch=False)

                        # cr_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.decoder_outputs,
                        #                                                          labels=self._question[:, 1:])
                        ml_loss = tf.identity(cr_loss)
                        ml_loss *= self._mask
                        ml_loss = tf.reduce_sum(ml_loss, axis=1)  # reduce over question dimension
                        ml_loss = tf.reduce_sum(ml_loss, axis=0)  # reduce over minibatch dimension
                        count = tf.reduce_sum(self._mask)
                        self.softmax_output = tf.nn.softmax(self.decoder_outputs, name="softmax")
                        self.argmax_output = tf.argmax(self.decoder_outputs, axis=2)
                        loss += ml_loss
                        totals += count

                ''' Update the dialog state '''
                self.outputs_wq, self.h_wq = tf.nn.dynamic_rnn(cell=self.rnn_cell_word, inputs=self.word_emb_question_encode,
                                                 dtype=tf.float32, sequence_length=self._seq_length_question-1, scope="forward_word")
                _, self.h_wa = tf.nn.dynamic_rnn(cell=self.rnn_cell_word, inputs=self.word_emb_answer,
                                                 dtype=tf.float32, scope="forward_word")
                _, self.h_c1 = tf.nn.dynamic_rnn(cell=self.rnn_cell_context, inputs=tf.expand_dims(self.h_wq, 1),
                                                 initial_state=dialogue_state,
                                                 dtype=tf.float32, scope="forward_context")
                _, dialogue_state = tf.nn.dynamic_rnn(cell=self.rnn_cell_context,
                                                      inputs=tf.expand_dims(self.h_wa, 1),
                                                      initial_state=self.h_c1,
                                                      dtype=tf.float32, scope="forward_context")
                _, self.h_pq = tf.nn.dynamic_rnn(cell=self.rnn_cell_pair, inputs=tf.expand_dims(self.h_wq, 1),
                                                 dtype=tf.float32, scope="forward_pair")
                _, h_pair = tf.nn.dynamic_rnn(cell=self.rnn_cell_pair,
                                                      inputs=tf.expand_dims(self.h_wa, 1),
                                                      initial_state=self.h_pq,
                                                      dtype=tf.float32, scope="forward_pair")

                q_att = compute_q_att(self.outputs_wq, dropout_keep)

                cur_turn = tf.add(cur_turn, 1)

                return cur_turn, loss, totals, q_att, h_pair, self._answer, att, dialogue_state, v_final, visdiag_embedding, beta

            _, loss_f, totals_f, _, _, _, self.att, self.dialogue_last_states, self.visual_features, self.visdiag_embedding, self.beta = tf.while_loop(cond_loop, dialog_flow,
                                                                                            [turn_i, loss_ini, totals_ini,
                                                                                             qatt_ini, pair_states_ini,
                                                                                             answer_ini,
                                                                                             self.att_ini,
                                                                                             dialogue_last_states_ini,
                                                                                             visual_features_ini,
                                                                                             visdiag_embedding_ini,
                                                                                             beta_ini])
            self.loss = loss_f / totals_f

    def create_sampling_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._q_his)[0]
        # sample_helper = tfc_seq.SampleEmbeddingHelper(embedding=self.dialogue_emb_weights,
        #                                               start_tokens=tf.fill([batch_size], start_token),
        #                                               end_token=stop_token)
        vis_fea = self.visual_features
        start_tokens = tf.fill([batch_size], start_token)
        initial_word_input = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=start_tokens)
        initial_input = tf.concat((initial_word_input, vis_fea), 1)

        def initial_fn():
            initial_elements_finished = tf.tile([False], [batch_size])  # all False at the initial step
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            # 选择logit最大的下标作为sample
            print("outputs", outputs)
            # Return -1s where we did not sample, and sample_ids elsewhere
            prediction_id = tf.to_int32(tf.reshape(tf.multinomial(outputs, 1), [-1]))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
            pred_embedding = tf.nn.embedding_lookup(self.dialogue_emb_weights, sample_ids)
            # 输入是h_i+o_{i-1}+c_i
            next_input = tf.concat((pred_embedding, vis_fea), 1)
            elements_finished = tf.equal(sample_ids, stop_token)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: initial_input, lambda: next_input)
            next_state = state
            return elements_finished, next_inputs, next_state

        my_sample_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                        helper=my_sample_helper,
                                        initial_state=self.visdiag_embedding,
                                        output_layer=self.decoder_projection_layer)

        (_, states, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        state_value = sample_id * 0
        if self.rl_module is not None:
            state_value = self.value_function(states)

        return sample_id, seq_length, state_value, self.att

    def create_greedy_graph(self, start_token, stop_token, max_tokens):

        batch_size = tf.shape(self._q_his)[0]
        print("greedy_turns", tf.shape(self._q_his)[1])
        # greedy_helper = tfc_seq.GreedyEmbeddingHelper(embedding=self.dialogue_emb_weights,
        #                                               start_tokens=tf.fill([batch_size], start_token),
        #                                               end_token=stop_token)
        vis_fea = self.visual_features
        start_tokens = tf.fill([batch_size], start_token)
        initial_word_input = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=start_tokens)
        initial_input = tf.concat((initial_word_input, vis_fea), 1)

        def initial_fn():
            initial_elements_finished = tf.tile([False], [batch_size])  # all False at the initial step
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            # 选择logit最大的下标作为sample
            print("outputs", outputs)
            # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
            # print("slot output_logits: ", output_logits)
            # prediction_id = tf.argmax(output_logits, axis=1)
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
            pred_embedding = tf.nn.embedding_lookup(self.dialogue_emb_weights, sample_ids)
            # 输入是h_i+o_{i-1}+c_i
            next_input = tf.concat((pred_embedding, vis_fea), 1)
            elements_finished = tf.equal(sample_ids, stop_token)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: initial_input, lambda: next_input)
            next_state = state
            return elements_finished, next_inputs, next_state

        my_greedy_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        decoder = BasicDecoderWithState(cell=self.decoder_cell,
                                        helper=my_greedy_helper,
                                        initial_state=self.visdiag_embedding,
                                        output_layer=self.decoder_projection_layer)
        # print(self.visdiag_embedding)

        (_, states, sample_id), _, seq_length = tfc_seq.dynamic_decode(decoder, maximum_iterations=max_tokens)

        state_value = sample_id * 0
        if self.rl_module is not None:
            state_value = self.value_function(states)

        return sample_id, seq_length, state_value, self.att

    def create_beam_graph(self, start_token, stop_token, max_tokens, k_best):

        # create k_beams
        decoder_initial_state = self.visdiag_embedding
        # decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        #     self.visdiag_embedding, multiplier=k_best)

        # Define a beam-search decoder
        batch_size = tf.shape(self._q_his)[0]
        vis_fea = self.visual_features
        vis_fea = tf.tile(tf.expand_dims(vis_fea, axis=1), [1, k_best, 1])
        # start_tokens = tf.fill([batch_size], start_token)
        # initial_word_input = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=start_tokens)
        # initial_input = tf.concat((initial_word_input, vis_fea), 1)

        def my_decoder(state, ids):
            word_embedding = tf.nn.embedding_lookup(self.dialogue_emb_weights, ids)
            next_input = tf.concat((word_embedding, vis_fea), 2)  # B, 20, 2348
            state_input = tf.transpose(state, [1, 0, 2])  # 20, B, 600
            # print('all', state_input[0])
            # print('all', state_input[0,:,:])
            state_output = []
            for i in range(k_best):
                # word_i = word_embedding[:, i, :]
                next_input_i = next_input[:, i, :]  # 64,2348

                _, state_output_i = tf.nn.dynamic_rnn(cell=self.decoder_cell,
                                              inputs=tf.expand_dims(next_input_i, 1),
                                              initial_state=state_input[i, :, :],
                                              dtype=tf.float32, scope="beam_search_decode")
                # print('i', state_output_i)
                state_output.append(tf.expand_dims(state_output_i, 1))
            state_output = tf.concat(state_output, axis=1)
            logits = self.decoder_projection_layer(state_output)
            return state_output, logits
        sample_id, seq_length = rnn.rnn_beam_search(batch_size, my_decoder, decoder_initial_state, max_tokens, k_best,
                    start_token, stop_token, name="rnn_beam_search")

        return sample_id, seq_length, self.att

    def get_loss(self):
        # return self.loss
        return self.loss

    def get_accuracy(self):
        return self.loss

    @staticmethod
    def is_seq2seq():
        return False


if __name__ == "__main__":
    import json

    with open("../../../../config/qgen/config.baseline.json", 'rb') as f_config:
        config = json.load(f_config)

    network = QGenNetworkHREDDecoder2(config["model"], num_words=111, policy_gradient=True)

    network.create_sampling_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_greedy_graph(start_token=1, stop_token=2, max_tokens=10)
    network.create_beam_graph(start_token=1, stop_token=2, max_tokens=10, k_best=5)
