import tensorflow as tf
import tensorflow.contrib.layers as tfc_layers
from neural_toolbox import rnn_v1, utils_v1
from generic.tf_utils.abstract_network import AbstractNetwork
from neural_toolbox import rnn
from neural_toolbox.gsf import *
from neural_toolbox.uaq_toolbox import compute_q_att, compute_current_att, maskedSoftmax, OD_compute


class GuesserNetwork_RAH(AbstractNetwork):
    def __init__(self, config, num_words, device='', reuse=False):
        AbstractNetwork.__init__(self, "guesser", device=device)

        mini_batch_size = None

        with tf.variable_scope(self.scope_name, reuse=reuse):
            # Misc
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            self._is_dynamic = tf.placeholder(tf.bool, name='is_dynamic')
            batch_size = None

            dropout_keep_scalar = float(config.get("dropout_keep_prob", 1.0))
            dropout_keep = tf.cond(self._is_training,
                                   lambda: tf.constant(dropout_keep_scalar),
                                   lambda: tf.constant(1.0))
            # Objects
            self._num_object = tf.placeholder(tf.int32, [mini_batch_size], name='obj_seq_length')
            self.obj_mask = tf.sequence_mask(self._num_object, dtype=tf.float32)
            # self.obj_mask = tf.sequence_mask(self._num_object, maxlen=20, dtype=tf.float32)
            # self.obj_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='obj_mask')
            self.obj_cats = tf.placeholder(tf.int32, [mini_batch_size, None], name='obj_cat')
            self.obj_spats = tf.placeholder(tf.float32, [mini_batch_size, None, config['spat_dim']], name='obj_spat')

            # Targets
            self.targets = tf.placeholder(tf.int32, [mini_batch_size], name="target_index")

            self.object_cats_emb = utils_v1.get_embedding(
                self.obj_cats,
                config['no_categories'] + 1,
                config['cat_emb_dim'],
                scope='cat_embedding')

            self.objects_input = tf.concat([self.object_cats_emb, self.obj_spats], axis=2)
            self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])

            with tf.variable_scope('obj_mlp'):
                h1 = utils_v1.fully_connected(
                    self.flat_objects_inp,
                    n_out=config['obj_mlp_units'],
                    activation='relu',
                    scope='l1')
                h2 = utils_v1.fully_connected(
                    h1,
                    n_out=config['dialog_emb_dim'],
                    activation='relu',
                    scope='l2')

            obj_embs = tf.reshape(h2, [-1, tf.shape(self.obj_cats)[1], config['dialog_emb_dim']])

            #####################
            #   UAQRAH PART
            #####################

            #####################
            #   WORD EMBEDDING
            #####################

            with tf.variable_scope('word_embedding', reuse=reuse):
                self.dialogue_emb_weights = tf.get_variable("dialogue_embedding_encoder",
                                                            shape=[num_words, config["dialogue"]["word_embedding_dim"]],
                                                            initializer=tf.random_uniform_initializer(-0.08, 0.08))

            #####################
            #   DIALOGUE
            #####################

            self._q_his = tf.placeholder(tf.int32, [batch_size, None, None], name='q_his')
            # self._q_his_mask = tf.placeholder(tf.float32, [batch_size, None, None], name='q_his_mask')
            self._a_his = tf.placeholder(tf.int32, [batch_size, None, None], name='a_his')
            self._q_his_lengths = tf.placeholder(tf.int32, [batch_size, None], name='q_his_lengths')
            self._q_turn = tf.placeholder(tf.int32, [batch_size], name='q_turn')
            self._max_turn = tf.placeholder(tf.int32, None, name='max_turn')

            bs = tf.shape(self._q_his)[0]

            self.rnn_cell_word = rnn.create_cell(config['dialogue']["rnn_word_units"],
                                                 layer_norm=config["dialogue"]["layer_norm"], reuse=tf.AUTO_REUSE,
                                                 cell=config['dialogue']["cell"], scope="forward_word")

            self.rnn_cell_context = rnn.create_cell(config['dialogue']["rnn_context_units"],
                                                    layer_norm=config["dialogue"]["layer_norm"], reuse=tf.AUTO_REUSE,
                                                    cell=config['dialogue']["cell"], scope="forward_context")
            self.rnn_cell_pair = rnn.create_cell(config['dialogue']["rnn_context_units"],
                                                 layer_norm=config["dialogue"]["layer_norm"], reuse=tf.AUTO_REUSE,
                                                 cell=config['dialogue']["cell"], scope="forward_pair")

            # ini
            dialogue_last_states_ini = self.rnn_cell_context.zero_state(bs, dtype=tf.float32)
            pair_states_ini = self.rnn_cell_pair.zero_state(bs, dtype=tf.float32)

            self.max_turn_loop = tf.cond(self._is_dynamic,
                                       lambda: self._max_turn,
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
                self.img_feature = tf.nn.l2_normalize(self._image, dim=-1, name="img_normalization")
            else:
                self.img_feature = self._image
            self.vis_dif = OD_compute(self.img_feature)

            self.att_ini = tf.ones([bs, 36])

            def compute_v_dif(att):
                max_obj = tf.argmax(att, axis=1)
                obj_oneh = tf.one_hot(max_obj, depth=36)  # 64,36
                vis_dif_select = tf.boolean_mask(self.vis_dif, obj_oneh)  # 64,36,36*2048 to 64,36*2048
                vis_dif_select = tf.reshape(vis_dif_select, [bs, 36, 2048])
                vis_dif_weighted = tf.reduce_sum(vis_dif_select * tf.expand_dims(att, -1), axis=1)
                vis_dif_weighted = tf.nn.l2_normalize(vis_dif_weighted, dim=-1, name="v_dif_normalization")
                return vis_dif_weighted

            turn_i = tf.constant(0, dtype=tf.int32)

            def uaqra_att_ini(vis_fea, q_fea, dialogue_state, config, is_training):
                with tf.variable_scope("q_guide_image_pooling"):
                    att_q = compute_current_att(vis_fea, q_fea, config, is_training, reuse=reuse)
                    att_q = tf.nn.softmax(att_q, axis=-1)

                with tf.variable_scope("h_guide_image_pooling"):
                    att_h = compute_current_att(self.img_feature, dialogue_state, config, is_training, reuse=reuse)
                    att_h = tf.nn.softmax(att_h, axis=-1)
                att = att_q + att_h
                return att

            def uaqra_att(vis_fea, q_fea, dialogue_state, att_prev, config, answer, m_num, is_training):
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

                return att

            att_list_ini = tf.expand_dims(self.att_ini, 0)
            hpair_list_ini = tf.expand_dims(pair_states_ini, 0)

            def cond_loop(cur_turn, att_prev, dialogue_state, att_list, hpair_list):
                return tf.less(cur_turn, self.max_turn_loop)

            def dialog_flow(cur_turn, att_prev, dialogue_state, att_list, hpair_list):

                #####################
                #   ENCODE CUR_TURN
                #####################

                self._question = self._q_his[:, cur_turn, :]
                self._answer = self._a_his[:, cur_turn]
                self._seq_length_question = self._q_his_lengths[:, cur_turn]

                self.word_emb_question = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=self._question)
                self.word_emb_question = tf.nn.dropout(self.word_emb_question, dropout_keep)

                self.word_emb_answer = tf.nn.embedding_lookup(params=self.dialogue_emb_weights, ids=self._answer)
                self.word_emb_answer = tf.nn.dropout(self.word_emb_answer, dropout_keep)


                ''' Update the dialog state '''
                seq_mask = tf.cast(tf.greater(self._seq_length_question, 1), self._seq_length_question.dtype)

                self.outputs_wq, self.h_wq = tf.nn.dynamic_rnn(cell=self.rnn_cell_word,
                                                               inputs=self.word_emb_question,
                                                               dtype=tf.float32,
                                                               sequence_length=self._seq_length_question-1,
                                                               scope="forward_word")
                _, self.h_wa = tf.nn.dynamic_rnn(cell=self.rnn_cell_word, inputs=self.word_emb_answer,
                                                 dtype=tf.float32,
                                                 sequence_length=seq_mask, scope="forward_word")
                _, self.h_c1 = tf.nn.dynamic_rnn(cell=self.rnn_cell_context, inputs=tf.expand_dims(self.h_wq, 1),
                                                 initial_state=dialogue_state,
                                                 dtype=tf.float32,
                                                 sequence_length=seq_mask, scope="forward_context")
                _, dialogue_state = tf.nn.dynamic_rnn(cell=self.rnn_cell_context,
                                                      inputs=tf.expand_dims(self.h_wa, 1),
                                                      initial_state=self.h_c1,
                                                      dtype=tf.float32,
                                                      sequence_length=seq_mask,  scope="forward_context")
                _, self.h_pq = tf.nn.dynamic_rnn(cell=self.rnn_cell_pair, inputs=tf.expand_dims(self.h_wq, 1),
                                                 dtype=tf.float32, scope="forward_pair")
                _, h_pair = tf.nn.dynamic_rnn(cell=self.rnn_cell_pair,
                                              inputs=tf.expand_dims(self.h_wa, 1),
                                              initial_state=self.h_pq,
                                              dtype=tf.float32, scope="forward_pair")

                q_att = compute_q_att(self.outputs_wq, dropout_keep)

                att = tf.cond(tf.equal(cur_turn, self.turn_0),
                              lambda: uaqra_att_ini(self.img_feature, q_att, dialogue_state, config["pooling"], self._is_training),
                              lambda: uaqra_att(self.img_feature, q_att, dialogue_state, att_prev,
                                config["pooling"], self._answer, self.m_num, self._is_training))

                att_list = tf.cond(tf.equal(cur_turn, self.turn_0),
                                   lambda: tf.expand_dims(att, 0),
                                   lambda: tf.concat([att_list, tf.expand_dims(att, 0)], 0))
                hpair_list = tf.cond(tf.equal(cur_turn, self.turn_0),
                                   lambda: tf.expand_dims(h_pair, 0),
                                   lambda: tf.concat([hpair_list, tf.expand_dims(h_pair, 0)], 0))

                cur_turn = tf.add(cur_turn, 1)

                return cur_turn, att, dialogue_state, att_list, hpair_list

            _, _, self.dialogue_last_states, self.att_list, self.hpair_list = tf.while_loop(
                cond_loop, dialog_flow,
                [turn_i, self.att_ini, dialogue_last_states_ini, att_list_ini, hpair_list_ini],
                shape_invariants=[turn_i.get_shape(), self.att_ini.get_shape(), dialogue_last_states_ini.get_shape(),
                                  tf.TensorShape([None, None, 36]), tf.TensorShape([None, None, 1200])])
            att_list = tf.transpose(self.att_list, perm=[1, 0, 2])  # 64,max_turn,36
            att_oneh = tf.one_hot(self._q_turn-1, depth=self.max_turn_loop)  # 64,max_turn
            self.att = tf.boolean_mask(att_list, att_oneh)  # 64,max_turn,36 to 64,36
            hpair_list = tf.transpose(self.hpair_list, [1, 0, 2])  # 64,max_turn,36
            self.h_pair = tf.boolean_mask(hpair_list, att_oneh)  # 64,max_turn,36 to 64,36
            visual_features = tf.reduce_sum(self.img_feature * tf.expand_dims(self.att, -1), axis=1)
            visual_features = tf.nn.l2_normalize(visual_features, dim=-1, name="v_fea_normalization")
            vis_dif_weighted = compute_v_dif(self.att)

            with tf.variable_scope("compute_beta"):
                self.h_pair = tf.nn.dropout(self.h_pair, dropout_keep)  # considering about the tanh activation
                beta = tfc_layers.fully_connected(
                    self.h_pair,
                    num_outputs=2,
                    activation_fn=tf.nn.softmax,
                    reuse=reuse,
                    scope="beta_computation")

            beta_0 = tf.tile(tf.expand_dims(tf.gather(beta, 0, axis=1), 1), [1, 2048])
            beta_1 = tf.tile(tf.expand_dims(tf.gather(beta, 1, axis=1), 1), [1, 2048])
            self.v_final = beta_0 * visual_features + beta_1 * vis_dif_weighted

            with tf.variable_scope("multimodal_fusion"):
                # concat
                self.visdiag_embedding = tfc_layers.fully_connected(
                    tf.concat([self.dialogue_last_states, self.v_final], axis=-1),
                    num_outputs=config['fusion']['projection_size'],
                    activation_fn=tf.nn.tanh,
                    reuse=reuse,
                    scope="visdiag_projection")

            scores = tf.matmul(obj_embs, tf.expand_dims(self.visdiag_embedding, axis=-1))
            scores = tf.reshape(scores, [-1, tf.shape(self.obj_cats)[1]])

            def masked_softmax(scores, mask):
                # subtract max for stability
                scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keep_dims=True), [1, tf.shape(scores)[1]])
                # compute padded softmax
                exp_scores = tf.exp(scores)
                exp_scores *= mask
                exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keep_dims=True)
                return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])

            self.softmax = masked_softmax(scores, self.obj_mask)
            self.selected_object = tf.argmax(self.softmax, axis=1)

            self.loss = tf.reduce_mean(utils_v1.cross_entropy(self.softmax, self.targets))
            self.error = tf.reduce_mean(utils_v1.error(self.softmax, self.targets))

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

    def viz(self):
        vis_att = tf.transpose(self.att_list, perm=[1, 0, 2])
        return [vis_att, self.loss]
