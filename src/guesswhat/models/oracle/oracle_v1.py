import tensorflow as tf

from neural_toolbox import rnn_v1, utils_v1

from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features

class OracleNetwork_v1(ResnetModel):

    def __init__(self, config, num_words, num_answers, device='', reuse=False):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            embeddings = []
            self.batch_size = None

            # QUESTION
            self._is_training = tf.placeholder(tf.bool, name="is_training")
            self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
            self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')

            word_emb = utils_v1.get_embedding(self._question,
                                           n_words=num_words,
                                           n_dim=int(config['question']["embedding_dim"]),
                                           scope="word_embedding")

            lstm_states, _ = rnn_v1.variable_length_LSTM(word_emb,
                                                   num_hidden=int(config['question']["no_LSTM_hiddens"]),
                                                   seq_length=self._seq_length)
            embeddings.append(lstm_states)

            # CATEGORY
            if config['inputs']['category']:
                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = utils_v1.get_embedding(self._category,
                                              int(config['category']["n_categories"]) + 1,  # we add the unkwon category
                                              int(config['category']["embedding_dim"]),
                                              scope="cat_embedding")
                embeddings.append(cat_emb)
                print("Input: Category")

            # SPATIAL
            if config['inputs']['spatial']:
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                embeddings.append(self._spatial)
                print("Input: Spatial")


            # IMAGE
            if config['inputs']['image']:
                self._image = tf.placeholder(tf.float32, [self.batch_size] + config['image']["dim"], name='image')
                self.image_out = get_image_features(
                    image=self._image, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['image']
                )
                embeddings.append(self.image_out)
                print("Input: Image")

            # CROP
            if config['inputs']['crop']:
                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['crop']["dim"], name='crop')
                self.crop_out = get_image_features(
                    image=self._crop, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['crop'])

                embeddings.append(self.crop_out)
                print("Input: Crop")


            # Compute the final embedding
            emb = tf.concat(embeddings, axis=1)

            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['MLP']['num_hiddens']
                l1 = utils_v1.fully_connected(emb, num_hiddens, activation='relu', scope='l1')

                self.pred = utils_v1.fully_connected(l1, num_classes, activation='softmax', scope='softmax')
                self.prediction = tf.argmax(self.pred, axis=1)

            self.loss = tf.reduce_mean(utils_v1.cross_entropy(self.pred, self._answer))
            self.error = tf.reduce_mean(utils_v1.error(self.pred, self._answer))
            self.accuracy = 1. - self.error

            print('Model... Oracle build!')

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy
