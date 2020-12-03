import numpy as np
from generic.utils.file_handlers import pickle_loader


class GloveEmbeddings(object):

    def __init__(self, file, glove_dim=300):
        self.glove = pickle_loader(file)
        self.glove_dim = glove_dim

    def get_embeddings(self, tokens):
        vectors = []
        for token in tokens:
            token = token.lower().replace("\'s", "")
            if token in self.glove:
                vectors.append(np.array(self.glove[token]))
            else:
                vectors.append(np.zeros((self.glove_dim,)))
        return vectors


def padder(list_of_tokens, seq_length=None, padding_symbol=0, max_seq_length=0):

    if seq_length is None:
        seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)

    padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_symbol)

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq)] = seq

    return padded_tokens, seq_length, max_seq_length


def padder_3d(list_of_tokens, max_seq_length=0, feature_size=0):
    seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)
    # print('seq_l', seq_length)

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)
    if feature_size == 0:
        feature_size = list_of_tokens[0][0].shape[0]

    padded_tokens = np.zeros(shape=(batch_size, max_seq_length, feature_size))

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq), :] = seq

    return padded_tokens, max_seq_length


def mask_generate(lengths, feature_size=0):
    seq_length = np.array([q.max() for q in lengths], dtype=np.int32)
    turn_length = np.array([q.shape[0] for q in lengths], dtype=np.int32)

    if feature_size == 0:
        feature_size = seq_length.max()

    B = len(lengths)
    max_seq_length = turn_length.max()

    padding_mask = np.zeros(shape=(B, max_seq_length, feature_size), dtype=np.float32)

    for i, (end_of_question, r) in enumerate(zip(lengths, [1] * B)):
        for j in range(max_seq_length):
            idx = end_of_question[j]
            padding_mask[i, j, :idx] = r  # gamma = 1

    return padding_mask


class DummyTokenizer(object):
    def __init__(self):
        self.padding_token = 0
        self.dummy_list = list()
        self.no_words = 10
        self.no_answers = 10
        self.unknown_answer = 0

    def encode_question(self, _):
        return self.dummy_list

    def encode_answer(self, _):
        return self.dummy_list