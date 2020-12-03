import math
import random
from multiprocessing import Semaphore

#Note from author : we put extra documentation as we beleive that this class can be very useful to other developers


def sem_iterator(l, sem):
    """
    Turn a list into a generator with a hidden semaphote (to limit the number off ongoing iteration)

    :param l:  list
    :param sem: semaphore
    """
    for e in l:
        sem.acquire()
        yield e


def split_batch(games, batch_size, use_padding):
    """
    Split a list of games into sublist of games of size batch_size

    :param games: list of games that is going to be used to create a batch
    :param batch_size: number of games used by batch
    :param use_padding: pad with already used games to fill the last batch
    :return: a list of list of games
    """
    i = 0
    is_done = False

    batch = []

    while not is_done:
        end = min(i + batch_size, len(games))
        selected_games = games[i:end]
        i += batch_size

        if i >= len(games):
            is_done = True
            if use_padding:
                no_missing = batch_size - len(selected_games)
                selected_games += games[:no_missing]

        batch.append(selected_games)

    return batch


class Iterator(object):
    """Provides an generic multithreaded iterator over the dataset."""

    def __init__(self, dataset, batch_size, batchifier, pool,
                 shuffle=False, use_padding=False, no_semaphore=20):

        # Filtered games
        games = dataset.get_data()
        games = batchifier.filter(games)
        games = batchifier.split(games)

        if shuffle:
            random.shuffle(games)

        self.batch_size = batch_size
        self.n_batches = int(math.ceil(1. * len(games) / self.batch_size))
        if use_padding:
            self.n_examples = self.n_batches * self.batch_size
        else:
            self.n_examples = len(games)

        batch = split_batch(games, batch_size, use_padding)

        # no proc
        # self.it = (batchifier.apply(b) for b in batch)

        # Multi_proc
        self.semaphores = Semaphore(no_semaphore)
        it_batch = sem_iterator(l=batch, sem=self.semaphores)
        self.process_iterator = pool.imap(batchifier.apply, it_batch)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        self.semaphores.release()
        return self.process_iterator.next()
        # return self.it.__next__()

    # trick for python 2.X
    def next(self):
        return self.__next__()


# TODO Fuse with Iterator
class BasicIterator(object):
    """Provides an generic multithreaded iterator over the dataset."""

    def __init__(self, games, batch_size, batchifier, shuffle=False, use_padding=False):

        # Filtered games
        games = batchifier.filter(games)
        games = batchifier.split(games)

        self.batch_size = batch_size
        self.n_batches = int(math.ceil(1. * len(games) / self.batch_size))
        if use_padding:
            self.n_examples = self.n_batches * self.batch_size
        else:
            self.n_examples = len(games)

        batch = split_batch(games, batch_size, use_padding)

        if shuffle:
            random.shuffle(games)

        # no proc
        self.it = (batchifier.apply(b) for b in batch)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.it.__next__()

    # trick for python 2.X
    def next(self):
        return self.__next__()
