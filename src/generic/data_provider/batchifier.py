import copy
from enum import Enum
import collections


class BatchifierSplitMode(Enum):
    NoSplit = 0
    SingleQuestion = 1
    DialogueHistory = 2
    last_question = 2

    @staticmethod
    def from_string(s):
        s = s.lower()
        if s == "no_split":
            return BatchifierSplitMode.NoSplit
        elif s == "question":
            return BatchifierSplitMode.SingleQuestion
        elif s == "dialogue":
            return BatchifierSplitMode.DialogueHistory
        else:
            assert False, "Invalid question type for batchifier. Was {}".format(s)


def batchifier_split_helper(games, split_mode):

    new_games = []

    # One sample = One full dialogue
    if split_mode == 0:
        new_games = games

    # One sample = One question
    elif split_mode == 1:
        for game in games:
            for i, q, a in zip(game.question_ids, game.questions, game.answers):
                new_game = copy.copy(game)  # Beware shallow copy!
                new_game.questions = [q]
                new_game.question_ids = [i]
                new_game.answers = [a]
                new_game.is_full_dialogue = False
                new_game.user_data = {"full_game": game}

                new_games.append(new_game)

    # One sample = Subset of questions
    elif split_mode == 2:
        for game in games:
            for i in range(len(game.question_ids)):
                new_game = copy.copy(game)  # Beware shallow copy!
                new_game.questions = game.questions[:i + 1]
                new_game.question_ids = game.question_ids[:i + 1]
                new_game.answers = game.answers[:i + 1]
                new_game.is_full_dialogue = len(game.question_ids) == len(new_game.question_ids)
                new_game.user_data = {"full_game": game}

                new_games.append(new_game)

    # elif split_mode == 2:
    #     for game in games:
    #         for i in range(len(game.question_ids)):
    #             new_game = copy.copy(game)  # Beware shallow copy!
    #             new_game.questions = game.questions[:i + 1]
    #             new_game.question_ids = game.question_ids[:i + 1]
    #             new_game.answers = game.answers[:i + 1]
    #             new_game.is_full_dialogue = len(game.question_ids) == len(new_game.question_ids)
    #             new_game.user_data = {"full_game": game}
    #
    #             new_games.append(new_game)

    return new_games


class AbstractBatchifier(object):

    def apply(self, games, skip_targets=False):
        pass

    def split(self, games):
        return games

    def filter(self, games):
        return games
