from generic.data_provider.iterator import BasicIterator
from generic.tf_utils.evaluator import Evaluator


class GuesserWrapper(object):

    def __init__(self, guesser, batchifier, tokenizer, listener):
        self.guesser = guesser
        self.batchifier = batchifier
        self.tokenizer = tokenizer
        self.listener = listener
        self.evaluator = None

    def initialize(self, sess):
        self.evaluator = Evaluator(self.guesser.get_sources(sess), self.guesser.scope_name)

    def find_object(self, sess, games):

        # the guesser may need to split the input
        iterator = BasicIterator(games,
                                 batch_size=len(games),
                                 batchifier=self.batchifier)

        # sample
        self.evaluator.process(sess, iterator, outputs=[], listener=self.listener, show_progress=False)
        results = self.listener.results()

        # Update games
        new_games = []
        # for game in games:
        for game in games:

            res = results[game.dialogue_id]
            # print("--")
            # print(att)

            game.id_guess_object = res["id_guess_object"]
            game.user_data.get("softmax", []).append(res["softmax"])
            game.status = "success" if res["success"] else "failure"
            game.is_full_dialogue = True

            new_games.append(game)

        return new_games
