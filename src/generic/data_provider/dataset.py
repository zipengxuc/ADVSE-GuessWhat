import copy


class AbstractDataset(object):
    def __init__(self, games):
        self.games = games

    def get_data(self, indices=list()):
        if len(indices) > 0:
            return [self.games[i] for i in indices]
        else:
            return self.games

    def n_examples(self):
        return len(self.games)


class CropDataset(AbstractDataset):
    """
    Each game contains no question/answers but a new object
    """

    def __init__(self, dataset, expand_objects):
        old_games = dataset.get_data()
        new_games = []

        for g in old_games:
            if expand_objects:
                new_games += self.split(g)
            else:
                new_games += self.update_ref(g)

        super(CropDataset, self).__init__(new_games)

    @staticmethod
    def load(dataset_cls, expand_objects, **kwargs):
        return CropDataset(dataset_cls(**kwargs), expand_objects=expand_objects)

    @staticmethod
    def split(game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)  # Beware shallow copy!

            # select new object
            new_game.object = obj

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)  # Beware shallow copy!
            new_game.image.id = obj.id

            games.append(new_game)

        return games

    @staticmethod
    def update_ref(game):

        new_game = copy.copy(game)  # Beware shallow copy!

        # Hack the image id to differentiate objects
        new_game.image = copy.copy(game.image)  # Beware shallow copy!
        new_game.image.id = game.object.id

        return [new_game]
