import numpy as np
import collections
import copy
from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat
from generic.data_provider.nlp_utils import padder, padder_3d


class GuesserBatchifier_RAH(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, status=list()):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.glove = glove

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def split(self, games):
        new_games = []

        for game in games:

            # Filter ill-formatted questions with stop_dialogues tokens
            if self.tokenizer.stop_dialogue_word in game.questions[-1]:
                new_game = copy.copy(game)
                game.questions[:-1] = game.questions[:-1]
                new_game.questions_ids = game.question_ids[:-1]
            else:
                new_game = game

            new_games.append(new_game)

        return new_games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games
        batch_size = len(games)

        for i, game in enumerate(games):

            # Encode question answers
            q_tokens = [self.tokenizer.encode(q, add_stop_token=True) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            # if self.generate:  # Add a dummy question at eval time to not ignore the last question
            #     q_tokens.append([])
            #     a_tokens.append([])
            #
            a_tokens, a_lengths, _ = padder(a_tokens, padding_symbol=self.tokenizer.padding_token, max_seq_length=1)

            # pad the question
            q_tokens_pad, q_lengths, _ = padder(q_tokens, padding_symbol=self.tokenizer.padding_token,
                                                max_seq_length=12)
            # print(q_tokens_pad.shape)
            batch["q_his"].append(q_tokens_pad)
            batch["q_his_lengths"].append(q_lengths)
            batch["a_his"].append(a_tokens)

            # Object embedding
            obj_spats, obj_cats = [], []
            for index, obj in enumerate(game.objects):

                bbox = obj.bbox
                spatial = get_spatial_feat(bbox, game.image.width, game.image.height)
                category = obj.category_id

                #                    1 point                 width         height
                bbox_coord = [bbox.x_left, bbox.y_upper, bbox.x_width, bbox.y_height]

                if obj.id == game.object.id and not skip_targets:
                    batch['target_category'].append(category)
                    batch['target_spatial'].append(spatial)
                    batch['target_index'].append(index)
                    batch['target_bbox'].append(bbox_coord)

                obj_spats.append(spatial)
                obj_cats.append(category)
            batch['obj_spat'].append(obj_spats)
            batch['obj_cat'].append(obj_cats)

            # image
            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        # Pad dialogue tokens
        batch["q_his"], max_turn = padder_3d(batch["q_his"])
        batch["q_his_lengths"], batch["q_turn"], batch["max_turn"] = padder(batch["q_his_lengths"], padding_symbol=1)
        batch["a_his"], _ = padder_3d(batch["a_his"], feature_size=1)
        # print(batch["q_turn"])

        # Pad objects
        batch['obj_spat'], _ = padder_3d(batch['obj_spat'])   # , max_seq_length=20)
        batch['obj_cat'], obj_length, _ = padder(batch['obj_cat'])  # , max_seq_length=20)
        batch['obj_seq_length'] = obj_length
        return batch
