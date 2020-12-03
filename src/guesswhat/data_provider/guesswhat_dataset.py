import gzip
import json
import copy
import os
import numpy as np
import PIL.Image as PImage
from PIL import ImageDraw

from generic.data_provider.dataset import AbstractDataset

try:
    import cocoapi.PythonAPI.pycocotools.mask as cocoapi
    use_coco = True
except ImportError:
    print("Coco API was not detected - advanced segmentation features cannot be used")
    use_coco = False
    pass

class Game_guesser_new(object):

    def __init__(self, id, object_id, id_guess_object, image, objects, qas, status,
                 success_turn, att=None):
        self.dialogue_id = id

        self.objects = []
        self.att = att
        for o in objects:
            self.image = Image(id=image["id"],
                               filename=image["file_name"],
                               width=image["width"],
                               height=image["height"],
                               url=None,
                               which_set=None,
                               rcnn=False,
                               image_builder=None)
            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=None,
                             segment=o['segment'],
                             crop_builder=None,
                             which_set=None,
                             image=None)

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self._object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status

        self.id_guess_object = id_guess_object

        self.is_full_dialogue = True
        self.success_turn = success_turn

        self.user_data = dict()


class Game_new(object):

    def __init__(self, id, object_id, id_guess_object, image, objects, qas, status, att=None):
        self.dialogue_id = id

        self.objects = []
        self.att = att
        for o in objects:
            self.image = Image(id=image["id"],
                               filename=image["file_name"],
                               width=image["width"],
                               height=image["height"],
                               url=None,
                               which_set=None,
                               rcnn=False,
                               image_builder=None)
            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=None,
                             segment=o['segment'],
                             crop_builder=None,
                             which_set=None,
                             image=None)

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self._object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status

        self.id_guess_object = id_guess_object

        self.is_full_dialogue = True

        self.user_data = dict()

class Game_new2(object):

    def __init__(self, id, object_id, guess_object_id, image, objects, qas, status, att=None):
        self.dialogue_id = id

        self.objects = []
        self.att = att

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status

        self.id_guess_object = guess_object_id

        self.is_full_dialogue = True

        self.user_data = dict()


class Game(object):

    def __init__(self, rcnn, id, object_id, guess_id, image, objects, qas, status, which_set, image_builder,
                 crop_builder, att=None):
        self.dialogue_id = id
        self.image = Image(id=image["id"],
                           filename=image["file_name"],
                           width=image["width"],
                           height=image["height"],
                           url=image["coco_url"],
                           which_set=which_set,
                           rcnn=rcnn,
                           image_builder=image_builder)
        self.objects = []
        self.att = att
        for o in objects:

            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=o['area'],
                             segment=o['segment'],
                             crop_builder=crop_builder,
                             which_set=which_set,
                             image=self.image)

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self._object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status

        self.id_guess_object = guess_id

        self.is_full_dialogue = True

        self.user_data = dict()

    # Optimization to pre-load image/crop inside the memory
    def bufferize(self):
        if self.image.image_loader is not None:
            self.image.image_loader.bufferize()
        if self.object.crop_loader is not None:
            self.object.crop_loader.bufferize()

    # Optimization to unload image/crop outside the memory
    def flush(self):
        if self.image.image_loader is not None:
            self.image.image_loader.flush()
        if self.object.crop_loader is not None:
            self.object.crop_loader.flush()

    @property
    def object(self):
        return self._object

    @object.setter
    def object(self, obj):

        assert isinstance(obj, Object), "Invalid object type"
        self._object = obj

    def show(self, img_raw_dir, display_index=False, display_mask=False):
        image_path = os.path.join(img_raw_dir, self.image.filename)

        img = PImage.open(image_path)
        draw = ImageDraw.Draw(img)

        for i, obj in enumerate(self.objects):
            if display_index:
                draw.text((obj.bbox.x_center, self.image.height - obj.bbox.y_center), str(i))
            if display_mask:
                print("Show mask: Not yet implemented... sry")

        img.show()

    def __str__(self):
        s = "Game = id: {} / status: {}\n".format(self.dialogue_id, self.status)
        s += " - {} \n".format(self.image)
        s += " - {} \n".format(self.object)
        s += " - Dialogue :\n"
        for i, (q, a) in enumerate(zip(self.questions, self.answers + [""])):
            s += "   {}) {} -> {}\n".format(i, q, a)
        return s


class Game_guesser(object):

    def __init__(self, rcnn, id, object_id, id_guess_object, image, objects, qas, status, which_set, image_builder,
                 crop_builder, att=None):
        self.dialogue_id = id
        self.image = Image(id=image["id"],
                           filename=image["file_name"],
                           width=image["width"],
                           height=image["height"],
                           url=image["coco_url"],
                           which_set=which_set,
                           rcnn=rcnn,
                           image_builder=image_builder)
        self.objects = []
        self.att = att
        for o in objects:

            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=o['area'],
                             segment=o['segment'],
                             crop_builder=crop_builder,
                             which_set=which_set,
                             image=self.image)

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self._object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status

        self.id_guess_object = id_guess_object

        self.is_full_dialogue = True
        self.success_turn = None

        self.user_data = dict()

    # Optimization to pre-load image/crop inside the memory
    def bufferize(self):
        if self.image.image_loader is not None:
            self.image.image_loader.bufferize()
        if self.object.crop_loader is not None:
            self.object.crop_loader.bufferize()

    # Optimization to unload image/crop outside the memory
    def flush(self):
        if self.image.image_loader is not None:
            self.image.image_loader.flush()
        if self.object.crop_loader is not None:
            self.object.crop_loader.flush()

    @property
    def object(self):
        return self._object

    @object.setter
    def object(self, obj):

        assert isinstance(obj, Object), "Invalid object type"
        self._object = obj

    def show(self, img_raw_dir, display_index=False, display_mask=False):
        image_path = os.path.join(img_raw_dir, self.image.filename)

        img = PImage.open(image_path)
        draw = ImageDraw.Draw(img)

        for i, obj in enumerate(self.objects):
            if display_index:
                draw.text((obj.bbox.x_center, self.image.height - obj.bbox.y_center), str(i))
            if display_mask:
                print("Show mask: Not yet implemented... sry")

        img.show()

    def __str__(self):
        s = "Game = id: {} / status: {}\n".format(self.dialogue_id, self.status)
        s += " - {} \n".format(self.image)
        s += " - {} \n".format(self.object)
        s += " - Dialogue :\n"
        for i, (q, a) in enumerate(zip(self.questions, self.answers + [""])):
            s += "   {}) {} -> {}\n".format(i, q, a)
        return s


class Image(object):
    def __init__(self, id, filename, width, height, url, which_set, rcnn, image_builder=None):
        self.id = id
        self.rcnn = rcnn
        self.filename = filename
        self.width = width
        self.height = height
        self.url = "http://cocodataset.org/#explore?id={}".format(id)
        self.old_url = url

        if rcnn:
            self.image_builder = image_builder
        else:
            if image_builder is not None:
                # self.filename = "{}.jpg".format(id)
                self.image_loader = image_builder.build(id, which_set=which_set, filename=self.filename, optional=False)

    def get_image(self, **kwargs):

        if self.rcnn:
            return self.image_builder.load(self.filename)
        else:
            if self.image_loader is not None:
                return self.image_loader.get_image(**kwargs)
            else:
                return None

    def __str__(self):
        return "Image = id: {} / url: {}".format(self.id, self.url)


class Bbox(object):
    def __init__(self, bbox, im_width, im_height):
        # Retrieve features (COCO format)
        self.x_width = bbox[2]
        self.y_height = bbox[3]
        self.x_left = bbox[0]

        self.x = bbox[0]
        self.y = bbox[1]
        self.width = bbox[2]
        self.height = bbox[3]

        self.x_right = self.x_left + self.x_width

        self.y_upper = im_height - bbox[1]
        self.y_lower = self.y_upper - self.y_height

        self.x_center = self.x_left + 0.5 * self.x_width
        self.y_center = self.y_lower + 0.5 * self.y_height

        self.coco_bbox = bbox

    def __str__(self):
        return "{0:5.2f}/{1:5.2f}".format(self.x_center, self.y_center)


class Object(object):
    def __init__(self, id, category, category_id, bbox, area, segment, crop_builder, image, which_set):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segment = segment

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        self.rle_mask = None
        if use_coco:
            self.rle_mask = cocoapi.frPyObjects(self.segment,
                                                h=image.height,
                                                w=image.width)

        self.crop_loader = None
        if crop_builder is not None:
            filename = "{}.jpg".format(image.id)
            self.crop_loader = crop_builder.build(id, filename=filename, which_set=which_set, bbox=bbox)
            self.crop_scale = crop_builder.scale

    def get_mask(self):
        assert self.rle_mask is not None, "Mask option are not available, please compile and link cocoapi (cf. cocoapi/PythonAPI/setup.py)"
        tmp_mask = cocoapi.decode(self.rle_mask)
        if len(tmp_mask.shape) > 2:  # concatenate several mask into a single one
            tmp_mask = np.sum(tmp_mask, axis=2)
            tmp_mask[tmp_mask > 1] = 1

        return tmp_mask.astype(np.float32)

    def get_crop(self, **kwargs):
        assert self.crop_loader is not None, "Invalid crop loader"
        return self.crop_loader.get_image(**kwargs)

    def __str__(self):
        return "Object = category: {} / center: {}".format(self.category, self.bbox)


class Dataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, which_set, image_builder=None, crop_builder=None, rcnn=False, games_to_load=float("inf")):
        file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
        games = []

        if games_to_load is None:
            games_to_load = float("inf")

        self.set = which_set

        with gzip.open(file) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))

                g = Game(rcnn=rcnn, id=game['id'],
                         object_id=game['object_id'],
                         guess_id=game.get('guess_id', -1),
                         objects=game['objects'],
                         qas=game['qas'],
                         image=game['image'],
                         status=game['status'],
                         which_set=which_set,
                         image_builder=image_builder,
                         crop_builder=crop_builder)

                games.append(g)

                # If no_games_to_load is defined : Loading a certain number of games
                if len(games) >= games_to_load:
                    break

        print("{} games were loaded...".format(len(games)))
        super(Dataset, self).__init__(games)


class Dataset_visg(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, file, image_builder=None, crop_builder=None, rcnn=False,
                 games_to_load=float("inf")):
        games = []

        if games_to_load is None:
            games_to_load = float("inf")

        self.set = 'visg'

        with open(file) as f:
            for line in f:
                game = json.loads(line.strip('\n'))

                g = Game(rcnn=rcnn, id=game['id'],
                         object_id=game['object_id'],
                         guess_id=game.get('guess_id', -1),
                         objects=game['objects'],
                         qas=game['qas'],
                         image=game['image'],
                         status=game['status'],
                         which_set='visg',
                         image_builder=image_builder,
                         crop_builder=crop_builder)

                games.append(g)

                # If no_games_to_load is defined : Loading a certain number of games
                if len(games) >= games_to_load:
                    break

        print("{} games were loaded...".format(len(games)))
        super(Dataset_visg, self).__init__(games)


class anaDataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, file, image_builder=None, crop_builder=None, rcnn=False,
                 games_to_load=float("inf")):
        # file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
        games = []

        if games_to_load is None:
            games_to_load = float("inf")

        self.set = "ana"

        with open(file) as f:
            for line in f:
                # line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))

                g = Game_guesser(rcnn=rcnn, id=game['id'],
                         object_id=game['object_id'],
                         id_guess_object=game.get('id_guess_object', -1),
                         objects=game['objects'],
                         qas=game['qas'],
                         image=game['image'],
                         status=game['status'],
                         which_set='ana',
                         image_builder=image_builder,
                         crop_builder=crop_builder)

                games.append(g)

                # If no_games_to_load is defined : Loading a certain number of games
                if len(games) >= games_to_load:
                    break

        print("{} games were loaded...".format(len(games)))
        super(anaDataset, self).__init__(games)


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

    @classmethod
    def load(cls, folder, which_set, image_builder=None, crop_builder=None, expand_objects=False, games_to_load=float("inf")):
        return CropDataset(Dataset(folder, which_set, image_builder, crop_builder, games_to_load), expand_objects=expand_objects)

    def split(self, game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)
            new_game.questions = [""]
            new_game.question_ids = [0]
            new_game.answers = [""]

            # update object reference
            new_game.object = [o for o in game.objects if o.id == obj.id][0]

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)
            new_game.image.id = obj.id

            games.append(new_game)

        return games

    def update_ref(self, game):

        new_game = copy.copy(game)  # Beware shallow copy!
        new_game.questions = [""]
        new_game.question_ids = [0]
        new_game.answers = [""]

        # Hack the image id to differentiate objects
        new_game.image = copy.copy(game.image)  # Beware shallow copy!
        new_game.image.id = game.object.id

        return [new_game]


def dump_samples_into_dataset(data, save_path, tokenizer, name="model", true_id=False):
    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for _, d in enumerate(data):
            dialogue = d["dialogue"]
            game = d["game"]
            object_id = d["object_id"]
            success = d["success"]
            prob_objects = d["prob_objects"]
            guess_object_id = d["guess_object_id"]

            sample = {}

            qas = []
            start = 1
            for k, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                        word == tokenizer.no_token or \
                        word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:k - 1])
                    a = tokenizer.decode([dialogue[k]])

                    prob_obj = list(prob_objects[len(qas), :len(game.objects)])
                    prob_obj = [str(round(p, 3)) for p in prob_obj]  # decimal are not supported y default in json encoder

                    qas.append({"question": q,
                                "answer": a[1:-1],
                                "id": k,
                                "p": prob_obj})

                    start = k + 1

            sample["id"] = game.dialogue_id if true_id else 0
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": [],  # no segment to avoid making the file to big
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["guess_object_id"] = guess_object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')


def dump_oracle(oracle_data, games, save_path, name="oracle"):
    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for game in games:

            qas = oracle_data[game.dialogue_id]
            sample = {}

            # check that question/answer are correctly sorted
            for qa, q_id in zip(qas, game.question_ids):
                assert qa["id"] == q_id

            for qo, qh in zip(qas, game.questions):
                assert qo["question"] == qh, "{} vs {}".format(qo, qh)

            sample["id"] = game.dialogue_id
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": o.segment,
                                  } for o in game.objects]

            sample["object_id"] = game.object.id
            sample["guess_object_id"] = game.object.id
            sample["status"] = game.status

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')


def dump_dataset(games, save_path, tokenizer, name="model"):

    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:

        for _, game in enumerate(games):

            sample = {}

            qas = []
            start = 1
            for k, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                        word == tokenizer.no_token or \
                        word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:k - 1])
                    a = tokenizer.decode([dialogue[k]])

                    prob_obj = list(prob_objects[len(qas), :len(game.objects)])
                    prob_obj = [str(round(p, 3)) for p in prob_obj]  # decimal are not supported y default in json encoder

                    qas.append({"question": q,
                                "answer": a[1:-1],
                                "id": k,
                                "p": prob_obj})

                    start = k + 1

            sample["id"] = game.dialogue_id if true_id else 0
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": [],  # no segment to avoid making the file to big
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["guess_object_id"] = guess_object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')