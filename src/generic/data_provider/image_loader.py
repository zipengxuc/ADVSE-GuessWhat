import os
from PIL import Image
import numpy as np
import h5py
import platform

from generic.data_provider.image_preprocessors import resize_image, scaled_crop_and_pad
from generic.data_provider import util
from tqdm import tqdm


# Why doing an image builder/loader?
# Well, there are two reasons:
#  - first you want to abstract the kind of image you are using (raw/conv/feature) when you are loading the dataset/batch.
#  One may just want... to load an image!
#  - One must optimize when to load the image for multiprocessing.
#       You do not want to serialize a 2Go of fc8 features when you create a process
#       You do not want to load 50Go of images at start
#
#   The Builder enables to abstract the kind of image you want to load. It will be used while loading the dataset.
#   The Loader enables to load/process the image when you need it. It will be used when you create the batch
#
#   Enjoy design patterns, it may **this** page of code complex but the it makes the whole project easier! Act Local, Think Global :P
#


class AbstractImgBuilder(object):
    def __init__(self, img_dir, is_raw, require_process=False):
        self.img_dir = img_dir
        self.is_raw = is_raw
        self.require_process = require_process

    def build(self, image_id, filename, **kwargs):
        return self

    def is_raw_image(self):
        return self.is_raw

    def require_multiprocess(self):
        return self.require_process


class AbstractImgLoader(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.buffer = None

    def get_image(self, **kwargs):
        if self.buffer is None:
            return self._get_image(**kwargs)
        else:
            return self.buffer

    def _get_image(self, **kwargs):
        pass

    def bufferize(self, **kwargs):
        if self.buffer is None:
            self.buffer = self.get_image(**kwargs)

    def flush(self):
        self.buffer = None


class DummyImgBuilder(AbstractImgBuilder, AbstractImgLoader):
    def __init__(self, img_dir, size=1000):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
        self.size = size

    def build(self, image_id, filename, **kwargs):
        return self

    def _get_image(self, **kwargs):
        return np.zeros(self.size)


class ErrorImgLoader(AbstractImgLoader):
    def __init__(self, img_path):
        AbstractImgLoader.__init__(self, img_path)

    def get_image(self, **kwargs):
        assert False, "The image/crop is not available in file: {}".format(self.img_path)


h5_basename = "features.h5"
h5_feature_key = "feature"  # feature
h5_idx_key = "index"  # h5_idx_key = "idx2img"


class h5FeatureBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, bufferize, scale):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
        self.bufferize = bufferize
        self.h5files = dict()
        self.img2idx = dict()
        self.scale = scale

    def build(self, image_id, filename, optional=True, which_set=None, **kwargs):

        # Is the h5 features split into train/val/etc. files or gather into a single file
        if which_set is not None:
            h5filename = which_set + "_" + h5_basename
        else:
            h5filename = h5_basename

        # Build full bath
        # h5filepath = os.path.join(self.img_dir, h5filename)
        h5filepath = self.img_dir

        # Retrieve
        if h5filename not in self.h5files:
            # Load file pointer to h5
            h5file = h5py.File(h5filepath, 'r')

            # hd5 requires continuous id while image_id can be very diverse.
            # We then need a mapping between both of them
            if h5_idx_key in h5file:
                # Retrieve id mapping from file
                img2idx = {id_img: id_h5 for id_h5, id_img in enumerate(h5file[h5_idx_key])}
            else:
                # Assume their is a perfect identity between image_id and h5_id
                no_images = h5file[h5_feature_key].shape[0]
                img2idx = {k: k for k in range(no_images)}

            self.h5files[h5filename] = h5file
            self.img2idx[h5filename] = img2idx
        else:
            h5file = self.h5files[h5filename]
            img2idx = self.img2idx[h5filename]

        if optional and image_id in img2idx or (not optional):
            loader = h5FeatureLoader(h5filepath, h5file=h5file, id=img2idx[image_id])
            if self.bufferize:
                loader.bufferize()
            return loader
        else:
            return None


class h5FeatureBuilder_rcnn(AbstractImgBuilder):
    def __init__(self, img_dir, bufferize, scale):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
        self.bufferize = bufferize
        self.h5files = dict()
        self.img2idx = dict()
        self.scale = scale

    def build(self, image_id, filename, optional=True, which_set=None, **kwargs):

        # Is the h5 features split into train/val/etc. files or gather into a single file
        if platform.node() == "cist-PowerEdge-R730":
            h5filepath = '/home/xuzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.hy'
            txtfilepath = '/home/xuzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.txt'
        elif platform.node() == "dell-PowerEdge-T630" or platform.node() == "DELL":
            h5filepath = '/home/xzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.hy'
            txtfilepath = '/home/xzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.txt'
        else:
            h5filepath = './data/features/rcnn/size,rcnn_arch,224.hy'
            txtfilepath = './data/features/rcnn/size,rcnn_arch,224.txt'

        h5filename = h5_basename

        # Retrieve
        if h5filename not in self.h5files:
            # Load file pointer to h5
            with open(txtfilepath, encoding='utf-8') as f:
                txt_data = f.read().split('\n')[:-1]
            h5file = h5py.File(h5filepath, 'r')
            # image_file = h5file["feature"]
            idx_to_name = {i: e.split("/")[6] for i, e in enumerate(txt_data)}
            img2idx = {v: k for k, v in idx_to_name.items()}

            self.h5files[h5filename] = h5file
            self.img2idx[h5filename] = img2idx
        else:
            h5file = self.h5files[h5filename]
            img2idx = self.img2idx[h5filename]

        if optional and image_id in img2idx or (not optional):
            # print(image_id)
            # print(img2idx.keys())
            loader = h5FeatureLoader_rcnn(h5filepath, h5file=h5file, id=img2idx[filename])
            if self.bufferize:
                loader.bufferize()
            return loader
        else:
            return None


'''as in the pytorch version'''


def _create_image_builder_rcnn(use_redis=False):

    if platform.node() == "cist-PowerEdge-R730":
        h5filepath = '/home/xuzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.hy'
        txtfilepath = '/home/xuzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.txt'
    elif platform.node() == "dell-PowerEdge-T630" or platform.node() == "DELL":
        h5filepath = '/home/xzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.hy'
        txtfilepath = '/home/xzp/guesswhat_v2/data/features/rcnn/size,rcnn_arch,224.txt'
    else:
        h5filepath = './data/features/rcnn/size,rcnn_arch,224.hy'
        txtfilepath = './data/features/rcnn/size,rcnn_arch,224.txt'
    image_file = h5py.File(h5filepath, "r")['att']
    with open(txtfilepath, encoding='utf-8') as f:
        txt_data = f.read().split('\n')[:-1]
    idx_to_name = {i: e.split("/")[6] for i, e in enumerate(txt_data)}
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    boxes_info = None
    image_shape = (36, 2048)
    if use_redis:
        redis = util.Redis('rcnn_image_features', way='str', shape=image_shape)
        if not redis.db.keys(pattern=redis.prefix + "*"):  # 判断是否存在该前缀的key
            print('Start loading features into memory.')
            N = image_file.shape[0]
            for i in tqdm(range(N)):
                redis[i] = image_file[i]
        else:
            print('Already loaded features into memory.')
        img_builder = Redis_image_builder_rcnn(redis, boxes_info, name_to_idx)
        print("finish create redis image builder")
    else:
        img_builder = H5py_image_builder_rcnn(image_file, boxes_info, name_to_idx)
        print("finish create h5py image builder")

    return img_builder


class H5py_image_builder_rcnn():
    def __init__(self, hfile, boxes, name2idx, require_process=False):
        self.hfile = hfile
        self.boxes = boxes
        self.name2idx = name2idx
        self.require_process = require_process

    def load(self, file_name):
        idx = self.name2idx[file_name]
        image = self.hfile[idx]
        # boxes = self.boxes[idx]["boxes"]
        return image

    def require_multiprocess(self):
        return self.require_process


class Redis_image_builder_rcnn():
    def __init__(self, redis, boxes, name2idx, require_process=False):
        self.redis = redis
        self.boxes = boxes
        self.name2idx = name2idx
        self.require_process = require_process
        # To be revised

    def load(self, filename):
        key = self.name2idx[filename]
        image = self.redis.get(key)
        return image

    def require_multiprocess(self):
        return self.require_process


def _create_crop_builder(crop_file):
    # type means "image" or "crop"
    # 分为存在hdf5和不存在的情况
    # try:
    # have extract h5py file ready
    image_f = h5py.File(crop_file, "r")
    image_file = image_f["feature"]
    image_key2id = image_f["index"]
    image_id2key = {v: k for k, v in enumerate(image_key2id)}

    # from h5py file get feature
    crop_builder = H5py_crop_builder(image_file, image_id2key)
    print("finish create h5py crop builder")
    # except:
    #     opt = {"pooling": config["crop_pooling"], "arch": config["crop_arch"],
    #            "feature": config["crop_feature"]}
    #     # no h5py file
    #     crop_builder = Raw_crop_builder(img_dir, transform, opt, is_cuda)
    #     print("finish create raw crop builder")

    return crop_builder


class H5py_crop_builder():
    def __init__(self, hfile, id2key):
        self.hfile = hfile
        self.id2key = id2key

    def load(self, object):
        id = object
        key = self.id2key[id]
        image = self.hfile[key]
        return image


# Load while creating batch
class h5FeatureLoader(AbstractImgLoader):
    def __init__(self, img_path, h5file, id):
        AbstractImgLoader.__init__(self, img_path)
        self.h5file = h5file
        self.id = id

    def _get_image(self, **kwargs):
        return self.h5file[h5_feature_key][self.id]

    # Make DeepCopy <=> shallow copy (as h5py file handler do not support deepcopy)
    def __deepcopy__(self, memo):
        return h5FeatureLoader(self.img_path, h5file=self.h5file, id=self.id)


class h5FeatureLoader_rcnn(AbstractImgLoader):
    def __init__(self, img_path, h5file, id):
        AbstractImgLoader.__init__(self, img_path)
        self.h5file = h5file
        self.id = id

    def _get_image(self, **kwargs):
        return self.h5file["att"][self.id]

    # Make DeepCopy <=> shallow copy (as h5py file handler do not support deepcopy)
    def __deepcopy__(self, memo):
        return h5FeatureLoader(self.img_path, h5file=self.h5file, id=self.id)


class RawImageBuilder(AbstractImgBuilder):
    def __init__(self, img_dir, width, height, channel=None):
        AbstractImgBuilder.__init__(self, img_dir, is_raw=True, require_process=True)
        self.width = width
        self.height = height
        self.channel = channel

    def build(self, image_id, filename, **kwargs):
        img_path = os.path.join(self.img_dir, filename)
        return RawImageLoader(img_path, self.width, self.height, channel=self.channel)


class RawImageLoader(AbstractImgLoader):
    def __init__(self, img_path, width, height, channel):
        AbstractImgLoader.__init__(self, img_path)
        self.width = width
        self.height = height
        self.channel = channel

    def _get_image(self, **kwargs):
        img = Image.open(self.img_path).convert('RGB')

        img = resize_image(img, self.width, self.height)
        img = np.array(img, dtype=np.float32)

        if self.channel is not None:
            img -= self.channel[None, None, :]

        return img


class RawCropBuilder(AbstractImgBuilder):
    def __init__(self, data_dir, width, height, scale, channel=None):
        AbstractImgBuilder.__init__(self, data_dir, is_raw=True, require_process=True)
        self.width = width
        self.height = height
        self.channel = channel
        self.scale = scale

    def build(self, object_id, filename, **kwargs):
        bbox = kwargs["bbox"]
        img_path = os.path.join(self.img_dir, filename)
        return RawCropLoader(img_path, self.width, self.height, scale=self.scale, bbox=bbox, channel=self.channel)


class RawCropLoader(AbstractImgLoader):
    def __init__(self, img_path, width, height, scale, bbox, channel):
        AbstractImgLoader.__init__(self, img_path)
        self.width = width
        self.height = height
        self.channel = channel
        self.bbox = bbox
        self.scale = scale

    def _get_image(self, **kwargs):
        img = Image.open(self.img_path).convert('RGB')

        crop = scaled_crop_and_pad(raw_img=img, bbox=self.bbox, scale=self.scale)
        crop = resize_image(crop, self.width, self.height)
        crop = np.array(crop, dtype=np.float32)

        # should it be before/after the padding?
        if self.channel is not None:
            img -= self.channel[None, None, :]

        return crop


def get_img_builder(config, image_dir, is_crop=False, bufferize=None):
    image_input = config["image_input"]

    scale = None
    if is_crop:
        scale = config["scale"]

    if image_input in ["fc8", "fc7"]:
        bufferize = bufferize if bufferize is not None else True
        loader = h5FeatureBuilder(image_dir, bufferize=bufferize, scale=scale)
        print("finish create vgg h5py image builder")

    elif image_input in ["rcnn"]:
        bufferize = bufferize if bufferize is not None else True
        loader = h5FeatureBuilder_rcnn(image_dir, bufferize=bufferize, scale=scale)
        print("finish create rcnn h5py image builder")

    elif image_input in ["conv", "raw_h5"]:
        bufferize = bufferize if bufferize is not None else False
        loader = h5FeatureBuilder(image_dir, bufferize=bufferize, scale=scale)

    elif image_input == "raw":
        if is_crop:
            loader = RawCropBuilder(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    scale=config["scale"],
                                    channel=config.get("channel", None))
        else:
            loader = RawImageBuilder(image_dir,
                                     height=config["dim"][0],
                                     width=config["dim"][1],
                                     channel=config.get("channel", None))
    else:
        assert False, "incorrect image input: {}".format(image_input)

    return loader

# Legacy code
# ---------------------------------------------
#
# class ConvBuilder(AbstractImgBuilder):
#     def __init__(self, img_dir):
#         AbstractImgBuilder.__init__(self, img_dir, is_raw=False)
#
#     def build(self, image_id, filename, **kwargs):
#         img_path = os.path.join(self.img_dir, "{}.npz".format(image_id))
#         return ConvLoader(img_path)
#
# class ConvLoader(AbstractImgLoader):
#     def __init__(self, img_path):
#         AbstractImgLoader.__init__(self, img_path)
#
#     def get_image(self, **kwargs):
#         return np.load(self.img_path)['x']
# class fcBuilder(AbstractImgBuilder):
#     def __init__(self, img_path):
#         AbstractImgBuilder.__init__(self, img_path, is_raw=False)
#         self.fc8 = pickle_loader(img_path)
#
#     # Trick to avoid serializing complete fc8 dictionary.
#     # We wrap the fc8 into a separate object which does not contain
#     # the *full* fc8 dictionary.
#
#     def build(self, image_id, filename, **kwargs):
#         """
#         :param image_id: (or object_id) id of the fc8 to load
#         :param filename: N/A
#         :param kwargs: optional -> do fc8 must be present in file
#         :return: fcLoader for the current id
#         """
#
#         one_fc8 = self.fc8.get(image_id, None)
#
#         assert one_fc8 is not None or kwargs.get("optional", True), \
#             "fc8 (id:{}) is missing in file: {}".format(image_id, self.img_dir)
#
#         return fcLoader(self.img_dir, one_fc8)
#
# class fcLoader(AbstractImgLoader):
#     def __init__(self,data_dir, fc8):
#         AbstractImgLoader.__init__(self, data_dir)
#         self.fc8 = fc8
#
#     def get_image(self, **kwargs):
#         assert self.fc8 is not None
#         return self.fc8
