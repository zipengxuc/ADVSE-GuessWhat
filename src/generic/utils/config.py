import os
import json

import tarfile

import shutil
import hashlib

from generic.utils.logger import create_logger

from generic.tf_utils.ckpt_loader import ExperienceManager


# ENTRY POINT
def load_config(args, user_data=None):

    # If the user provide a xp_id, load the configuration from id_xp
    if args.load_checkpoint is not None:

        xp_id = args.load_checkpoint
        xp_dir = os.path.join(args.out_dir, args.load_checkpoint)
        with open(os.path.join(xp_dir, "config.json"), 'rb') as f_config:
            config = json.loads(f_config.read().decode('utf-8'))

        xp_manager = ExperienceManager.load_from_xp_id(xp_dir=xp_dir)

    # Otherwise, load the configuration from the config flag
    else:
        with open(args.config, 'rb') as f_config:
            config = json.loads(f_config.read().decode('utf-8'))

        # xp_identifier
        xp_id = get_config_hash(config)

        # Create directory
        xp_dir = os.path.join(args.out_dir, xp_id)
        if not os.path.isdir(xp_dir):
            os.makedirs(xp_dir)

        # copy config file
        shutil.copy(args.config, os.path.join(xp_dir, 'config.json'))

        # copy compressed source
        src_path = os.getcwd().split("/src", 1)[0]
        src_path = os.path.join(src_path, 'src')
        src_out = os.path.join(xp_dir, 'src.tar.gz')
        with tarfile.open(src_out, "w:gz") as tar:
            tar.add(src_path, arcname=os.path.basename(src_path))

        xp_manager = ExperienceManager(xp_id=xp_id, xp_dir=xp_dir,
                                       config=config, args=args,
                                       user_data=user_data)

    # create logger
    logger = create_logger(os.path.join(xp_dir, 'train.log'))
    logger.info("Config Hash : {}".format(xp_id))
    logger.info("Config Name : {}".format(config["name"]))

    # display config
    logger.info(config)

    # display args
    if args is not None:
        for key, val in vars(args).items():
            logger.info("{} : {}".format(key, val))

    # set seed
    set_seed(config)

    return config, xp_manager


def get_config_hash(config):
    str_config = json.dumps(config, sort_keys=True)  # Ensure that config hash are consistent (ordered keys)
    hash = hashlib.md5(json.dumps(str_config, sort_keys=True).encode('utf-8')).hexdigest()
    return hash


def get_config_from_xp(exp_dir, identifier):
    config_path = os.path.join(exp_dir, identifier, 'config.json')
    if not os.path.exists(config_path):
        raise RuntimeError("Couldn't find config")

    with open(config_path, 'r') as f:
        return json.load(f)


def get_recursively(search_dict, field, no_field_recursive=False):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:

            if no_field_recursive \
                    and (isinstance(value, dict) or isinstance(key, list)):
                continue

            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field, no_field_recursive)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field, no_field_recursive)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found


def set_seed(config):
    import numpy as np
    import tensorflow as tf
    seed = config["seed"]
    if seed > -1:
        np.random.seed(seed)
        tf.set_random_seed(seed)
