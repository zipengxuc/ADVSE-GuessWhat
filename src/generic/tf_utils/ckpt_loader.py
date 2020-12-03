import os
import tensorflow as tf
import collections
import logging
import argparse
import json


class ExperienceManager(object):

    status_filename = "status.json"
    params_filename = "params.ckpt"

    def __init__(self, xp_id, xp_dir, args, config, user_data=None):

        self.id = xp_id

        self.dir_xp = xp_dir
        self.dir_best_ckpt = os.path.join(xp_dir, "best")
        self.dir_last_ckpt = os.path.join(xp_dir, "last")

        if user_data is None:
            user_data = dict()
        assert isinstance(user_data, dict)

        self.score_tracking = False

        self.data = dict(
            hash_id=xp_id,
            config=config,
            args=args.__dict__,
            score_name=None,
            max_is_best=None,
            epoch=0,
            best_epoch=0,
            best_score=None,
            extra_losses=collections.defaultdict(list),
            user_data=user_data)

        self.stop_epoch = 0

    def configure_score_tracking(self, tracking_score, max_is_best):

        if max_is_best:
            init_score = float("-inf")
        else:
            init_score = float("inf")

        self.data["max_is_best"] = max_is_best
        self.data["score_name"] = tracking_score
        self.data["best_score"] = init_score

        self.score_tracking = True

    @staticmethod
    def load_from_xp_id(xp_dir):

        xp_id = os.path.basename(xp_dir)
        xp_manager = ExperienceManager(xp_id, xp_dir,
                                       args=argparse.Namespace(),  # dummy
                                       config=dict())  # dummy

        status_path = os.path.join(xp_dir, ExperienceManager.status_filename)
        with open(status_path, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
            xp_manager.data = data

        return xp_manager

    def load_checkpoint(self, sess, saver, load_best=False):

        logger = logging.getLogger()

        # Retrieve ckpt path
        if load_best:
            dir_ckpt = self.dir_best_ckpt
        else:
            dir_ckpt = self.dir_last_ckpt

        if os.path.exists(os.path.join(dir_ckpt, 'checkpoint')):

            # Load xp ckpt
            ckpt_path = os.path.join(dir_ckpt, self.params_filename)
            saver.restore(sess, ckpt_path)

            # Load xp state
            status_path = os.path.join(self.dir_xp, self.status_filename)
            with open(status_path, 'rb') as f:
                self.data = json.loads(f.read().decode('utf-8'))

            logger.info("Best previous {} : {}".format(self.data["score_name"], self.data["best_score"]))

        else:
            logger.warning("Checkpoint could not be found in directory: '{}'.".format(dir_ckpt))

        return self.data["epoch"]

    def _save(self, sess, saver, dir_ckpt):

        # Create directory
        if not os.path.isdir(dir_ckpt):
            os.makedirs(dir_ckpt)

        # Save checkpoint
        saver.save(sess, os.path.join(dir_ckpt, 'params.ckpt'))

        logger = logging.getLogger()
        logger.info("checkpoint saved... Directory: {}".format(dir_ckpt))

    def save_checkpoint(self, sess, saver, epoch, losses):

        assert self.score_tracking, "Score tracking is not configured!"

        # retrieve current score
        assert self.data["score_name"] in losses, "Missing tracking score {} in losses. Got {}".format(self.data["score_name"], losses.keys())
        running_score = losses[self.data["score_name"]]

        # update data
        self.data["epoch"] = epoch

        for key, value in losses.items():
            self.data["extra_losses"][key].append(value)

        # save best checkpoint
        max_is_best = self.data["max_is_best"]
        if (max_is_best and running_score > self.data["best_score"]) or \
                (not max_is_best and running_score < self.data["best_score"]):
            self.data["best_epoch"] = epoch
            self.data["best_score"] = running_score

            self._save(sess, saver, self.dir_best_ckpt)
            self.stop_epoch = 0
        else:
            self.stop_epoch = self.stop_epoch + 1

        # save current checkpoint
        self._save(sess, saver, self.dir_last_ckpt)

        # Save status
        status_path = os.path.join(self.dir_xp, self.status_filename)
        with open(status_path, 'w') as f_out:
            f_out.write(json.dumps(self.data, allow_nan=True))

        return self.stop_epoch

    def update_user_data(self, user_data):

        status_path = os.path.join(self.dir_xp, self.status_filename)

        self.data["user_data"] = {**self.data["user_data"], **user_data}

        with open(status_path, 'w') as f_out:
            f_out.write(json.dumps(self.data, allow_nan=True))


def create_resnet_saver(networks):

    if not isinstance(networks, list):
        networks = [networks]

    resnet_vars = dict()
    for network in networks:

        start = len(network.scope_name) + 1
        for v in network.get_resnet_parameters():
            resnet_vars[v.name[start:-2]] = v
    return tf.train.Saver(resnet_vars)




