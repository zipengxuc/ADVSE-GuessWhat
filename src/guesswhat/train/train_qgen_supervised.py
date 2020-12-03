import os
import argparse
import logging

from distutils.util import strtobool

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.utils.config import load_config, get_config_from_xp
from generic.data_provider.image_loader import get_img_builder, _create_image_builder_rcnn
from generic.data_provider.nlp_utils import GloveEmbeddings
from generic.utils.thread_pool import create_cpu_pool
from guesswhat.train.eval_listener import QGenListener

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.qgen.qgen_factory import create_qgen
from guesswhat.models.guesser.guesser_factory import create_guesser
from guesswhat.models.oracle.oracle_factory import create_oracle

from guesswhat.data_provider.looper_batchifier import LooperBatchifier
from guesswhat.models.looper.basic_looper import BasicLooper

from guesswhat.models.qgen.qgen_wrapper import QGenWrapper
from guesswhat.models.oracle.oracle_wrapper import OracleWrapper
from guesswhat.models.guesser.guesser_wrapper import GuesserWrapper

from guesswhat.train.utils import compute_qgen_accuracy


if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('QGen network baseline!')

    parser.add_argument("-data_dir", type=str, default="data", help="Directory with data")
    parser.add_argument("-out_dir", default="out/qgen", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, default="config/qgen/config.rnn.json", help='Config file')
    parser.add_argument("-dict_file", type=str, default="data/dict.json", help="Dictionary file name")
    parser.add_argument("-glove_file", type=str, default="glove_dict.pkl", help="Glove file name")
    parser.add_argument("-img_dir", type=str, default='data/features/vgg16/image.hdf5', help='Directory with images')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=4, help="No thread to load batch")
    parser.add_argument("-train_epoch", type=int, default=40)
    parser.add_argument("-early_stop", type=int, default=5)
    parser.add_argument("-no_games_to_load", type=int, default=float("inf"), help="No games to use during training Default : all")
    parser.add_argument("-skip_training",  type=lambda x: bool(strtobool(x)), default="False", help="Start from checkpoint?")
    parser.add_argument("-load_new",  type=lambda x: bool(strtobool(x)), default="True", help="Start from checkpoint?")

    args = parser.parse_args()

    config, xp_manager = load_config(args)
    logger = logging.getLogger()

    # Load config
    batch_size = config['optimizer']['batch_size']
    if args.load_new and config['model']['image']['image_input'] == "rcnn":
        rcnn = True
        print("rcnn!")
    else:
        rcnn = False
    no_epoch = args.train_epoch

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    image_builder, crop_builder = None, None
    use_resnet, use_multiproc = False, False
    if rcnn:
        image_builder = _create_image_builder_rcnn()
    elif config["model"]['inputs'].get('image', False):
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_builder, crop_builder, rcnn, args.no_games_to_load)
    validset = Dataset(args.data_dir, "valid", image_builder, crop_builder, rcnn, args.no_games_to_load)
    testset = Dataset(args.data_dir, "test", image_builder, crop_builder, rcnn, args.no_games_to_load)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(args.dict_file)

    # Build Network
    logger.info('Building network..')
    network, batchifier_cstor = create_qgen(config["model"], num_words=tokenizer.no_words)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config["optimizer"])  # output:[loss, accuracy]

    ###############################
    #  START  TRAINING
    #############################

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # CPU/GPU option
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    with tf.Session(config=config_gpu) as sess:

        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))

        sess.run(tf.global_variables_initializer())
        if args.continue_exp or args.load_checkpoint is not None:
            start_epoch = xp_manager.load_checkpoint(sess, saver)
        else:
            start_epoch = 0

        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = batchifier_cstor(tokenizer, sources, status=('success',), supervised=True)
        xp_manager.configure_score_tracking("valid_loss", max_is_best=False)

        idx, _, _, _ = network.create_greedy_graph(start_token=tokenizer.start_token, stop_token=tokenizer.stop_token, max_tokens=10)
        listener = QGenListener(require=idx)

        for t in range(start_epoch, no_epoch):
            if args.skip_training:
                logger.info("Skip training...")
                break
            logger.info('Epoch {}..'.format(t + 1))

            # Create cpu pools (at each iteration otherwise threads may become zombie - python bug)
            cpu_pool = create_cpu_pool(args.no_thread, use_process=image_builder.require_multiprocess())

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            [train_loss, _] = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False)
            [valid_loss] = evaluator.process(sess, valid_iterator, outputs=outputs, listener=listener)

            for qt in listener.get_questions()[:5]:
                logger.info(tokenizer.decode(qt))

            logger.info("Training loss   : {}".format(train_loss))
            logger.info("Validation loss : {}".format(valid_loss))

            stop_flag = xp_manager.save_checkpoint(sess, saver, epoch=t,
                                                   losses=dict(train_loss=train_loss, valid_loss=valid_loss))
            if stop_flag >= args.early_stop:
                logger.info("==================early stopping===================")
                break

        # Load early stopping
        logger.info("==================test===================")
        xp_manager.load_checkpoint(sess, saver, load_best=True)
        cpu_pool = create_cpu_pool(args.no_thread, use_process=use_multiproc)

        # Create Listener
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size,
                                 batchifier=batchifier,
                                 shuffle=False)
        [test_loss, _] = evaluator.process(sess, test_iterator, outputs=outputs)

        logger.info("Testing loss: {}".format(test_loss))

        # Save the test scores
        xp_manager.update_user_data(
            user_data={
                "test_loss": test_loss,
            }
        )

    tf.reset_default_graph()
    with tf.Session(config=config_gpu) as sess_loop:
        # compute the loop accuracy
        logger.info("==================loop===================")
        mode_to_evaluate = ["sampling", "greedy", "beam"]
        cpu_pool = create_cpu_pool(args.no_thread, use_process=False)

        train_batchifier = LooperBatchifier(tokenizer, generate_new_games=True)
        eval_batchifier = LooperBatchifier(tokenizer, generate_new_games=False)
        oracle_dir = "out/oracle/"
        oracle_checkpoint = "6e0ec7f150b27f46296406853f498af6"
        guesser_dir = "out/guesser/"
        guesser_checkpoint = "c48036b430ebca1c44a25188edb05034"
        oracle_config = get_config_from_xp(oracle_dir, oracle_checkpoint)
        guesser_config = get_config_from_xp(guesser_dir, guesser_checkpoint)

        qgen_network, qgen_batchifier_cstor = create_qgen(config["model"], num_words=tokenizer.no_words)
        qgen_var = [v for v in tf.global_variables() if "qgen" in v.name]  # and 'rl_baseline' not in v.name
        for v in qgen_var:
            print(v.name)
        qgen_saver = tf.train.Saver(var_list=qgen_var)

        oracle_network, oracle_batchifier_cstor = create_oracle(oracle_config["model"], num_words=tokenizer.no_words-1)
        oracle_var = [v for v in tf.global_variables() if "oracle" in v.name]
        oracle_saver = tf.train.Saver(var_list=oracle_var)

        guesser_network, guesser_batchifier_cstor, guesser_listener = create_guesser(guesser_config["model"],
                                                                                     num_words=tokenizer.no_words-1)
        guesser_var = [v for v in tf.global_variables() if "guesser" in v.name]
        guesser_saver = tf.train.Saver(var_list=guesser_var)

        oracle_saver.restore(sess_loop, os.path.join(oracle_dir, oracle_checkpoint, 'best', 'params.ckpt'))
        guesser_saver.restore(sess_loop, os.path.join(guesser_dir, guesser_checkpoint, 'best', 'params.ckpt'))
        qgen_saver.restore(sess_loop, os.path.join(xp_manager.dir_best_ckpt, 'params.ckpt'))

        oracle_split_mode = 1
        oracle_batchifier = oracle_batchifier_cstor(tokenizer, sources=oracle_network.get_sources(sess_loop), split_mode=oracle_split_mode)
        oracle_wrapper = OracleWrapper(oracle_network, oracle_batchifier, tokenizer)

        guesser_batchifier = guesser_batchifier_cstor(tokenizer, sources=guesser_network.get_sources(sess_loop))
        guesser_wrapper = GuesserWrapper(guesser_network, guesser_batchifier, tokenizer, guesser_listener)

        qgen_batchifier = qgen_batchifier_cstor(tokenizer, sources=qgen_network.get_sources(sess_loop), generate=True)
        qgen_wrapper = QGenWrapper(qgen_network, qgen_batchifier, tokenizer,
                                   max_length=12,
                                   k_best=20)

        xp_manager.configure_score_tracking("valid_accuracy", max_is_best=True)

        loop_config = {}  # fake config
        loop_config['loop'] = {}
        loop_config['loop']['max_question'] = 5
        game_engine = BasicLooper(loop_config,
                                  oracle_wrapper=oracle_wrapper,
                                  guesser_wrapper=guesser_wrapper,
                                  qgen_wrapper=qgen_wrapper,
                                  tokenizer=tokenizer,
                                  batch_size=64)

        logger.info(">>>  New Objects  <<<")
        compute_qgen_accuracy(sess_loop, trainset, batchifier=train_batchifier, looper=game_engine,
                              mode=mode_to_evaluate, cpu_pool=cpu_pool, batch_size=batch_size,
                              name="ini.new_object", save_path=xp_manager.dir_xp, store_games=True)

        logger.info(">>>  New Games  <<<")
        compute_qgen_accuracy(sess_loop, testset, batchifier=eval_batchifier, looper=game_engine,
                              mode=mode_to_evaluate, cpu_pool=cpu_pool, batch_size=batch_size,
                              name="ini.new_images", save_path=xp_manager.dir_xp, store_games=True)

