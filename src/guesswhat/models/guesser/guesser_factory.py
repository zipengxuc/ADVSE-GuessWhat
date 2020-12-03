from guesswhat.models.guesser.guesser_RAH import GuesserNetwork_RAH
from guesswhat.data_provider.guesser_qa_batchifier import GuesserBatchifier_RAH
from guesswhat.train.eval_listener import CropAccuracyListener, AccuracyListener


# factory class to create networks and the related batchifier

def create_guesser(config, num_words, reuse=False):

    network_type = config["type"]

    if network_type == "RAH":
        network = GuesserNetwork_RAH(config, num_words=num_words, reuse=reuse)
        batchifier = GuesserBatchifier_RAH
        listener = AccuracyListener(require=network.softmax)

    else:
        assert False, "Invalid network_type: should be: baseline/oracle"

    return network, batchifier, listener

