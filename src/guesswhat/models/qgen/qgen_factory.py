from guesswhat.models.qgen.qgen_uaqrah_network import QGenNetworkHREDDecoderUAQRAH
from guesswhat.data_provider.qgen_hred_batchifier import HREDBatchifier


def create_qgen(config, num_words, rl_module=None):

    network_type = config["type"]

    if network_type == "UAQRAH":
        network = QGenNetworkHREDDecoderUAQRAH(config, num_words, rl_module=rl_module)
        batchifier = HREDBatchifier
    else:
        assert False, "Invalid network_type: should be: baseline/oracle"

    return network, batchifier

