from guesswhat.models.oracle.oracle_v1 import OracleNetwork_v1
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier

# stupid factory class to create networks

def create_oracle(config, num_words, reuse=False, device=''):

    network_type = config["type"]
    num_answers = 3

    if network_type == "v1":
        network = OracleNetwork_v1(config, num_words=num_words, num_answers=num_answers, device=device, reuse=reuse)
        batchifier = OracleBatchifier
    else:
        assert False, "Invalid network_type: should be: baseline/film"

    return network, batchifier


