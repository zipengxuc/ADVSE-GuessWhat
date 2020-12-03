import os
import argparse
import time
from jinja2 import Template
from shutil import copyfile
import numpy as np

from generic.utils.file_handlers import pickle_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-data_dir", type=str, help="Directory with the all the logs")
    parser.add_argument("-out_dir", type=str, help="Directory where the html/config will be dumped")
    parser.add_argument("-metric", type=str, default="error", help='Define your metric (error/accuracy/loss)')
    parser.add_argument("-round", type=int, default=3, help='Round after X decimal')
    parser.add_argument("-only_finished", type=bool, default=True, help='Only report logs with test scores')
    parser.add_argument("-max_hour", type=int, help='Only report logs with test scores')

    args = parser.parse_args()


    directory_path=args.data_dir #"/home/sequel/fstrub/guesswhat_film/out/oracle"
    output_dir = args.out_dir # "/home/sequel/fstrub/guesswhat_film/out/oracle/html_results"
    metric = args.metric #
    round_value = args.round #

    html_file = os.path.join(output_dir, "scores_{}.html".format(metric))
    config_dir = os.path.join(output_dir, "config")

    log_directories = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]

    no_res = "n/a"
    now = time.time()

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    experiences = []

    for log_directory in log_directories:

        # retrieve log file
        hash_id = os.path.basename(log_directory)

        # Bufferize log file
        pkl_file = os.path.join(directory_path, log_directory, "status.pkl")
        try:
            data = pickle_loader(pkl_file)
            config = data["config"]

            if args.max_hour is not None and\
                 os.stat(pkl_file).st_mtime > now - args.max_hour * 3600:
                print("Ignore file : {} - status file is too old".format(pkl_file))
                continue

        except FileNotFoundError:
            print("Ignore file : {} - status file not found".format(pkl_file))
            continue

        print("Parsing status.pkl in directory {}...".format(log_directory))

        # copy config file
        copyfile(src=os.path.join(directory_path,log_directory, "config.config.baseline.json"), dst=os.path.join(config_dir, "{}.json".format(hash_id)))

        # retrieve info
        best_accuracy_idx = np.argmax(data[metric]["valid"])
        train_score, valid_score, test_score = data[metric]["train"][best_accuracy_idx], \
                                               data[metric]["valid"][best_accuracy_idx], \
                                               data[metric]["test"]

        # store basic info
        xp = dict()
        xp["hash_id"] = data["hash_id"]
        xp["name"] = config["name"]
        xp["epoch"] = best_accuracy_idx+1
        xp["train"] = train_score
        xp["valid"] = valid_score
        xp["test"] = test_score

    if args.only_finished:
        experiences = [xp for xp in experiences if xp["test"] != no_res]

    with open("results.template.html", "r") as out_file:
        template_str = out_file.read()
        template = Template(template_str)

    with open(html_file, "w") as out_file:
        out_file.write(template.render(metric=metric, experiences=experiences))

