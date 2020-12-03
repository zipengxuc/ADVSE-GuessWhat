import os
import re
import argparse
from jinja2 import Template
from shutil import copyfile


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-data_dir", type=str, help="Directory with the all the logs")
    parser.add_argument("-out_dir", type=str, help="Directory where the html/config will be dumped")
    parser.add_argument("-metric", type=str, default="error", help='Define your metric (error/accuracy/loss)')
    parser.add_argument("-round", type=int, default=3, help='Round after X decimal')
    parser.add_argument("-only_finished", type=bool, default=True, help='Only report logs with test scores')


    args = parser.parse_args()


    directory_path=args.data_dir #"/home/sequel/fstrub/guesswhat_film/out/oracle"
    output_dir = args.out_dir # "/home/sequel/fstrub/guesswhat_film/out/oracle/html_results"
    metric = args.metric #
    round_value = args.round #

    html_file = os.path.join(output_dir, "scores_{}.html".format(metric))
    config_dir = os.path.join(output_dir, "config")

    log_directories = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]



    locate_ckpt_save_regex = re.compile(r".* checkpoint saved.*$")
    locate_ckpt_load_regex = re.compile(r".* Restoring parameters from.*$")
    locate_epoch_regex = re.compile(r".* Epoch (\d+).*$")
    locate_name_regex = re.compile(r".* Config name : (.*)$")

    locate_train_str = r".* :: Training {} *: ([-+]?\d*\.\d+|\d+)"
    locate_valid_str = r".* :: Validation {} *: ([-+]?\d*\.\d+|\d+)"
    locate_test_str = r".* :: Testing {} *: ([-+]?\d*\.\d+|\d+)"

    no_res = "n/a"



    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    experiences = []

    for log_directory in log_directories:

        # retrieve log file
        hash_id = os.path.basename(log_directory)




        # Bufferize log file
        log_file = os.path.join(directory_path,log_directory, "train.log")
        try:
            with open(log_file, "r") as log:
                lines = log.readlines()
        except FileNotFoundError:
            print("Ignore directory : {} - Log file not found".format(log_directory))
            continue

        print("Parsing log in directory {}...".format(log_directory))

        # copy config file
        copyfile(src=os.path.join(directory_path,log_directory, "config.config.baseline.json"), dst=os.path.join(config_dir, "{}.json".format(hash_id)))

        # Look for scores in log file
        ckpt_save_idx = None
        ckpt_load_idx = None
        best_epoch_idx = None
        best_epoch = None
        config_name = ""

        xp = {}
        current_epoch_idx, current_epoch = 0, 0
        for i, line in enumerate(lines):

            # Store current epoch
            if locate_epoch_regex.search(line):
                current_epoch_idx = i
                current_epoch =  int(locate_epoch_regex.match(line).group(1))

            # Store current best ckpt/epoch
            if locate_ckpt_save_regex.search(line):
                ckpt_save_idx = i
                best_epoch_idx = current_epoch_idx
                best_epoch = current_epoch

            # Store when ckpt is loaded
            if locate_ckpt_load_regex.search(line):
                ckpt_load_idx = i

            # Store config name
            if locate_name_regex.search(line):
                config_name = locate_name_regex.match(line).group(1)


        # store basic info
        xp["hash_id"] = hash_id
        xp["name"] = config_name
        xp["epoch"] = best_epoch
        xp["train"] = no_res
        xp["valid"] = no_res
        xp["test"] = no_res

        # store scores
        if ckpt_save_idx is not None:
            buffer = lines[best_epoch_idx:ckpt_save_idx]

            locate_train_regex = re.compile(locate_train_str.format(metric))
            locate_valid_regex = re.compile(locate_valid_str.format(metric))

            for line in buffer:

                if locate_train_regex.search(line):
                    xp["train"] = round(float(locate_train_regex.match(line).group(1)),round_value)

                if locate_valid_regex.search(line):
                    xp["valid"] = round(float(locate_valid_regex.match(line).group(1)),round_value)

        if ckpt_load_idx is not None \
                and ckpt_save_idx < ckpt_load_idx : # Check that best validation score is not posterior to the test score

            buffer = lines[ckpt_load_idx:ckpt_load_idx + 10] #  magic number...

            locate_test_regex = re.compile(locate_test_str.format(metric))

            for line in buffer:

                if locate_test_regex.search(line):
                    xp["test"] = round(float(locate_test_regex.match(line).group(1)),round_value)

        experiences.append(xp)

    if args.only_finished:
        experiences = [xp for xp in experiences if xp["test"] != no_res]


    with open("results.template.html", "r") as out_file:
        template_str = out_file.read()
        template = Template(template_str)


    with open(html_file, "w") as out_file:
        out_file.write(template.render(metric=metric, experiences=experiences))

