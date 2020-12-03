from tqdm import tqdm


class BasicLooper(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']

        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper

    def process(self, sess, iterator, mode, save_path=None, name=None, optimizer=None, store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        games = []

        score, total_elem = 0, 0
        for game_data in tqdm(iterator):

            ongoing_games = game_data["raw"]
            att_dict = {}
            beta_dict = {}

            # Step 1: generate question/answer
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                ongoing_games, att_dict= self.qgen.sample_next_question(sess, ongoing_games, att_dict, beta_dict, mode=mode)
                # ongoing_games, att_dict, beta_dict = self.qgen.sample_next_question(sess, ongoing_games, att_dict, beta_dict, mode=mode)

                # Step 1.2: Answer the question
                ongoing_games = self.oracle.answer_question(sess, ongoing_games)

                # Step 1.3 Check if all dialogues are finished
                if all([g.user_data["has_stop_token"] for g in ongoing_games]):
                    break

            # Step 2 : Find the object
            ongoing_games = self.guesser.find_object(sess, ongoing_games)
            # games.extend(ongoing_games)
            if store_games:
                dump_dataset(ongoing_games, att_dict,
                # dump_dataset(ongoing_games, att_dict, beta_dict,
                             save_path=save_path,
                             # tokenizer=looper.tokenizer,
                             name=name + "." + mode)

            # Step 3 : Apply gradient
            if optimizer is not None:
                self.qgen.policy_update(sess, ongoing_games, optimizer=optimizer)

            # Step 4 : Compute score
            score += sum([g.status == "success" for g in ongoing_games])

            # Free the memory used for optimization -> DO NOT REMOVE! cd bufferize in looper_batchifier
            # for game in ongoing_games:
            #     game.flush()

        score = 1.0 * score / iterator.n_examples

        return score, games


class BasicLooperC(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']

        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper

    def process(self, sess, iterator, mode, save_path=None, name=None, optimizer=None, store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        games = []

        score, total_elem = 0, 0
        for game_data in tqdm(iterator):

            ongoing_games = game_data["raw"]
            att_dict = {}
            beta_dict = {}
            dis_lists = []

            # Step 1: generate question/answer
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                ongoing_games, att_dict = self.qgen.sample_next_question(sess, ongoing_games, att_dict, beta_dict,
                                                                         mode=mode)
                # ongoing_games, att_dict, beta_dict = self.qgen.sample_next_question(sess, ongoing_games, att_dict, beta_dict, mode=mode)

                # Step 1.2: Answer the question
                ongoing_games = self.oracle.answer_question(sess, ongoing_games)

                _, distributes = self.guesser.find_object(sess, ongoing_games)
                if no_question == 0:
                    dis_lists.append([[1.0/20]*20]*len(distributes))
                dis_lists.append(distributes)

                # Step 1.3 Check if all dialogues are finished
                if all([g.user_data["has_stop_token"] for g in ongoing_games]):
                    break

            # Step 2 : Find the object
            ongoing_games, _ = self.guesser.find_object(sess, ongoing_games)
            # games.extend(ongoing_games)
            if store_games:
                dump_dataset(ongoing_games, att_dict,
                             # dump_dataset(ongoing_games, att_dict, beta_dict,
                             save_path=save_path,
                             # tokenizer=looper.tokenizer,
                             name=name + "." + mode)

            # Step 3 : Apply gradient
            if optimizer is not None:
                self.qgen.policy_update(sess, ongoing_games, dis_lists, optimizer=optimizer)

            # Step 4 : Compute score
            score += sum([g.status == "success" for g in ongoing_games])

            # Free the memory used for optimization -> DO NOT REMOVE! cd bufferize in looper_batchifier
            # for game in ongoing_games:
            #     game.flush()

        score = 1.0 * score / iterator.n_examples

        return score, games


class BasicLooperCR(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']

        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper

    def process(self, sess, iterator, mode, save_path=None, name=None, optimizer=None, store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        games = []

        score, total_elem = 0, 0
        for game_data in tqdm(iterator):

            ongoing_games = game_data["raw"]
            att_dict = {}
            beta_dict = {}
            dis_lists = []
            q_flags = []

            # Step 1: generate question/answer
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                ongoing_games, att_dict, q_flag = self.qgen.sample_next_question(sess, ongoing_games, att_dict,
                                                                                 beta_dict,
                                                                                 mode=mode)
                # ongoing_games, att_dict, beta_dict = self.qgen.sample_next_question(sess, ongoing_games, att_dict, beta_dict, mode=mode)
                q_flags.append(q_flag)
                # Step 1.2: Answer the question
                ongoing_games = self.oracle.answer_question(sess, ongoing_games)

                _, distributes = self.guesser.find_object(sess, ongoing_games)
                if no_question == 0:
                    dis_lists.append([[1.0 / 20] * 20] * len(distributes))
                dis_lists.append(distributes)

                # Step 1.3 Check if all dialogues are finished
                if all([g.user_data["has_stop_token"] for g in ongoing_games]):
                    break

            # Step 2 : Find the object
            ongoing_games, _ = self.guesser.find_object(sess, ongoing_games)
            # games.extend(ongoing_games)
            if store_games:
                dump_dataset(ongoing_games, att_dict,
                             # dump_dataset(ongoing_games, att_dict, beta_dict,
                             save_path=save_path,
                             # tokenizer=looper.tokenizer,
                             name=name + "." + mode)

            # Step 3 : Apply gradient
            if optimizer is not None:
                self.qgen.policy_update(sess, ongoing_games, dis_lists, q_flags, optimizer=optimizer)

            # Step 4 : Compute score
            score += sum([g.status == "success" for g in ongoing_games])

            # Free the memory used for optimization -> DO NOT REMOVE! cd bufferize in looper_batchifier
            # for game in ongoing_games:
            #     game.flush()

        score = 1.0 * score / iterator.n_examples

        return score, games


def dump_dataset(games, att_dict, save_path, name="model"):
# def dump_dataset(games, att_dict, beta_dict, save_path, name="model"):
    import gzip
    import os
    import json

    with open(os.path.join(save_path, 'guesswhat_att.' + name + '.json'), 'a', encoding='utf-8') as f:

        for _, game in enumerate(games):

            sample = {}

            qas = []
            for id, question, answers in zip(game.question_ids, game.questions, game.answers):
                qas.append({"question": question,
                            "answer": answers,
                            "id": id,
                            "p": 0})

            sample["id"] = game.dialogue_id
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url,
                "file_name": game.image.filename
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment": o.segment,  # no segment to avoid making the file too big
                                  } for o in game.objects]

            sample["object_id"] = game.object.id
            sample["id_guess_object"] = game.id_guess_object
            sample["status"] = game.status
            sample["att"] = att_dict[game.dialogue_id]
            # sample["beta"] = beta_dict[game.dialogue_id]

            f.write(json.dumps(sample))
            f.write('\n')
