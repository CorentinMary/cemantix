# agents classes
import logging
import numpy as np
import pandas as pd


class Agent:
    """Parent class for agents."""

    def __init__(
        self,
        model,
        eps_breakpoint=(30, 0.1),
        sigma_maxwords=5,
        sigma_threshold=500,
        start_list=[],
        blacklist_path=None,
    ) -> None:
        self.model = model
        self.eps_breaktemp = eps_breakpoint[0]
        self.eps_breakprob = eps_breakpoint[1]
        self.eps_slope1 = (1 - self.eps_breakprob) / (0 - self.eps_breaktemp)
        self.eps_slope2 = (self.eps_breakprob - 0) / (self.eps_breaktemp - 100)
        self.sigma_maxwords = sigma_maxwords
        self.sigma_threshold = sigma_threshold
        self.sigma_slope = (self.sigma_maxwords - 1) / (1000 - self.sigma_threshold)
        self.start_list = start_list
        blacklist_df = pd.read_json(blacklist_path)
        blacklist_df.columns = ["word"]
        self.blacklist = list(blacklist_df.word.values)
        self.picked_word_list = [""]
        self.round_no = 0
        self.df_result = pd.DataFrame(columns=["word", "temp", "prog"])

    def preprocess(self, word):
        if word.endswith("s") and word[:-1] in self.model.index_to_key:
            return word[:-1]
        return word

    def pick_random_word(self) -> str:
        """Picks a random word in the model's vocabulary
        Returns:
            str: word picked.
        """
        vocab_length = len(self.model.index_to_key)
        word = ""
        while word in self.picked_word_list or word in self.blacklist:
            n = np.random.randint(1, vocab_length)
            word = self.preprocess(self.model.index_to_key[n])
        return word

    def _eps(self, temp: float):
        if temp <= 0:
            return 1
        elif temp <= self.eps_breaktemp:
            return 1 + self.eps_slope1 * temp
        else:
            return self.eps_breakprob + self.eps_slope2 * (temp - self.eps_breaktemp)

    def _sigma(self, prog: float):
        if prog <= self.sigma_threshold:
            return 1
        else:
            return int(1 + self.sigma_slope * (prog - self.sigma_threshold))

    def update(self, result):
        if result != -1:
            self.df_result = self.df_result.append(
                pd.DataFrame(
                    {
                        "word": result["word"],
                        "temp": float(result["temp"].replace(",", ".")),
                        "prog": float(result["prog"].replace(",", "."))
                        if result["prog"] != ""
                        else 0,
                    },
                    index=[0],
                ),
                ignore_index=True,
            ).sort_values(by="temp", ascending=False)
            self.round_no += 1
        else:
            self.blacklist.append(self.picked_word_list[-1])


class CemantixBandit(Agent):
    """Agent using bandit strategy."""

    def __init__(
        self,
        model,
        eps_breakpoint: tuple = (30, 0.1),
        sigma_maxwords: int = 5,
        sigma_threshold: int = 500,
        start_list: list = [],
        blacklist_path: str = None,
    ) -> None:
        """
        :param model:
            pretrained word2vec model
        :param eps_breakpoint: tuple, defalts to (30, 0.1).
            breakpoint coordinates thresholds to use for exploration function.
        :param sigma_maxwords: int, defaults to 5.
            maximum number of words to use for similarity search.
        :param sigma_threshold: int, defaults to 50.
            progression threshold above which to use similarity search.
        :param start_list: list, defaults to [].
            words to start playing the game with.
        :param blacklist_path: str, defaults to None.
            path to the list of words unknown to the game that should be avoided to gain time.
        """
        super().__init__(
            model,
            eps_breakpoint,
            sigma_maxwords,
            sigma_threshold,
            start_list,
            blacklist_path,
        )

    def search_similarity(self, prog):
        # Drawing the number of words to use for similarity search
        sigma_bound = self._sigma(prog)
        sigma = np.random.randint(1, sigma_bound + 1)
        similar_words = [
            self.preprocess(w[0])
            for w in self.model.most_similar(list(self.df_result.word.values[:sigma]))
            if self.preprocess(w[0]) not in self.picked_word_list
        ]
        if len(similar_words):
            word = similar_words[0]
            logging.info(f"Using similarity ({sigma}) with {word}")
        else:
            word = self.pick_random_word()
            logging.info(f"Exploring with {word}")

        return word

    def pick_word(self):
        if len(self.start_list):
            word = self.start_list[0]
            self.start_list.remove(word)
        elif self.round_no == 0:
            word = self.pick_random_word()
        else:
            # Choosing between exploration and similarity search
            best_temp = self.df_result.temp.values[0]
            best_prog = self.df_result.prog.values[0]
            eps = self._eps(best_temp)
            explore = np.random.binomial(1, eps)
            if explore == 1:
                word = self.pick_random_word()
                logging.info(f"Exploring with {word}")
            else:
                word = self.search_similarity(best_prog)
        self.picked_word_list.append(word)

        return word


class CemantixGangster(Agent):
    """More complex agent using bandit strategy with a random choice of words to use for similarity."""

    def __init__(
        self,
        model,
        eps_breakpoint: tuple = (30, 0.1),
        sigma_maxwords: int = 5,
        sigma_threshold: int = 500,
        gamma_threshold: int = 750,
        start_list: list = [],
        blacklist_path: str = [],
    ) -> None:
        """
        :param model:
            pretrained word2vec model
        :param eps_breakpoint: tuple, defalts to (30, 0.1).
            breakpoint coordinates thresholds to use for exploration function.
        :param sigma_maxwords: int, defaults to 5.
            maximum number of words to use for similarity search.
        :param sigma_threshold: int, defaults to 50.
            progression threshold above which to use similarity search.
        :param gamma_threshold: int, defaults to 750.
            progression threshold for selection of words used in similarity search.
        :param start_list: list, defaults to [].
            words to start playing the game with.
        :param blacklist_path: str, defaults to None.
            path to the list of words unknown to the game that should be avoided to gain time.
        """
        super().__init__(
            model,
            eps_breakpoint,
            sigma_maxwords,
            sigma_threshold,
            start_list,
            blacklist_path,
        )
        self.gamma_threshold = gamma_threshold

    def search_similarity(self, prog):
        # Drawing the number of words to use for similarity search
        sigma_bound = self._sigma(prog)
        sigma = np.random.randint(1, sigma_bound + 1)
        # Drawing the words to use for similarity search
        # if there are enough words above the gamma_threshold, draw them randomly; else use the top sigma words
        suitable_words = list(
            self.df_result.loc[self.df_result.prog >= self.gamma_threshold].word.values
        )
        if len(suitable_words) >= sigma:
            selected_words = list(
                np.random.choice(suitable_words, size=sigma, replace=False)
            )
        else:
            selected_words = list(self.df_result.word.values[:sigma])
        similar_words = [
            self.preprocess(w[0])
            for w in self.model.most_similar(selected_words)
            if self.preprocess(w[0]) not in self.picked_word_list
        ]
        if len(similar_words):
            word = similar_words[0]
            logging.info(f"Using similarity ({sigma}) with {word}")
        else:
            word = self.pick_random_word()
            logging.info(f"Exploring with {word}")

        return word

    def pick_word(self):
        if len(self.start_list):
            word = self.start_list[0]
            self.start_list.remove(word)
        elif self.round_no == 0:
            word = self.pick_random_word()
        else:
            # Choosing between exploration and similarity search
            best_temp = self.df_result.temp.values[0]
            best_prog = self.df_result.prog.values[0]
            eps = self._eps(best_temp)
            explore = np.random.binomial(1, eps)
            if explore == 1:
                word = self.pick_random_word()
                logging.info(f"Exploring with {word}")
            else:
                word = self.search_similarity(best_prog)
        self.picked_word_list.append(word)

        return word


class CemantixPirate(Agent):
    """Even more complex agent where the number of words selected in similarity search decreases when getting closer to the solution."""

    def __init__(
        self,
        model,
        eps_breakpoint: tuple = (30, 0.1),
        sigma_maxwords: int = 5,
        sigma_threshold: int = 500,
        gamma_threshold: int = 750,
        tau_init: int = 20,
        tau_final: int = 5,
        tau_threshold: int = 500,
        start_list: list = [],
        blacklist_path: str = None,
    ) -> None:
        """
        :param model:
            pretrained word2vec model
        :param eps_breakpoint: tuple, defalts to (30, 0.1).
            breakpoint coordinates thresholds to use for exploration function.
        :param sigma_maxwords: int, defaults to 5.
            maximum number of words to use for similarity search.
        :param sigma_threshold: int, defaults to 50.
            progression threshold above which to use similarity search.
        :param gamma_threshold: int, defaults to 750.
            progression threshold for selection of words used in similarity search.
        :param tau_init: int, defaults to 20.
            initial number of words to select for similarity search.
        :param tau_final: int, defaults to 5.
            final number of words to select for similarity search.
        :param tau_threshold: int, defaults to 500.
            progression threshold above which to decrease the number of words to select for similarity search.
        :param start_list: list, defaults to [].
            words to start playing the game with.
        :param blacklist_path: str, defaults to None.
            path to the list of words unknown to the game that should be avoided to gain time.
        """
        super().__init__(
            model,
            eps_breakpoint,
            sigma_maxwords,
            sigma_threshold,
            start_list,
            blacklist_path,
        )
        self.gamma_threshold = gamma_threshold
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.tau_threshold = tau_threshold
        self.tau_slope = (tau_final - tau_init) / (1000 - tau_threshold)

    def _tau(self, prog):
        if prog <= self.tau_threshold:
            return self.tau_init
        else:
            return int(self.tau_init + self.tau_slope * (prog - self.tau_threshold))

    def search_similarity(self, prog):
        # Drawing the number of words to use for similarity search
        sigma_bound = self._sigma(prog)
        sigma = np.random.randint(1, sigma_bound + 1)
        tau = self._tau(prog)
        # Drawing the words to use for similarity search
        # if there are enough words above the gamma_threshold, draw them randomly; else use the top sigma words
        suitable_words = list(
            self.df_result.loc[self.df_result.prog >= self.gamma_threshold].word.values
        )
        if len(suitable_words) >= sigma:
            selected_words = list(
                np.random.choice(suitable_words, size=sigma, replace=False)
            )
        else:
            selected_words = list(self.df_result.word.values[:sigma])
        similar_words = [
            self.preprocess(w[0])
            for w in self.model.most_similar(selected_words, topn=tau)
            if self.preprocess(w[0]) not in self.picked_word_list
        ]
        if len(similar_words):
            word = similar_words[0]
            logging.info(f"Using similarity ({sigma}) with {word}")
        else:
            word = self.pick_random_word()
            logging.info(f"Exploring with {word}")

        return word

    def pick_word(self):
        if len(self.start_list):
            word = self.start_list[0]
            self.start_list.remove(word)
        elif self.round_no == 0:
            word = self.pick_random_word()
        else:
            # Choosing between exploration and similarity search
            best_temp = self.df_result.temp.values[0]
            best_prog = self.df_result.prog.values[0]
            eps = self._eps(best_temp)
            explore = np.random.binomial(1, eps)
            if explore == 1:
                word = self.pick_random_word()
                logging.info(f"Exploring with {word}")
            else:
                word = self.search_similarity(best_prog)

        self.picked_word_list.append(word)
        return word
