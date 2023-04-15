# Python classes for playing the cemantix game

import datetime as dt
import logging
import numpy as np
import os
import pandas as pd
import time
import warnings

from gensim.models import KeyedVectors
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from urllib.request import urlretrieve

warnings.filterwarnings("ignore")

def load_model(embedding):
    try:
        model = KeyedVectors.load_word2vec_format(f"./artifacts/{embedding}", binary=True, unicode_errors="ignore")
    except:
        logging.info("Specified embedding not found in artifacts/ folder. Downloading...")
        try:
            urlretrieve(f"https://embeddings.net/embeddings/{embedding}", f"./artifacts/{embedding}")
        except:
            logging.error("Failed to download the embedding. Please check that the specified embedding is available on 'https://embeddings.net/embeddings/'")
        model = KeyedVectors.load_word2vec_format(f"./artifacts/{embedding}", binary=True, unicode_errors="ignore")
    
    return model



class CemantixScraper():
    """A class that opens the cemantix game, makes guesses and stores the result in a dataframe.
    """

    def __init__(self, executable_path=f"../artifacts/chromedriver.exe", game_url="https://cemantix.certitudes.org/"):
        self.executable_path = executable_path
        self.game_url = game_url

    def start_game(self):
        """Launches the driver and starts the game.
        """
        self.driver = webdriver.Chrome(executable_path=self.executable_path)
        self.driver.get(self.game_url)
        # closing the dialog box 
        close_dialog_button = self.driver.find_elements(by=By.ID, value="dialog-close")
        if len(close_dialog_button):
            close_dialog_button[0].click()

    def make_guess(self, word):
        """Plays the word provided.

        Args:
            word (str): word to play.
        """
        input_box = self.driver.find_elements(by=By.ID, value="cemantix-guess")
        assert len(input_box)!=0, "Input box not found." 

        input_box[0].send_keys(word)
        input_box[0].send_keys(Keys.ENTER)
    
    def get_result(self, print_error=False):
        """Returns -1 if the word provided doesn't exist, else the score obtained with the word.

        Args:
            print_error (bool): whether to print an error message or not when the word is not recognised.

        Returns:
            dict: word,  temperature and progression scores obtained. -1 the word produces an error.
        """
        time.sleep(3)
        error_msg = self.driver.find_elements(by=By.ID, value="cemantix-error")
        assert len(error_msg)!=0, "Error message element not found."

        if error_msg[0].text != '':
            logging.info("The word provided was not recognized by the game.")
            return -1
        
        guessed = self.driver.find_elements(by=By.ID, value="cemantix-guessed")
        assert len(guessed)!=0, "Guessed not found."
        result = {
            "word": guessed[0].find_elements(by=By.XPATH, value="./td")[1].text,
            "temp": guessed[0].find_elements(by=By.XPATH, value="./td")[2].text,
            "prog": guessed[0].find_elements(by=By.XPATH, value="./td")[4].text
        }
        return result
    
    def close(self):
        self.driver.close()



class CemantixGame():
    """Plays Cemantix.
    """

    def __init__(self, executable_path=f"../artifacts/chromedriver.exe", game_url="https://cemantix.certitudes.org/") -> None:
        self.game = CemantixScraper(executable_path, game_url)
        self.result = pd.DataFrame(columns=["word", "temp", "prog"])
        self.win = False

    def play_list(self, word_list) -> pd.DataFrame:
        """Plays the words listed.

        Args:
            word_list (list): list of words to play

        Returns:
            pd.DataFrame: results of the game after playing.
        """
        self.game.start_game()
        for i, word in tqdm(enumerate(word_list)):
            self.game.make_guess(word)
            res = self.game.get_result()
            if res != -1:
                self.result = self.result.append(pd.DataFrame(res, index=[0]), ignore_index=True)
        
        return self.result

    def play_strategy(self, agent, max_iter=1000) -> pd.DataFrame:
        """Plays the strategy of a given agent

        Args:
            agent: an agent implementing a strategy.
        
        Returns:
            pd.DataFrame: results of the game after playing.
        """
        i = 0
        self.game.start_game()
        while i < max_iter and not self.win:
            word = agent.pick_word()
            self.game.make_guess(word)
            res = self.game.get_result(print_error=True)
            agent.update(res)
            if res != -1:
                self.result = self.result.append(pd.DataFrame(res, index=[0]), ignore_index=True)
                i += 1
                if res["prog"] == '1000':
                    self.win = True
                    logging.info("GAME WON!")
        
        return self.result
        
    def save_result(self, path="./results/", name="cemantix") -> None:
        self.result.to_csv(f"{path}{name}_{dt.date.today().strftime('%Y%m%d')}.csv", index=False, sep=";")

    def end(self) -> None:
        self.result.truncate()
        self.game.close()



class CemantixBandit():
    """Agent using bandit strategy.
    """
    def __init__(self, model, eps_breakpoint=(30, 0.1), sigma_maxwords=5, sigma_threshold=500, start_list=[], **kwargs) -> None:
        """
        Args:
            model: pretrained word2vec model
            eps_breakpoint (tuple, optional): breakpoint coordinates thresholds to use for exploration function. Defaults to (30, 0.1).
            sigma_maxwords (int, optional): maximum number of words to use for similarity search. Defaults to 5.
            sigma_threshold (int, optional): progression threshold above which to use similarity search. Defaults to 50.
        """
        self.model = model

        self.eps_breaktemp = eps_breakpoint[0]
        self.eps_breakprob = eps_breakpoint[1] 
        self.eps_slope1 = (1 - self.eps_breakprob) / (0 - self.eps_breaktemp)
        self.eps_slope2 = (self.eps_breakprob - 0) / (self.eps_breaktemp - 100)

        self.sigma_maxwords = sigma_maxwords
        self.sigma_threshold = sigma_threshold
        self.sigma_slope = (self.sigma_maxwords - 1) / (1000 - self.sigma_threshold)

        self.start_list = start_list

        self.picked_word_list = ['']
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
        word = ''
        while word in self.picked_word_list:
            n = np.random.randint(1, vocab_length)
            word = self.preprocess(self.model.index_to_key[n])

        return word
    
    def eps(self, temp):
        if temp <= 0:
            return 1
        elif temp <= self.eps_breaktemp:
            return 1 + self.eps_slope1 * temp
        else:
            return self.eps_breakprob + self.eps_slope2 * (temp - self.eps_breaktemp)
    
    def sigma(self, prog):
        if prog <= self.sigma_threshold:
            return 1
        else:
            return int(1 + self.sigma_slope * (prog - self.sigma_threshold))

    def pick_word(self):
        if len(self.start_list):
            word = self.start_list[0]
            self.start_list.remove(word)
        elif self.round_no == 0:
            word = self.pick_random_word()
        else:
            # Choosing between exploration and similarity search
            best_temp = self.df_result.temp.values[0]
            eps = self.eps(best_temp)
            explore = np.random.binomial(1, eps)
            if explore == 1:
                word = self.pick_random_word()
                logging.info(f"Exploring with {word}")
            else:
                # Drawing the number of words to use for similarity search
                best_prog = self.df_result.prog.values[0]
                sigma_bound = self.sigma(best_prog)
                sigma = np.random.randint(1, sigma_bound+1)
                similar_words = [
                    self.preprocess(w[0]) for w in self.model.most_similar(list(self.df_result.word.values[:sigma]))
                    if self.preprocess(w[0]) not in self.picked_word_list
                ]
                if len(similar_words):
                    word = similar_words[0]
                    logging.info(f"Using similarity ({sigma}) with {word}")
                else:
                    word = self.pick_random_word()
                    logging.info(f"Exploring with {word}")

        self.picked_word_list.append(word)
        return word

    def update(self, result):
        if result != -1:
            self.df_result = (
                self.df_result
                .append(
                    pd.DataFrame({
                        "word": result["word"],
                        "temp": float(result["temp"].replace(',', '.')),
                        "prog": float(result["prog"].replace(',', '.')) if result["prog"] != '' else 0
                    }, index=[0]), ignore_index=True
                )
                .sort_values(by="temp", ascending=False)
            )
            self.round_no += 1
        


class CemantixGangster():
    """More complex agent using bandit strategy.
    """
    def __init__(self, model, eps_breakpoint=(30, 0.1), sigma_maxwords=5, sigma_threshold=500, gamma_threshold=750, start_list=[], **kwargs) -> None:
        """
        Args:
            model: pretrained word2vec model
            eps_breakpoint (tuple, optional): breakpoint coordinates thresholds to use for exploration function. Defaults to (30, 0.1).
            sigma_maxwords (int, optional): maximum number of words to use for similarity search. Defaults to 5.
            sigma_threshold (int, optional): progression threshold above which to use similarity search. Defaults to 500.
            gamma_threshold (int, optional): progression threshold for selection of words used in similarity search. Defaults to 750.
            start_list (list, optional): list of words to start the game with. Defaults to [].
        """
        self.model = model

        self.eps_breaktemp = eps_breakpoint[0]
        self.eps_breakprob = eps_breakpoint[1] 
        self.eps_slope1 = (1 - self.eps_breakprob) / (0 - self.eps_breaktemp)
        self.eps_slope2 = (self.eps_breakprob - 0) / (self.eps_breaktemp - 100)

        self.sigma_maxwords = sigma_maxwords
        self.sigma_threshold = sigma_threshold
        self.sigma_slope = (self.sigma_maxwords - 1) / (1000 - self.sigma_threshold)

        self.gamma_threshold = gamma_threshold

        self.start_list = start_list

        self.picked_word_list = ['']
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
        word = ''
        while word in self.picked_word_list:
            n = np.random.randint(1, vocab_length)
            word = self.preprocess(self.model.index_to_key[n])

        return word
    
    def eps(self, temp):
        if temp <= 0:
            return 1
        elif temp <= self.eps_breaktemp:
            return 1 + self.eps_slope1 * temp
        else:
            return self.eps_breakprob + self.eps_slope2 * (temp - self.eps_breaktemp)
    
    def sigma(self, prog):
        if prog <= self.sigma_threshold:
            return 1
        else:
            return int(1 + self.sigma_slope * (prog - self.sigma_threshold))

    def pick_word(self):
        if len(self.start_list):
            word = self.start_list[0]
            self.start_list.remove(word)
        elif self.round_no == 0:
            word = self.pick_random_word()
        else:
            # Choosing between exploration and similarity search
            best_temp = self.df_result.temp.values[0]
            eps = self.eps(best_temp)
            explore = np.random.binomial(1, eps)
            if explore == 1:
                word = self.pick_random_word()
                logging.info(f"Exploring with {word}")
            else:
                # Drawing the number of words to use for similarity search
                best_prog = self.df_result.prog.values[0]
                sigma_bound = self.sigma(best_prog)
                sigma = np.random.randint(1, sigma_bound+1)
                # Drawing the words to use for similarity search
                # if there are enough words above the gamma_threshold, draw them randomly; else use the top sigma words
                suitable_words = list(self.df_result.loc[self.df_result.prog >= self.gamma_threshold].word.values)
                if len(suitable_words) >= sigma:
                    selected_words = list(np.random.choice(suitable_words, size=sigma, replace=False))
                else:
                    selected_words = list(self.df_result.word.values[:sigma])
                similar_words = [
                    self.preprocess(w[0]) for w in self.model.most_similar(selected_words)
                    if self.preprocess(w[0]) not in self.picked_word_list
                ]
                if len(similar_words):
                    word = similar_words[0]
                    logging.info(f"Using similarity ({sigma}) with {word}")
                else:
                    word = self.pick_random_word()
                    logging.info(f"Exploring with {word}")

        self.picked_word_list.append(word)
        return word

    def update(self, result):
        if result != -1:
            self.df_result = (
                self.df_result
                .append(
                    pd.DataFrame({
                        "word": result["word"],
                        "temp": float(result["temp"].replace(',', '.')),
                        "prog": float(result["prog"].replace(',', '.')) if result["prog"] != '' else 0
                    }, index=[0]), ignore_index=True
                )
                .sort_values(by="temp", ascending=False)
            )
            self.round_no += 1
        
