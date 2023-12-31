# Python classes for the cemantix game environment

import datetime as dt
import logging
import numpy as np
import os
import pandas as pd
import time
import warnings

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CemantixScraper:
    """A class that opens the cemantix game, makes guesses and stores the result in a dataframe."""

    def __init__(
        self,
        executable_path=f"../artifacts/chromedriver.exe",
        game_url="https://cemantix.certitudes.org/",
    ):
        self.executable_path = executable_path
        self.game_url = game_url

    def start_game(self):
        """Launches the driver and starts the game."""
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
        assert len(input_box) != 0, "Input box not found."

        input_box[0].send_keys(word)
        input_box[0].send_keys(Keys.ENTER)

    def get_result(self):
        """Returns -1 if the word provided doesn't exist, else the score obtained with the word.

        Args:
            print_error (bool): whether to print an error message or not when the word is not recognised.

        Returns:
            dict: word,  temperature and progression scores obtained. -1 the word produces an error.
        """
        time.sleep(5)
        error_msg = self.driver.find_elements(by=By.ID, value="cemantix-error")
        assert len(error_msg) != 0, "Error message element not found."

        if error_msg[0].text != "":
            logging.info("The word provided was not recognized by the game.")
            return -1

        guessed = self.driver.find_elements(by=By.ID, value="cemantix-guessed")
        assert len(guessed) != 0, "Guessed not found."
        result = {
            "word": guessed[0].find_elements(by=By.XPATH, value="./td")[1].text,
            "temp": guessed[0].find_elements(by=By.XPATH, value="./td")[2].text,
            "prog": guessed[0].find_elements(by=By.XPATH, value="./td")[4].text,
        }
        return result

    def close(self):
        self.driver.close()


class CemantixGame:
    """Plays Cemantix."""

    def __init__(
        self,
        executable_path=f"../artifacts/chromedriver.exe",
        game_url="https://cemantix.certitudes.org/",
    ) -> None:
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
                self.result = self.result.append(
                    pd.DataFrame(res, index=[0]), ignore_index=True
                )

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
            res = self.game.get_result()
            agent.update(res)
            if res != -1:
                self.result = self.result.append(
                    pd.DataFrame(res, index=[0]), ignore_index=True
                )
                i += 1
                if res["prog"] == "1000":
                    self.win = True
                    logging.info("GAME WON!")

        return self.result

    def save_result(self, path="./results/", name="cemantix") -> None:
        self.result.to_csv(
            f"{path}{name}_{dt.date.today().strftime('%Y%m%d')}.csv",
            index=False,
            sep=";",
        )

    def end(self) -> None:
        self.result.truncate()
        self.game.close()
