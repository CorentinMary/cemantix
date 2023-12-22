# Cemantix game solver
import datetime as dt
import json
import logging
import os
import yaml

from src import *

AGENT_MAPPING = {
    "bandit": CemantixBandit,
    "gangster": CemantixGangster,
    "pirate": CemantixPirate,
}

os.chdir(os.path.abspath(os.path.dirname(__file__)))
with open("config.yaml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
logging.basicConfig(
    filename=f"./logs/cemantix_{dt.datetime.now().strftime(format='%Y-%m-%d_%H%M')}.log",
    level=logging.INFO,
)


def main(config):
    assert config["agent_type"] in AGENT_MAPPING.keys(), "Unknown agent_type."
    model = load_model(
        embedding=config["word2vec"], blacklist_path=config["blacklist_path"]
    )
    game = CemantixGame(
        executable_path="./artifacts/chromedriver.exe", game_url=config["game_url"]
    )
    agent = AGENT_MAPPING.get(config["agent_type"])(
        model=model, blacklist_path=config["blacklist_path"], **config["agent_params"]
    )
    game.play_strategy(agent, max_iter=config["max_iter"])
    game.save_result()
    game.end()

    # updating blacklist
    with open(config["blacklist_path"], "w") as file:
        json.dump(agent.blacklist, file)


main(config)
