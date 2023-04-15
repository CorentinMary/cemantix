# Cemantix game solver
import logging
import os
import yaml

from src import *

os.chdir(os.path.abspath(os.path.dirname(__file__)))
with open("config.yaml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
logging.basicConfig(filename=f"./logs/cemantix_{dt.datetime.now().strftime(format='%Y-%m-%d_%Hh%M')}.log", level=logging.INFO)

def main():
    model = load_model(embedding=config['word2vec'])
    game = CemantixGame(executable_path="./artifacts/chromedriver.exe",game_url=config["game_url"])
    if config["agent_type"] == "bandit":
        agent = CemantixBandit(model=model, **config["agent_params"])
    elif config["agent_type"] == "gangster":
        agent = CemantixGangster(model=model, **config["agent_params"])
    else:
        raise ValueError("Unknown agent_type")
    game.play_strategy(agent, max_iter=config["max_iter"])
    game.save_result()
    game.end()

main()
