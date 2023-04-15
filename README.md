# Cemantix Solver

## Description
Ever heard of the game cemantix? Check it out: https://cemantix.certitudes.org/

This repo combines scraping, (soft) Reinforcement Learning and (pretrained) Natural Language Processing to automatically play (and win) the game.

## Explanation
### Environment
The environment is defined as the game page on which the player can make a guess (the action) and receive an information on how close he is to the solution (the reward).
The CemantixScraper class is used to simulate actions and extract rewards from the web page.

### Bandit agent
The first agent works using the well-known bandit strategy: at each turn, it chooses between exploring, that is picking a random word; or using its previous picks to make a similar guess.
This is done using two random variables:
- "explore" which is a bernoulli variable with probability eps where eps is a function decreasing with the best score obtained so far. This means that the closer the agent is to the solution, the less likely it is to explore.
- "sigma" which is used to pick the number of words to use for the similarity search. Similarly, the closer the agent is to the solution, the more words it will use for similarity search.
One of the drawbacks of this strategy is that only the words closest to the solution are picked for similarity which can lead to a dead end in the search. This is why the next agent was created...

### Gangster agent
This agent is very similar to the first agent but it adds a new level of randomness in the choice of the words used for similarity search via a new random variable called "gamma", making the agent even better.

### Pirate agent
Coming soon...

## Try it out!
Run the following command lines to test it:
```
conda create -n cemantix python=3.8 -y
conda activate cemantix
pip install -r requirements.txt
cemantix.bat
```

## Troubleshooting
You may encounter an issue with the chromedriver executable if its version does not match your version of chrome.
If that is the case you can check your version of chrome on chrome://version/ and then download the corresponding chromedriver on https://chromedriver.chromium.org/downloads and updload it in the artifacts/ folder.

