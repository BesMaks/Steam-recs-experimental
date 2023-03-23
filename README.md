# Video Game Recommendations

## Problem statement

As a (hypothetical) junior data scientist at Steam, my role involves drawing insights from data and creating data-driven tools to boost sales and revenue. 

Firstly, I have been tasked with **providing useful recommendations for the marketing team**, who are currently reviewing their avertising contracts with various publishers.

As a second component, I have been asked to **build a user recommendation engine** using collaborative filtering methods to incentivise purchases. Steam (hypothetically) currently uses a content based model to genereate recommendations and is keen to see which method performs better.

Finally, I will be looking at **item-to-item recommendations**, providing similar items which can be listed on a game's page.

## Components


* **Data**

The data was obtained from https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

* **Ressources**

A collection of [helper functions](https://github.com/nadinezab/video-game-recs/blob/master/resources.py) were defined in the project and saved in `resources.py` for future use.


## Main technologies and packages used

* Python version 3.6.9 
* Pandas version 0.25.1 
* Seaborn version 0.9.0
* Scikit-learn version 0.22.1 
* Lightfm version 1.15 


## Results and recommendations

We built a user game recommendation engine, which takes in a user id and generates `k` games that the user is predicted to like. 
Once integrated with the Steam interface, this can be used directly to display recommended games on a user's homepage. We would need to conduct A/B testing to see if it outperforms the existing games recommender. Based on domain knowledge, the recommendations appear sensible.

<img src="/Images/useritem.png" alt="User item predictions" width = "500" >

We looked at item similarity and built an 'item-to-item' recommendation enginem, which generates `k` similar items using cosine similarity. Once integrated with the Steam interface, this can be used directly to display similar games on a given game's page. We would need to conduct A/B testing to see if it outperforms the existing similar games recommender. Based on domain knowledge, the recommendations appear sensible.

<img src="/Images/similaritems.png" alt="Items similar to American Truck Simulator" width = "500" >
