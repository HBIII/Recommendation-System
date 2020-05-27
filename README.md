# Recommendation-System

## Overview

Built 3 different types of recommendation system to predict the ratings/stars for given user ids and business ids. Used the validation dataset to evaluate the accuracy of recommendation systems.

1)	**Item-based CF recommendation system**

    Implemented item-based recommendation system using **Pearson similarity** formula.

2)	**Model-based recommendation system**

    Implemented Model-based recommendation system using **XGBregressor**(a regressor based on the decision tree) to train a model and       use the validation dataset to validate your result. 

3)	**Hybrid recommendation system**

    Implemented Hybrid recommendation system by taking weighted average of item-based CF and Model-based recommendation system by using     the below formula:

      ***final score*** = **𝛼 × score**<sub>𝑖𝑡𝑒𝑚_𝑏𝑎𝑠𝑒𝑑</sub>   + **(1−𝛼) × 𝑠𝑐𝑜𝑟𝑒**<sub>𝑚𝑜𝑑𝑒𝑙_𝑏𝑎𝑠𝑒𝑑</sub>

I used RMSE to check the accuracy of my recommendation systems.

Item-based CF recommendation system: 1.05
Model-based recommendation system: 0.99
Hybrid recommendation system: 0.98


