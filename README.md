## QSAR models explainability

I trained models (CatBoost, Decision tree, Convolutional GNN) in order to predict certain ADMET properties of molecules against given biological targets. Then, I used explainability methods to gather insights into what was important for the models. I used LIME, GradCAM, feature importance of CatBoost (as it is a kind of a random forest).

One notebook per dataset is available.

[link to datasets](https://drive.google.com/file/d/1v7hi1tDJ2zxvNTsVqfkh5UAQxxr-5WjR/view?usp=sharing)

## Results summary

I was not able to determine the quality of GradCAM and feature importance results, due to lack of domain-specific knowledge. I was able to see some consistent patterns, however, in the parts of molecules highlighted by GradCAM.

I think I was able to gather evidence that LIME is not effective in this scenario. The space of fingerprints has a very high dimensionality (2048), which seems too high for this method to work.

## How to run

I used Python 3.10.
In order to run the code, you need to install all the required packages:
```
pip install -r requirements.txt
```