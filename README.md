
The goal of this repository is to test whether we can use BERT to do a regression, or in other words whether can BERT learn numerical values of bits of text. It looks like it can. 

**Generate Data:**

I was not able to find any data sets for this, so I generated it myself. I took data from https://www.kaggle.com/c/house-prices-advanced-regression-techniques and generated description paragraphs. 

I did not use all the features. The features that I used:

`MoSold`\
`YrSold`\
`1stFlrSF`\
`2ndFlrSF`\
`FullBath`\
`HalfBath`\
`LotArea`\
`OverallQual`\
`OverallCond`

And sample generated paragraph can look like this:

`
The overall material and finish of the house is good. Lot area is 8450 sq ft. The house is in average condition. Sold in February 2008. 2 full bathrooms and 1 half bathroom. First floor square footage is 856 sq. ft. and second floor square footage is 854 sq. ft.
`

Or

`
Lot area is 8500 square meters. The overall material and finish of the house is below average. First floor area is 649 square feet and second floor area is 668 square feet. Sold in July 2008. 1 full bathroom. The house is in below average condition.
`

The sentences are shuffled. The values are plugged in to different templates, for example square feet or meters can be to give measure of area. Also if a measurement is 0 for example for second floor area or half bathroom that portion is left out from the paragraph.

After the data paragraphs are generated we training them on prices given in the dataset. We use very much a standard BERT approach.

**BERT Model**

I used `bert-base-uncased`. I added one linear layer on top of the BERT model. I did not freeze any layers when fine tuning. I trained 5 models, each model was trained on 4/5 of the training data, and remaining 1/5 was used for validation. I kept the model that had best validation loss score. Therefore, I ended up with 5 models. The final result was average score of all model predictions. I went with more or less standard parameters and did not explore the results of many parameters. I did normalize the Sales Price to be between 0 and 1 for the training set.

**Results:**

The resulting model scores is 0.21 to 0.23 (the predictions are not deterministic), which is not great and is below 80% of kaggle submissions. However, we are clearly handicapped, extracting information from a text is harder, and we are using only subset of the features. Also the model has not been tuned. Catboost algorithm that uses the same features scores about 0.18. Using just mean of the sale price gives 0.426. Therefore, the results look reasonable.

**How to run:**

1) Put `train.csv` and `test.csv` that can be downloaded from kaggle competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data to

`/data`

2) `python data/make_text_descriptions.txt`

This will generate files that have text descriptions.

3) `python src/bert_model/run_bert_model.py`

This will train the models, and save them to `src/bert_model/generated_data`, it will also save min and max data for the SalePrice, we need these undo normalization in the next step.

4) `python src/bert_model/predictor.py`

This will used the models generated in the last step to make a predictions file: `house_prices_pred.csv`











