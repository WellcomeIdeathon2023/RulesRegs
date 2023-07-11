
# Clinical trial recommendation using Bidirectional Encoder Representations from Transformers (BERT) and Collaborative Filtering.

The project involves using a fine-tuned BERT transformer model and collaborative filtering to recommend clinical trials,Idea is to combine power of machine learning models and Human experiences into consideration to recommendation for clinical trial in a more inclusive way(without naturally affecting or segregating them).User gives in search query which is classified into a mental health condition, This condition as query is then used along with collaborative filtering techniques to recommend clinical trials that are suitable for the individual. Data is collected from the ClinicalTrials.gov API by querying for conditions related to mental health. The collected data includes several features such as the trial ID, title, condition, eligibility criteria, description, keywords, status, location, start date, and related links. The data is preprocessed and the features 'EligibilityCriteria', 'DetailedDescription', 'Keyword', and 'BriefTitle' are concatenated and used as input, while the mental health condition is used as the target or label for training the BERT model for classification.


## API Reference

#### Get all items

```http
  GET ClinicalTrials.gov/api/query/field_values?expr=heart+attack&field=Condition
```


## Environment Variables

pip install -r requirements.txt
## Installation

pip install flask

    
## How to train the BERT model?

[](https://linktodocumentation)




If you want to fine tune the model for other task, you need to follow the same data-structure as in selected_columns.csv, that is features(comma-seperated text) and label (numerically encoded).

Note: Prediction from the BERT model is numeric, internally it's mapped with their respectic categorical counterpart.

Collaborative filtering is item - user based(BreifTile - UserID).Used Simulated Data to make the matrix factorization.

Future Vision:

Ratings is considered for making collaboarative filtering but feedback (categorical data can be used as sentiment and then encoded and scored based on severity.)




## Support

For support, email ray.ankit1201@outlook.com


## Demo

Insert gif or link to demo

file:///Users/ankit/Downloads/demo.gif
