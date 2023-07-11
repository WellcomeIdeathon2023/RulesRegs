from flask import Flask,render_template,request
import pickle
import numpy as np
from transformers import AutoTokenizer # Tokenisation
from transformers import AutoModelForSequenceClassification # Classification

pt = pickle.load(open('pt.pkl','rb'))
ratings_with_clinical = pickle.load(open('ratings_with_clinical.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def recommend_ui():
    return render_template('index.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('q')
    model_path = "/Users/ankit/Downloads/Ideathon/trial_Recommender"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    input_text = user_input
    tokenized_text = tokenizer(input_text,
                           truncation=True,
                           is_split_into_words=False,
                           return_tensors='pt')

    outputs = model(**tokenized_text)
    predicted_label = outputs.logits.argmax(-1)
    predicted_label_value = predicted_label.item()

    if predicted_label_value == 0:
        condition_name = "Anxiety"
    elif predicted_label_value == 1:
        condition_name = "Autism"
    elif predicted_label_value == 2:
        condition_name = "Schizophrenia"
    else:
        condition_name = "Unknown"

    name_filter = ratings_with_clinical['Condition'] == condition_name
    trial_value = ratings_with_clinical.loc[name_filter, 'BriefTitle'].values[0]
    
    # index fetch
    index = np.where(pt.index==trial_value)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    data = []
    for i in similar_items:
        item = []
        temp_df = ratings_with_clinical[ratings_with_clinical['BriefTitle'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('BriefTitle')['BriefTitle'].values))
        item.extend(list(temp_df.drop_duplicates('BriefTitle')['DetailedDescription'].values))
        item.extend(list(temp_df.drop_duplicates('BriefTitle')['StartDate'].values))

        data.append(item)

    print(data)

    return render_template('index.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)