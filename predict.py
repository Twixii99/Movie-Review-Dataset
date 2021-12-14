import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

model_file = 'model_xgb_model.bin'
with open(model_file, 'rb') as m_in:
    cv, model = pickle.load(m_in)

app = Flask('weather_prediction')

def clean(data):
  cleaned_data = []
  for item in data:
    item = item.lower()
    item = re.sub(r'[^\w\s]|[\d+]', '', item)
    cleaned_data.append(item)
  return 

def advanced_clean(data):
  for idx, sample in enumerate(data):
    __tokens__ = []
    tokens = nltk.word_tokenize(sample)
    for token in tokens:
      if token not in english_stopwords:
        token_pos = get_wordnet_pos(token)
        lemma = lemmatizer.lemmatize(token, token_pos)
        if  lemma in words and lemma not in english_stopwords and lemma in words:
          __tokens__.append(lemma)
    data[idx] = ''.join(__token__ + ' ' for __token__ in __tokens__)
    
@app.route('/predict', methods=['POST'])
def predict():
    review = request.get_json()

    review = claen([review])
    review = advanced_clean([review])
    X = cv.transform([review])
    y_pred = model.predict(xgb.DMatrix(X, feature_names=cv.get_feature_names()))
    decison = y_pred >= 0.5

    result = {
        'Raining probability': float(y_pred),
        'Will it rain ?': bool(decision)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
