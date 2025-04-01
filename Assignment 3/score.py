import pickle
import sklearn
import sys
from typing import Tuple

def score(text:str, model:sklearn.estimator, threshold:float) -> Tuple[bool, float]:
    '''score a trained model on a text and return the prediction and propensity'''
    prediction = model.predict([text])[0]
    propensity = model.predict_proba([text])[0][1]

    return prediction, propensity

if __name__ == '__main__':
    text, model_path, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
    model = pickle.load(open(model_path, 'rb'))
    
    # score the text
    prediction, propensity = score(text, model, threshold)
    
    print(f'Prediction: {prediction}')
    print(f'Propensity: {propensity}')
    print(f'Threshold: {threshold}')
    print(f'Text: {text}')
    print(f'Model: {model_path}')
    
    sys.exit(0)