from flask import Flask, request, jsonify
import requests
import json

from torch_utils import transform, get_prediction

app=Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            pm_data=json.loads(request.get_data(),encoding='utf-8')
            pm_data=pm_data['pm']
            tensor=transform(pm_data)
            prediction=get_prediction(tensor)
            data={'prediction':prediction.item(), 'class_name':str(prediction.item())}
            
            return jsonify(data)
        except:
            return jsonify({'error':'error during prediction'})