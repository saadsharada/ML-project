from django.shortcuts import render
import joblib
from joblib import load
model = load('SavedModels/knn.joblib')

def index(request):
    return render(request, 'index.html')

def getPredictions(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age):

    prediction = model.predict([
        [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]
    ])
    
    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def result(request):
    pregnancies = float(request.GET['pregnancies'])
    glucose = float(request.GET['glucose'])
    bloodpressure = float(request.GET['bloodpressure'])
    skinthickness = float(request.GET['skinthickness'])
    insulin = float(request.GET['insulin'])
    bmi = float(request.GET['bmi'])
    dpf = float(request.GET['dpf'])
    age = float(request.GET['age'])
    result = getPredictions(pregnancies, glucose, bloodpressure,
                            skinthickness, insulin, bmi, dpf, age)

    return render(request, 'result.html', {'result': result})