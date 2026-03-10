import joblib
import pandas as pd

model = joblib.load('models/logistic_model.joblib')


features = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

new_patient = pd.DataFrame([[40,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]], columns=features)

prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)  
pos_prob = probability[0][1] * 100
neg_prob = probability[0][0] * 100


if prediction[0] == 1:
    print("Result:possitive ")
    print(f"probability {pos_prob:.2f}%")
else:
    print("Result:negative")
    print(f"probabilty {neg_prob:.2f}%")


