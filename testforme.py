import joblib

bundle = joblib.load("models/demand_model.joblib")
print(type(bundle))        # should be <class 'dict'>
print(bundle.keys())       # should show ['model', 'features']
