import joblib

model = joblib.load("models/price_model_v1.joblib")

print(type(model))

try:
    print("Best iteration:", model.best_iteration)
except:
    pass

print("Feature importances:")
print(model.feature_importances_)