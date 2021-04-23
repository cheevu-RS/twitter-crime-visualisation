from tensorflow import keras
import numpy as np
from preprocess_tweets import preprocess_string
import sys
model = keras.models.load_model("tweet_crime_classifier_model")
class_names = ["AntiSocialBehaviour","Theft","CriminalDamage","DrugOffences","PossessionOfWeapons","PublicOrder","VehicleCrime","ViolentCrime","CyberCrime","Terrorism","NonCrime"]
print(model)
def predict_class(s):
    probabilities = model.predict([[preprocess_string(s)]])
    return class_names[np.argmax(probabilities[0])-1]

print(predict_class(sys.argv[1]))
