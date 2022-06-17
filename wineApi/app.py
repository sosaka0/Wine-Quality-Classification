# 1. Library imports
import uvicorn
from fastapi import FastAPI
from wine import wines
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Wine Quality Classification'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'***': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_wine(data:wines):
    
    fixedacidity=data.fixedacidity
    volatileacidity=data.volatileacidity
    citricacid=data.citricacid
    residualsugar=data.residualsugar
    chlorides=data.chlorides
    freesulfurdioxide=data.freesulfurdioxide
    totalsulfurdioxide=data.totalsulfurdioxide
    density=data.density
    pH=data.pH
    sulphates=data.sulphates
    alcohol=data.alcohol

   
    prediction = classifier.predict([[fixedacidity,volatileacidity,citricacid,residualsugar,chlorides,freesulfurdioxide,totalsulfurdioxide,density,pH,sulphates,alcohol]])
    if(prediction[0]>0.5):
        prediction="Best Quality"
    else:
        prediction="Low Quality"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload