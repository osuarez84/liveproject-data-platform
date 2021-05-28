import sklearn
import numpy as np
from joblib import dump, load
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict

# Define input object model
class InputItem(BaseModel):
    feature_vector: List[float]
    score: bool

# Define output object model
class OutputItem(BaseModel):
    is_inlier: bool
    anomaly_score: Optional[float] = None



clf = load('model.joblib')
app = FastAPI()

@app.post("/prediction", response_model=OutputItem)
async def prediction(item: InputItem):
    # Init the response object
    item_out = {}

    # Always compute prediction
    pred_not_bool = clf.predict(np.array(item.feature_vector).reshape(1, -1))
    item_out["is_inlier"] = 'true' if pred_not_bool == 1 else 'false'

    # If score is needed then compute score and add it to the item
    if item.score == 1:
        scores = clf.score_samples(np.array(item.feature_vector).reshape(1, -1))
        item_out["anomaly_score"] = scores

    return item_out


@app.get("/model_information")
async def model_info():
    return clf.get_params()
