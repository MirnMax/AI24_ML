import pickle
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel


class FeatureSet(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: float


def medinc_regressor(x: dict) -> dict:
    with open("final_model.sav", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open("ohe.sav", 'rb') as ohe_file:
        ohe = pickle.load(ohe_file)
        
    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    x_df = pd.DataFrame(x, index=[0])
    x_df_ohe = x_df.drop(columns=cat_cols).copy()
    x_df_ohe[ohe.get_feature_names_out()] = ohe.transform(x_df[cat_cols])
    
    res = loaded_model.predict(x_df_ohe)[0]
    return {"prediction": res}

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["medinc_regressor"] = medinc_regressor
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict")
async def predict(feature_set: FeatureSet):
    return ml_models["medinc_regressor"](feature_set.model_dump())