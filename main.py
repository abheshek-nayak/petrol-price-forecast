import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI, File, UploadFile, Request, Response, HTTPException
from fastapi.templating import Jinja2Templates
from io import BytesIO
import uvicorn
import numpy as np


app = FastAPI()

#. Load trained Pipeline
my_model = load_model('model')
templates = Jinja2Templates(directory='templates')

@app.get('/')
async def func(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})

@app.post('/petrol_price')
async def create_upload_file(request: Request,file: UploadFile = File(...)):

    try:
        contents = file.file.read()
        buffer = BytesIO(contents)
        df = pd.read_csv(buffer)
    except:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        buffer.close()
        file.file.close()
    
    
    df['Date'] =pd.to_datetime(df['Date'])
    df['month'] = [i.month for i in df['Date']]
    df['year'] = [i.year for i in df['Date']]
    df['week'] = [i.week for i in df['Date']]
    df['Series'] = np.arange(1,len(df)+1)
    forecast_df = predict_model(my_model,df)
    result = forecast_df.drop(['Prediction', 'month', 'year', 'week', 'Series'],axis=1)
    result.rename(columns={'Label':'Forecast','Date':'Date(YYYY-DD-MM)'},inplace=True)
    
    return templates.TemplateResponse('home.html',{'request': request, 'data': result.to_html()})
    

    


if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)