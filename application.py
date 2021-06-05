from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
file_path= 'Self_LinearRegressionModel.pkl'
model=pickle.load(open(file_path,'r'))
File_path = '_All_Cities_Cleaned.csv'
house=pd.read_csv(File_path)

@app.route('/',methods=['GET','POST'])
def index():
    seller_types=sorted(house['seller_type'].unique())
    property_types=sorted(house['property_type'].unique())
    layout_types=sorted(house['layout_type'].unique())
    locality=sorted(house['locality'].unique())
    furnish_types=sorted(house['furnish_type'].unique())
    city=sorted(house['city'].unique())
    seller_types.insert(0,'Select Seller type')
    return render_template('index_1.html',seller_typess=seller_types, property_typess=property_types, layout_typess=layout_types,
    locality=locality,furnish_typess=furnish_types,city=city)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    seller_typ=request.form.get('seller_types')
    property_typ=request.form.get('property_types')
    layout_typ=request.form.get('layout_types')
    locality=request.form.get('locality')
    furnish_typ=request.form.get('furnish_types')
    city=request.form.get('city')
    bedroom=request.form.get('bedroom')
    area=request.form.get('area')
    bathroom=request.form.get('bathroom')
    
    prediction=model.predict(pd.DataFrame(columns=['seller_type','bedroom','layout_type','property_type','locality','area','furnish_type','bathroom','city'],
                              data=np.array([seller_typ,bedroom,layout_typ,property_typ,locality,area,furnish_typ,bathroom,city]).reshape(1, 9)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()
