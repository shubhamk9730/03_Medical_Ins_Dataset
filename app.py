from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pkl
model = pkl.load(open('model_DT.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_class():
    age= request.form['age']
    sex= request.form['sex']                #{'male':1, 'female':0}
    bmi= request.form['bmi']
    children= request.form['children']
    smoker= request.form['smoker']          #{'yes':1, 'no':0}
    region= request.form['region']          #use get dummies
    project_data = {'sex': {'male':1, 'female':0},'smoker': {'yes':1, 'no':0}, 'columns': ['age','sex','bmi','children','smoker','region_northeast','region_northwest','region_southeast','region_southwest']}
    df= pd.DataFrame(columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region_northeast','region_northwest', 'region_southeast', 'region_southwest'])
    column_names = df.columns
    reg = 'region_'+region
    region_index = np.where(column_names == reg)[0][0]

    arr = np.zeros(shape=(9))
    arr[0]= age
    arr[1]= project_data['sex'][sex]
    arr[2]= bmi
    arr[3]= children
    arr[4]= project_data['smoker'][smoker]
    arr[region_index]= 1
    return str(model.predict([arr])[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070, debug=True)