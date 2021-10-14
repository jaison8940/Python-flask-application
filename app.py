from flask import Flask,send_file,render_template,request
import pickle
from flask_sqlalchemy import SQLAlchemy
import pymysql
import secrets
import pandas as pd
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

conn = "mysql+pymysql://{0}:{1}@{2}/{3}".format(secrets.dbuser,secrets.dbpass,secrets.dbhost,secrets.dbname)
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = conn
db = SQLAlchemy(app)
results_df = pd.read_sql('SELECT * FROM results',conn,index_col = 'id')
# print(results_df)


class results(db.Model):
    '''
    This is results class
    '''
    id = db.Column(db.Integer, primary_key = True)
    Pclass = db.Column(db.Integer)
    Age = db.Column(db.Integer)
    SibSp = db.Column(db.Integer)
    Parch = db.Column(db.Integer)
    Gender = db.Column(db.Integer)
    Prediction = db.Column(db.Integer)

# db.create_all()

fig,ax=plt.subplots(figsize=(6,6))
ax=sns.set(style="darkgrid")

@app.route('/visualize',methods=['POST','GET'])
def visualize():
    sns.countplot(x="Prediction", hue="Gender",data=results_df,palette='RdBu_r')
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='img/png')



@app.route('/',methods=['POST','GET'])
def home():    
    if request.method == 'POST':
        data = request.get_json()
        model = pickle.load(open('model.pkl','rb'))
        pclass =  int(data['pclass'])
        age = int(data['age'])
        parch = int(data['parch'])
        sibsp = int(data['sibsp'])
        gender = int(data['gender'])
        predict = model.predict([[pclass,age,sibsp,parch,gender]])
        print(predict)
        result = results(Pclass = pclass, Age = age, SibSp = sibsp, Parch = parch, Gender = gender, Prediction = int(predict[0]))
        db.session.add(result)
        db.session.commit()
        return {"result": int(predict[0])}    
    
    

    return render_template('index.html')

    

   

if __name__ == '__main__':
    app.run(debug=True)