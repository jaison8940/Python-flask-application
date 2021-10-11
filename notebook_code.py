import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sqlalchemy import create_engine,Column, Integer, String
from sqlalchemy.orm import declarative_base,sessionmaker
import pymysql
import secrets
import csv


conn = "mysql+pymysql://{0}:{1}@{2}/{3}".format(secrets.dbuser,secrets.dbpass,secrets.dbhost,secrets.dbname)
engine = create_engine(conn, echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class tianic_dataset(Base):
    __tablename__ = 'titanic_dataset'
    id = Column(Integer, primary_key = True)
    PassengerId = Column(Integer)
    Name = Column(String(50))
    Sex = Column(String(50))
    Fare = Column(String(50))
    Embarked = Column(String(50))
    Cabin = Column(String(50))
    Ticket = Column(String(50))
    Pclass = Column(Integer)
    Age = Column(Integer)
    SibSp = Column(Integer)
    Parch = Column(Integer)
    Gender = Column(Integer)
    Survived = Column(Integer)
    
Base.metadata.create_all(engine)

with open('titanic_train.csv','r') as file:
    data = csv.DictReader(file)
    for d in data:
        row = tianic_dataset(PassengerId = d['PassengerId'],Name = d['Name'],Sex = d['Sex'],Ticket = d['Ticket'],Fare = d['Fare'],Cabin = d['Cabin'],Embarked = d['Embarked'],Pclass = d['Pclass'], Age = d['Age'], SibSp = d['SibSp'], Parch = ['Parch'], Gender = d['Sex'], Survived = d['Survived'])  
        session.add(row)
session.commit()

df = pd.read_sql('SELECT * FROM titanic_dataset',engine,index_col = 'id')
print(df.head())
df.replace(r'^\s*$', np.nan, regex=True,inplace=True)
df['Age'].replace(0, np.nan, inplace=True)

# print(df.isna().sum())
df['Age'].fillna(df['Age'].mean(),inplace=True)
df.drop('Cabin',axis=1,inplace=True)
gender=pd.get_dummies(df['Sex'],drop_first=True)
df['Gender'] = gender
df.drop(['PassengerId','Name','Sex','Ticket','Fare','Embarked'],axis=1,inplace=True)

x = df.drop('Survived',axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr=LogisticRegression()
lr.fit(x_train,y_train)
f = open('model.pkl','wb')
pickle.dump(lr,f)
f.close()
