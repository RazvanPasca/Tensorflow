import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#dataframe
diabetes = pd.read_csv('02-TensorFlow-Basics/pima-indians-diabetes.csv')
# print(diabetes.head())

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']
#normalized columns
diabetes[cols_to_norm]=diabetes[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))
print(diabetes.head)

#we want to create feature columns and numeric columns off the columns in the dataframe

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#There are 4 groups:A,B,C,D
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
#assume we have a column with all the countries in the world in it ->we don't want to write all the countries
#or we don't know all the groupings as done below, depending on how many cats we expect
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)

#How to convert a continous column in a categorical column -- feature engineering
diabetes['Age'].hist(bins=20)
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_bucket]

#now perform the TRAIN/TEST split
x_data = diabetes.drop('Class',axis = 1) #we remove the class from the data frame
labels = diabetes['Class']
X_train,X_test,y_train,y_test = train_test_split(x_data,labels,test_size= 0.3,random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
# model.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
# results = model.evaluate(eval_input_Func)
# print(results)

#we want the predictions now
#we DON'T pass the Y value
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)
# predictions = model.predict(pred_input_func)
#we get a list of dictionaries, for each example we have a dictionary with relevant stats and class
# pred_list = list(predictions)

#LETS GO DNN MODEL
#10 layers, 10 neurons each, all connected to each others
embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4) #otherwise we get error on feature columns
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,embedded_group_col,age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, num_epochs=1, shuffle=False, batch_size=10)
print(dnn_model.evaluate(eval_input_func))