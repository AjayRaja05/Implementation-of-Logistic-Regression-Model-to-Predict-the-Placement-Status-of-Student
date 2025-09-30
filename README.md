# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AJAYRAJA RATHINAM T
RegisterNumber: 212224240006
```
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
# 1.Placement Data


<img width="1914" height="186" alt="Screenshot 2025-09-30 110814" src="https://github.com/user-attachments/assets/002564c7-140b-4382-a984-2fb3359dbbaf" />

# 2.Salary Data


<img width="1142" height="253" alt="image" src="https://github.com/user-attachments/assets/ab739dbc-0f17-45f5-9ee2-5dd695d0152f" />

# 3.Checking null function()


<img width="171" height="576" alt="image" src="https://github.com/user-attachments/assets/4d0d8cc3-f6ed-4d01-805c-7f10ec8c5147" />

# 4.Duplicate Data


<img width="111" height="33" alt="image" src="https://github.com/user-attachments/assets/a8449c63-73c0-4da0-8e1f-e385264e2b7d" />

# 5.Print Data


<img width="1053" height="486" alt="image" src="https://github.com/user-attachments/assets/3c1b35d0-e063-423c-b54b-f7d3cc775cf3" />

# 6.Data Status


<img width="180" height="541" alt="image" src="https://github.com/user-attachments/assets/ac6eb52e-c0d1-4a3f-aa5e-5cb499a91f67" />

# 7.Prediction Array


<img width="647" height="55" alt="image" src="https://github.com/user-attachments/assets/7e262233-cf06-4433-abfc-5e00a1fedbaf" />

# 8.Accuracy Value


<img width="167" height="50" alt="image" src="https://github.com/user-attachments/assets/33d03414-4cf6-4dff-a4f8-11cd74dff61b" />

# 9.Confusion Matrix


<img width="184" height="60" alt="image" src="https://github.com/user-attachments/assets/ade79bc2-19e4-4504-97fb-8277e154e63c" />

# 10.Classification Report


<img width="545" height="205" alt="image" src="https://github.com/user-attachments/assets/e8641937-08a4-4f28-a40c-a64314c9972c" />

# 11.Prediction


<img width="1549" height="89" alt="image" src="https://github.com/user-attachments/assets/2aa4c963-ba13-4133-bc34-34084a35f434" />













## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
