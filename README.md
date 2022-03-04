# prediction-of-number-of-students
#Prediction using Supervised Machine Learning The task to predict the percentage of a student based on the number of study hours. 
#Question: What will be the predicted score if the student studies for 9.25 hours/day?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df.head()
#We will explore the dataset in two ways:
#First we will know the structure of the dataset.
#Second,the statistical properties of dataset.
#Structure
df.info()
#Statistical
df.describe()
sns.scatterplot(x="Hours", y="Scores", data=df)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Scatter Plot of Scores VS Hours")
plt.grid()
plt.show()
#We can see the two variables of scatterplot are correlated. We see the no. of hours spending in studying increases the scores obtained.
#This shows there is a linear relationship between hours and scores.
Now after relationship we obtain the model and fit it. Then we use it for predict the scores using the no. of hours spent studying. 
#This can be done using a Simple Regression Model. Hours spending on study is independent variable whereas Scores obtained is dependent Variable.

One Independent variable regression model is:

Outcome = A + B(Predictor) where a is the intercept and B is the slope of regression line.

Score = A + B*(Hours spent on studying)
# We will reshape our data divide into attributes and labels.
x=df.iloc[:,:-1]
y=df.iloc[:,1]
#Splitting the data into testing and training datasets and then training algorithm (we will use skikit-Learn)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#0.3 means 30% data used for testing
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train.values, y_train.values)
y_predicted=model.predict(x_train.values)
print("Training algorithm is complete")
A=model.intercept_
B=model.coef_
print("The regression equation is:-")
print("y=",A,"+",B,"* x")
# We use seaborn library
sns.regplot(x="Hours",y="Scores",data=df,color="Red")
plt.title("Regression Plot(Hours Vs Score)")
#Prediction for the test dataset using the model.predict()
y_predict=model.predict(x_test.values)
print("The predicted score based on the test dataset is:-",y_predict)

#Let's compare the predicted scores of the test dataset with their original scores. We will create a dataframe with two column of predict score and test score.

compare=pd.DataFrame({"Actual":y_test,"Predicted":y_predict})
compare
#We will compare with bar graph for better visualisation
compare.plot(kind='bar', color=("red","yellow"))
plt.xlabel("Test Score")
plt.ylabel("Score")
plt.title("comparison of Actual Vs Predicted")

#now we wil check the fit of regression model using the following attributes:
from sklearn import metrics
print("Mean Absolute Error:-",metrics.mean_absolute_error(y_test,y_predict))
print("Mean Squared Error (MSE):-",metrics.mean_squared_error(y_test,y_predict))
print("Root Mean Square Error (RMSE):-",np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print("R-square:-",metrics.r2_score(y_test,y_predict))

x=9.25
model.predict([[9.25]])
result=float(model.predict([[9.25]]))
print("Score:-",round(result,0))
