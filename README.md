# Supervised-Learning--Linear-Regression-Polynomial-Regression-Regularization-Methods


Copy of Linear Regression
Copy of Linear Regression_

[ ]
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
quantbruce_real_estate_price_prediction_path = kagglehub.dataset_download('quantbruce/real-estate-price-prediction')

print('Data source import complete.')

📈 Linear Regression
In Machine Learning and this notebook we use Scikit-learn a lot.

آپلود عکس

What is scikit-learn used for?
Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.

What is linear regression used for?
Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.

Making Predictions with Linear Regression
Given the representation is a linear equation, making predictions is as simple as solving the equation for a specific set of inputs.

Let’s make this concrete with an example. Imagine we are predicting weight (y) from height (x). Our linear regression model representation for this problem would be:

y = B0 + B1 * x1

or

weight =B0 +B1 * height

Where B0 is the bias coefficient and B1 is the coefficient for the height column. We use a learning technique to find a good set of coefficient values. Once found, we can plug in different height values to predict the weight.

For example, lets use B0 = 0.1 and B1 = 0.5. Let’s plug them in and calculate the weight (in kilograms) for a person with the height of 182 centimeters.

weight = 0.1 + 0.5 * 182

weight = 91.1

You can see that the above equation could be plotted as a line in two-dimensions. The B0 is our starting point regardless of what height we have. We can run through a bunch of heights from 100 to 250 centimeters and plug them to the equation and get weight values, creating our line.

آپلود عکس

Now that we know how to make predictions given a learned linear regression model, let’s look at some rules of thumb for preparing our data to make the most of this type of model.

📤 Import & Install Libraries

[ ]
!pip install hvplot
Collecting hvplot
  Downloading hvplot-0.11.3-py3-none-any.whl.metadata (15 kB)
Requirement already satisfied: bokeh>=3.1 in /usr/local/lib/python3.11/dist-packages (from hvplot) (3.7.2)
Requirement already satisfied: colorcet>=2 in /usr/local/lib/python3.11/dist-packages (from hvplot) (3.1.0)
Requirement already satisfied: holoviews>=1.19.0 in /usr/local/lib/python3.11/dist-packages (from hvplot) (1.20.2)
Requirement already satisfied: numpy>=1.21 in /usr/local/lib/python3.11/dist-packages (from hvplot) (2.0.2)
Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from hvplot) (24.2)
Requirement already satisfied: pandas>=1.3 in /usr/local/lib/python3.11/dist-packages (from hvplot) (2.2.2)
Requirement already satisfied: panel>=1.0 in /usr/local/lib/python3.11/dist-packages (from hvplot) (1.6.3)
Requirement already satisfied: param<3.0,>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from hvplot) (2.2.0)
Requirement already satisfied: Jinja2>=2.9 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (3.1.6)
Requirement already satisfied: contourpy>=1.2 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (1.3.2)
Requirement already satisfied: narwhals>=1.13 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (1.37.1)
Requirement already satisfied: pillow>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (11.2.1)
Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (6.0.2)
Requirement already satisfied: tornado>=6.2 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (6.4.2)
Requirement already satisfied: xyzservices>=2021.09.1 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->hvplot) (2025.4.0)
Requirement already satisfied: pyviz-comms>=2.1 in /usr/local/lib/python3.11/dist-packages (from holoviews>=1.19.0->hvplot) (3.0.4)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.3->hvplot) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.3->hvplot) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.3->hvplot) (2025.2)
Requirement already satisfied: bleach in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (6.2.0)
Requirement already satisfied: linkify-it-py in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (2.0.3)
Requirement already satisfied: markdown in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (3.8)
Requirement already satisfied: markdown-it-py in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (3.0.0)
Requirement already satisfied: mdit-py-plugins in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (0.4.2)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (2.32.3)
Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (4.67.1)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->hvplot) (4.13.2)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from Jinja2>=2.9->bokeh>=3.1->hvplot) (3.0.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas>=1.3->hvplot) (1.17.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.11/dist-packages (from bleach->panel>=1.0->hvplot) (0.5.1)
Requirement already satisfied: uc-micro-py in /usr/local/lib/python3.11/dist-packages (from linkify-it-py->panel>=1.0->hvplot) (1.0.3)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py->panel>=1.0->hvplot) (0.1.2)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->panel>=1.0->hvplot) (3.4.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->panel>=1.0->hvplot) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->panel>=1.0->hvplot) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->panel>=1.0->hvplot) (2025.4.26)
Downloading hvplot-0.11.3-py3-none-any.whl (170 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.3/170.3 kB 7.8 MB/s eta 0:00:00
Installing collected packages: hvplot
Successfully installed hvplot-0.11.3

[ ]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

%matplotlib inline

💾 Check out the Data

[ ]
df=pd.read_csv('Real estate.csv')

[ ]
df.head()


[ ]
df.shape
(414, 8)

[ ]
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 414 entries, 0 to 413
Data columns (total 8 columns):
 #   Column                                  Non-Null Count  Dtype  
---  ------                                  --------------  -----  
 0   No                                      414 non-null    int64  
 1   X1 transaction date                     414 non-null    float64
 2   X2 house age                            414 non-null    float64
 3   X3 distance to the nearest MRT station  414 non-null    float64
 4   X4 number of convenience stores         414 non-null    int64  
 5   X5 latitude                             414 non-null    float64
 6   X6 longitude                            414 non-null    float64
 7   Y house price of unit area              414 non-null    float64
dtypes: float64(6), int64(2)
memory usage: 26.0 KB

[ ]
df.corr()


[ ]
sns.heatmap(df.corr(), annot=True,cmap='Reds')

📊 Exploratory Data Analysis (EDA)

[ ]
sns.pairplot(df)

📈 Training a Linear Regression Model
X and y arrays

[ ]
X=df.drop('Y house price of unit area', axis=1)

y=df['X4 number of convenience stores']

[ ]
print("X=",X.shape,"\ny=", y.shape)
X= (414, 7) 
y= (414,)
🧱 Train Test Split
Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.


[ ]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

[ ]
X_train.shape
(289, 7)

[ ]
X_test.shape
(125, 7)
✔️ Linear Regression

[ ]
model = LinearRegression()

[ ]
model.fit(X_train, y_train)

✔️ Model Evaluation

[ ]
model.coef_
array([-1.49344835e-17, -9.09342046e-15, -1.36338423e-16,  1.73472348e-18,
        1.00000000e+00,  1.28927721e-14,  1.08238203e-14])

[ ]
pd.DataFrame(model.coef_, X.columns, columns=['Coedicients'])

✔️ Predictions from our Model

[ ]
y_pred = model.predict(X_test)
✔️ Regression Evaluation Metrics
Here are three common evaluation metrics for regression problems:

Mean Absolute Error (MAE) is the mean of the absolute value of the errors:
1n∑i=1n|yi−y^i|
Mean Squared Error (MSE) is the mean of the squared errors:
1n∑i=1n(yi−y^i)2
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:
1n∑i=1n(yi−y^i)2−−−−−−−−−−−−√
📌 Comparing these metrics:

MAE is the easiest to understand, because it's the average error.
MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
All of these are loss functions, because we want to minimize them.


[ ]
pd.DataFrame({'Y_Test': y_test,'Y_Pred':y_pred, 'Residuals':(y_test-y_pred) }).head(5)


[ ]
MAE= metrics.mean_absolute_error(y_test, y_pred)
MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE= np.sqrt(MSE)

[ ]
MAE
4.231748536250847e-15

[ ]
MSE
2.718688400256278e-29

[ ]
RMSE
np.float64(5.214104333685967e-15)

[ ]
df['X4 number of convenience stores'].mean()
np.float64(4.094202898550725)
Residual Histogram
Often for Linear Regression it is a good idea to separately evaluate residuals
(y−y^)
and not just calculate performance metrics (e.g. RMSE).

Let's explore why this is important...

The residual eerors should be random and close to a normal distribution.

آپلود عکس

آپلود عکس


[ ]
test_residual= y_test - y_pred

[ ]
pd.DataFrame({'Error Values': (test_residual)}).hvplot.kde()


[ ]
sns.displot(test_residual, bins=25, kde=True)

Residual plot shows residual error VS. true y value.

[ ]
sns.scatterplot(x=y_test, y=test_residual)

plt.axhline(y=0, color='r', ls='--')

Residualplot showing a clear pattern, indicating Linear Regression no valid!
Finished, but you can copy this notebook and start practicing.
Colab paid products - Cancel contracts here
