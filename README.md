# Predicting Fuel Efficiency of Vehicles:
In this series, we'd be going from data collection to deploying the Machine Learning model:

Data Collection - we are using the classic Auto MPG dataset from UCI ML Repository, I have already collected data and stored it in auto-mpg.data .
Define Problem Statement - We'll frame the problem based on the dataset description and initial exploration.
EDA - Carry our exploratory analysis to figure out the important features and creating new combination of features.
Data Preparation - Using step 4, create a pipeline of tasks to transform the data to be loaded into our ML models.
Selecting and Training ML models - Training a few models to evaluate their predictions using cross-validation.
Hyperparameter Tuning - Fine tune the hyperparameters for the models that showed promising results.
Saving Model: Using Pickle module we save our model.
Whole code is divided into two notebooks EDA.ipynb and model_prediction.ipynb. As, the name suggests , first notebook contains all the analysis of data required to make an idea and same is used in other notebook to build a model.

# Problem Statement:
Data contains MPG variable which is continuous data and tells us about the efficiency of fuel consumption of a vehicle. Our aim here is to predict the MPG value for a vehicle given we have other attributes.

In two different notebooks total solution is completed first is for EDA and second for Modelling.
First Notebook
EDA
Check for data types of columns  

Check for Null values Distribution seems to be left skewed so let's use median as impute method

median=data['Horsepower'].median()
data['Horsepower']=data['Horsepower'].fillna(median)
data.info()
Check for outliers

Look for category distribution in categorical columns As no column has dtype as object but few columns value are repetitive i.e cylinders and Origin Origin varable looks like a country code. Let's convert it by mapping it to specific values.

origin_dict={
 1:'India',
 2:'USA',
 3:'Germany'
 }
 data.head()
Plot for correlation We will use pairplots to get an intuition of potential correlations. Pairplot gives you brief overview as how variables relate to each other.  We can see that [Weight, Horsepower,Displacement] has a relation with our target variable. So, we will hypothesize that group of given attributes affect the target variables.

Checking Correaltion Matrix with MPG  We saw that [Weight, Horsepower,Displacement,Cylinders] are negatively affecting the target
Look for new variables: Displacement on power,Weight on cylinder,Acceleration on power,Acceleration on cylinder

 data['displacement_on_power'] = data['Displacement'] / data['Horsepower']
data['weight_cylinder'] = data['Weight'] / data['Cylinders']
data['acc_power'] = data['Acceleration'] / data['Horsepower']
data['acc_cylinder'] = data['Acceleration'] / data['Cylinders']

corr_matrix= data.corr()
corr_matrix['MPG'].sort_values(ascending=False)]
 Here we observe that two new features have strong effect on the target i.e acc_cylinder and acc_power. For same reason it is advisable to create new features and check their relationship with target so we can obtain more information about data.

# Second Notebook
Selecting and training models

Select and train a few algorithms.
Evaluation using mean squared error.
Model Evaluation using Cross Validation.
Hyperparameter Tuning
Check Feature Importance
Evaluate final model
Save the model
# We have created two pipelines:

numeric_pipeline:Function to process numerical transformations Argument: data->original dataframe. Returns: num_attrs-> numerical dataframe num_pipeline->numerical pipeline object

data_pipeline: Complete transformation pipeline for both numerical and categorical data. Argument: data-> original dataframe Returns: prepared_data-> transformed data, ready to use

# Traning Models:
Linear Regression  Through linear regression we get mean squared error of 2.95 which is good but still we make decision after comparing it with other models.
Decision Tree  Although the error has reduced but still we will go with one more model and compare the performance
Random Forest  Here error which we received is 0 but no model can be perfect. This means overfitting has occured. Because of similar scenario, we don't touch our test data until we are sure of the efficiency of our model.
Model Validation using Cross Validation
K-cross valiation technique divides the training data into K distinct subsets called folds, then it trains on individual fold and evaluate the model K times, picking a different fold for evaluation every time and training on other K-1 folds.

Result is an array containing the K evaluation scores. Result of all three models. 

As we can see that RandomForestRegressor provides us with the minimum error so we will continue with the same.

# Hyperparameter Tuning using GridSearchCV
Hyperparameters are parameters that are not directly learnt with the model. It is possible and recommended to search the hyper-parameter space or best cross validation score. The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter. For instance, the following param_grid

To find the names and current values for all parameters for a given estimator use:

from sklearn.model_selection import GridSearchCV

# We create 2 grids to be explored
param_grid=[
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg=RandomForestRegressor()
grid_search= GridSearchCV(forest_reg,
                          param_grid,
                          scoring='neg_mean_squared_error',
                          return_train_score=True,
                          cv=10)
grid_search.fit(prepared_data,data_label)
After, finding out the best parameter we view features importance and found that features [Weight, Model Year, Horsepower, Displacement, Cylinders] have got the larger number.
