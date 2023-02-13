# Deep-learning-for-prediction-of-car-sales-prices
Linear Regression model

This project consists of having a dataset obtained from Kaggle with car sales data, which has columns such as the car name, year, starting price, current price, kilometers, fuel, type of seller which will be dealership, private, etc., the transmission, i.e., manual or automatic, and the owner which is a binary value and it indicates that the owner is known or unknown.
All of this data will be the inputs of the neural network to be trained, and what we will do is train it so that it can then accurately predict the prices of vehicles for sale through a regression model.

Heatmaps:
Example:
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(cars_df.corr(), annot = True)
Heatmaps are plots or graphs that allow us to identify all the possible correlations that can be traced with the other data.

Difference with the other two exercises:
In this case, there are several inputs among which we will select to predict the price of a vehicle.
We must create a separate dataframe with all the data that we find convenient to use as predictor or "independent" variables, and then select which one we want to predict.

Note: It may be that when we build the x and y dataframe, the data will be on a different scale. We will notice this by selecting the x and y dataframe in the variable explorer of the environment we are using and seeing that there are numbers that are not in concordance, i.e., values that are very separated from each other. For this, we will use a tool from sklearn to scale the data. When we check the newly created dataframe again, we will have all the data ordered from 0 to 1.

Model Training:
The training phase is performed in the following lines of code:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_scaled,test_size=0.25)
What this does is divide the data into training and testing, the data that the function takes as input are x_scaled, y_scaled, where x are the input features and y the target values, test_size is a parameter that in this case is set to 0.25, which indicates that 25% of the data is used for testing and 75% for training, the output parameters are: x_train, x_test, y_train, and y_test, these are 4 arrays that represent the training data for the input features, test data for the input features, and the same for target values.

Training phase:
In this case, we have in the input the 3 columns that we selected as predictors or independent variables, we added two more layers, and the output will always be linear.

Training parameters:
Batch_size: As established at 50, it means that every 50 examples from the data set, the model updates the weights.

validation_split: Validation is a process used to evaluate a model's performance. During training, a fraction of the data is reserved to be used as validation data, in this case 20%. After each epoch, the model is evaluated on the validation data to determine its performance. This is used to avoid overfitting, which occurs when a model is too closely fit to the training data and its performance decreases on new or unknown data.
