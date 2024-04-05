# Housing-price-prediction

Predicting housing prices is a crucial task in real estate and finance sectors, as it enables stakeholders to make informed decisions. This report presents a comprehensive analysis of housing price prediction using machine learning techniques. Through this study, we demonstrate the potential of machine learning in assisting with property valuation.

## INTRODUCTION
In this report, we present the methodology used in a housing price prediction project utilizing TensorFlow and Keras. The primary objective of the project is to develop a predictive model that can accurately estimate housing prices based on various features associated with the properties. The project leverages the power of TensorFlow to capture complex patterns in the housing data.

## About the data
The dataset used for this project was obtained from Kaggle's House Prices dataset. The dataset has been originally divided into train and test set and each contains a collection of housing-related features, including both numerical and categorical variables, as well as the corresponding target variable, which is the sale price of the houses. There are 81 features, including our response variable “SalePrice”. Our aim in this project is to fit a model that can accurately predict the sale prices of the houses based on the given features. 

## METHODOLOGY
Exploratory Data Analysis
1. Data Cleaning:  The initial step involved identifying and handling missing values.It turned out that most of the features had no missing values, so they were left as it is.
2. Understanding the features: To better understand what kind of model to use it is important to understand the features at hand. First, the response variable ‘SalePrice’ was checked. It was found that it is normally distributed and has an average of 180921.2. Next, the numerical variables were checked for their distributions. Majority of them followed the normal distribution. Since the more important ones already followed a normal distribution, we proceeded with the next step - understanding the categorical features. A bar plot was created using OneHotEncoder to see how many unique categories each of these categorical variables have. OneHotEncoder transforms these categorical variables into a binary format, creating new binary columns for each category. It converts the different categories into as many separate binary columns, where each column corresponds to one category and contains 1 or 0 depending on whether that category is present for a particular data point. While most of them had 5 or less categories, ‘Neighbourhood’ had the highest number of unique categories.
3. Model Architecture
This dataset contains a mix of numeric, categorical, and missing features therefore, TensorFLow (decision trees-based approach) has been used, instead of simple regression, which supports all these feature types natively, and no preprocessing is required. It was used because of the following reasons:
●	TensorFlow provides high-level APIs like Keras for building and training neural networks with ease.
●	Keras is a high-level neural networks API that runs on top of various deep learning frameworks, including TensorFlow.
●	Keras makes it easy to utilize pre-trained models, such as those from the TensorFlow Hub or other sources for projects such as Housing Price Prediction
●	Keras simplifies the process of building and training neural networks, making it easier to create models that can learn from the data and make accurate predictions.
The dataset was converted from Pandas format (pd.DataFrame) into TensorFlow Datasets format (tf.data.Dataset). In this project we have used a Random Forest Model because this is the most well-known of the Decision Forest training algorithms. It’s a collection of decision trees, each trained independently on a random subset of the training dataset (sampled with replacement). 

## Training and Evaluation
Before training the dataset, 20% was manually separated for validation. Out of bag (OOB) score has been used to validate the model. The data points not used to train a particular tree or model because they were left out when forming the training set for that tree are the out of bag points and the OOB score is a way to see how well the model is performing on these "left out" data points.

## RESULTS
Performance Metrics: The model's performance was evaluated using metrics such as mean squared error (MSE) and root mean squared error (RMSE). These metrics provided insights into the accuracy and fit of the model. It was observed that the RMSE decreased with increase in the number of trees. As the number of trees increases, the overall predictive power of the ensemble tends to improve. However, after a certain point, adding more trees may lead to diminishing returns in terms of reducing RMSE on the validation or test set. This is because additional trees might contribute only marginally to the ensemble's performance and could potentially lead to overfitting. In this model’s case, the RMSE rapidly decreases at first and then almost stabilizes. Therefore, we can say that the model is performing well.


## Predictions: 
The trained model was used to make predictions on the test data, which were comparatively close to the actual sale prices. In addition to this, we found out the importance of variables using metrics such as “NUM_AS_ROOT” and “SUM_SCORE” among others and it was found that OverallQual comes out to be the most important feature whereas, GarageFinish is not so important.

## CONCLUSION
In this project, we employed TensorFlow Keras to develop a deep learning model for housing price prediction. The methodology encompassed data preprocessing, model architecture design, training, and evaluation. The results indicated that the random forest model is effectively able to predict housing prices based on the given features. Further refinements and model iterations could potentially enhance the predictive performance. 
