# TANZANIA WATER WELLS PREDICTION

![image](https://github.com/josephinemaro/Phase3-Project/assets/162449289/813a5ac3-2232-4b5c-9506-715c288e1ed7)


## Overview
This project aims to analyze data from Tanzanian water wells to identify patterns and factors that contribute to well functionality status. By developing predictive models and segmenting wells based on their characteristics and performance, we aim to provide actionable insights to improve well maintenance, optimize resource allocation, and enhance water access. These insights will support decision-making processes to ensure the sustainability and efficiency of water supply systems.
## Business Understanding
### Problem Statement
In Tanzania, maintaining the functionality of water wells is critical for ensuring sustainable access to clean water. Through this prediction, stakeholders can prioritize maintenance efforts, allocate resources efficiently, and safeguard access to clean water for local communities.
### Objective
Our goal is to develop a robust predictive model that accurately identifies patterns and factors that contribute to well functionality status. 
### Stakeholders
Tanzanian government agencies for water resource management, NGOs, Local communities
### Business Impact
Resource allocation. 
Mitigate the risk of waterborne diseases
Economic development 
Ensuring uninterrupted water supply for agriculture and other activities.

## Data Understanding
### Dataset
The dataset used in this analysis comprises information about existing water wells in Tanzania. The dataset comprises 59,400 entries distributed across 40 columns. Among these, 31 columns are identified as categorical, while 9 are numeric. There are four CSV files provided: "Training set values" includes data on independent features for the training set, "Training set labels" contains information on the dependent variable, "Test set values" comprises values for prediction, and "Submission format" is furnished for adhering to the required format of the results in the context of a data science competition.
### Data Analysis
#### Data Selection:
To identify the key characteristics influencing water well performance, I conducted a thorough analysis employing techniques such as exploratory data analysis, correlation analysis, and domain expertise. Through this process, I identified a subset of attributes with significant correlations to successful outcomes while eliminating redundant or unnecessary variables.

#### Data Cleaning:
Certain features exhibited similarities. To reduce dimensionality and mitigate multicollinearity concerns, I retained the most general columns from each group. Validity checks were then performed to identify duplicate values and outliers in the dataset. Duplicated records were retained since they did not necessarily represent identical wells but may have been constructed under the same project. Following data cleaning, efforts were made to ensure uniformity by formatting and enhancing the readability of columns. Custom functions were defined to facilitate these tasks.
## Modeling
We employed various machine learning techniques to predict water well functionality status based on provided data. 
Each model underwent training on the training set and subsequent evaluation on the testing set to gauge its predictive accuracy and generalization ability.
### Models Used
Logistic regression served as our initial baseline model, providing a probabilistic interpretation of predictions with an accuracy of approximately 66.4%. Following this, we explored the K-Nearest Neighbors (KNN) algorithm, achieving an accuracy of approximately 67.2%. 
Decision Tree and Random Forest classifiers were also employed, with the Decision Tree model yielding the highest accuracy of approximately 71.0%. Random Forest model achieved an accuracy of approximately 68.0%.
### Hyperparameter Tuning
We employed hyperparameter tuning techniques, particularly using GridSearchCV, to systematically explore a range of hyperparameters and identify the optimal configuration for our models. In the random forest classifier, we defined a grid of hyperparameters including criteria for splitting, maximum depth, maximum number of features, and the number of estimators. GridSearchCV then exhaustively searched through this parameter grid, evaluating each combination through cross-validation to determine the set of hyperparameters that yielded the best performance.
## Evaluation
The predictiction of the functionality status of Tanzanian water wells incorporated various machine learning models. Four models were evaluated: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest. The evaluation metrics used were accuracy, precision, recall, F1-score, and root mean squared error (RMSE).

Logistic Regression: Achieved an accuracy of approximately 66.4%. It performed moderately well in identifying functional and non-functional wells but struggled with wells needing repair, with a low F1-score for this class.

K-Nearest Neighbors (KNN): Achieved an accuracy of approximately 67.2%. Similar to Logistic Regression, it performed moderately well for functional and non-functional wells but struggled with wells needing repair, resulting in a low F1-score for this class.
![image](https://github.com/josephinemaro/Phase3-Project/assets/162449289/8e9985f4-d06a-4c26-a2a2-45140ce30e7d)

Decision Tree: Outperformed the other models, achieving an accuracy of approximately 71.0%. It performed better in identifying functional and non-functional wells but still faced challenges in accurately predicting wells needing repair, with a low F1-score for this class.

Random Forest: Achieved an accuracy of approximately 68.0%. While it performed well for functional and non-functional wells, it struggled significantly with wells needing repair, resulting in a low F1-score for this class.

![image](https://github.com/josephinemaro/Phase3-Project/assets/162449289/dac8245f-fcf7-4e87-8557-13bd22ba4edf)

The bar graph represents the feature importances from the tuned Random Forest model used for predicting water wells functionality. Each bar corresponds to a different feature from the dataset, with the height indicating the importance of that feature in making predictions.

## Conclusion
Overall, the Decision Tree model showed the best performance among the models evaluated, with the highest accuracy and better precision and recall for functional and non-functional wells. However, all models struggled to accurately predict wells needing repair, indicating a need for further improvement in identifying and addressing maintenance issues. The project highlights the importance of using machine learning models to predict water well functionality, which can help stakeholders prioritize maintenance and repair efforts to ensure sustainable access to clean water.
