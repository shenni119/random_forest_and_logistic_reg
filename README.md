Goal:

I tried to understand what features cause a successful tutor-student pairing/relationship. Success is a binary dependent variable: 1 if the tutor and student had 3 or more sessions together, 0 otherwise. I then had to communicate my findings to non-technical colleagues, including the sales and tutor sourcing team.

What project does (general): 
1. Data wrangling: extracting features from unstructured data, one hot encoding of select features, aggregating transactional data
2. Model training: logistic regression and random forest model that use mostly tutor features to predict if a tutor-student pairing will be successful.

Inputs (required)
1. csv file, where each row is a tutor session. Must have unique tutor id. Features include tutor's experience in a given topic, tutor's historic rating, and even unstructure multi-line text description of the tutor and student comments.

Inputs (optional)
1. csv file, additional information about tutors or clients (in case you want to control for things like the hours the client already purchased).

Outputs:
1. A text file that outlines the statistically significant features (from logistic regression) that impact the liklihood of tutor-student pairing success.
2. A text file that outlines features with the highest mean decrease impurity (most important) that impact the liklihood of tutor-student pairing success.
3. A trained logistic regression and random forest model for predicting if a tutor-student pairing will be successful. #1 and #2 are the focus of this project. 
4. Accuracy for both models.
