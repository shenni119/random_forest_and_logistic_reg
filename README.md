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
1. Printed text, outlining the statistically significant features (from logistic regression) that impact the liklihood of tutor-student pairing success.
2. Printed text, ranking features with the highest mean decrease impurity (most important, but doesn't provide statistical significance or directionality) that impact the liklihood of tutor-student pairing success.
3. A record of the final data that was used, called "prior_exp_all_logit_data.csv"
4. A trained logistic regression and random forest model for predicting if a tutor-student pairing will be successful.
5. Accuracy for both models.

Example of the output is below (all academic subject features below can be ignored. They are there to prevent overfitting because subject variation naturally tend to lead to shorter or longer tutor-student pairings):

__________________Logistic Regression Results__________________

Accuracy of logistic regression classifier on test set: 0.76

__________________Logistic Regression Feature Coeffients__________________
Warning: Maximum number of iterations has been exceeded.
         Current function value: 0.490332
         Iterations: 35
                           Logit Regression Results
==============================================================================
Dep. Variable:        successful_pair   No. Observations:                15781
Model:                          Logit   Df Residuals:                    15732
Method:                           MLE   Df Model:                           48
Date:                Mon, 14 Jan 2019   Pseudo R-squ.:                  0.2924
Time:                        20:17:07   Log-Likelihood:                -7737.9
converged:                      False   LL-Null:                       -10936.
                                        LLR p-value:                     0.000
================================================================================================
                                   coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------
kaplan_binary                   -0.4373      0.510     -0.857      0.392      -1.438       0.563
teach for america_binary         9.6806    105.377      0.092      0.927    -196.854     216.215
princeton review_binary          0.5729      0.268      2.140      0.032       0.048       1.097
chegg_binary                    -0.0603      0.440     -0.137      0.891      -0.922       0.802
wyzant_binary                    0.4637      0.195      2.379      0.017       0.082       0.846
sylvan_binary                   -0.1323      0.181     -0.732      0.464      -0.486       0.222
spanish teacher_binary          24.2153   1.86e+05      0.000      1.000   -3.65e+05    3.65e+05
classroom experience_binary     -0.0662      0.737     -0.090      0.928      -1.511       1.379
french teacher_binary           14.2231   2690.582      0.005      0.996   -5259.220    5287.666
math teacher_binary              0.8677      0.284      3.052      0.002       0.310       1.425
highly recommend_binary         -0.0661      0.372     -0.178      0.859      -0.795       0.662
very patient_binary              0.2098      0.547      0.383      0.702      -0.863       1.282
tutor_avg_rating                 0.2851      0.050      5.666      0.000       0.187       0.384
prior_exp_all                   -0.0008      0.000     -4.027      0.000      -0.001      -0.000
prior_exp_subject                0.0060      0.001      9.688      0.000       0.005       0.007
fivestar_comment_percent         0.0050      0.001      4.172      0.000       0.003       0.007
tutor_plan_substantial20         0.2149      0.053      4.079      0.000       0.112       0.318
Credits First Purchase Hours     0.0104      0.001      7.642      0.000       0.008       0.013
teacher_flag_binary              0.0600      0.053      1.139      0.255      -0.043       0.163
instant_flag                    -3.2500      0.082    -39.468      0.000      -3.411      -3.089
...
================================================================================================

__________________Random Forest Results__________________

Average baseline accuracy:  0.501
Average random forest accuracy: 0.704

____________________Random Forest Feature Importance______________________

                                importance
instant_flag                  3.083213e-01
prior_exp_subject             1.608916e-01
prior_exp_all                 1.282443e-01
fivestar_comment_percent      1.063064e-01
tutor_avg_rating              9.671205e-02
Credits First Purchase Hours  7.195228e-02
tutor_plan_substantial20      1.370608e-02
teacher_flag_binary           1.155626e-02
...
wyzant_binary                 1.651849e-03
sylvan_binary                 1.627205e-03
...
princeton review_binary       1.012056e-03
SAT Reading                   8.513859e-04
very patient_binary           7.015285e-04
math teacher_binary           6.830091e-04
highly recommend_binary       6.736205e-04
French                        6.076404e-04
Spanish                       5.112869e-04
GRE Analytical Writing        4.916097e-04
chegg_binary                  3.584503e-04
PSAT Writing Skills           3.348056e-04
classroom experience_binary   2.959832e-04
kaplan_binary                 2.670603e-04
PSAT Critical Reading         2.554120e-04
ACT Writing                   2.513988e-04
PSAT Mathematics              2.396513e-04
teach for america_binary      4.492313e-05
spanish teacher_binary        1.333828e-06
french teacher_binary         3.019507e-07

