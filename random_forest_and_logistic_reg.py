import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

#FIELD NAME(S) OF COLUMNS THAT CONTAINS UNSTRUCTURED TEXT INTO WHICH YOU WANT TO LOOK FOR KEYWORDS
text_field_1a="tutor_other_tutoring_comp"
text_field_1b="tutor_other_comp"
text_field_1c='director_note'

#KEYWORDS: ALL KEYWORDS BELOW WILL BECOME DUMMY VARIABLES IN THE LOGISTIC REGRESSION
# 1: THE KEYWORD IS PRESENT IN THE COLUMNS NOTED ABOVE FOR THE ROW, 0=KEYWORD NOT PRESENT
key_word_lst=['kaplan','teach for america',\
    'princeton review','chegg','wyzant','sylvan','spanish teacher',\
    'classroom experience','french teacher','math teacher',\
    'highly recommend','very patient']

#LINE 23-173 IS SPENT CLEANING AND PREPPING THE FINAL DATASET FOR THE LOGIT REGRESS- SKIP TO LINE 169 FOR THE REGRESSION/TESTING CODE
filename='online_sessions_103.csv'
filename2='client_purchase_hrs.csv'
filename3='online_sessions_prior_exp_FINAL.csv'
filename4='timeliness2.csv'
reference_file5='tutors.csv'


pd_filename3 = pd.read_csv(filename3)
regressfile1='tutor_info_FINAL.csv'
regressfile2='test_scores.csv'

#IMPORT RELEVANT CSVS
pd_tutor_info = pd.read_csv(regressfile1,sep='\t')\
    .rename(columns={'tutors.tutor_id':'tutoring_sessions.tutor_id'})
pd_punctuality = pd.read_csv(filename4,sep='\t')\
    .rename(columns={'tutor_id':'tutoring_sessions.tutor_id'})
pd_testscores = pd.read_csv(regressfile2)\
    .rename(columns={'Tutors Tutor ID':'tutoring_sessions.tutor_id'})

#DELINEATING A VALUE IN ONE CELL THAT SHOULD BE IN 3
pd_filename3[['prior_exp_all', 'prior_exp_subject', 'fivestar_comment_percent']]\
    = pd_filename3['incre_values'].str.split('|', expand=True)
string_to_numeric=['prior_exp_all','prior_exp_subject','fivestar_comment_percent']
for col in string_to_numeric:
    pd_filename3[col] = pd.to_numeric(pd_filename3[col], errors='coerce')
pd_filename3['fivestar_comment_percent']=pd_filename3['fivestar_comment_percent']*100

pd_filename3['tutor_plan_substantial20']= np.where(\
    pd_filename3['tutor_plan_wrd_count']>19\
    ,1,0)
pd_punctuality["tutor_pct_10min_orMore_late"]=\
    (pd_punctuality["tutor_pct_10_minutes_late"]+pd_punctuality["tutor_pct_15_minutes_late"])


#TRANSFORMING TRANSACTIONAL DATA TO AGGREGATE DATA FOR REGRESSION
tutor_client_pair=pd_filename3.groupby(['tutoring_sessions.tutor_id',\
    'tutoring_sessions.client_id']).agg({\
    'tutoring_sessions.session_id': 'count',\
    'tutoring_sessions.subject': 'first',\
    'tutor_avg_rating':'mean',\
    'tutoring_sessions.instant_flag':'first',\
    '5_star_client_note_substantial': 'max',\
    'tutor_plan_substantial20': 'max',\
    'tutoring_sessions.client_rating': 'mean',\
    'Credits First Purchase Hours':'max',\
    'prior_exp_all':'first',\
    'prior_exp_subject':'first',\
    'fivestar_comment_percent':'first'}).\
    rename(columns={'tutoring_sessions.session_id':'pair_session_count',\
    'tutoring_sessions.subject':'subject'}).\
    reset_index()

pair_merged_a=pd.merge(tutor_client_pair,pd_tutor_info,\
    on='tutoring_sessions.tutor_id', how='left')
pair_merged_b=pd.merge(pair_merged_a,pd_punctuality,\
    on='tutoring_sessions.tutor_id', how='left')
pd_tutor_director_notes = pd.read_csv(reference_file5,sep='\t',error_bad_lines=False)\
    .rename(columns={'tutors.tutor_id':'tutoring_sessions.tutor_id'})
pd_tutor_director_note_shorten=pd_tutor_director_notes[\
    ['tutoring_sessions.tutor_id','director_note']]
pair_merged_c=pd.merge(pair_merged_b,pd_tutor_director_note_shorten,\
    on='tutoring_sessions.tutor_id', how='left')
pair_merged=pd.merge(pair_merged_c,pd_testscores,\
    on='tutoring_sessions.tutor_id', how='left')

#STILL CREATING DATASET - COMBINING RELEVANT UNSTRUCTURED TEXT COLS INTO ONE COLUMN
pair_merged["text_field_1"] = pair_merged[text_field_1a]+\
    " "+ pair_merged[text_field_1b]+ ' '+pair_merged[text_field_1c]

prior_exp_lst=[]
for word in key_word_lst:
    new_col='{}_binary'.format(word)
    pair_merged[new_col]=np.where(\
        pair_merged["text_field_1"].str.contains(\
        word, na=False),\
        1,0)
    tutors_grouped=pair_merged.groupby('tutoring_sessions.tutor_id')\
        .agg({new_col: 'first'})
    Total = tutors_grouped[new_col].sum()
    # print ('tally of tutors with prior exp in {}: {}'.format(word,Total))
    prior_exp_lst.append(new_col)

teacher_tag_list=['Early Childhood','Elementary School','Middle School',
    "Math","Currently Teaching","Secondary School","Science"]

pair_merged['teacher_flag_binary']=np.where(\
    pair_merged['tutor_flags_concat'].isin(teacher_tag_list),\
    1,0)

tutor_client_pair=pair_merged
dep_var='successful_pair'
tutor_client_pair[dep_var]=np.where(\
    tutor_client_pair['pair_session_count']>=3\
    ,1,0)
tutor_client_pair['instant_flag']= np.where(\
    tutor_client_pair['tutoring_sessions.instant_flag']=='Instant Flag'\
    ,1,0)

ind_var_a=[\
    'tutor_avg_rating',\
    'prior_exp_all','prior_exp_subject', 'fivestar_comment_percent',\
    'tutor_plan_substantial20','Credits First Purchase Hours','teacher_flag_binary',\
    'instant_flag']
ind_var0=prior_exp_lst+ind_var_a

#CREATING DUMMY VARIABLES FROM FIELD WITH CATEGORICAL VARIABLES
dummy_categorical_field='subject'
dummies = pd.get_dummies(tutor_client_pair[dummy_categorical_field])
data_regress=tutor_client_pair.join(dummies)

#REMOVING ALL ROWS THAT DOESN'T INVOLVE THE DUMMY VARIABLES WE ARE INTERESTED IN
categorical_var_dummies=["SAT","PSAT","PSAT Critical Reading",\
    "PSAT Mathematics","PSAT Writing Skills","SAT Math",\
    "SAT Reading","SAT Writing and Language",\
    "ACT","ACT English","ACT Math","ACT Reading",\
    "ACT Science","ACT Writing",\
    "GRE",'GRE Quantitative','GRE Verbal','GRE Analytical Writing',\
    "Algebra","Algebra 2","Geometry","Pre-Algebra","Pre-Calculus",\
    "Spanish","Spanish 1","Conversational Spanish","Spanish 2",\
    "French","French 1","Conversational French"]
identifier_fields=['tutoring_sessions.tutor_id','tutoring_sessions.client_id']
ROW_FILTER=ind_var0+categorical_var_dummies
data_regress_ROW_FILTER=data_regress\
    [data_regress['subject'].isin(ROW_FILTER)]

#PREPPING FINAL DF FOR REGRESS: REMOVING 1 ADDITIONAL FIELD FOR COLINEARITY
colin_remove_var="Conversational French"
colin_categorical_var_dummies= [x for x in categorical_var_dummies if x != colin_remove_var]
ind_var_regress=ind_var0+colin_categorical_var_dummies
data_regress_COL_ROW_FILTER=data_regress_ROW_FILTER[\
    identifier_fields+ind_var_regress+[dep_var]]

#CUSTOM FILTERS FOR REGRESS DATA
custom_filter='prior_exp_all'
filename_session_min=5
data_regress_CUSTOM_COL_ROW_FILTER = data_regress_COL_ROW_FILTER[\
    data_regress_COL_ROW_FILTER[custom_filter] >= \
    filename_session_min]

#FINAL FILTERS FOR NULL OR INF VALUES
data_regress_FINAL=data_regress_CUSTOM_COL_ROW_FILTER[\
    ~data_regress_CUSTOM_COL_ROW_FILTER.isin([np.nan, np.inf, -np.inf]).any(1)]
data_regress_FINAL.to_csv('{}_logit_data.csv'\
    .format(custom_filter))

###############################lOGISTIC REGRESSION#############################
#TESTING ACCURACY OF LOGISTIC REGRESSION
X_train, X_test, y_train, y_test = train_test_split(\
    data_regress_FINAL[ind_var_regress],\
    data_regress_FINAL[dep_var], test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print ('__________________Logistic Regression Results__________________')
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# CONFUSION MATRIX
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
# print(classification_report(y_test, y_pred))

#LOGISTIC REGRESSION FEATURES COEFFICIENTS
print ('__________________Logistic Regression Feature Coeffients__________________')
logit_model=sm.Logit(data_regress_FINAL[dep_var],\
    data_regress_FINAL[ind_var_regress].astype(float))
result=logit_model.fit()
#LOGISTIC REG COEFFICIENT
print(result.summary())
#LOGISTIC REG COEFFICIENT CONVERTED INTO ODD RATIOS
# print(np.exp(result.params))

# #EXPORTING LOGISTIC REG RESULTS
# text_file1 = open("{}_logit_output.txt".format(custom_filter), "w")
# text_file1.write(str(result.summary()))
# text_file1.close()
# #LOGISTIC REGRESSION CREATING TXT FILE THAT EXPLAIN COEFFICIENTS IN MORE UNDERSTANDABLE TERMS
# text_file2 = open("{}_odds_ratio.txt".format(custom_filter), "w")
# text_file2.write(str(np.exp(result.params)))
# text_file2.close()

####################RANDOM FOREST MODEL########################
#SPLITTING DATA INTO TRAINING AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test \
    = train_test_split(data_regress_FINAL[ind_var_regress]\
    , data_regress_FINAL[dep_var], test_size = 0.3, random_state = 20)

indep_var_list = list(data_regress_FINAL[ind_var_regress].columns)

#Getting baseline projection based on average successful matches
baseline_projection_series = y_test
baseline_projection_df=baseline_projection_series.to_frame()
baseline_projection_df['baseline_projection']=baseline_projection_series.mean()
baseline_err = abs(baseline_projection_df['baseline_projection'] - y_test)
print ('__________________Random Forest Results__________________')
print('Average baseline accuracy: ', 1-round(np.mean(baseline_err), 3))

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 1000, random_state = 20)
rf_model.fit(X_train, y_train)

#PREDICT OUTCOME USING RANDOM FOREST
rf_prediction = rf_model.predict(X_test)

rf_error = abs(rf_prediction - y_test)
print('Average random forest accuracy:', 1-round(np.mean(rf_error), 3))

print ('____________________Random Forest Feature Importance______________________')

for name, importance in zip(indep_var_list, rf_model.feature_importances_):
    print(name, "=", importance)
