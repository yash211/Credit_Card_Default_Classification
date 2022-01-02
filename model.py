import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedStratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE

train_df=pd.read_csv('dataset/train.csv')
test_df=pd.read_csv('dataset/test.csv')

print(train_df.dtypes)
df_des=train_df.describe() #Need to standardize,missing values

#dropping name
train_df=train_df.drop(columns="name",axis=1)
test_df=test_df.drop(columns="name",axis=1)
#checking missing values
print(train_df.isnull().sum())

 
#replacing nan with Yes for owning car
train_df["owns_car"]=train_df["owns_car"].replace(np.nan,"Y")

def visual_bars(col):
    col_counts=train_df[col].value_counts()
    plt.bar(col_counts.index,col_counts,color=['green','red','purple','blue'],width=0.5)
    plt.xlabel(col)
    plt.ylabel("Percentage distribution of " + col)
    plt.title("Counts of data points")
    plt.show()

visual_bars("occupation_type")
visual_bars("owns_car")
visual_bars("no_of_children")
visual_bars("total_family_members")
visual_bars("prev_defaults")
visual_bars("default_in_last_6months")
visual_bars("credit_card_default")

#replacing nan with Yes for owning car
train_df["owns_car"]=train_df["owns_car"].replace(np.nan,"Y")

#replacing nan with 0 because its most occuring,later on will see for overfit
train_df["no_of_children"]=train_df["no_of_children"].replace(np.nan,0)

#no_of_days_employed
train_df["no_of_days_employed"]=train_df["no_of_days_employed"].replace(np.nan,train_df["no_of_days_employed"].mean())


#replacing with mean family members
train_df["total_family_members"]=train_df["total_family_members"].replace(np.nan,train_df["total_family_members"].median())

#migrant worker
print(train_df["migrant_worker"].value_counts())
train_df["migrant_worker"]=train_df["migrant_worker"].replace(np.nan,0)
#yearly_debt_payments
train_df["yearly_debt_payments"]=train_df["yearly_debt_payments"].replace(np.nan,train_df["yearly_debt_payments"].mean())
#credit_score
train_df["credit_score"]=train_df["credit_score"].replace(np.nan,train_df["credit_score"].mean())

print(train_df.isnull().sum()==0)  #handled missing values

#removing customer_id
train_df=train_df.drop(columns="customer_id",axis=1)

#handling categorical variables-One hot encoding for occupation_type and label for only columns cotaining 2 values 
print(train_df["gender"].value_counts()) 
train_df["gender"]=train_df["gender"].replace('XNA','F')
print(train_df["owns_car"].value_counts())
#train_df["owns_car"]=train_df["owns_car"].map({'N':1,'Y':0})
owns_car=pd.get_dummies(train_df['owns_car'],drop_first=True)
gender=pd.get_dummies(train_df['gender'],drop_first=True)
#train_df["gender"]=train_df["gender"].map({'F':1,'M':0})
print(train_df["owns_house"].value_counts())
owns_house=pd.get_dummies(train_df['owns_house'],drop_first=True)
#train_df["owns_house"]=train_df["owns_house"].map({'Y':1,'N':0})
types=pd.get_dummies(train_df['occupation_type'],drop_first=True)
#aligning columns 
target=train_df["credit_card_default"]
train_df=train_df.drop(columns='credit_card_default',axis=1)

train_df=pd.concat([train_df,gender,owns_car,owns_house,types, target], axis = 1)

#From data description,we can see there are poosible outliers in dataset

#print(ffeatures)
def outlier_vis(feature):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(feature)
    plt.show()
#No outliers columns are-age,credit_limit_used,credit_score
#One or more but less than maximum outlier columns are-net_yearly_income,employed,migrant worker,credit_limit,prev and last six month default
outlier_col=['no_of_children','net_yearly_income','no_of_days_employed','total_family_members','yearly_debt_payments','credit_limit']
for col in outlier_col:
    outlier_vis(train_df[col])
    
outlier_vis(test_df['default_in_last_6months'])

#Normalizing
child_sqrt=np.sqrt(train_df['no_of_children'])
print(child_sqrt.skew())   #1.0958.......i can still improve

outlier_vis(child_sqrt)

#using yeo
#no_of_children
print(train_df['no_of_children'].skew())
# train_df['no_of_children'] = np.where(train_df['no_of_children'] > 3,1 , train_df['no_of_children'])  ..1.38->skewness
pow=PowerTransformer(method='yeo-johnson', standardize=True)
child=train_df['no_of_children'].values
child=child.reshape(-1,1)
child_yeo = pow.fit_transform(child)
outlier_vis(child_yeo)
sns.displot(child_yeo,kde=True)
train_df['no_of_children']=child_yeo
print(train_df['no_of_children'].skew())

#total_family_members
family_sqrt=np.sqrt(train_df['total_family_members'])
train_df['total_family_members']=family_sqrt
print(family_sqrt.skew())
outlier_vis(family_sqrt)

#yearly_debt_payments
print(train_df['yearly_debt_payments'].quantile(0.50)) 
print(train_df['yearly_debt_payments'].quantile(0.95)) 
train_df['yearly_debt_payments'] = np.where(train_df['yearly_debt_payments'] > 62784.5215,29122.265 , train_df['yearly_debt_payments'])
outlier_vis(train_df['yearly_debt_payments'])
print(train_df['yearly_debt_payments'].skew())

#net_yearly_income
print(train_df['net_yearly_income'].quantile(0.50))
print(train_df['net_yearly_income'].quantile(0.95))
train_df['net_yearly_income'] = np.where(train_df['net_yearly_income'] > 352349.00, 171714.91, train_df['net_yearly_income'])
outlier_vis(train_df['net_yearly_income'])
sns.displot(train_df['net_yearly_income'],kde=True)
print(train_df['net_yearly_income'].skew())

#no_of_days_employed
print(train_df['no_of_days_employed'].skew())
print(train_df['no_of_days_employed'].quantile(0.95))
train_df['no_of_days_employed'] = np.where(train_df['no_of_days_employed'] > 6100, 3000, train_df['no_of_days_employed'])
outlier_vis(train_df['no_of_days_employed'])
sns.displot(train_df['no_of_days_employed'],kde=True)
print(train_df['no_of_days_employed'].skew())

#credit_limit
print(train_df['credit_limit'].skew())
print(train_df['credit_limit'].quantile(0.50))
print(train_df['credit_limit'].quantile(0.95))
train_df['credit_limit'] = np.where(train_df['credit_limit'] > 80300, 35600, train_df['credit_limit'])
#climit_sqrt=np.log(train_df['credit_limit'])
outlier_vis(train_df['credit_limit'])
sns.displot(train_df['credit_limit'],kde=True)
print(train_df['credit_limit'].skew())

#Defaults
# pow1=PowerTransformer(method='yeo-johnson', standardize=True)
# pow6=PowerTransformer(method='yeo-johnson', standardize=True)
# prevD=train_df['prev_defaults'].values.reshape(-1,1)
# prev6D=train_df['default_in_last_6months'].values.reshape(-1,1)
# prevD_yeo=pow1.fit_transform(prevD)
# prev6D_yeo=pow6.fit_transform(prev6D)
# train_df['prev_defaults']=prevD_yeo
# train_df['default_in_last_6months']=prev6D_yeo
# print(train_df['prev_defaults'].skew(),train_df['default_in_last_6months'].skew())

#print(pdef.skew())
print(train_df['prev_defaults'].skew(),train_df['default_in_last_6months'].skew())

#---------------Handling Imbalance------------------
future_train_df=train_df.copy()
train_df=train_df.drop(columns=["occupation_type","gender","owns_car","owns_house"],axis=1)
#train_df=train_df.drop(columns=['default_in_last_6months'],axis=1)

plt.figure(figsize=(20,20))
sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':9})

sm_X=train_df.drop(columns=['credit_card_default']).values#
#imp=['credit_score','prev_defaults','default_in_last_6months']
#sm_newX=train_df[imp].values
sm_y=train_df['credit_card_default'].values
sm_y=sm_y.reshape(-1,1)


X_strain, X_stest, y_strain, y_stest = train_test_split(sm_X, sm_y, test_size=0.2, random_state=42)
print(X_strain.shape)
sm = SMOTE(random_state=27)
X_smotetrain, y_smotetrain = sm.fit_resample(X_strain, y_strain)

#--------------Scaling
scale=StandardScaler()
scaledX=scale.fit_transform(sm_X)
X_train, X_test, y_train, y_test = train_test_split(scaledX, sm_y, test_size=0.25, random_state=27)
print(X_strain.shape)
scaled_sm = SMOTE(random_state=27)
X_smoteStrain, y_smoteStrain = scaled_sm.fit_resample(X_train, y_train)

#----------------------------------------------Model Building-----------------------------

# slrc=LogisticRegression()
# slrc.fit(X_smoteStrain,y_smoteStrain)
# pred_slrc=slrc.predict(X_test)
# print("Logistic regression Accuracy",accuracy_score(y_test,pred_slrc))          #overfitting

# # Function
# # error_rate = []

# # for i in range (1,40):
# #     sknn = KNeighborsClassifier(n_neighbors = i)
# #     sknn.fit(X_smoteStrain, y_smoteStrain)
# #     pred_sknn = sknn.predict(X_test)
# #     error_rate.append(np.mean(pred_sknn != y_test))

# # # Plot error rate
# # plt.figure(figsize = (10,6))
# # plt.plot(range(1,40), error_rate, color = 'blue', linestyle = '--', marker = 'o', 
# #         markerfacecolor = 'green', markersize = 10)

# # plt.title('Error Rate vs K Value')
# # plt.xlabel('K')
# # plt.ylabel('Error Rate')
# # plt.show()
# # sknn=KNeighborsClassifier(n_neighbors=sqrt(X_smoteStrain.shape[0]*X_smoteStrain.shape[1]))
# # sknn.fit(X_smoteStrain,y_smoteStrain)
# # pred_sknn=sknn.predict(X_test)
# # print("K nearest Neighbours Accuracy",accuracy_score(y_test,pred_sknn))          #neigh->1=0.9615,2->0.9648,4->0.9533,3->0.94

##-------------------------RF on Scaled
srfc=RandomForestClassifier(n_estimators=500)
srfc.fit(X_smoteStrain,y_smoteStrain)
pred_srfc=srfc.predict(X_test)
print("Random Forest Scaled Accuracy",accuracy_score(y_test,pred_srfc))

##-------------------------LinearSVC on scaled
slsvc=LinearSVC()
slsvc.fit(X_smoteStrain,y_smoteStrain)
pred_slsvc=slsvc.predict(X_test)
print("Linear SVC Accuracy",accuracy_score(y_test,pred_slsvc))


##--------------------------Decision Tree
dsm=DecisionTreeClassifier()
dsm.fit(X_smotetrain,y_smotetrain)
pred_dec=dsm.predict(X_stest) 
print("Decision Tree Accuracy",accuracy_score(y_stest,pred_dec))   #97%

##--------------------------RandomForest
rfc=RandomForestClassifier()
rfc.fit(X_smotetrain,y_smotetrain)
pred_rfc=rfc.predict(X_stest)
print("Random Forest Accuracy",accuracy_score(y_stest,pred_rfc))  #0.9811..without tuning
# rfc_impF=rfc.feature_importances_
# c=0
# for col in train_df.columns:
#     print(col+"   ->",rfc_impF[c])
#     c=c+1

##--------------------------Naive Bayes
# nbc=MultinomialNB()
# nbc.fit(X_smotetrain,y_smotetrain)
# pred_nbc=nbc.predict(X_stest)
# print("Naive Bayes Classifier Accuracy",accuracy_score(y_stest,pred_nbc))   #0.24

#------------------------Logistic Regression
lrc=LogisticRegression()
lrc.fit(X_smotetrain,y_smotetrain)
pred_lrc=lrc.predict(X_stest)
print("Logistic regression Accuracy",accuracy_score(y_stest,pred_lrc))   #92%

##-----------------------Linear SVC
lsvc=LinearSVC()
lsvc.fit(X_smotetrain,y_smotetrain)
pred_lsvc=lsvc.predict(X_stest)
print("Linear SVC Accuracy",accuracy_score(y_stest,pred_lsvc))           #90%

# ##----------------------SVC
# svc=SVC()
# svc.fit(X_smotetrain,y_smotetrain)
# pred_svc=svc.predict(X_stest)
# print("SVC Accuracy",accuracy_score(y_stest,pred_svc))                   #51%

##----------------------GradientBoosting'
gbc=GradientBoostingClassifier()
gbc.fit(X_smotetrain,y_smotetrain)
pred_gbc=gbc.predict(X_stest)
print("Gradient Boosting Accuracy",accuracy_score(y_stest,pred_gbc))     #97.92

##----------------------AdaBoosting
abc=AdaBoostClassifier()
abc.fit(X_smotetrain,y_smotetrain)
pred_abc=abc.predict(X_stest)
print("Ada Boosting Accuracy",accuracy_score(y_stest,pred_abc))          #97.97

##---------------------Bagging 
bag=BaggingClassifier()
bag.fit(X_smotetrain,y_smotetrain)
pred_bag=bag.predict(X_stest)
print("Bagging Classifier",accuracy_score(y_stest,pred_bag))            #97.84

##Hyperparameter Tuning for RF
rfHy=RandomForestClassifier()
n_estimators=[300,350,400,450,500]
max_features = ['sqrt', 'log2']
parameters=dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
grid=GridSearchCV(estimator=rfHy,param_grid=parameters,scoring='accuracy',error_score=0,cv=cv)
rf_grid=grid.fit(X_smotetrain,y_smotetrain)
print(rf_grid.best_score_,rf_grid.best_params_)
rf_grid_pred=grid.predict(X_stest)

rfHyk=RandomForestClassifier()
cvk = KFold(n_splits=14,random_state=None)
grid1=GridSearchCV(estimator=rfHyk,param_grid=parameters,scoring='accuracy',error_score=0,cv=cvk)
rfk_grid=grid1.fit(X_smotetrain,y_smotetrain)
print(rfk_grid.best_score_,rfk_grid.best_params_)


##Preparing testing data

#--------------------------------------------missing values-------------------------------------
print(test_df.isnull().sum())
test_df["owns_car"]=test_df["owns_car"].replace(np.nan,"Y")
test_df["no_of_children"]=test_df["no_of_children"].replace(np.nan,0)
test_df["no_of_days_employed"]=test_df["no_of_days_employed"].replace(np.nan,test_df["no_of_days_employed"].mean())
test_df["total_family_members"]=test_df["total_family_members"].replace(np.nan,test_df["total_family_members"].median())
test_df["migrant_worker"]=test_df["migrant_worker"].replace(np.nan,0)
test_df["yearly_debt_payments"]=test_df["yearly_debt_payments"].replace(np.nan,test_df["yearly_debt_payments"].mean())
test_df["credit_score"]=test_df["credit_score"].replace(np.nan,test_df["credit_score"].mean())


#--------------------------------------------Visualizing----------------
outlier_col=['no_of_children','net_yearly_income','no_of_days_employed','total_family_members','yearly_debt_payments','credit_limit']
for col in outlier_col:
    outlier_vis(test_df[col])

child_pow=PowerTransformer(method='yeo-johnson', standardize=True)
test_child=test_df['no_of_children'].values
test_child=test_child.reshape(-1,1)
#test_df['no_of_children'] = np.where(test_df['no_of_children'] > 2,1 , test_df['no_of_children'])
test_df['no_of_children']=pow.transform(test_child)
test_df['total_family_members']=np.sqrt(test_df['total_family_members'])
print(test_df['credit_limit'].quantile(0.50))
print(test_df['credit_limit'].quantile(0.95))
test_df['yearly_debt_payments'] = np.where(test_df['yearly_debt_payments'] > 62784.5215,29122.265 , test_df['yearly_debt_payments'])
test_df['net_yearly_income'] = np.where(test_df['net_yearly_income'] > 401773.15, 172714.91, test_df['net_yearly_income'])
test_df['no_of_days_employed'] = np.where(test_df['no_of_days_employed'] > 6100, 3000, test_df['no_of_days_employed'])
test_df['credit_limit'] = np.where(test_df['credit_limit'] > 94464.20, 35600, test_df['credit_limit'])

outlier_vis(test_df['no_of_children'])
print(test_df['default_in_last_6months'].skew())
for col in outlier_col:
    outlier_vis(test_df[col])
#-------------------------------------------Categorical----------------------------- 
print(test_df["gender"].value_counts()) 
test_df["gender"]=test_df["gender"].replace('XNA','F')
print(test_df["owns_car"].value_counts())
#train_df["owns_car"]=train_df["owns_car"].map({'N':1,'Y':0})
test_owns_car=pd.get_dummies(test_df['owns_car'],drop_first=True)
test_gender=pd.get_dummies(test_df['gender'],drop_first=True)
#train_df["gender"]=train_df["gender"].map({'F':1,'M':0})
print(test_df["owns_house"].value_counts())
test_owns_house=pd.get_dummies(test_df['owns_house'],drop_first=True)
#train_df["owns_house"]=train_df["owns_house"].map({'Y':1,'N':0})
test_types=pd.get_dummies(test_df['occupation_type'],drop_first=True)
#--aligning dataset....why?It is clear visually
test_df=pd.concat([test_df,test_gender,test_owns_car,test_owns_house,test_types], axis = 1)

#Removing useless features
test_df=test_df.drop(columns=["occupation_type","gender","owns_car","owns_house"],axis=1)
customers=test_df.customer_id
test_df=test_df.drop(columns=['customer_id'],axis=1)
#test_df=test_df[imp]
#initial_testdf=test_df

##scaled test
#scaled_test=scale.transform(test_df)

#--------------Model.prediction,submission file creation-----------
test_predict=rfc.predict(test_df)
#scaled_predict=slrc.predict(test_df)
submission = pd.DataFrame({
        "customer_id": customers,
        "credit_card_default":test_predict
    })

submission.to_csv('sample_submission.csv', index=False)


##ada-91.12,bag=91.14,