# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

![image](https://github.com/user-attachments/assets/e2e152bb-14f4-4ad0-8459-f77f1b0156bb)


df_null_sum=df.isnull().sum()
df_null_sum

![image](https://github.com/user-attachments/assets/a9720f0d-b7c9-4b50-9641-831b9977ad50)


df.dropna()

![image](https://github.com/user-attachments/assets/dee7a048-1a1f-4147-b74f-41fd483b8b9f)


max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals

![image](https://github.com/user-attachments/assets/f9aa11d2-d524-44c2-9d82-10824f987b84)


from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()

![image](https://github.com/user-attachments/assets/461ee3cc-439b-4d5a-8059-9da8e512ea69)


sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

![image](https://github.com/user-attachments/assets/1cfa3768-a7ca-4353-94ed-d96360c2db7f)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

![image](https://github.com/user-attachments/assets/4a02e41d-527c-4850-afb4-12f8638694e4)


from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()

![image](https://github.com/user-attachments/assets/635a9150-d71f-4082-85f5-a4355a606ca0)


from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()

![image](https://github.com/user-attachments/assets/e9431485-9584-42fe-a30f-9800bd5bf4dd)


df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3


![image](https://github.com/user-attachments/assets/72e00abd-c42f-4d2d-9e6e-472c60c821bc)


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()


![image](https://github.com/user-attachments/assets/1e581673-715b-4dd4-a052-ad29bb12df9f)


df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()

![image](https://github.com/user-attachments/assets/5bfdcd36-b31a-45ef-a61c-7bd4880c6d59)


import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

![image](https://github.com/user-attachments/assets/d4f1b3b6-8304-488f-9195-2b817637bcd0)


df

![image](https://github.com/user-attachments/assets/fc7837f0-775a-4477-833a-42cc6fce0135)


df.info()

![image](https://github.com/user-attachments/assets/c4bbfd6c-11e9-4527-809b-27ede5b7c6e2)


df_null_sum=df.isnull().sum()
df_null_sum

![image](https://github.com/user-attachments/assets/d08e2cda-7349-41ed-a811-f533a69bfbdd)


categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]


![image](https://github.com/user-attachments/assets/cd06e95f-a3c1-48d9-a0c4-b7661cee0c11)


df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]


![image](https://github.com/user-attachments/assets/ee44a0ea-e3a6-4823-a2aa-a3754add4543)


import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

![image](https://github.com/user-attachments/assets/6b760750-7d26-4950-80e3-91c498910a65)


df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![image](https://github.com/user-attachments/assets/cab75060-1063-4635-9212-083db0caa1d7)


X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)

![image](https://github.com/user-attachments/assets/e1b395e9-dcfa-4fec-878c-d12bc61b2603)


selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


![image](https://github.com/user-attachments/assets/2d86b2ef-9615-45bf-9a15-2fc0d6c8c634)


y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")

![image](https://github.com/user-attachments/assets/847aeb1c-3807-4001-b9f8-297411943fdc)



!pip install skfeature-chappers


![image](https://github.com/user-attachments/assets/f1a6b7d3-cad7-45ca-bd4b-bb551360cbb5)


import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]



![image](https://github.com/user-attachments/assets/524a1f69-7dbb-408b-a8f5-60617b3be021)


df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![image](https://github.com/user-attachments/assets/232f8cdc-cbf6-4837-b60a-573c1576297f)


k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)

![image](https://github.com/user-attachments/assets/9c2de862-5afd-4f40-bb6b-2c81b9586cfe)


import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

![image](https://github.com/user-attachments/assets/965017b6-8629-4925-9d97-4d9943d3740e)


df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

![image](https://github.com/user-attachments/assets/da6a65f6-bfb7-4ace-bea7-b1f4b2fdc4f4)


X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

![image](https://github.com/user-attachments/assets/45355bbf-124d-4296-a341-55440cc593fe)


selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)


![image](https://github.com/user-attachments/assets/12400199-741c-481b-be87-51f347b4042e)



# RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file has been done successfully.
       
