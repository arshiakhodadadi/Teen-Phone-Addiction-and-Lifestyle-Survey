# Data call
from pandas import read_csv ,get_dummies ,core
df = read_csv(r'C:\Users\ok\Desktop\Program\python\CSV dataset\teen_phone_addiction_dataset.csv')
df.drop(columns = ['Name'],inplace=True,axis=1)

# Remove the ID column and replace it with names 
df.drop(axis=1,inplace=True,columns=['ID'])

# Converting string values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df['Gender_numeric'] = df['Gender'].map({'Male': 0, 'Female': 1 ,'Other' : 2})
df.drop(axis=1,inplace=True,columns=['Gender'])

from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()

Location = encode.fit_transform(df['Location'].astype('str'))
df['Location'] = Location

school_grade = []
for th in df['School_Grade']:
    school_grade.append(th.replace('th' ,''))

from numpy import array
school_grade = array(school_grade)
df['School_Grade'] = school_grade.astype('int')

df = get_dummies(df,columns=['Phone_Usage_Purpose'])
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

categorical_ix = df.select_dtypes(include=['bool','object']).columns
print(categorical_ix)

# Data information
print(f'Data dimensions :{df.shape}\n{df.info()}\ndescribe :{df.describe()}')

from matplotlib import pyplot as plt

ax = df.hist(
    figsize=(10, 8),     
    bins=20,              
    xlabelsize=8,         
    ylabelsize=8,         
    grid=False,           
    color='skyblue',     
    edgecolor='black'     
)

plt.tight_layout() 
plt.show()

# Investigating the relationship between features and targets to eliminate negative and ineffective relationships
correlations = df.corr()
correlations['Addiction_Level'].sort_values(ascending=False)
print(correlations)

import seaborn as sns

corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of Feature Correlations')
plt.show()

print(df.isnull().sum())
# Creating appropriate data and converting raw data to standards
def data(df):
    y = df.pop('Addiction_Level')
    X = df
    return X,y

X ,y = data(df)

from sklearn.model_selection import train_test_split
X_train ,X_test ,y_train ,y_test = train_test_split(X ,y ,test_size=0.33 ,random_state = 87)

# Delete outlier
from sklearn.neighbors import LocalOutlierFactor

def outlier(X_train,y_train):
    # Local Outlier Factor
    X_train = X_train.values
    y_train = y_train.values
    lof = LocalOutlierFactor(n_neighbors=10)
    X_sel = lof.fit_predict(X_train)
    mask = X_sel != -1
    X_train , y_train = X_train[mask,:],y_train[mask]

    return X_train ,y_train
X_train_new ,y_train_new = outlier(X_train,y_train)

print(X_train.shape ,X_train_new.shape) # Since deleting the outlier does not make any changes, we do not consider it.

# Standardization
from sklearn.preprocessing import StandardScaler

def standard(X_train ,X_test):
    scaler = StandardScaler()
    Standard_features = ['Daily_Usage_Hours' ,'Sleep_Hours' ,'Weekend_Usage_Hours']
    for f in Standard_features:
        X_train[f] = scaler.fit_transform(X_train[[f]])
        X_test[f] = scaler.transform(X_test[[f]])
    return X_train ,X_test

X_train ,X_test = standard(X_train ,X_test)

# RobustScaler and QuantileTransformer
from sklearn.preprocessing import QuantileTransformer ,RobustScaler

def quantile(X_train ,X_test):
    scaler_quantile = QuantileTransformer(output_distribution='normal')
    quantile_features = ['Exercise_Hours' ,'Time_on_Social_Media' ,'Time_on_Gaming' ,'Time_on_Education']
    for f in quantile_features:
        X_train[f] = scaler_quantile.fit_transform(X_train[[f]])
        X_test[f] = scaler_quantile.transform(X_test[[f]])
    
    scaler_robust = RobustScaler()
    robust_features = ['Screen_Time_Before_Bed' ,'Phone_Checks_Per_Day']
    for f in robust_features:
        X_train[f] = scaler_robust.fit_transform(X_train[[f]])
        X_test[f] = scaler_robust.transform(X_test[[f]])
    return X_train ,X_test

X_train ,X_test = quantile(X_train ,X_test)

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

def min_max(X_train ,X_test):
    scaler = MinMaxScaler()
    min_max_features = ['Family_Communication' ,'Apps_Used_Daily' ,'Social_Interactions' ,'Academic_Performance' ,
                        'Self_Esteem' ,'Depression_Level' ,'Anxiety_Level' ,'Age']
    for f in min_max_features:
        X_train[f] = scaler.fit_transform(X_train[[f]])
        X_test[f] = scaler.transform(X_test[[f]])
    return X_train ,X_test
X_train ,X_test = min_max(X_train ,X_test)
print(X_train)

# Model construction: Linear model
from sklearn.linear_model import LinearRegression ,Ridge ,Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

models = {'LinearRegression':LinearRegression(),
        'Ridge': Ridge(),
        'Lasso':Lasso(),
        'RandomForestRegressor':RandomForestRegressor(),
        'xgb':xgb.XGBRegressor(),
        'SVR':SVR(),
        'DecisionTreeRegressor':DecisionTreeRegressor(),
        'KNeighborsRegressor':KNeighborsRegressor()}

from sklearn.metrics import r2_score ,mean_absolute_error ,mean_squared_error ,explained_variance_score

scores = {'r2_score':r2_score ,
          'mean_absolute_error':mean_absolute_error,
          'mean_squared_error':mean_squared_error,
          'explained_variance_score':explained_variance_score}


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
import pandas as pd

scores = {
    'R2 Score': r2_score,
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'Explained Variance': explained_variance_score
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    model_scores = {}
    for metric_name, metric_func in scores.items():
        model_scores[metric_name] = metric_func(y_test, yhat)
    results[model_name] = model_scores

results_df = pd.DataFrame(results).T 

results_df_sorted = results_df.sort_values(by='R2 Score', ascending=False)

print("\n📊Accuracy of models based on various metrics :\n")
print(results_df_sorted.round(4))

# email :arshia.khodadad.ir@gmail.com
# dev : Arshia Khodadadi