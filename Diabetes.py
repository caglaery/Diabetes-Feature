from os import umask
from random import random
from tabnanny import check
from turtle import Turtle
import numpy as np
import pandas as pd
import seaborn as sns
from PIL.ImageColor import colormap
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from scipy.constants import value
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler,RobustScaler
from statsmodels.tools import categorical
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format",lambda x:"%.3f"% x)
pd.set_option("display.width",500)
df=pd.read_csv("PythonProject/datasets/diabetes.csv")
#Görev 1 : Keşifçi Veri Analizi
df.head(20)
df.describe().T
df.value_counts()
df.isnull().sum()
df.shape
df.info()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols,num_cols,cat_but_car=grab_col_names(df)

def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    print("#########################################")

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("##############################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)

for col in num_cols:
    num_summary(df,col)

df.head()
#NUMERİK DEĞİŞKENLERİN TARGETA GÖRE ANALİZİ
def target_sum_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")
for col in num_cols:
 target_sum_with_num(df,"Outcome",col)

#KORELASYON
df.corr()
f, ax = plt.subplots(figsize=[18,23])
sns.heatmap(df.corr(),annot=True,fnt=".2f",ax=ax,cmap="magma")
ax.set_title("Correlation Matrix",fontsize=20)
plt.show()
#MISSING VALUES
df.head()
zero_columns=[col for col in df.columns if df[col].min()==0 and col not in ["Pregnancies","Outcome"]]
for col in zero_columns:
    df[col]=df[col].replace(0,np.nan)
df.head()
df.shape
df.isnull().sum()
def missing_values_table(dataframe,na_name=False):
    na_columns=[col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    na_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio=(dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df=pd.concat([na_miss,np.round(ratio,2)],axis=1,keys=["n_miss","ratio"])
    print(missing_df,end="\n")
    if na_name:
        return na_columns

na_columns=missing_values_table(df,na_name=True)
#Eksik Değerlerin Bağımlı Değişlen İle İlişkisinin İncelenmesi
def missing_vs_target(dataframe,target,na_columns):
    temp_df=dataframe.copy()
    for col in na_columns:
        temp_df[col+"_NA_FLAG"]=np.where(temp_df[col].isnull(),1,0)
    na_flags=temp_df.loc[:,temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN":temp_df.groupby(col)[target].mean(),
                            "Count":temp_df.groupby(col)[target].count()}),end="\n\n\n")

missing_vs_target(df,"Outcome",zero_columns)
#Missing Valueların Doldurulması
for col in zero_columns:
    df.loc[df[col].isnull(),col]=df[col].median()

df.isnull().sum()
df.head()
#OUTLIER-AYKIRI DEĞER ANALİZİ
def outlier_thresholds(dataframe,col_name,q1=0.05,q3=0.95):
    quartile1=dataframe[col_name].quantile(q1)
    quartile3=dataframe[col_name].quantile(q3)
    iqr=quartile3-quartile1
    up_limit=quartile3+ 1.5*iqr
    low_limit=quartile1- 1.5*iqr
    return low_limit,up_limit


def check_outlier(dataframe,col_name):
    low_limit,up_limit=outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit)|(dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outlier(dataframe,col_name,index=False):
    low,up=outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up)|(dataframe[col_name]<low)].shape[0]>10:
        print((dataframe[(dataframe[col_name]>up)|(dataframe[col_name]<low)].head()))
    else:
        print((dataframe[(dataframe[col_name]>up)|(dataframe[col_name]<low)]))
     if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe,col_name):
    low_limit,up_limit=outlier_thresholds(dataframe,col_name)
    df_without_outlier=dataframe[~(dataframe[col_name]>up_limit)|(dataframe[col_name]<low_limit)]
    return df_without_outlier

def replace_with_thresholds(dataframe,variable,q1=0.05,q3=0.95):
    low_limit , up_limit =outlier_thresholds(dataframe,variable,q1=0.05,q3=0.95)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable]= up_limit

for col in df.columns:
    print(col,check_outlier(df,col))
    if check_outlier(df,col):
        replace_with_thresholds(df,col)
df.head()
cat_cols
#ÖZELLİK ÇIKARIMLARI
#############################################
#1-Yaş Değişkeninin Kategoriye Ayrılıp Yeni Değişken Oluşturulması
df.loc[(df["Age"]>=21)&(df["Age"]<50),"NEW_AGE_CAT"]="mature"
df.loc[(df["Age"]>=50),"NEW_AGE_CAT"]="senior"
#2-BMI Değişkeninin Kategoriye Ayrılıp Yeni Değişken Oluşturulması
df["NEW_BMI"]=pd.cut(df["BMI"],bins=[0,18.5,24.9,29.9,100],labels=["Underweight","Healthy","Overweight","Obese"])
#3-Glikoz Değişkeninin Kategoriye Ayrılıp Yeni Değişken Oluşturulması
df["NEW_GLUCOSE"]=pd.cut(df["Glucose"],bins=[0,140,200,300],labels=["Normal","Prediabetes","Diabetes"])
#4-Yaş ve BMI Değişkenlerinin Birleştirilip Yeni Değişken Oluşturulması
df.loc[(df["Age"]>=21)&(df["Age"]<50)&(df["BMI"]<18.5),"NEW_AGE_BMI"]="underweightmature"
df.loc[(df["Age"]>=50)&(df["BMI"]<18.5),"NEW_AGE_BMI"]="underweightsenior"
df.loc[(df["BMI"]>=18.5)&(df["BMI"]<25)&(df["Age"]>=21)&(df["Age"]<50),"NEW_AGE_BMI"]="healthymature"
df.loc[(df["BMI"]>=18.5)&(df["BMI"]<25)&(df["Age"]>=50),"NEW_AGE_BMI"]="healthysenior"
df.loc[(df["BMI"]>=25)&(df["BMI"]<30)&(df["Age"]>=21)&(df["Age"]<50),"NEW_AGE_BMI"]="overweightmature"
df.loc[(df["BMI"]>=25)&(df["BMI"]<30)&(df["Age"]>=50),"NEW_AGE_BMI"]="overweightsenior"
df.loc[(df["Age"]>=21)&(df["Age"]<50)&(df["BMI"]>=30),"NEW_AGE_BMI"]="obesemature"
df.loc[(df["Age"]>=50)&(df["BMI"]>=30),"NEW_AGE_BMI"]="obesesenior"
#5-Yaş ve Glikoz Değişkenlerinin Birleştirilip Yeni Değişken Oluşturulması
df.loc[(df["Age"]>=21)&(df["Age"]<50)&(df["Glucose"]<70),"NEW_AGE_GLUCOSE"]="lowmature"
df.loc[(df["Age"]>=50)&(df["Glucose"]<70),"NEW_AGE_GLUCOSE"]="lowsenior"
df.loc[(df["Age"]>=21)&(df["Age"]<50)&(df["Glucose"]>=70)&(df["Glucose"]<100),"NEW_AGE_GLUCOSE"]="normalmature"
df.loc[(df["Age"]>=50)&(df["Glucose"]>=70)&(df["Glucose"]<100),"NEW_AGE_GLUCOSE"]="normalsenior"
df.loc[(df["Age"]>=21)&(df["Age"]<50)&(df["Glucose"]>=100)&(df["Glucose"]<=125),"NEW_AGE_GLUCOSE"]="hiddenmature"
df.loc[(df["Age"]>=50)&(df["Glucose"]>=100)&(df["Glucose"]<=125),"NEW_AGE_GLUCOSE"]="hiddensenior"
df.loc[(df["Age"]>=21)&(df["Age"]<50)&(df["Glucose"]>125),"NEW_AGE_GLUCOSE"]="highmature"
df.loc[(df["Age"]>=50)&(df["Glucose"]>125),"NEW_AGE_GLUCOSE"]="highsenior"
#Insulin Değeri İle Kategorik Değişken Üretme
def set_insulin(dataframe,col_name="Insulin"):
    if 16<=dataframe[col_name]<=166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"]=df.apply(set_insulin,axis=1)
df["NEW_GLUCOSE*PREGNANCIES"]=df["Glucose"]*df["Pregnancies"]
df["NEW_GLUCOSE*AGE"]=df["Glucose"]*df["Age"]
df["NEW_GLUCOSE*BLOODPRESSURE"]=df["Glucose"]*df["BloodPressure"]
df.columns=[col.upper() for col in df.columns]
df.head()
df.shape
#ENCODING
#1-Değişken tiplerinin ayrılması
#LABEL ENCODING
num_cols,cat_cols,cat_but_car=grab_col_names(df)
binary_cols=[col for col in df.columns if df[col].dtype=="O"and df[col].nunique()==2]
def label_encoder(dataframe,binary_col):
    labelencoder=LabelEncoder()
    dataframe[binary_col]=labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
for col in binary_cols:
    df=label_encoder(df,col)

df.head()
#ONE-HOT ENCODING
cat_cols=[col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
def one_hot_encoder(dataframe,categorical_cols,drop_first=False):
    dataframe=pd.get_dummies(dataframe,columns=categorical_cols,drop_first=drop_first)
    return dataframe
df=one_hot_encoder(df,cat_cols,drop_first=True)
df.head(30)
df = df.replace({True: 1, False: 0})
print(df)
#Standartlaştırma
scaler=StandardScaler()
df[num_cols]=scaler.fit_transform(df[num_cols])
df.head()
#MODELLEME
y=df["OUTCOME"]
X=df.drop("OUTCOME",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=17)
rf_model=RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred,y_test),2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test),2)}")
print(f"F1: {round(f1_score(y_pred,y_test),2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test),2)}")

#FEATURE IMPORTANCE
def plot_importance(model,features,num=len(X),save=False):
    feature_imp=pd.DataFrame({"Value":model.feature_importances_,"Feature":features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10,10))
    sns.set_theme(font_scale=1)
    sns.barplot(x="Value",y="Feature",data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()

    if save:
        plt.savefig("importance.png")
    plt.show(block=True)

    plot_importance(rf_model,X)