import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import funciones as fn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

nuevos = pd.read_csv('Data/datos_nuevos_creditos.csv')
historicos = pd.read_csv('Data/datos_historicos.csv')

#Separación de variables explicativas con variable objetivo.
df_x=historicos.iloc[:,:-1]
df_y=historicos.iloc[:,-1]
df_x.drop(columns=["ID"],inplace=True)

cat=df_x.select_dtypes(include="object").columns
historicos[cat]
num=df_x.select_dtypes(exclude="object").columns
historicos[num]

#get_dummies
df_dum=pd.get_dummies(df_x,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
x_train=pd.DataFrame(xnor,columns=df_dum.columns)
x_train.columns

#Selección de variables a traves de los modelos descritos a continuación
mr = LinearRegression()
mdtr= DecisionTreeRegressor()
mrfr= RandomForestRegressor()
mgbr=GradientBoostingRegressor()
modelos= [ mr, mdtr, mrfr, mgbr]

# "función" para buscar el mejor threshold que logre seleccionar las variables para cada modelo.
df_resultado = pd.DataFrame()
thres=0.1
for i in range(10):
    df_actual=0
    var_names=fn.sel_variables(modelos, x_train, df_y, threshold="{}*mean".format(thres))
    xtrain=x_train[var_names]
    accu_xtrain=fn.medir_modelos(modelos,"r2",xtrain,df_y,10)
    df=accu_xtrain.mean(axis=0)
    df_actual = pd.DataFrame(df, columns=['threshold {}'.format(thres)])
    df_resultado = pd.concat([df_resultado, df_actual], axis=1)
    thres+=0.1
    thres=round(thres,2)


#Gráfica de los resultados __________________________________
df=df_resultado.T
plt.figure(figsize=(10,10))
sns.lineplot(data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.ylabel("r2")
plt.title("Variacion threshold")

df.idxmax(axis=0)

modelos= [mrfr]
var_names=fn.sel_variables(modelos, x_train, df_y, threshold="0.1*mean")
var_names

modelos= [mgbr]
var_names2=fn.sel_variables(modelos, x_train, df_y, threshold="0.1*mean")
var_names2


#### Random forest
xtrainf1=x_train[var_names]

param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
    'max_depth': [5, 10, 20],  # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
    # Número máximo de características a considerar en cada división
     # Método de selección de muestras para el entrenamiento de cada árbol
}

rfrtuning=RandomForestRegressor()
grid_search1=GridSearchCV(rfrtuning, param_grid, scoring="r2",cv=10, n_jobs=-1)
grid_result1=grid_search1.fit(xtrainf1, df_y)

pd.set_option('display.max_colwidth', 100)
resultados1=grid_result1.cv_results_
grid_result1.best_params_
pd_resultados1=pd.DataFrame(resultados1)
pd_resultados1[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

rfr_final=grid_result1.best_estimator_

eval=cross_val_score(rfr_final,xtrainf1,df_y,cv=20,scoring="r2")

np.mean(eval)

#Grafiquemos las predicciones
ypredrfr=rfr_final.predict(xtrainf1)
sns.set_theme(style="ticks")
dict={"y":df_y,"ypredrfr":ypredrfr}
df1=pd.DataFrame(dict)
df1=df1.stack().reset_index()
df1.drop(columns=["level_0"],inplace=True)
df1.columns=["tipo","valor"]
df1["tipo"]=df1["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df1, x="valor", hue="tipo")

# Metricas
mse = mean_squared_error(df_y, ypredrfr)
print("Mean Squared Error (MSE):", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
mae = mean_absolute_error(df_y, ypredrfr)
print("Mean Absolute Error (MAE):", mae)
r2 = r2_score(df_y, ypredrfr)
print("R-squared (R2):", r2)

lista1=[mse, rmse,mae,r2]


### Gradient Boosting

xtrainf=x_train[var_names2]

param_grid2 = {
    'n_estimators': [50, 100, 200],  # Número de árboles en el ensamble
    'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
    'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol base
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
}

rfrtuning=GradientBoostingRegressor()
grid_search2=GridSearchCV(rfrtuning, param_grid2, scoring="r2",cv=10, n_jobs=-1)
grid_result2=grid_search2.fit(xtrainf, df_y)

gbr_final=grid_result2.best_estimator_

eval2=cross_val_score(gbr_final,xtrainf,df_y,cv=20,scoring="r2")

np.mean(eval2)

#Para el gradient boosting
ypredgbr=gbr_final.predict(xtrainf)
sns.set_theme(style="ticks")
dict={"y":df_y,"ypredrfr":ypredgbr}
df2=pd.DataFrame(dict)
df2=df2.stack().reset_index()
df2.drop(columns=["level_0"],inplace=True)
df2.columns=["tipo","valor"]
df2["tipo"]=df2["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df2, x="valor", hue="tipo")

# Metricas
mse = mean_squared_error(df_y, ypredgbr)
print("Mean Squared Error (MSE):", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
mae = mean_absolute_error(df_y, ypredgbr)
print("Mean Absolute Error (MAE):", mae)
r2 = r2_score(df_y, ypredgbr)
print("R-squared (R2):", r2)

lista2=[mse,rmse,mae,r2]

lista3= ["Mean Squared Error (MSE)","Root Mean Squared Error (RMSE)","Mean Absolute Error (MAE)","R-squared (R2)"]
dict={"metrica":lista3,"rfr":lista1,"gbr":lista2}
dffinal=pd.DataFrame(dict)
dffinal

#######Predicciones para tabla de nuevos########
## Random forest
nuevos.info()
x=nuevos.iloc[:,:-1]
y=nuevos.iloc[:,-1]

#get_dummies
df_dum=pd.get_dummies(x,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
xtest=pd.DataFrame(xnor,columns=df_dum.columns)
xtest.drop(columns=["ID"],inplace=True)
len(xtest.columns)
len(xtrainf1.columns)
plt.hist(y)
xtest=xtest.reindex(columns=xtrainf.columns)
ypredtestrfr=gbr_final.predict(xtest)

plt.hist(ypredtestrfr, bins=50)

dict = {"ID":nuevos["ID"], "int_rc":ypredtestrfr}
excel=pd.DataFrame(dict)
excel["int_rc"]=excel["int_rc"].apply(lambda x: x + 15 )

excel.to_excel("Predicciones_rf.xlsx",index=False)

## Gradient boosting
nuevos.info()
x=nuevos.iloc[:,:-1]
y=nuevos.iloc[:,-1]

#get_dummies
df_dum=pd.get_dummies(x,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
xtest=pd.DataFrame(xnor,columns=df_dum.columns)
xtest.drop(columns=["ID"],inplace=True)
len(xtest.columns)
len(xtrainf.columns)
plt.hist(y)
xtest=xtest.reindex(columns=xtrainf.columns)
ypredtestgbr=gbr_final.predict(xtest)

plt.hist(ypredtestrfr, bins=50)

dict = {"ID":nuevos["ID"], "int_rc":ypredtestgbr}
excel=pd.DataFrame(dict)
excel["int_rc"]=excel["int_rc"].apply(lambda x: x + 15 )

excel.to_excel("Predicciones_gb.xlsx",index=False)