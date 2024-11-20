import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
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

tabla_nuevos='https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_nuevos_creditos.csv'
tabla_hist='https://raw.githubusercontent.com/juancamiloespana/LEA3_FIN/main/data/datos_historicos.csv'
tabla_nuevos=pd.read_csv(tabla_nuevos)
tabla=pd.read_csv(tabla_hist)

#Separación de variables explicativas con variable objetivo.
dfx=tabla.iloc[:,:-1]
dfy=tabla.iloc[:,-1]
dfx.drop(columns=["ID"],inplace=True)


cat=dfx.select_dtypes(include="object").columns
tabla[cat]
num=dfx.select_dtypes(exclude="object").columns
tabla[num]

#get_dummies
df_dum=pd.get_dummies(dfx,columns=cat)
df_dum.info()

#Escalamos las variables
scaler=StandardScaler()
scaler.fit(df_dum)
xnor=scaler.transform(df_dum)
x=pd.DataFrame(xnor,columns=df_dum.columns)
x.columns

#Selección de variables a traves de los modelos descritos a continuación
mr = LinearRegression()
mdtr= DecisionTreeRegressor()
mrfr= RandomForestRegressor()
mgbr=GradientBoostingRegressor()
modelos= [ mr, mdtr, mrfr, mgbr]

def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)

    metric_modelos.columns=["reg","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos

def sel_variables(modelos,X,y,threshold):

    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)

    return var_names_ac

# "función" para buscar el mejor threshold que seleccina las variables para cada modelo.------------------
df_resultado = pd.DataFrame()
thres=0.1
for i in range(10):
    df_actual=0
    var_names=sel_variables(modelos, x, dfy, threshold="{}*mean".format(thres))
    xtrain=x[var_names]
    accu_xtrain=medir_modelos(modelos,"r2",xtrain,dfy,10)
    df=accu_xtrain.mean(axis=0)
    df_actual = pd.DataFrame(df, columns=['threshold {}'.format(thres)])
    df_resultado = pd.concat([df_resultado, df_actual], axis=1)
    thres+=0.15
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
var_names=sel_variables(modelos, x, dfy, threshold="0.1*mean")
var_names

modelos= [mgbr]
var_names2=sel_variables(modelos, x, dfy, threshold="0.1*mean")
var_names2


#### Random forest
xtrainf1=x[var_names]

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
grid_result1=grid_search1.fit(xtrainf1, dfy)

pd.set_option('display.max_colwidth', 100)
resultados1=grid_result1.cv_results_
grid_result1.best_params_
pd_resultados1=pd.DataFrame(resultados1)
pd_resultados1[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

rfr_final=grid_result1.best_estimator_

eval=cross_val_score(rfr_final,xtrainf1,dfy,cv=20,scoring="r2")

np.mean(eval)

# Metricas
mse = mean_squared_error(dfy, ypredrfr)
print("Mean Squared Error (MSE):", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
mae = mean_absolute_error(dfy, ypredrfr)
print("Mean Absolute Error (MAE):", mae)
r2 = r2_score(dfy, ypredrfr)
print("R-squared (R2):", r2)

lista1=[mse, rmse,mae,r2]

#Grafiquemos las predicciones
ypredrfr=rfr_final.predict(xtrainf)
sns.set_theme(style="ticks")
dict={"y":dfy,"ypredrfr":ypredrfr}
df1=pd.DataFrame(dict)
df1=df1.stack().reset_index()
df1.drop(columns=["level_0"],inplace=True)
df1.columns=["tipo","valor"]
df1["tipo"]=df1["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df1, x="valor", hue="tipo")

### Gradient Boosting

xtrainf=x[var_names2]

param_grid2 = {
    'n_estimators': [50, 100, 200],  # Número de árboles en el ensamble
    'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
    'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol base
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
}

rfrtuning=GradientBoostingRegressor()
grid_search2=GridSearchCV(rfrtuning, param_grid2, scoring="r2",cv=10, n_jobs=-1)
grid_result2=grid_search2.fit(xtrainf, dfy)

gbr_final=grid_result2.best_estimator_

eval2=cross_val_score(gbr_final,xtrainf,dfy,cv=20,scoring="r2")

np.mean(eval2)

#Para el gradient boosting
ypredgbr=gbr_final.predict(xtrainf)
sns.set_theme(style="ticks")
dict={"y":dfy,"ypredrfr":ypredgbr}
df2=pd.DataFrame(dict)
df2=df2.stack().reset_index()
df2.drop(columns=["level_0"],inplace=True)
df2.columns=["tipo","valor"]
df2["tipo"]=df2["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df2, x="valor", hue="tipo")

# Metricas
mse = mean_squared_error(dfy, ypredgbr)
print("Mean Squared Error (MSE):", mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
mae = mean_absolute_error(dfy, ypredgbr)
print("Mean Absolute Error (MAE):", mae)
r2 = r2_score(dfy, ypredgbr)
print("R-squared (R2):", r2)

lista2=[mse,rmse,mae,r2]

lista3= ["Mean Squared Error (MSE)","Root Mean Squared Error (RMSE)","Mean Absolute Error (MAE)","R-squared (R2)"]
dict={"metrica":lista3,"rfr":lista1,"gbr":lista2}
dffinal=pd.DataFrame(dict)
dffinal

#######Predicciones para tabla de nuevos########
tabla_nuevos.info()
x=tabla_nuevos.iloc[:,:-1]
y=tabla_nuevos.iloc[:,-1]

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
ypredtestrfr=gbr_final.predict(xtest)

plt.hist(ypredtestrfr, bins=50)