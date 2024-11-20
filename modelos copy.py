import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import xgboost as xgb
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
import funciones as fn


#importar datos
nuevos = pd.read_csv('Data/datos_nuevos_creditos.csv')
historicos = pd.read_csv('Data/datos_historicos.csv')

nuevos.info()
historicos.info()
historicos.drop(columns=["ID"],inplace=True)

# Datos de entrenamiento
y = historicos['NoPaidPerc']
x = historicos.drop('NoPaidPerc', axis=1)

## Mtz de correlación
scaler = None
df_corr, sc = fn.procesar_datos(historicos, scaler)

df_corr = df_corr.select_dtypes(include=['float64', 'int64'])
corr_mat = df_corr.corr()

plt.figure(figsize=(20, 8))
sns.heatmap(corr_mat, annot=True, cmap='OrRd', fmt=".2f", linewidths=0.5)
plt.title('Matriz de correlación')
plt.show()

# Obtener correlaciones más fuertes
correlations = (
    corr_mat['NoPaidPerc']
    .drop('NoPaidPerc')
    .reset_index() 
)

correlations.columns = ['Variable', 'Correlation']
correlations['Abs Correlation'] = correlations['Correlation'].abs()
scv = correlations.sort_values(by='Abs Correlation', ascending=False).head(10)

print("Correlaciones más fuertes con la variable objetivo:\n")
print(scv)

#joblib.dump(df_corr, 'salidas/df_cor.pkl')



#################################################################################
################################################################################
################################################################################

df_vs = df_corr.drop(columns=['NoPaidPerc'])

# Selección de las variables deseadas
variables_seleccionadas = [
    'MaritalStatus_Widowed',
    'MaritalStatus_Married',
    'DebtRatio',
    'Education_Bachelor',
    'EmploymentLength',
    'HomeOwnership_Own'
]

df_vs = df_vs[variables_seleccionadas]

### Metricas de los modelos con las variables seleccionadas 
lr = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()

# Lista de modelos
modelos = [lr, dtr, rfr, gbr]

# Medir los modelos
accu_x = fn.medir_modelos(modelos,"r2",df_corr,y,20) ## base con todas las variables 
accu_xtrain = fn.medir_modelos(modelos,"r2",df_vs,y,20) ### base con variables seleccionadas

# Dataframe con los resultados
accu = pd.concat([accu_x,accu_xtrain],axis=1)
accu.columns = ['rl', 'dt', 'rf', 'gb',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']

#Promedio para cada modelo
np.mean(accu, axis=0)

#Gráfico de F1 score para modelos con todas las variables y modelos con variables seleccionadas
sns.boxplot(data=accu_x, palette="Set3")
sns.boxplot(data=accu_xtrain, palette="Set3")
sns.boxplot(data=accu, palette="Set3")

##### Entrenamiento de los modelos

# param_grid = {
#     'n_estimators': [100, 200, 300],  # Número de árboles en el bosque
#     'max_depth': [5, 10, 20],  # Profundidad máxima de cada árbol
#     'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
#     'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
#     # Número máximo de características a considerar en cada división
#      # Método de selección de muestras para el entrenamiento de cada árbol
# }

# rfrtuning = RandomForestRegressor()
# grid_search1 = GridSearchCV(rfrtuning, param_grid, scoring="r2",cv=10, n_jobs=-1)
# grid_result1 = grid_search1.fit( , dfy)

# pd.set_option('display.max_colwidth', 100)
# resultados1 = grid_result1.cv_results_
# grid_result1.best_params_
# pd_resultados1 = pd.DataFrame(resultados1)
# pd_resultados1[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

# rfr_final = grid_result1.best_estimator_ 
# joblib.dump(rfr_final, "rfr_final.pkl") 

param_grid2 = {
    'n_estimators': [50, 100, 200],  # Número de árboles en el ensamble
    'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
    'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol base
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
    'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
}

rfrtuning = GradientBoostingRegressor()
grid_search2 = GridSearchCV(rfrtuning, param_grid2, scoring="r2",cv=10, n_jobs=-1)
grid_result2 = grid_search2.fit(df_vs, y)

gbr_final = grid_result2.best_estimator_ 
joblib.dump(gbr_final, "gbr_final.pkl") 


# eval = cross_val_score(rfr_final,xtrainf,dfy,cv=20,scoring="r2")
eval2 = cross_val_score(gbr_final,df_vs,y,cv=20,scoring="r2")

# np.mean(eval)
np.mean(eval2)


# #Grafiquemos las predicciones
# ypredrfr = rfr_final.predict(xtrainf)
# sns.set_theme(style="ticks")
# dict = {"y":dfy,"ypredrfr":ypredrfr}
# df1 = pd.DataFrame(dict)
# df1 = df1.stack().reset_index()
# df1.drop(columns=["level_0"],inplace=True)
# df1.columns = ["tipo","valor"]
# df1["tipo"] = df1["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

# sns.histplot(data=df1, x="valor", hue="tipo")

#Para el gradient boosting
ypredgbr = gbr_final.predict(df_vs)
sns.set_theme(style="ticks")
dict = {"y":y,"ypredrfr":ypredgbr}
df2 = pd.DataFrame(dict)
df2 = df2.stack().reset_index()
df2.drop(columns=["level_0"],inplace=True)
df2.columns=["tipo","valor"]
df2["tipo"]=df2["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

sns.histplot(data=df2, x="valor", hue="tipo")

## Metricas comparacion
# MSE
mse = mean_squared_error(y, ypredgbr)
print("Mean Squared Error (MSE):", mse)
# RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
# MAE
mae = mean_absolute_error(y, ypredgbr)
print("Mean Absolute Error (MAE):", mae)
# R2
r2 = r2_score(y, ypredgbr)
print("R-squared (R2):", r2)

lista1 = [mse, rmse,mae,r2]




