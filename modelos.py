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

#importar datos
nuevos = pd.read_csv('Data/datos_nuevos_creditos.csv')
historicos = pd.read_csv('Data/datos_historicos.csv')

nuevos.info()
historicos.info()

# nuevos['int_rc'] = 0.3
# df_ejemplo=df_nuevos[['ID','int_rc']]
# df_ejemplo.to_csv('data\\grupo_1.csv')

x_train = historicos.iloc[:,:-1]
y_train = historicos["NoPaidPerc"]

# columnas
num = x_train.select_dtypes(exclude='object').columns
cat = x_train.select_dtypes(include='object').columns

column_transformer = ColumnTransformer([
    ('scaler', StandardScaler(), num),
    ('onehot', OneHotEncoder(), cat)
     ], remainder='passthrough')

x_train = column_transformer.fit_transform(x_train)

# Separación de variables explicativas con variable objetivo.
dfx = historicos.iloc[:,:-1]
dfy = historicos.iloc[:,-1]
dfx.drop(columns=["ID"],inplace=True)

cat2 = dfx.select_dtypes(include="object").columns
historicos[cat2]
num2 = dfx.select_dtypes(exclude="object").columns
historicos[num2]

# Dummies
df_dum = pd.get_dummies(dfx,columns=cat2)
df_dum.info()

# Escalar las variables
scaler = StandardScaler()
scaler.fit(df_dum)
xnor = scaler.transform(df_dum)
x = pd.DataFrame(xnor,columns=df_dum.columns)
x.columns
#x contiene el dataframe ya listo para ser entrenado

# Selección de variables a traves de los modelos descritos a continuación 
lr = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()
modelos= [lr, dtr, rfr, gbr]

####################################
#  función selección de variables  #
####################################
def sel_variables(modelos,X,y,threshold):
    var_names_ac = np.array([])
    for modelo in modelos:
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    return var_names_ac

var_names = sel_variables(modelos, x, dfy, threshold="2.2*mean")
var_names.shape

# Variables elegidas incialmente con el threshold 2.2 
xtrain = x[var_names] #5 variables

###########################
#  función medir modelos  #
###########################
def medir_modelos(modelos,scoring,X,y,cv):
    metric_modelos = pd.DataFrame()
    for modelo in modelos:
        scores = cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores = pd.DataFrame(scores)
        metric_modelos = pd.concat([metric_modelos,pdscores],axis=1)
    metric_modelos.columns = ["reg","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos

# Medir los modelos
accu_x = medir_modelos(modelos,"r2",x,dfy,20) ## base con todas las variables 
accu_xtrain = medir_modelos(modelos,"r2",xtrain,dfy,20) ### base con variables seleccionadas

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

#### función para buscar el mejor threshold que seleccina las variables para cada modelo
df_resultado = pd.DataFrame()
thres = 0.1
for i in range(10):
    df_actual = 0
    var_names = sel_variables(modelos, x, dfy, threshold="{}*mean".format(thres))
    xtrain = x[var_names]
    accu_xtrain = medir_modelos(modelos,"r2",xtrain,dfy,10)
    df=accu_xtrain.mean(axis=0)
    df_actual = pd.DataFrame(df, columns=['threshold {}'.format(thres)])
    df_resultado = pd.concat([df_resultado, df_actual], axis=1)
    thres += 0.15
    thres = round(thres,2)

# resultado = pd.DataFrame()  # Inicializa el DataFrame para almacenar resultados
# thres = 0.1  # Umbral inicial
# for i in range(10):
#     var_names = sel_variables(modelos, x, dfy, threshold="{}*mean".format(thres))
#     xtrain = x[var_names]
#     accu_xtrain = medir_modelos(modelos, "r2", xtrain, dfy, 10)
#     df_actual = pd.DataFrame(accu_xtrain.mean(axis=0), columns=['threshold {}'.format(thres)])
#     resultado = pd.concat([resultado, df_actual], axis=1)
#     thres += 0.15
#     thres = round(thres, 2)


# Gráfica de los resultados 
df = df_resultado.T
plt.figure(figsize=(10,10))
sns.lineplot(data=df)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.ylabel("r2")
plt.title("Variacion threshold")

#Mejor threshold para cada modelo
df.idxmax(axis=0)

# #Los dos modelos a tunear son random_forest y decision_tree con un trheshold de 2.2 por la media,
# modelos= [lr, dtr, rfr, gbr]
# var_names = sel_variables(modelos, x, dfy, threshold="0.1*mean")
# var_names.shape
# #Finalmente se escogen 5 variables para entrenar el modelo, se determino este número
# #ya que según la gráfica presentan un desempeño casi igual al threshold con mayor rendimiento

# #tabla final
# xtrainf=x[var_names] 
# #Al final se deja este threshold que da como resultado 5 variables

# #Volvemos a medir el modelo pero con las 5 variables y todas las variables
# #accu_x=funciones.medir_modelos(modelos,"f1",x,dfy,20) ## base con todas las variables 
# accu_xtrainf = medir_modelos(modelos,"r2",xtrainf,dfy,10) ### base con variables seleccionadas
# accu_xtrainf.mean()

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

# param_grid2 = {
#     'n_estimators': [50, 100, 200],  # Número de árboles en el ensamble
#     'learning_rate': [0.01, 0.1, 0.5],  # Tasa de aprendizaje
#     'max_depth': [3, 5, 7],  # Profundidad máxima de cada árbol base
#     'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
#     'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas para estar en un nodo hoja
# }

# rfrtuning = GradientBoostingRegressor()
# grid_search2 = GridSearchCV(rfrtuning, param_grid2, scoring="r2",cv=10, n_jobs=-1)
# grid_result2 = grid_search2.fit(xtrainf, dfy)

# gbr_final = grid_result2.best_estimator_ 
# joblib.dump(gbr_final, "gbr_final.pkl") 


# eval = cross_val_score(rfr_final,xtrainf,dfy,cv=20,scoring="r2")
# eval2 = cross_val_score(gbr_final,xtrainf,dfy,cv=20,scoring="r2")

# np.mean(eval)
# np.mean(eval2)


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

# #Para el gradient boosting
# ypredgbr = gbr_final.predict(xtrainf)
# sns.set_theme(style="ticks")
# dict = {"y":dfy,"ypredrfr":ypredgbr}
# df2 = pd.DataFrame(dict)
# df2 = df2.stack().reset_index()
# df2.drop(columns=["level_0"],inplace=True)
# df2.columns=["tipo","valor"]
# df2["tipo"]=df2["tipo"].apply(lambda x: "real"if x=="y" else "predicho")

# sns.histplot(data=df2, x="valor", hue="tipo")

# ## Metricas comparacion
# # MSE
# mse = mean_squared_error(dfy, ypredrfr)
# print("Mean Squared Error (MSE):", mse)
# # RMSE
# rmse = np.sqrt(mse)
# print("Root Mean Squared Error (RMSE):", rmse)
# # MAE
# mae = mean_absolute_error(dfy, ypredrfr)
# print("Mean Absolute Error (MAE):", mae)
# # R2
# r2 = r2_score(dfy, ypredrfr)
# print("R-squared (R2):", r2)

# lista1 = [mse, rmse,mae,r2]




