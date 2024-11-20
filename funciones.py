####### Prueba
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ### para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 
from itertools import product
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
####Este archivo contienen funciones utiles a utilizar en diferentes momentos del proyecto.

#Esta función permite hacer una prueba chi-cuadrado a las variables categóricas 
def prueba_chicuadrado(tabla):
    tabla_cat = tabla.select_dtypes(include=['object']).copy()
    cat_var1=tuple(tabla.select_dtypes(include=["object"]))
    cat_var2=tuple(tabla.select_dtypes(include=["object"]))
    cat_var_prod = list(product(cat_var1, cat_var2, repeat=1))
    result=[]
    for i in cat_var_prod:
        result.append((i[0],i[1],list(ss.chi2_contingency(pd.crosstab(
                            tabla_cat[i[0]], tabla_cat[i[1]])))[1]))
    resultado=[x for x in result if x[2]>0.05]
    return resultado
    
#Esta función permite ver los valores únicos por variable
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('#---------------------------------------------#')
    
###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas
def ejecutar_sql (nombre_archivo, cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close
  cur.executescript(sql_as_string)
  
 
def imputar_f(df,list_cat):  
        
    
    df_c=df[list_cat]
    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer(strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)

    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)

    df =pd.concat([df_n,df_c],axis=1)
    return df


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



def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos

#La función se modifica para adaptarlo a nuestro programa 
def preparar_datos (df):
   
    

    #######Cargar y procesar nuevos datos ######
   
    
    #### Cargar modelo y listas 
    
   
    #list_cat=joblib.load("list_cat.pkl")
    list_dummies=joblib.load("salidas\\list_dummies.pkl")
    var_names=joblib.load("salidas\\var_names.pkl")
    scaler=joblib.load( "salidas\\scaler.pkl") 

    ####Ejecutar funciones de transformaciones
    
    #df=imputar_f(df,list_cat)
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    df_dummies= df_dummies.loc[:,~df_dummies.columns.isin(['EmployeeID'])]
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    X=X[var_names]
    
    
    
    
    return X

#### Función para variar el treshold y sacar el mejor desempeño###

#Procesamiento de las bases de datos
def procesar_datos(df, scaler=None):
    
    df_dummizado = pd.get_dummies(df, columns=['HomeOwnership', 'Education', 'MaritalStatus'], dtype=int)
    
    if scaler is None:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_dummizado), columns=df_dummizado.columns)
    else:
        df_scaled = pd.DataFrame(scaler.transform(df_dummizado), columns=df_dummizado.columns)
    
    return df_scaled, scaler