"""Funciones de limpieza y procesamiento del proyecto."""
import numpy as np
import pandas as pd

def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados, nulos y atípicos extremos mediante 3*IQR."""
    df_clean = df.copy()
    
    # Quitar duplicados por identificador
    if "id_persona" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["id_persona"])
        
    # Filtrar atípicos en 'ingreso' por criterio 3*IQR
    q1, q3 = df_clean["ingreso"].quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_superior = q3 + 3 * iqr
    df_clean = df_clean[(df_clean["ingreso"].isna()) | (df_clean["ingreso"] <= limite_superior)]
    
    # Quitar nulos en variables clave
    df_clean = df_clean.dropna(subset=["ingreso", "anios_educ"]).copy()
    
    return df_clean

def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la transformación logarítmica del ingreso y categoriza grupos de edad."""
    df_proc = df.copy()
    
    # Crear log_ingreso
    df_proc["log_ingreso"] = np.log(df_proc["ingreso"])
    
    # Crear grupos de edad
    df_proc["grupo_edad"] = pd.cut(
        df_proc["edad"], 
        bins=[17, 29, 45, 65], 
        labels=["Jóvenes", "Adultos", "Mayores"]
    )
    
    return df_proc
