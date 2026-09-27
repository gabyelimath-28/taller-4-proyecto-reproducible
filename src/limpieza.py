"""Funciones de limpieza y procesamiento del proyecto."""
import numpy as np
import pandas as pd
import hashlib

def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados, nulos y atípicos extremos mediante 3*IQR."""
    df_clean = df.copy()
    if "id_persona" in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["id_persona"])
        
    q1, q3 = df_clean["ingreso"].quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_superior = q3 + 3 * iqr
    df_clean = df_clean[(df_clean["ingreso"].isna()) | (df_clean["ingreso"] <= limite_superior)]
    df_clean = df_clean.dropna(subset=["ingreso", "anios_educ"]).copy()
    return df_clean

def agregar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la transformación logarítmica del ingreso y categoriza grupos de edad."""
    df_proc = df.copy()
    df_proc["log_ingreso"] = np.log(df_proc["ingreso"])
    df_proc["grupo_edad"] = pd.cut(
        df_proc["edad"], 
        bins=[17, 29, 45, 65], 
        labels=["Jóvenes", "Adultos", "Mayores"]
    )
    return df_proc

def anonimizar(df: pd.DataFrame, sal: str = "clave-secreta-ecuador-2026") -> pd.DataFrame:
    """Anonimiza identificadores directos e indirectos cumpliendo LOPDP."""
    df_a = df.copy()
    if "cedula" in df_a.columns:
        df_a["id_seudonimo"] = df_a["cedula"].apply(
            lambda x: hashlib.sha256((str(x) + sal).encode("utf-8")).hexdigest()[:12]
        )
    columnas_sensibles = [c for c in ["nombre", "fecha_nac", "cedula"] if c in df_a.columns]
    return df_a.drop(columns=columnas_sensibles)
