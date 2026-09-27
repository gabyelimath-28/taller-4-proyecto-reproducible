"""Funciones de limpieza, procesamiento y anonimización del proyecto."""
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

def anonimizar(df: pd.DataFrame, sal: str) -> pd.DataFrame:
    """Elimina datos identificables, seudonimiza la cédula y generaliza variables sensibles."""
    df_anon = df.copy()
    
    # 1. Seudonimizar cédula con hash SHA-256 + sal
    def hash_cedula(val):
        return hashlib.sha256(f"{val}{sal}".encode("utf-8")).hexdigest()[:12]
    
    if "cedula" in df_anon.columns:
        df_anon["id_seudonimo"] = df_anon["cedula"].apply(hash_cedula)
    
    # 2. Eliminar identificadores directos
    columnas_a_borrar = [c for c in ["nombre", "fecha_nac", "cedula"] if c in df_anon.columns]
    df_anon = df_anon.drop(columns=columnas_a_borrar)
    
    # 3. Generalizar edad en rangos de 10 años y redondear ingresos a la decena
    if "edad" in df_anon.columns:
        df_anon["rango_edad"] = pd.cut(df_anon["edad"], bins=range(10, 90, 10), right=False, labels=[f"{i}-{i+9}" for i in range(10, 80, 10)])
    if "ingreso" in df_anon.columns:
        df_anon["ingreso_redondeado"] = (df_anon["ingreso"] / 10).round() * 10
        
    return df_anon
