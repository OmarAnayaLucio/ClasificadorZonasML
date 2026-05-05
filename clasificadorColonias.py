"""
=============================================================================
CLASIFICADOR DE COLONIAS POR POTENCIAL INMOBILIARIO (PLUSVALÍA)
=============================================================================
Clasifica colonias en: Caliente | Tibia | Fría
Criterio: Potencial de venta inmobiliaria desde perspectiva de marketing
          (plusvalía, gentrificación, seguridad, transporte, demanda, etc.)

Versión: 1.1 (corregida nomenclatura: Tibia, no Tibio)
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import warnings
import os
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────

# ── Pega aquí tu URL de Google Sheets (o ruta local .xlsx / .csv) ──
FILE_PATH = "https://docs.google.com/spreadsheets/d/1Om_F39iZP2w4ZuipC0MIyRAXxjSDgLm6eKP-b6rhos8/edit?gid=2127774564#gid=2127774564"

SHEET_NAME  = 0                          # solo aplica si es .xlsx local
OUTPUT_PATH = "colonias_clasificadas.xlsx"
MODEL_PATH  = "modelo_colonias.pkl"

# Nombres exactos de columnas en tu archivo
COL_CODIGO  = "d_codigo"
COL_MNPIO   = "D_mnpio"
COL_ASENTA  = "d_asenta"
COL_ESTADO  = "d_estado"
COL_TIPO    = "d_tipo_asenta"
COL_ZONA    = "d_zona"
COL_TARGET  = "Tipo Zona"              # columna con etiquetas: Caliente, Tibia, Fría o #N/A

# Etiquetas válidas (exactamente como están escritas en tu archivo)
VALID_LABELS = {"Caliente", "Tibia", "Fría"}

# ─────────────────────────────────────────────
# LECTOR UNIVERSAL DE FUENTE DE DATOS
# ─────────────────────────────────────────────

def _gsheet_id_and_gid(url: str) -> tuple:
    """Extrae spreadsheet_id y gid de una URL de Google Sheets."""
    match_id  = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    match_gid = re.search(r"gid=(\d+)", url)
    sheet_id  = match_id.group(1)  if match_id  else None
    gid       = match_gid.group(1) if match_gid else "0"
    return sheet_id, gid

def load_data(source: str) -> pd.DataFrame:
    """
    Carga datos desde múltiples fuentes de forma automática:
      1. Google Sheets público  → export CSV directo (sin credenciales)
      2. Google Sheets privado  → gspread con OAuth (abre browser 1 vez)
      3. Archivo local .xlsx    → pd.read_excel
      4. Archivo local .csv     → pd.read_csv
    """
    if "docs.google.com/spreadsheets" in source:
        sheet_id, gid = _gsheet_id_and_gid(source)
        if not sheet_id:
            raise ValueError(f"No se pudo extraer el ID del Sheet de: {source}")

        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        try:
            print(f"      Intentando lectura pública...")
            df = pd.read_csv(export_url, dtype=str)
            print(f"      ✓ Lectura pública exitosa ({len(df):,} filas)")
            return df
        except Exception as e_public:
            print(f"      ✗ Modo público falló ({e_public})")

        print("      Intentando autenticación OAuth (abrirá el navegador)...")
        try:
            import gspread
            from gspread_dataframe import get_as_dataframe
            gc = gspread.oauth()
            sh = gc.open_by_key(sheet_id)
            worksheet = next((ws for ws in sh.worksheets() if str(ws.id) == str(gid)), sh.sheet1)
            df = get_as_dataframe(worksheet, dtype=str).dropna(how="all")
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
            print(f"      ✓ Lectura OAuth exitosa ({len(df):,} filas)")
            return df
        except ImportError:
            raise ImportError("Instala: pip install gspread gspread-dataframe google-auth-oauthlib")
        except Exception as e_oauth:
            raise ConnectionError(f"No se pudo acceder al Sheet.\nError público: {e_public}\nError OAuth: {e_oauth}")

    if source.endswith(".csv"):
        return pd.read_csv(source, encoding="utf-8-sig", dtype=str, low_memory=False)
    else:
        return pd.read_excel(source, sheet_name=SHEET_NAME, dtype=str)

# ─────────────────────────────────────────────
# KEYWORDS DE PLUSVALÍA INMOBILIARIA
# ─────────────────────────────────────────────

HOT_KEYWORDS = [
    "residencial", "jardines", "lomas", "pedregal", "polanco",
    "santa fe", "interlomas", "del valle", "narvarte", "condesa",
    "roma", "coyoacán", "bosques", "campestre", "country", "club",
    "real", "grand", "premium", "elite", "luxury", "villas",
    "hacienda", "rancho", "fraccionamiento", "privada", "condominio",
    "torre", "parques", "rinconada", "prado", "arboledas",
    "insurgentes", "reforma", "satelite", "echegaray",
    "florida", "peñas", "fuentes", "manantiales", "vista hermosa",
    "altavista", "tlalpan", "cumbres", "metepec", "juriquilla",
    "vertiz", "del bosque", "san angel", "pedregal",
    "san jerónimo", "jardín", "del parque", "san carlos"
]

WARM_KEYWORDS = [
    "colonia", "unidad", "ampliación", "nueva", "nuevo", "centro",
    "san juan", "san miguel", "san pedro", "santa maría", "santa cruz",
    "san antonio", "san francisco", "el carmen", "la paz", "el sol",
    "el paraíso", "el mirador", "el palmar", "morelos", "hidalgo",
    "independencia", "insurgentes", "reforma", "revolución",
    "san marcos", "san pablo", "san jose", "benito", "juárez",
    "las palmas", "los pinos", "el roble", "el fresno", "las flores",
    "magisterial", "popular", "nueva generación", "industrial",
    "comercial", "las americas", "las torres", "buenavista",
    "lázaro cárdenas", "cuauhtémoc", "venustiano", "federal",
    "estación", "las plazas", "portales"
]

COLD_KEYWORDS = [
    "ejido", "ranchería", "rancho viejo", "barrio", "pueblo",
    "ladera", "cerro", "barranca", "pedregal viejo", "terrero",
    "pastizal", "milpa", "tlaxcala", "campesino", "colectivo",
    "comunal", "paraje", "calvario", "cementerio", "pantéon",
    "agostadero", "boshindo", "bovini", "barrancas", "cañada",
    "chiquito", "viejo", "antiguo", "abandonado"
]

def keyword_score(text: str, keywords: list) -> int:
    """Cuenta cuántos keywords del grupo aparecen en el texto."""
    text = str(text).lower()
    return sum(1 for kw in keywords if kw in text)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features de plusvalía inmobiliaria."""
    feat = pd.DataFrame()
    nombre = (df[COL_ASENTA].fillna("") + " " + df[COL_MNPIO].fillna("")).str.lower()
    tipo_raw = df[COL_TIPO].fillna("Desconocido").str.strip().str.lower()
    zona_raw = df[COL_ZONA].fillna("Rural").str.strip().str.lower()

    feat["es_urbano"] = (zona_raw == "urbano").astype(int)

    tipo_mapa = {
        "fraccionamiento": 5, "zona residencial": 5, "residencial": 5,
        "condominio": 4, "colonia": 3, "unidad habitacional": 3,
        "barrio": 2, "pueblo": 2, "ejido": 1, "ranchería": 1,
        "rancho": 1, "paraje": 1, "desconocido": 2,
    }
    feat["tipo_score"] = tipo_raw.map(tipo_mapa).fillna(2)

    feat["score_hot"]  = nombre.apply(lambda x: keyword_score(x, HOT_KEYWORDS))
    feat["score_warm"] = nombre.apply(lambda x: keyword_score(x, WARM_KEYWORDS))
    feat["score_cold"] = nombre.apply(lambda x: keyword_score(x, COLD_KEYWORDS))
    feat["hot_vs_cold"] = feat["score_hot"] - feat["score_cold"]

    gentry_kw = ["residencial", "jardines", "bosques", "lomas", "pedregal",
                 "country", "real", "grand", "premium", "villas", "hacienda",
                 "fraccionamiento", "condominio", "privada", "torre", "vista"]
    feat["gentry_flag"] = nombre.apply(lambda x: int(any(kw in x for kw in gentry_kw)))

    marginal_kw = ["ejido", "ranchería", "barrio", "pueblo", "ladera",
                   "cerro", "barranca", "milpa", "tlaxcala", "campesino",
                   "rancho", "paraje", "calvario", "agostadero"]
    feat["marginal_flag"] = nombre.apply(lambda x: int(any(kw in x for kw in marginal_kw)))

    feat["len_nombre"] = df[COL_ASENTA].fillna("").str.len()
    feat["tiene_numero"] = df[COL_ASENTA].fillna("").str.contains(r'\d').astype(int)

    feat["estado"] = df[COL_ESTADO].fillna("Desconocido")
    feat["mnpio"]  = df[COL_MNPIO].fillna("Desconocido")
    feat["tipo"]   = df[COL_TIPO].fillna("Desconocido")

    feat["cp_raiz"] = pd.to_numeric(df[COL_CODIGO].astype(str).str[:2], errors="coerce").fillna(0).astype(int)
    feat["cp_region"] = pd.cut(
        feat["cp_raiz"],
        bins=[0, 16, 22, 42, 60, 70, 80, 90, 100],
        labels=["CDMX_Norte", "CDMX_Sur", "Edomex_Urb",
                "Centro_Pais", "Occidente", "Norte", "Pacifico", "Otro"],
        right=False
    ).astype(str)
    return feat

def encode_categoricals(feat: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """Label-encode columnas categóricas."""
    cat_cols = ["estado", "mnpio", "tipo", "cp_region"]
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            feat[col] = le.fit_transform(feat[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            feat[col] = feat[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    return feat, encoders

def build_model():
    """Ensemble Voting de 3 algoritmos."""
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.08, max_depth=4, subsample=0.8, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))])
    model = VotingClassifier(estimators=[("gb", gb), ("rf", rf), ("lr", lr)], voting="soft", weights=[3, 2, 1])
    return model

def main():
    print("=" * 60)
    print("  CLASIFICADOR INMOBILIARIO DE COLONIAS (Caliente | Tibia | Fría)")
    print("=" * 60)

    print("\n[1/6] Cargando datos...")
    df = load_data(FILE_PATH)
    print(f"      Total registros: {len(df):,}")
    print(f"      Columnas: {list(df.columns)}")

    print("\n[2/6] Separando datos etiquetados / por predecir...")
    df[COL_TARGET] = df[COL_TARGET].fillna("#N/A").astype(str).str.strip()
    # Normalizar posibles variantes (ej. "Frio" -> "Fría", pero aquí solo respetamos lo escrito)
    # Si alguien escribió "Frio" sin tilde, lo corregimos:
    df[COL_TARGET] = df[COL_TARGET].replace("Frio", "Fría")

    mask_labeled = df[COL_TARGET].isin(VALID_LABELS)
    df_train = df[mask_labeled].copy()
    df_pred  = df[~mask_labeled].copy()

    print(f"      Etiquetados:       {len(df_train):,}")
    print(f"      Por predecir:      {len(df_pred):,}")
    print(f"      Distribución actual:\n{df_train[COL_TARGET].value_counts()}")

    if len(df_train) == 0:
        raise ValueError("No se encontraron registros con etiquetas válidas. "
                         "Verifica que la columna 'Tipo Zona' contenga 'Caliente', 'Tibia' o 'Fría'.")

    print("\n[3/6] Generando features de plusvalía inmobiliaria...")
    feat_train_raw = extract_features(df_train)
    feat_pred_raw  = extract_features(df_pred)

    feat_train_enc, encoders = encode_categoricals(feat_train_raw.copy(), fit=True)
    feat_pred_enc, _         = encode_categoricals(feat_pred_raw.copy(), encoders=encoders, fit=False)

    ALL_COLS = [c for c in feat_train_enc.columns]  # todas las columnas numéricas
    X_train = feat_train_enc[ALL_COLS].values.astype(float)
    y_train = df_train[COL_TARGET].values
    X_pred  = feat_pred_enc[ALL_COLS].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_pred  = imputer.transform(X_pred)

    print("\n[4/6] Validando modelo (5-fold CV)...")
    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"      Accuracy CV:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"      Scores:       {np.round(cv_scores, 3)}")

    print("\n[5/6] Entrenando modelo final...")
    model.fit(X_train, y_train)

    # Reporte en entrenamiento (con nombres correctos)
    y_pred_train = model.predict(X_train)
    print("\n      Reporte en datos de entrenamiento:")
    print(classification_report(y_train, y_pred_train, target_names=["Caliente", "Fría", "Tibia"], zero_division=0))

    joblib.dump({"model": model, "encoders": encoders, "imputer": imputer}, MODEL_PATH)
    print(f"      Modelo guardado: {MODEL_PATH}")

    print("\n[6/6] Prediciendo colonias sin clasificar...")
    if len(df_pred) > 0:
        preds = model.predict(X_pred)
        proba = model.predict_proba(X_pred)
        class_names = model.classes_  # serán ['Caliente','Fría','Tibia']

        df_pred = df_pred.copy()
        df_pred[COL_TARGET] = preds
        df_pred["confianza_%"] = (proba.max(axis=1) * 100).round(1)
        for i, cls in enumerate(class_names):
            df_pred[f"prob_{cls}"] = (proba[:, i] * 100).round(1)

    # Unir resultados
    df_final = pd.concat([df_train, df_pred], ignore_index=True)
    df_final["confianza_%"] = df_final.get("confianza_%", None)
    df_final.loc[mask_labeled.values[:len(df_final)], "confianza_%"] = None
    df_final["fuente"] = "Modelo ML"
    df_final.loc[mask_labeled.values[:len(df_final)], "fuente"] = "Dato original"

    df_final.to_excel(OUTPUT_PATH, index=False)
    print(f"\n  ✓ Archivo guardado: {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("  DISTRIBUCIÓN FINAL COMPLETA")
    print("=" * 60)
    print(df_final[COL_TARGET].value_counts())
    pct = df_final[COL_TARGET].value_counts(normalize=True) * 100
    print("\nPorcentajes:")
    print(pct.round(1))

    if len(df_pred) > 0:
        print(f"\nConfianza promedio predicciones: {df_pred['confianza_%'].mean():.1f}%")
        low_conf = df_pred[df_pred['confianza_%'] < 60]
        print(f"Predicciones con confianza < 60%: {len(low_conf):,} (revisar manualmente)")

    print("\n  Pipeline completado exitosamente.")
    return df_final

if __name__ == "__main__":
    df_resultado = main()
