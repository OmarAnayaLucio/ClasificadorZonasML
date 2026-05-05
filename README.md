# 🏡 Clasificador de Colonias por Potencial Inmobiliario (Plusvalía)

Clasifica automáticamente colonias mexicanas en **Caliente**, **Tibia** o **Fría** según su potencial de venta inmobiliaria, usando técnicas de **machine learning** y criterios de **marketing urbano** (gentrificación, exclusividad, demanda, transporte, seguridad implícita).

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Tabla de contenido

- [¿Qué hace?](#qué-hace)
- [Criterios de clasificación](#criterios-de-clasificación)
- [Requisitos](#requisitos)
- [Instalación y uso](#instalación-y-uso)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Entrenamiento y predicción](#entrenamiento-y-predicción)
- [Resultados](#resultados)
- [Contribuir](#contribuir)
- [Autor](#autor)

---

## 🚀 ¿Qué hace?

Este proyecto entrena un modelo de **ensemble voting** (GradientBoosting + RandomForest + LogisticRegression) para clasificar colonias mexicanas en tres categorías inmobiliarias:

- 🔥 **Caliente**: alta plusvalía, gentrificación activa, demanda constante, zonas premium o en auge.
- 🌤️ **Tibia**: demanda media, potencial de crecimiento, zonas en transición.
- ❄️ **Fría**: baja plusvalía, marginación, poco interés inversor.

**Entrada**: archivo Excel/CSV o Google Sheets con las columnas:
- `d_codigo` (código postal)
- `D_mnpio` (municipio)
- `d_asenta` (nombre de colonia/asentamiento)
- `d_estado` (estado)
- `d_tipo_asenta` (tipo de asentamiento)
- `d_zona` (Urbano / Rural)
- `Tipo Zona` (opcional: `Caliente`, `Tibia`, `Fría` o `#N/A` para predecir)

**Salida**: mismo archivo con la columna `Tipo Zona` completada, más columnas de confianza y probabilidades.

---

## 🧠 Criterios de clasificación

El modelo usa **8 grupos de características** derivadas de los datos:

| # | Característica | ¿Qué mide? |
|---|----------------|-------------|
| 1 | **Tipo de asentamiento** | Fraccionamiento > Colonia > Barrio > Ejido (escala 1-5) |
| 2 | **Palabras clave calientes** | Presencia de “residencial”, “lomas”, “polanco”, “country”, etc. |
| 3 | **Palabras clave tibias** | “colonia”, “centro”, “popular”, “industrial” |
| 4 | **Palabras clave frías** | “ejido”, “ranchería”, “barranca”, “milpa” |
| 5 | **Diferencial hot vs cold** | Score_hot - Score_cold |
| 6 | **Gentrificación flag** | Bandera por términos de exclusividad (villas, privada, torre…) |
| 7 | **Marginalidad flag** | Bandera por términos de baja inversión (cerro, calvario, paraje…) |
| 8 | **Ubicación geográfica** | Estado, municipio, región del CP (CDMX_Norte, Edomex_Urb, etc.) |
| 9 | **Urbano / Rural** | Urbano favorece plusvalía |
| 10 | **Longitud del nombre** | Nombres más largos = zonas más exclusivas |
| 11 | **¿Tiene número?** | Ej. “Ampliación 5 de Mayo” → menos plusvalía |

Estas características se combinan mediante un **ensemble de aprendizaje supervisado** entrenado con datos históricos etiquetados.

---

## 💻 Requisitos

- Python 3.8 o superior
- Librerías (ver `requirements.txt`): # ClasificadorZonasML
