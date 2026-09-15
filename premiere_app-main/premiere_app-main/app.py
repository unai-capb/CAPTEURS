# app_controle_metrologique_capteurs_commentaires.py
# Lancement :
#   pip install -r requirements.txt
#   streamlit run app_controle_metrologique_capteurs_commentaires.py

from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

st.set_page_config(page_title="Contrôle de cohérence et analyse des conditions ambiantes", page_icon="", layout="wide")

# Palette qualitative à fort contraste (facile à distinguer) pour tous les
# graphiques catégoriels (courbes multi-capteurs, barres, camemberts, etc.).
px.defaults.color_discrete_sequence = px.colors.qualitative.Safe

# Échelle "qualité de l'air" : vert foncé (bon / bas) -> jaune -> rouge (mauvais / élevé).
AIR_QUALITY_COLORSCALE = [
    [0.0, "#1b5e20"],
    [0.5, "#f2c200"],
    [1.0, "#b71c1c"],
]

# Échelle divergente utilisée pour la température et l'humidité :
# vert foncé au centre = zone de confort ("normal"), rouge aux extrêmes
# (trop bas comme trop haut).
COMFORT_COLORSCALE = [
    [0.00, "#b71c1c"],
    [0.20, "#ef6c00"],
    [0.50, "#1b5e20"],
    [0.80, "#ef6c00"],
    [1.00, "#b71c1c"],
]


def build_comfort_zrange(zmid, data_min, data_max, half_span_floor):
    """
    Calcule un intervalle [zmin, zmax] symétrique autour de `zmid` afin que
    le point central du COMFORT_COLORSCALE (vert foncé) corresponde bien à
    la valeur "normale" `zmid`, et que le rouge corresponde aux extrêmes.
    """
    half_span = max(abs(data_max - zmid), abs(zmid - data_min), half_span_floor, 1e-6)
    return zmid - half_span, zmid + half_span


VARIABLES = {
    "temperature": {"label": "Température [°C]", "unit": "°C", "tolerance": 0.4, "keywords": ["vint1", "temperature", "température", "°c"]},
    "humidite": {"label": "Humidité [%HR]", "unit": "%HR", "tolerance": 2.0, "keywords": ["vint2", "humidite", "humidité", "%hr", "% hr"]},
    "co2": {"label": "CO₂ [ppm]", "unit": "ppm", "tolerance": None, "keywords": ["vint3", "co2", "co₂", "ppm"]},
}

SECTION_LABELS = {
    "A": "Avant installation : capteurs ensemble",
    "B": "Préparation / installation : exclue",
    "C": "Installé : mesures effectives",
    "D": "Fin / retrait : exclue",
    "E": "De nouveau ensemble",
}
SECTION_USE = {"A": "Contrôle fiabilité", "B": "Exclue", "C": "Analyse logements", "D": "Exclue", "E": "Contrôle fiabilité"}
CALIBRATION_SECTIONS = ["A", "E"]


def normalize_colname(col):
    return " ".join(str(col).strip().split()).lower()


def find_column(columns, keywords):
    for col in columns:
        norm = normalize_colname(col)
        if any(k.lower() in norm for k in keywords):
            return col
    return None


def standardize_one_sheet(df, sensor_name):
    df = df.copy()
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]
    date_col = find_column(df.columns, ["date"])
    temp_col = find_column(df.columns, VARIABLES["temperature"]["keywords"])
    hum_col = find_column(df.columns, VARIABLES["humidite"]["keywords"])
    co2_col = find_column(df.columns, VARIABLES["co2"]["keywords"])

    missing = []
    if date_col is None:
        missing.append("Date")
    if temp_col is None:
        missing.append("Température")
    if hum_col is None:
        missing.append("Humidité")
    if co2_col is None:
        missing.append("CO2")
    if missing:
        raise ValueError(f"Colonnes non reconnues dans '{sensor_name}' : {missing}. Colonnes disponibles : {list(df.columns)}")

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "temperature": pd.to_numeric(df[temp_col], errors="coerce"),
        "humidite": pd.to_numeric(df[hum_col], errors="coerce"),
        "co2": pd.to_numeric(df[co2_col], errors="coerce"),
    }).dropna(subset=["Date"]).sort_values("Date")
    out["capteur"] = sensor_name
    return out[["Date", "capteur", "temperature", "humidite", "co2"]]


@st.cache_data
def load_sensor_file(uploaded_file):
    """
    Charge le fichier principal de mesures.

    Formats acceptés :
    - XLSX / XLS : chaque feuille est considérée comme un capteur ;
    - CSV : le fichier entier est considéré comme un capteur, nommé d'après
      le nom du fichier. Le séparateur est détecté automatiquement.
    """
    filename = getattr(uploaded_file, "name", "capteur")
    suffix = Path(filename).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix in {".xlsx", ".xls"}:
        try:
            sheets = pd.read_excel(BytesIO(raw), sheet_name=None)
        except ImportError as e:
            if suffix == ".xls":
                raise ImportError(
                    "La lecture des fichiers .xls nécessite le paquet 'xlrd'. "
                    "Installez-le avec : pip install xlrd"
                ) from e
            raise
        return {
            name: standardize_one_sheet(df, name)
            for name, df in sheets.items()
        }

    if suffix == ".csv":
        # Lecture robuste des CSV.
        #
        # Certains CSV exportés depuis Excel commencent par une ligne spéciale :
        #     sep=;
        #
        # Cette ligne n'est pas l'en-tête réel du tableau. Si elle n'est pas
        # retirée, pandas peut interpréter les colonnes comme
        # ['sep=', 'Unnamed: 1'].
        last_error = None

        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text_csv = raw.decode(encoding)
            except UnicodeDecodeError as e:
                last_error = e
                continue

            lines = text_csv.splitlines()

            while lines and not lines[0].strip():
                lines.pop(0)

            if not lines:
                raise ValueError("Le fichier CSV est vide.")

            first_line = lines[0].strip()

            # Cas typique d'un CSV Excel : sep=; ou sep=,
            sep_match = re.match(
                r"^sep\s*=\s*(.)\s*$",
                first_line,
                flags=re.IGNORECASE
            )

            if sep_match:
                separator = sep_match.group(1)
                csv_content = "\n".join(lines[1:])

                if not csv_content.strip():
                    raise ValueError(
                        "Le fichier CSV contient une directive 'sep=' mais aucune donnée."
                    )

                df = pd.read_csv(
                    BytesIO(csv_content.encode(encoding)),
                    sep=separator,
                    decimal=",",
                    encoding=encoding,
                )

            else:
                # Aucun sep= : pandas détecte automatiquement le séparateur
                df = pd.read_csv(
                    BytesIO(raw),
                    sep=None,
                    engine="python",
                    decimal=",",
                    encoding=encoding,
                )

            # Supprime les colonnes entièrement vides créées par certains exports
            df = df.dropna(axis=1, how="all")

            sensor_name = Path(filename).stem or "capteur_csv"

            return {
                sensor_name: standardize_one_sheet(
                    df,
                    sensor_name
                )
            }

        if last_error is not None:
            raise last_error

    raise ValueError(
        f"Format non pris en charge : {suffix or 'sans extension'}. "
        "Formats acceptés : .xlsx, .xls et .csv."
    )


@st.cache_data
def load_external_temperature(uploaded_file):
    """
    Lit un CSV de température extérieure.

    Le fichier attendu contient au minimum :
    - une colonne de date ;
    - une colonne de température extérieure.

    Les séparateurs ';' et les décimales avec virgule sont pris en charge.
    """
    df = pd.read_csv(
        uploaded_file,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    df.columns = [" ".join(str(c).strip().split()) for c in df.columns]

    date_col = find_column(df.columns, ["date"])
    temp_col = find_column(
        df.columns,
        ["température extérieure", "temperature exterieure",
         "temp extérieure", "temp exterieure", "extérieur", "exterieur"],
    )

    if date_col is None or temp_col is None:
        raise ValueError(
            "Le CSV extérieur doit contenir une colonne Date et une colonne "
            "Température extérieure. Colonnes détectées : "
            f"{list(df.columns)}"
        )

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
        "temperature_exterieure": pd.to_numeric(df[temp_col], errors="coerce"),
    })
    return (
        out.dropna(subset=["Date", "temperature_exterieure"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )


def merge_external_temperature(df, external_df, tolerance="90min"):
    """
    Aligne la température extérieure sur les dates du capteur par voisin le
    plus proche. Les points trop éloignés restent manquants.
    """
    if external_df is None or external_df.empty or df.empty:
        return df.copy()

    left = df.sort_values("Date").copy()
    right = external_df.sort_values("Date").copy()

    return pd.merge_asof(
        left,
        right,
        on="Date",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )


def summarize_sheets(cleaned):
    rows = []
    for sensor, df in cleaned.items():
        rows.append({
            "capteur": sensor, "début": df["Date"].min(), "fin": df["Date"].max(), "n_mesures": len(df),
            "temperature_min": df["temperature"].min(), "temperature_max": df["temperature"].max(),
            "humidite_min": df["humidite"].min(), "humidite_max": df["humidite"].max(),
            "co2_min": df["co2"].min(), "co2_max": df["co2"].max(),
        })
    return pd.DataFrame(rows)


def common_period(cleaned):
    return max(df["Date"].min() for df in cleaned.values()), min(df["Date"].max() for df in cleaned.values())


def datetime_selector(label, default_value, min_value, max_value, key):
    """
    Streamlit ne possède pas st.datetime_input.
    On combine donc st.date_input et st.time_input.
    """
    default_value = pd.to_datetime(default_value)
    min_value = pd.to_datetime(min_value)
    max_value = pd.to_datetime(max_value)
    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input(f"Date {label}", value=default_value.date(), min_value=min_value.date(), max_value=max_value.date(), key=f"{key}_date")
    with c2:
        t = st.time_input(f"Heure {label}", value=default_value.time(), key=f"{key}_time")
    return pd.Timestamp(datetime.combine(d, t))


def assign_sections(cleaned, section_dates):
    assigned = {}
    intervals = [(sec, pd.to_datetime(start), pd.to_datetime(end)) for sec, (start, end) in section_dates.items()]
    for sensor, df in cleaned.items():
        tmp = df.copy()
        tmp["section"] = "hors_section"
        for sec, start, end in intervals:
            tmp.loc[(tmp["Date"] >= start) & (tmp["Date"] < end), "section"] = sec
        assigned[sensor] = tmp
    return assigned


def get_section_counts(assigned):
    rows = []
    for sensor, df in assigned.items():
        counts = df["section"].value_counts()
        row = {s: int(counts.get(s, 0)) for s in ["A", "B", "C", "D", "E", "hors_section"]}
        row["capteur"] = sensor
        row["A+E_calibration"] = row["A"] + row["E"]
        row["B+D_exclues"] = row["B"] + row["D"]
        row["Total_sections"] = row["A"] + row["B"] + row["C"] + row["D"] + row["E"]
        row["Total_fichier"] = len(df)
        rows.append(row)
    return pd.DataFrame(rows)[["capteur", "A", "B", "C", "D", "E", "hors_section", "A+E_calibration", "B+D_exclues", "Total_sections", "Total_fichier"]]


def resample_sensor(df, freq):
    if df.empty:
        return pd.DataFrame()
    sensor = df["capteur"].iloc[0]
    tmp = (df.set_index("Date")[list(VARIABLES.keys())]
             .sort_index()
             .resample(freq)
             .mean()
             .interpolate(method="time", limit_area="inside"))
    tmp["capteur"] = sensor
    return tmp.reset_index()


def build_wide_tables(assigned, freq="30min", sections=None):
    parts = []
    for _, df in assigned.items():
        tmp = df.copy()
        if sections is not None:
            tmp = tmp[tmp["section"].isin(sections)].copy()
        if tmp.empty:
            continue
        r = resample_sensor(tmp, freq)
        if not r.empty:
            parts.append(r)
    if not parts:
        return {}
    long_df = pd.concat(parts, ignore_index=True)
    return {var: long_df.pivot_table(index="Date", columns="capteur", values=var, aggfunc="mean").sort_index() for var in VARIABLES}


def tolerance_series(var, ref):
    if var == "co2":
        return 50 + 0.03 * ref.abs()
    return pd.Series(VARIABLES[var]["tolerance"], index=ref.index)


def compute_classic_metrics(wide_tables):
    rows = []
    for var, table in wide_tables.items():
        if table.empty or table.shape[1] < 2:
            continue
        for sensor in table.columns:
            others = [c for c in table.columns if c != sensor]
            val = table[sensor]
            ref = table[others].mean(axis=1)
            valid = pd.concat([val, ref], axis=1).dropna()
            if valid.empty:
                continue
            val = valid.iloc[:, 0]
            ref = valid.iloc[:, 1]
            err = val - ref
            abs_err = err.abs()
            conform = abs_err <= tolerance_series(var, ref)
            rows.append({
                "variable": var,
                "variable_label": VARIABLES[var]["label"],
                "capteur": sensor,
                "n_points": len(valid),
                "biais_moyen": err.mean(),
                "MAE": abs_err.mean(),
                "RMSE": np.sqrt(np.mean(err ** 2)),
                "ecart_max_abs": abs_err.max(),
                "tolerance": "±50 ppm ±3 % lecture" if var == "co2" else f"±{VARIABLES[var]['tolerance']} {VARIABLES[var]['unit']}",
                "pct_points_conformes": conform.mean() * 100,
                "diagnostic_classique": "OK" if conform.mean() >= 0.80 else "À vérifier",
            })
    return pd.DataFrame(rows)


def compute_zscore_metrics(wide_tables, z1=1.0, z2=2.0, z3=3.0):
    rows, ztables, spread_rows = [], {}, []
    for var, table in wide_tables.items():
        if table.empty or table.shape[1] < 2:
            continue
        mu = table.mean(axis=1)
        sigma = table.std(axis=1, ddof=0).replace(0, np.nan)
        ztab = table.sub(mu, axis=0).div(sigma, axis=0)
        ztables[var] = ztab
        s = sigma.dropna()
        spread_rows.append({
            "variable": var,
            "variable_label": VARIABLES[var]["label"],
            "ecart_type_moyen_entre_capteurs": s.mean(),
            "ecart_type_max_entre_capteurs": s.max(),
        })
        for sensor in table.columns:
            z = ztab[sensor].dropna()
            if z.empty:
                continue
            pct2 = (z.abs() > z2).mean() * 100
            rows.append({
                "variable": var,
                "variable_label": VARIABLES[var]["label"],
                "capteur": sensor,
                "n_points": len(z),
                "z_moyen": z.mean(),
                "z_abs_moyen": z.abs().mean(),
                "z_abs_max": z.abs().max(),
                f"pct_abs_z_sup_{z1}": (z.abs() > z1).mean() * 100,
                f"pct_abs_z_sup_{z2}": pct2,
                f"pct_abs_z_sup_{z3}": (z.abs() > z3).mean() * 100,
                "diagnostic_zscore": "Très cohérent" if pct2 < 5 else "Acceptable" if pct2 < 10 else "Suspect",
            })
    return pd.DataFrame(rows), pd.DataFrame(spread_rows), ztables


def compute_variation_metrics(wide_tables):
    rows, delta_tables = [], {}
    for var, table in wide_tables.items():
        delta = table.diff().dropna(how="all")
        delta_tables[var] = delta
        if delta.empty or delta.shape[1] < 2:
            continue
        corr = delta.corr()
        for sensor in delta.columns:
            others = [c for c in delta.columns if c != sensor]
            val = delta[sensor]
            ref = delta[others].mean(axis=1)
            valid = pd.concat([val, ref], axis=1).dropna()
            if valid.empty:
                continue
            err = valid.iloc[:, 0] - valid.iloc[:, 1]
            mcorr = corr.loc[sensor, others].mean()
            rows.append({
                "variable": var,
                "variable_label": VARIABLES[var]["label"],
                "capteur": sensor,
                "biais_variation": err.mean(),
                "MAE_variation": err.abs().mean(),
                "RMSE_variation": np.sqrt(np.mean(err ** 2)),
                "corr_moyenne_variations": mcorr,
                "diagnostic_variation": "Bonne dynamique" if mcorr >= .80 else "Dynamique moyenne" if mcorr >= .50 else "Dynamique faible",
            })
    return pd.DataFrame(rows), delta_tables


def minmax_score(series, higher_is_better=False):
    s = pd.to_numeric(series, errors="coerce")
    if s.max() == s.min():
        return pd.Series(100.0, index=s.index)
    return 100 * (s - s.min()) / (s.max() - s.min()) if higher_is_better else 100 * (s.max() - s) / (s.max() - s.min())


def build_global_ranking(classic, zmetrics, vmetrics):
    if classic.empty:
        return pd.DataFrame()
    c = classic.groupby("capteur", as_index=False).agg(
        MAE_moyenne=("MAE", "mean"),
        RMSE_moyen=("RMSE", "mean"),
        pct_conforme_moyen=("pct_points_conformes", "mean"),
    )
    if not zmetrics.empty:
        col_z2 = [col for col in zmetrics.columns if col.startswith("pct_abs_z_sup_2")]
        zcol = col_z2[0] if col_z2 else "z_abs_moyen"
        z = zmetrics.groupby("capteur", as_index=False).agg(z_abs_moyen=("z_abs_moyen", "mean"), pct_z2_moyen=(zcol, "mean"))
        c = c.merge(z, on="capteur", how="left")
    else:
        c["z_abs_moyen"], c["pct_z2_moyen"] = np.nan, np.nan
    if not vmetrics.empty:
        v = vmetrics.groupby("capteur", as_index=False).agg(corr_variations_moyenne=("corr_moyenne_variations", "mean"))
        c = c.merge(v, on="capteur", how="left")
    else:
        c["corr_variations_moyenne"] = np.nan
    c["score_MAE"] = minmax_score(c["MAE_moyenne"], False)
    c["score_RMSE"] = minmax_score(c["RMSE_moyen"], False)
    c["score_z"] = minmax_score(c["pct_z2_moyen"].fillna(c["pct_z2_moyen"].median()), False)
    c["score_variations"] = minmax_score(c["corr_variations_moyenne"].fillna(c["corr_variations_moyenne"].median()), True)
    c["score_global"] = 0.30 * c["score_MAE"] + 0.20 * c["score_RMSE"] + 0.30 * c["score_z"] + 0.20 * c["score_variations"]
    c = c.sort_values("score_global", ascending=False).reset_index(drop=True)
    c["rang"] = np.arange(1, len(c) + 1)
    c["diagnostic_global"] = np.where(c["score_global"] >= 75, "Très cohérent", np.where(c["score_global"] >= 50, "Acceptable", "À vérifier"))
    return c


def prepare_section_c(dfC):
    """Ajoute les variables temporelles utilisées dans l’analyse de la section C."""
    dfC = dfC.copy()
    dfC["Hour"] = dfC["Date"].dt.hour
    dfC["DayOfWeek"] = dfC["Date"].dt.dayofweek
    dfC["DayName"] = dfC["Date"].dt.day_name()
    dfC["Date_only"] = dfC["Date"].dt.date
    dfC["Week"] = dfC["Date"].dt.isocalendar().week.astype(int)
    dfC["Month"] = dfC["Date"].dt.month_name()
    return dfC


def compute_c_alerts(dfC, temp_min, temp_max, hum_min, hum_max, co2_warn, co2_crit, confort_override=None):
    """
    Compte les dépassements des seuils de confort et de qualité d’air.

    `confort_override` (Series booléenne, optionnelle) : si fournie (ex. issue
    du diagramme de Givoni), remplace le calcul de confort thermique conjoint
    température + humidité, qui sinon est dérivé des plages fixes.
    """
    n = len(dfC)
    if n == 0:
        return {}, pd.DataFrame()
    if confort_override is not None:
        hors_confort = int((~confort_override.reindex(dfC.index).fillna(False)).sum())
    else:
        hors_confort = int((
            ~(
                (dfC["temperature"] >= temp_min) & (dfC["temperature"] <= temp_max)
                & (dfC["humidite"] >= hum_min) & (dfC["humidite"] <= hum_max)
            )
        ).sum())
    values = {
        "Température > max": int((dfC["temperature"] > temp_max).sum()),
        "Température < min": int((dfC["temperature"] < temp_min).sum()),
        "Humidité > max": int((dfC["humidite"] > hum_max).sum()),
        "Humidité < min": int((dfC["humidite"] < hum_min).sum()),
        "Hors confort thermique (T + HR conjoint)": hors_confort,
        "CO₂ alerte": int(((dfC["co2"] >= co2_warn) & (dfC["co2"] < co2_crit)).sum()),
        "CO₂ critique": int((dfC["co2"] >= co2_crit).sum()),
    }
    summary = pd.DataFrame({
        "Alerte": list(values.keys()),
        "Occurrences": list(values.values()),
        "% du total": [100 * v / n for v in values.values()],
    })
    return values, summary


def comfort_score_section_c(dfC, temp_min, temp_max, hum_min, hum_max, co2_warn, co2_crit, confort_override=None):
    """
    Calcule un score synthétique de confort et de qualité d’air sur 100.

    Correction méthodologique par rapport à la version précédente : le
    confort thermique est désormais évalué de manière CONJOINTE
    (température ET humidité simultanément dans la plage acceptable),
    au lieu de deux pourcentages indépendants sommés séparément. Un
    logement peut être "température OK" 90 % du temps et "humidité OK"
    90 % du temps sans jamais l'être aux deux en même temps — l'ancienne
    version ne détectait pas ce cas. `confort_override` permet d'utiliser
    à la place le résultat du diagramme de Givoni.
    """
    if dfC.empty:
        return np.nan, {}
    temp_ok = ((dfC["temperature"] >= temp_min) & (dfC["temperature"] <= temp_max)).mean() * 100
    hum_ok = ((dfC["humidite"] >= hum_min) & (dfC["humidite"] <= hum_max)).mean() * 100
    if confort_override is not None:
        confort_thermique = confort_override.reindex(dfC.index).fillna(False).mean() * 100
    else:
        confort_thermique = (
            (dfC["temperature"] >= temp_min) & (dfC["temperature"] <= temp_max)
            & (dfC["humidite"] >= hum_min) & (dfC["humidite"] <= hum_max)
        ).mean() * 100
    co2_ok = (dfC["co2"] < co2_warn).mean() * 100
    co2_critical = (dfC["co2"] >= co2_crit).mean() * 100
    score = 0.55 * confort_thermique + 0.45 * co2_ok - 0.25 * co2_critical
    return float(np.clip(score, 0, 100)), {
        "temp_ok_%": temp_ok,
        "humidite_ok_%": hum_ok,
        "confort_thermique_%": confort_thermique,
        "co2_ok_%": co2_ok,
        "co2_critique_%": co2_critical,
    }


def make_section_c_report(dfC, sensor, temp_min, temp_max, hum_min, hum_max, co2_warn, co2_crit, score,
                           methode_confort="Plage fixe (température & humidité)", confort_details=None):
    if dfC.empty:
        return "Aucune donnée disponible en section C."
    pct_temp = ((dfC["temperature"] >= temp_min) & (dfC["temperature"] <= temp_max)).mean() * 100
    pct_hum = ((dfC["humidite"] >= hum_min) & (dfC["humidite"] <= hum_max)).mean() * 100
    pct_warn = (dfC["co2"] >= co2_warn).mean() * 100
    pct_crit = (dfC["co2"] >= co2_crit).mean() * 100
    diagnostic = "satisfaisante" if score >= 85 else "moyenne" if score >= 65 else "à surveiller"
    confort_thermique_pct = (confort_details or {}).get("confort_thermique_%", np.nan)
    return f"""# Rapport automatique — Section C

Capteur / logement : {sensor}
Période : {dfC['Date'].min()} → {dfC['Date'].max()}
Nombre de mesures : {len(dfC)}

Température moyenne : {dfC['temperature'].mean():.1f} °C
Humidité moyenne : {dfC['humidite'].mean():.1f} %HR
CO₂ moyen : {dfC['co2'].mean():.0f} ppm
CO₂ maximal : {dfC['co2'].max():.0f} ppm

Méthode de seuils de confort thermique : {methode_confort}
Température dans la plage [{temp_min}, {temp_max}] °C : {pct_temp:.1f} %
Humidité dans la plage [{hum_min}, {hum_max}] %HR : {pct_hum:.1f} %
Confort thermique conjoint (T + HR simultanément) : {confort_thermique_pct:.1f} %
CO₂ supérieur à {co2_warn} ppm : {pct_warn:.1f} %
CO₂ supérieur à {co2_crit} ppm : {pct_crit:.1f} %

Score de qualité ambiante : {score:.0f}/100
Diagnostic : qualité ambiante {diagnostic}.

Cette analyse décrit le logement associé au capteur et ne compare pas directement les capteurs entre eux.
"""



def time_in_interval(hour, start_hour, end_hour):
    """Retourne True pour les heures comprises dans un intervalle, y compris s'il traverse minuit."""
    if start_hour <= end_hour:
        return (hour >= start_hour) & (hour < end_hour)
    return (hour >= start_hour) | (hour < end_hour)


def add_occupancy_status(df, weekday_start, weekday_end, weekend_occupied=False,
                         weekend_start=9, weekend_end=18):
    """Ajoute les colonnes occupation, nuit et week-end selon le calendrier choisi."""
    out = df.copy()
    hour_decimal = out["Date"].dt.hour + out["Date"].dt.minute / 60
    is_weekend = out["Date"].dt.dayofweek >= 5
    occupied_week = (~is_weekend) & time_in_interval(hour_decimal, weekday_start, weekday_end)
    occupied_weekend = is_weekend & weekend_occupied & time_in_interval(hour_decimal, weekend_start, weekend_end)
    out["Occupation"] = np.where(occupied_week | occupied_weekend, "Occupé", "Inoccupé")
    out["Est_weekend"] = is_weekend
    out["Est_nuit"] = time_in_interval(hour_decimal, 22, 6)
    return out


def sensor_reliability_summary(sensor, ranking_ae, ranking_a, ranking_e):
    """Produit un indicateur utilisable en C à partir des contrôles A, E et A+E."""
    def score_of(df):
        if df is None or df.empty or sensor not in set(df["capteur"]):
            return np.nan
        return float(df.loc[df["capteur"] == sensor, "score_global"].iloc[0])

    score_ae = score_of(ranking_ae)
    score_a = score_of(ranking_a)
    score_e = score_of(ranking_e)
    # Ordre de repli corrigé : A+E si disponible, sinon la moyenne de A et E
    # s'ils sont tous les deux connus, sinon celui des deux qui est disponible.
    # (Auparavant, un score A valide était ignoré si A+E était manquant et
    # que E l'était aussi, ce qui produisait "Indéterminée" à tort.)
    if pd.notna(score_ae):
        base = score_ae
    elif pd.notna(score_a) and pd.notna(score_e):
        base = (score_a + score_e) / 2
    elif pd.notna(score_e):
        base = score_e
    elif pd.notna(score_a):
        base = score_a
    else:
        base = np.nan
    if pd.isna(base):
        level, message = "Indéterminée", "Pas assez de données de contrôle pour qualifier ce capteur."
    elif base >= 75:
        level, message = "Bonne", "Les résultats en C peuvent être interprétés avec une bonne confiance métrologique."
    elif base >= 50:
        level, message = "Moyenne", "Attention : le capteur peut présenter un décalage. Interprétez surtout les tendances."
    else:
        level, message = "Faible", "Attention, le capteur semble donner des valeurs décalées ou incohérentes."

    evolution = "Non calculable"
    if pd.notna(score_a) and pd.notna(score_e):
        delta = score_e - score_a
        evolution = "Stable" if abs(delta) < 10 else "Amélioration en E" if delta > 0 else "Dégradation en E"
    return {
        "score_AE": score_ae, "score_A": score_a, "score_E": score_e,
        "niveau": level, "message": message, "evolution": evolution,
    }


def fmt_score(value):
    return "N/D" if pd.isna(value) else f"{value:.0f}/100"


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")



# TEMPÉRATURE EXTÉRIEURE VIA API MÉTÉO (Open-Meteo — gratuite, sans clé)


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_ville_meteo(nom_ville):
    """Convertit un nom de ville en coordonnées GPS (API de géocodage Open-Meteo)."""
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": nom_ville, "count": 1, "language": "fr"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            return None
        res = data["results"][0]
        return {
            "lat": res["latitude"],
            "lon": res["longitude"],
            "nom": res.get("name", nom_ville),
            "pays": res.get("country", ""),
        }
    except Exception:
        return None


@st.cache_data(show_spinner="Récupération de la météo historique…", ttl=3600)
def fetch_historical_temperature(lat, lon, date_debut, date_fin):
    """
    Récupère la température extérieure horaire historique via l'API gratuite
    Open-Meteo Archive, sur la période couvrant les mesures des capteurs.
    Contrairement à une API de prévision, l'API "archive" fournit les
    valeurs réellement observées aux dates passées — c'est ce qu'il faut
    ici, puisqu'on compare à des mesures déjà enregistrées.
    """
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": pd.Timestamp(date_debut).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(date_fin).strftime("%Y-%m-%d"),
                "hourly": "temperature_2m",
                "timezone": "auto",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        if not hourly.get("time"):
            return None
        out = pd.DataFrame({
            "Date": pd.to_datetime(hourly["time"]),
            "temperature_exterieure": hourly["temperature_2m"],
        })
        return (
            out.dropna(subset=["Date", "temperature_exterieure"])
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )
    except Exception:
        return None


@st.cache_data(show_spinner="Récupération des prévisions météo…", ttl=1800)
def fetch_forecast_temperature(lat, lon, date_debut, date_fin):
    """
    Récupère la température extérieure horaire prévue via l'API gratuite
    Open-Meteo Forecast. Utilisée pour les dates actuelles ou futures
    (jusqu'à ~16 jours à l'avance selon la disponibilité du modèle).
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": pd.Timestamp(date_debut).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(date_fin).strftime("%Y-%m-%d"),
                "hourly": "temperature_2m",
                "timezone": "auto",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        if not hourly.get("time"):
            return None
        out = pd.DataFrame({
            "Date": pd.to_datetime(hourly["time"]),
            "temperature_exterieure": hourly["temperature_2m"],
        })
        return (
            out.dropna(subset=["Date", "temperature_exterieure"])
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )
    except Exception:
        return None


def fetch_external_temperature_range(lat, lon, date_debut, date_fin):
    """
    Récupère la température extérieure sur une période donnée, qu'elle soit
    entièrement passée, entièrement future, ou à cheval sur les deux :
    - la partie antérieure à aujourd'hui est récupérée via l'API historique
      (Archive) ;
    - la partie à partir d'aujourd'hui est récupérée via l'API de prévision
      (Forecast, ~16 jours à l'avance selon la disponibilité du modèle).
    Les deux morceaux sont ensuite concaténés en une seule série continue.
    """
    today = pd.Timestamp.now().normalize()
    date_debut = pd.Timestamp(date_debut).normalize()
    date_fin = pd.Timestamp(date_fin).normalize()
    if date_debut > date_fin:
        date_debut, date_fin = date_fin, date_debut

    frames = []
    hist_end = min(date_fin, today - pd.Timedelta(days=1))
    if date_debut <= hist_end:
        hist = fetch_historical_temperature(lat, lon, date_debut, hist_end)
        if hist is not None and not hist.empty:
            frames.append(hist)

    fut_start = max(date_debut, today)
    if date_fin >= fut_start:
        fut = fetch_forecast_temperature(lat, lon, fut_start, date_fin)
        if fut is not None and not fut.empty:
            frames.append(fut)

    if not frames:
        return None
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )



# PSYCHROMÉTRIE ET DIAGRAMME DE GIVONI


ATM_PRESSURE_PA = 101325.0  # pression atmosphérique standard au niveau de la mer


def saturation_vapor_pressure(temp_c):
    """Pression de vapeur saturante (Pa) à une température donnée — formule de Magnus-Tetens."""
    temp_c = np.asarray(temp_c, dtype=float)
    return 611.2 * np.exp(17.62 * temp_c / (243.12 + temp_c))


def humidity_ratio_from_rh(temp_c, rh_percent, p_atm=ATM_PRESSURE_PA):
    """
    Calcule l'humidité absolue (kg d'eau / kg d'air sec) à partir de la
    température sèche (°C) et de l'humidité relative (%). C'est l'axe des
    ordonnées classique du diagramme de l'air humide (diagramme de Givoni).
    """
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_percent, dtype=float), 0, 100) / 100.0
    e = rh * saturation_vapor_pressure(temp_c)
    e = np.minimum(e, p_atm - 1.0)  # garde-fou numérique (évite division par ~0)
    return 0.622 * e / (p_atm - e)


# Bornes indicatives des zones de confort du diagramme de Givoni (activité
# sédentaire, habillement d'été). Les températures de bascule par vitesse
# d'air (27 / 29 / 31 / 33 °C) reprennent les ordres de grandeur usuellement
# rapportés dans la littérature bioclimatique (travaux de Givoni ; guides de
# confort d'été passif en climat tropical humide). Ce sont des valeurs
# indicatives et ajustables, pas une norme figée — à recaler sur votre
# propre référentiel si besoin.
GIVONI_DEFAULTS = {
    "t_min": 20.0,
    "w_min": 0.004,
    "w_max": 0.012,
    "t_max_dry": {
        "0 m/s": 27.0, "0,5 m/s": 29.0, "1 m/s": 31.0, "1,5 m/s": 33.0,
    },
    "reduction_humide": 3.0,
}

GIVONI_SPEED_ORDER = ["0 m/s", "0,5 m/s", "1 m/s", "1,5 m/s"]

GIVONI_ZONE_COLORS = {
    "0 m/s": "#e64a19",
    "0,5 m/s": "#f57c00",
    "1 m/s": "#ff9800",
    "1,5 m/s": "#ffca28",
}


def givoni_zone_polygon(t_min, t_max_dry, t_max_humid, w_min, w_max):
    """
    Construit le quadrilatère (température, humidité absolue) d'une zone de
    confort de Givoni : plus tolérant en température à faible humidité,
    moins tolérant à forte humidité (bascule vers la gauche en haut).
    """
    return [
        (t_min, w_min),
        (t_max_dry, w_min),
        (t_max_humid, w_max),
        (t_min, w_max),
    ]


def build_givoni_zones(t_min, w_min, w_max, t_max_dry_by_speed, reduction_humide):
    """Construit les 4 polygones de zones de confort (un par vitesse d'air), imbriqués."""
    zones = {}
    for speed_label, t_max_dry in t_max_dry_by_speed.items():
        t_max_humid = t_max_dry - reduction_humide
        zones[speed_label] = givoni_zone_polygon(t_min, t_max_dry, t_max_humid, w_min, w_max)
    return zones


def point_in_convex_polygon(t, w, polygon):
    """Test point-dans-polygone convexe, vectorisé (t, w = tableaux numpy de même taille)."""
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    sign = None
    inside = np.ones_like(t, dtype=bool)
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (w - y1) - (y2 - y1) * (t - x1)
        s = cross >= 0
        if sign is None:
            sign = s
        inside &= (s == sign)
    return inside


def classify_givoni_comfort(temp_c, rh_percent, zones, speed_order=GIVONI_SPEED_ORDER):
    """
    Pour chaque mesure, renvoie la vitesse d'air minimale nécessaire pour
    atteindre le confort ("0 m/s", "0,5 m/s", ...), ou "Hors confort" si
    aucune des zones ne la contient (même à 1,5 m/s).
    """
    w = humidity_ratio_from_rh(temp_c, rh_percent)
    t_values = np.asarray(temp_c, dtype=float)
    labels = np.full(len(t_values), "Hors confort", dtype=object)
    still_out = np.ones(len(t_values), dtype=bool)
    for label in speed_order:
        inside = point_in_convex_polygon(t_values, w, zones[label]) & still_out
        labels[inside] = label
        still_out = still_out & ~inside
    index = temp_c.index if hasattr(temp_c, "index") else None
    return pd.Series(labels, index=index)


def make_givoni_figure(zones, speed_order, scatter_df=None, t_range=(16, 36)):
    """Construit le diagramme de Givoni : zones de confort + isolignes d'humidité relative + points mesurés."""
    fig = go.Figure()

    t_axis = np.linspace(t_range[0], t_range[1], 100)
    for rh in [20, 40, 60, 80, 100]:
        w_line = humidity_ratio_from_rh(t_axis, rh)
        fig.add_trace(go.Scatter(
            x=t_axis, y=w_line, mode="lines",
            line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dot"),
            name=f"{rh}% HR", hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=t_range[1], y=humidity_ratio_from_rh(np.array([t_range[1]]), rh)[0],
            text=f"{rh}%", showarrow=False, font=dict(size=9, color="gray"),
            xanchor="left",
        )

    # Zones dessinées de la plus large (1,5 m/s) à la plus étroite (0 m/s)
    # pour que les zones les plus restrictives restent visibles au premier plan.
    for label in reversed(speed_order):
        poly = zones[label] + [zones[label][0]]
        xs, ys = zip(*poly)
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines", fill="toself",
            fillcolor=GIVONI_ZONE_COLORS.get(label, "#cccccc"),
            opacity=0.55, line=dict(color=GIVONI_ZONE_COLORS.get(label, "#cccccc")),
            name=label,
        ))

    if scatter_df is not None and not scatter_df.empty:
        fig.add_trace(go.Scatter(
            x=scatter_df["temperature"], y=scatter_df["w"], mode="markers",
            marker=dict(size=4, color="rgba(20,20,20,0.35)"),
            name="Mesures",
            hovertemplate="T=%{x:.1f} °C<br>w=%{y:.4f} kg/kg<extra></extra>",
        ))

    fig.update_layout(
        title="Diagramme de Givoni — zones de confort selon la vitesse d'air",
        xaxis_title="Température sèche intérieure (°C)",
        yaxis_title="Humidité absolue (kg eau / kg air sec)",
        xaxis=dict(range=list(t_range)),
        height=520,
        legend=dict(orientation="h"),
    )
    return fig


# INTERFACE


st.title("Contrôle de cohérence et analyse des conditions ambiantes")
st.markdown("---")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Importer les mesures (XLSX, XLS ou CSV)",
        type=["xlsx", "xls", "csv"],
        help=(
            "Excel : chaque feuille est traitée comme un capteur. "
            "CSV : le fichier est traité comme un seul capteur."
        ),
    )

    st.markdown("---")
    st.subheader("Température extérieure (facultatif)")
    source_temp_ext = st.radio(
        "Source",
        ["Aucune", "Fichier CSV", "API météo (Open-Meteo, historique ou prévision)"],
        horizontal=False,
    )
    external_temperature_file = None
    ville_meteo_ext = None
    if source_temp_ext == "Fichier CSV":
        external_temperature_file = st.file_uploader(
            "Importer la température extérieure (CSV)",
            type=["csv"],
            help=(
                "Format attendu : une colonne Date et une colonne "
                "Température extérieure. Le fichier temperature.csv fourni "
                "est directement compatible."
            ),
        )
    elif source_temp_ext == "API météo (Open-Meteo, historique ou prévision)":
        ville_meteo_ext = st.text_input(
            "Ville proche des logements",
            placeholder="ex : Bayonne, Hasparren, Biarritz…",
        )
        st.caption(
            "La période à récupérer (passée et/ou future) se choisit plus "
            "bas, une fois le fichier Excel chargé. Les dates passées "
            "utilisent la météo réellement observée (API Open-Meteo "
            "Archive) ; les dates à partir d'aujourd'hui utilisent les "
            "prévisions (API Open-Meteo Forecast, ~16 jours à l'avance). "
            "Nécessite un accès internet sur la machine qui exécute "
            "l'application."
        )

    st.markdown("---")
    freq = st.selectbox("Pas temporel", ["30min", "15min", "1H"], index=0)
    z1 = st.number_input("Seuil z faible", value=1.0, step=0.5)
    z2 = st.number_input("Seuil z alerte", value=2.0, step=0.5)
    z3 = st.number_input("Seuil z fort", value=3.0, step=0.5)

    st.markdown("---")
    st.subheader("Tolérances des métriques classiques (A+E)")
    st.caption(
        "Ces tolérances définissent le seuil de conformité utilisé dans "
        "l'onglet « Classique » (auparavant fixées en dur dans le code)."
    )
    tol_temp_classic = st.number_input(
        "Tolérance — Température (°C)",
        value=float(VARIABLES["temperature"]["tolerance"]), step=0.1,
    )
    tol_hum_classic = st.number_input(
        "Tolérance — Humidité (%HR)",
        value=float(VARIABLES["humidite"]["tolerance"]), step=0.5,
    )
    VARIABLES["temperature"]["tolerance"] = tol_temp_classic
    VARIABLES["humidite"]["tolerance"] = tol_hum_classic

    st.markdown("---")
    st.subheader("Seuils de la section C")
    temp_min = st.number_input("Température min (°C)", value=16.0, step=0.5)
    temp_max = st.number_input("Température max (°C)", value=26.0, step=0.5)
    hum_min = st.number_input("Humidité min (%HR)", value=30.0, step=1.0)
    hum_max = st.number_input("Humidité max (%HR)", value=60.0, step=1.0)
    co2_warn = st.number_input("CO₂ — seuil d’alerte (ppm)", value=800, step=50)
    co2_crit = st.number_input("CO₂ — seuil critique (ppm)", value=1000, step=50)

    st.markdown("---")
    st.subheader("Méthode des seuils de confort thermique")
    methode_confort = st.radio(
        "Méthode",
        ["Plage fixe (température & humidité)", "Diagramme de Givoni (bioclimatique)"],
        help=(
            "La plage fixe évalue température et humidité indépendamment. "
            "Le diagramme de Givoni évalue leur combinaison conjointe, avec "
            "un effet favorable de la ventilation (vitesse d'air) sur la "
            "tolérance à la chaleur."
        ),
    )
    if methode_confort == "Diagramme de Givoni (bioclimatique)":
        vitesse_air = st.selectbox(
            "Vitesse d'air supposée dans le logement",
            GIVONI_SPEED_ORDER,
            index=0,
            help=(
                "Une vitesse d'air plus élevée (brassage d'air, ventilation "
                "naturelle) élargit la zone de confort acceptable en "
                "température."
            ),
        )
        with st.expander("Paramètres avancés du diagramme de Givoni"):
            st.caption(
                "Valeurs par défaut indicatives, inspirées de la littérature "
                "bioclimatique (Givoni). Ajustables si votre référentiel diffère."
            )
            g_t_min = st.number_input(
                "Température min de confort (°C)",
                value=GIVONI_DEFAULTS["t_min"], step=0.5,
            )
            g_w_min = st.number_input(
                "Humidité absolue min (kg/kg)",
                value=GIVONI_DEFAULTS["w_min"], step=0.001, format="%.3f",
            )
            g_w_max = st.number_input(
                "Humidité absolue max (kg/kg)",
                value=GIVONI_DEFAULTS["w_max"], step=0.001, format="%.3f",
            )
            g_reduction = st.number_input(
                "Réduction du confort en air humide (°C)",
                value=GIVONI_DEFAULTS["reduction_humide"], step=0.5,
            )
            g_t_max = {}
            cols_g = st.columns(2)
            for i, (label, default_t) in enumerate(GIVONI_DEFAULTS["t_max_dry"].items()):
                g_t_max[label] = cols_g[i % 2].number_input(
                    f"T max — {label}", value=default_t, step=0.5,
                    key=f"givoni_tmax_{label}",
                )
    else:
        vitesse_air = None
        g_t_min = GIVONI_DEFAULTS["t_min"]
        g_w_min = GIVONI_DEFAULTS["w_min"]
        g_w_max = GIVONI_DEFAULTS["w_max"]
        g_reduction = GIVONI_DEFAULTS["reduction_humide"]
        g_t_max = dict(GIVONI_DEFAULTS["t_max_dry"])

# --- Validations des seuils (corrige l'absence de contrôle de cohérence) ---
if temp_min >= temp_max:
    st.sidebar.error("La température min doit être strictement inférieure à la température max.")
    st.stop()
if hum_min >= hum_max:
    st.sidebar.error("L'humidité min doit être strictement inférieure à l'humidité max.")
    st.stop()
if co2_warn >= co2_crit:
    st.sidebar.error("Le seuil CO₂ d'alerte doit être strictement inférieur au seuil critique.")
    st.stop()
if g_w_min >= g_w_max:
    st.sidebar.error("L'humidité absolue min (Givoni) doit être strictement inférieure à la max.")
    st.stop()

if uploaded_file is None:
    st.info("Importe un fichier XLSX, XLS ou CSV pour commencer.")
    st.stop()

try:
    cleaned = load_sensor_file(uploaded_file)
except Exception as e:
    st.error(f"Erreur lors de la lecture : {e}")
    st.stop()

summary = summarize_sheets(cleaned)
auto_start, auto_end = common_period(cleaned)
min_date, max_date = summary["début"].min(), summary["fin"].max()

external_temperature = None
if source_temp_ext == "Fichier CSV" and external_temperature_file is not None:
    try:
        external_temperature = load_external_temperature(
            external_temperature_file
        )
        st.sidebar.success(
            f"Température extérieure chargée : "
            f"{len(external_temperature)} mesures"
        )
    except Exception as e:
        st.sidebar.error(
            f"Erreur dans le fichier de température extérieure : {e}"
        )
elif source_temp_ext == "API météo (Open-Meteo, historique ou prévision)" and ville_meteo_ext:
    geo = geocode_ville_meteo(ville_meteo_ext)
    if geo is None:
        st.sidebar.error(
            "Ville introuvable ou API injoignable (vérifiez la connexion internet)."
        )
    else:
        st.sidebar.markdown("**Période météo à récupérer**")
        st.sidebar.caption(
            "Par défaut : la période couverte par le fichier de mesures. "
            "Choisissez des dates futures pour obtenir des prévisions."
        )
        meteo_date_debut = st.sidebar.date_input(
            "Date de début",
            value=pd.Timestamp(min_date).date(),
            min_value=pd.Timestamp(min_date).date() - pd.Timedelta(days=365 * 5),
            max_value=pd.Timestamp.now().date() + pd.Timedelta(days=16),
            key="meteo_date_debut",
        )
        meteo_date_fin = st.sidebar.date_input(
            "Date de fin",
            value=max(pd.Timestamp(max_date).date(), pd.Timestamp.now().date()),
            min_value=pd.Timestamp(min_date).date() - pd.Timedelta(days=365 * 5),
            max_value=pd.Timestamp.now().date() + pd.Timedelta(days=16),
            key="meteo_date_fin",
        )
        if meteo_date_debut > meteo_date_fin:
            st.sidebar.error("La date de début doit précéder la date de fin.")
        else:
            external_temperature = fetch_external_temperature_range(
                geo["lat"], geo["lon"], meteo_date_debut, meteo_date_fin
            )
            if external_temperature is None or external_temperature.empty:
                st.sidebar.error(
                    "Impossible de récupérer la météo pour cette période "
                    "(API injoignable, ou date trop éloignée dans le passé "
                    "pour l'archive / dans le futur pour la prévision)."
                )
            else:
                nb_jours_futurs = int((
                    pd.Timestamp(meteo_date_fin) - pd.Timestamp.now().normalize()
                ).days)
                mention_prevision = (
                    " (inclut des prévisions)" if nb_jours_futurs >= 0 else ""
                )
                st.sidebar.success(
                    f"Météo chargée pour {geo['nom']}, {geo['pays']} : "
                    f"{len(external_temperature)} mesures horaires"
                    f"{mention_prevision}"
                )

st.subheader("1. Feuilles détectées")
st.dataframe(summary, width="stretch")


# DÉFINITION DES SECTIONS


st.subheader("2. Définition des sections A/B/C/D/E")

st.info("""
**Comment définir les sections ?**

L'étude est basée sur cinq périodes :

🟢 **A — Avant installation**  
Les capteurs sont placés ensemble avant leur installation.  
Cette période sert à vérifier leur cohérence initiale.

🔴 **B — Installation / préparation**  
Les capteurs sont manipulés ou déplacés vers les logements.  
Cette période est exclue, car l'installation n'est pas simultanée.

🔵 **C — Mesures effectives**  
Les capteurs sont installés dans les différents logements.  
Cette période sert à analyser les logements, mais pas à comparer directement les capteurs entre eux.

🔴 **D — Retrait**  
Les capteurs sont récupérés ou déplacés.  
Cette période est exclue, car le retrait n'est pas forcément simultané.

🟢 **E — De nouveau ensemble**  
Les capteurs sont replacés ensemble après la campagne de mesure.  
Cette période sert à vérifier leur cohérence finale.

Le contrôle métrologique utilise uniquement **A + E**.
""")

############################################################################################################
############################################################################################################


mode = st.radio("Mode de définition", ["Manuel", "Automatique"], horizontal=True)

default_A_start = auto_start
default_A_end = auto_start + pd.Timedelta(hours=16, minutes=30)
default_B_start = default_A_end
default_B_end = default_B_start + pd.Timedelta(hours=16)
default_E_end = auto_end
default_E_start = auto_end - pd.Timedelta(days=5)
default_D_end = default_E_start
default_D_start = default_D_end - pd.Timedelta(days=1)
default_C_start = default_B_end
default_C_end = default_D_start

if mode == "Manuel":
    st.info("""
     **Mode manuel recommandé**

    C'est le mode le plus adapté ici, car je peux exclure précisément les périodes
    de manipulation, d'installation et de retrait.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🟢 A — Avant installation")
        st.caption("Période où les capteurs sont ensemble et immobiles. Utilisée pour le contrôle initial.")
        A_start = datetime_selector("début A", default_A_start, min_date, max_date, "A_start")
        A_end = datetime_selector("fin A", default_A_end, min_date, max_date, "A_end")

        st.markdown("### 🔴 B — Installation / préparation")
        st.caption("Période exclue : capteurs manipulés ou installés à des moments différents.")
        B_start = datetime_selector("début B", default_B_start, min_date, max_date, "B_start")
        B_end = datetime_selector("fin B", default_B_end, min_date, max_date, "B_end")

        st.markdown("### 🔵 C — Mesures effectives")
        st.caption("Période d'analyse des logements. Les capteurs ne doivent pas être comparés entre eux ici.")
        C_start = datetime_selector("début C", default_C_start, min_date, max_date, "C_start")
        C_end = datetime_selector("fin C", default_C_end, min_date, max_date, "C_end")

    with c2:
        st.markdown("### 🔴 D — Retrait")
        st.caption("Période exclue : récupération ou déplacement des capteurs.")
        D_start = datetime_selector("début D", default_D_start, min_date, max_date, "D_start")
        D_end = datetime_selector("fin D", default_D_end, min_date, max_date, "D_end")

        st.markdown("### 🟢 E — De nouveau ensemble")
        st.caption("Période où les capteurs sont replacés ensemble. Utilisée pour le contrôle final.")
        E_start = datetime_selector("début E", default_E_start, min_date, max_date, "E_start")
        E_end = datetime_selector("fin E", default_E_end, min_date, max_date, "E_end")
else:
    st.info("""
    ⚙️ **Mode automatique**

    L'application découpe automatiquement les données :
    - A commence au premier instant commun ;
    - B suit A ;
    - E termine au dernier instant commun ;
    - D précède E ;
    - C correspond à la période centrale.

    Ce mode est pratique, mais le mode manuel est plus précis ici.
    """)
    col1, col2, col3, col4 = st.columns(4)
    nb_A = col1.number_input("Durée A en jours", min_value=0.1, value=1.0, step=0.5)
    nb_B = col2.number_input("Durée B en jours", min_value=0.1, value=1.0, step=0.5)
    nb_D = col3.number_input("Durée D en jours", min_value=0.1, value=1.0, step=0.5)
    nb_E = col4.number_input("Durée E en jours", min_value=0.1, value=5.0, step=0.5)
    A_start, A_end = auto_start, auto_start + pd.Timedelta(days=float(nb_A))
    B_start, B_end = A_end, A_end + pd.Timedelta(days=float(nb_B))
    E_end, E_start = auto_end, auto_end - pd.Timedelta(days=float(nb_E))
    D_end, D_start = E_start, E_start - pd.Timedelta(days=float(nb_D))
    C_start, C_end = B_end, D_start

section_dates = {"A": (A_start, A_end), "B": (B_start, B_end), "C": (C_start, C_end), "D": (D_start, D_end), "E": (E_start, E_end)}
section_table = pd.DataFrame([{"section": s, "description": SECTION_LABELS[s], "utilisation": SECTION_USE[s], "début": a, "fin": b} for s, (a, b) in section_dates.items()])

st.markdown("### Sections utilisées")
st.dataframe(section_table, width="stretch")

st.success("""
**Vérification logique des sections**

- **A + E** : utilisées pour comparer les capteurs lorsqu'ils sont ensemble ;
- **B + D** : exclues, car les capteurs sont manipulés ou déplacés ;
- **C** : utilisée uniquement pour analyser les logements.

L'objectif est de comparer les capteurs uniquement lorsqu'ils sont dans le même environnement.
""")

if (section_table["début"] >= section_table["fin"]).any():
    st.error("Certaines sections ont une date de début supérieure ou égale à la date de fin.")
    st.stop()

assigned = assign_sections(cleaned, section_dates)
section_counts = get_section_counts(assigned)

st.subheader("3. Nombre de mesures par section")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mesures A+E", int(section_counts["A+E_calibration"].sum()))
m2.metric("Mesures C", int(section_counts["C"].sum()))
m3.metric("Exclues B+D", int(section_counts["B+D_exclues"].sum()))
m4.metric("Hors section", int(section_counts["hors_section"].sum()))
st.dataframe(section_counts, width="stretch")
st.caption("Ce tableau sert à vérifier que le masquage temporel est correct. Si A ou E contient trop peu de mesures, le contrôle de fiabilité sera moins robuste.")

wide_AE = build_wide_tables(assigned, freq=freq, sections=CALIBRATION_SECTIONS)
classic = compute_classic_metrics(wide_AE)
zmetrics, spread, ztables = compute_zscore_metrics(wide_AE, z1, z2, z3)
vmetrics, delta_tables = compute_variation_metrics(wide_AE)
ranking = build_global_ranking(classic, zmetrics, vmetrics)

# Contrôles séparés pour distinguer la fiabilité initiale (A) et finale (E).
def ranking_for_sections(section_list):
    wide = build_wide_tables(assigned, freq=freq, sections=section_list)
    c = compute_classic_metrics(wide)
    z, _, _ = compute_zscore_metrics(wide, z1, z2, z3)
    v, _ = compute_variation_metrics(wide)
    return build_global_ranking(c, z, v)

ranking_A = ranking_for_sections(["A"])
ranking_E = ranking_for_sections(["E"])

tabs = st.tabs(["Vue sections", "Classique", "Z-score / écart-type", "Variations", "Corrélations", "Classement", "Analyse section C", "Rapport"])

with tabs[0]:
    st.subheader("Vue visuelle du masquage A/B/C/D/E")
    st.info("Cette vue permet de vérifier visuellement que les sections sont bien placées. Les sections A et E doivent correspondre aux périodes où les capteurs sont ensemble.")
    var = st.selectbox("Variable", list(VARIABLES.keys()), format_func=lambda x: VARIABLES[x]["label"])
    parts = [df[["Date", "capteur", "section", var]].rename(columns={var: "valeur"}) for df in assigned.values()]
    long_df = pd.concat(parts, ignore_index=True)
    fig = px.scatter(long_df, x="Date", y="valeur", color="section", facet_row="capteur", height=900, title=f"Découpage - {VARIABLES[var]['label']}")
    fig.update_yaxes(matches=None)
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    st.subheader("Analyse classique : biais, MAE, RMSE")
    st.info("""
      **Important**

    Un écart entre deux capteurs ne signifie pas forcément qu'un capteur est défectueux.

    - **Décalage constant, ou offset** : les capteurs suivent les mêmes variations avec une différence fixe. Une recalibration peut souvent corriger ce problème.
    - **Décalage variable** : le capteur ne suit pas les autres de manière stable. Il est alors plus suspect.

    C'est pourquoi l'application calcule aussi les z-scores, les corrélations et les variations.
    """)
    st.dataframe(classic, width="stretch")
    if not classic.empty:
        st.download_button("Télécharger métriques classiques", to_csv(classic), "metriques_classiques.csv", "text/csv")
        st.plotly_chart(px.bar(classic, x="capteur", y="MAE", color="variable_label", barmode="group", title="MAE par capteur"), width="stretch")

with tabs[2]:
    st.subheader("Analyse statistique : z-score et écart-type")
    st.info("""
    Le z-score indique à quel point un capteur s'éloigne du comportement collectif.

    - |z| < 1 : mesure proche du groupe ;
    - |z| entre 1 et 2 : écart modéré ;
    - |z| > 2 : mesure atypique ;
    - |z| > 3 : mesure fortement atypique.
    """)
    st.markdown("### Dispersion entre capteurs")
    st.dataframe(spread, width="stretch")
    st.markdown("### Z-score")
    st.dataframe(zmetrics, width="stretch")
    if not zmetrics.empty:
        zcols = [c for c in zmetrics.columns if c.startswith("pct_abs_z_sup_")]
        zcol = st.selectbox("Colonne z-score", zcols) if zcols else None
        if zcol:
            st.plotly_chart(px.bar(zmetrics, x="capteur", y=zcol, color="variable_label", barmode="group", title=f"Pourcentage de points atypiques : {zcol}"), width="stretch")

with tabs[3]:
    st.subheader("Analyse des variations")
    st.info("Cette analyse compare les variations successives des capteurs. Elle distingue un simple offset constant d'un comportement réellement différent.")
    st.dataframe(vmetrics, width="stretch")
    if not vmetrics.empty:
        st.plotly_chart(px.bar(vmetrics, x="capteur", y="corr_moyenne_variations", color="variable_label", barmode="group", title="Corrélation moyenne des variations"), width="stretch")

with tabs[4]:
    st.subheader("Corrélations sur A + E")
    st.info("La corrélation mesure si les capteurs évoluent ensemble. Une corrélation élevée ne signifie pas forcément que les valeurs sont identiques.")
    for var, table in wide_AE.items():
        st.markdown(f"### {VARIABLES[var]['label']}")
        corr = table.corr()
        st.dataframe(corr, width="stretch")
        st.plotly_chart(px.imshow(corr, text_auto=".3f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title=f"Corrélation - {VARIABLES[var]['label']}"), width="stretch")

with tabs[5]:
    st.subheader("Classement global")
    st.info("Le score global combine MAE, RMSE, z-score et corrélations des variations. Il est plus robuste qu'un seul indicateur isolé.")
    st.dataframe(ranking, width="stretch")
    if not ranking.empty:
        st.download_button("Télécharger classement", to_csv(ranking), "classement_global.csv", "text/csv")
        st.plotly_chart(px.bar(ranking, x="capteur", y="score_global", color="diagnostic_global", title="Score global des capteurs"), width="stretch")

with tabs[6]:
    st.subheader("Analyse détaillée de la section C : qualité ambiante")
    st.info(
        "Les filtres temporels, l’occupation, le décalage temporel et la fiabilité "
        "métrologique sont appliqués avant le calcul des indicateurs et graphiques."
    )

    selected_sensor = st.selectbox(
        "Choisir un capteur / logement",
        list(assigned.keys()),
        key="sensor_C_analysis",
    )
    raw_dfC = assigned[selected_sensor][
        assigned[selected_sensor]["section"] == "C"
    ].copy()

    if raw_dfC.empty:
        st.warning("Aucune donnée en section C pour ce capteur.")
    else:
        st.markdown("### Paramètres d’analyse de la pièce")
        p1, p2, p3 = st.columns(3)

        with p1:
            standard_start = st.number_input(
                "Début d’occupation standard — semaine",
                0.0, 23.5, 8.0, 0.5,
                key="occ_std_start",
            )
            standard_end = st.number_input(
                "Fin d’occupation standard — semaine",
                0.0, 24.0, 18.0, 0.5,
                key="occ_std_end",
            )

        with p2:
            weekend_occupied_std = st.checkbox(
                "Occupation standard le week-end",
                value=False,
                key="occ_weekend_std",
            )
            weekend_start_std = st.number_input(
                "Début week-end",
                0.0, 23.5, 9.0, 0.5,
                key="occ_we_start",
            )
            weekend_end_std = st.number_input(
                "Fin week-end",
                0.0, 24.0, 18.0, 0.5,
                key="occ_we_end",
            )

        with p3:
            custom_occ = st.checkbox(
                "Affiner l’occupation pour cette pièce",
                value=False,
                key=f"custom_occ_{selected_sensor}",
            )
            time_shift = st.slider(
                "Décalage de la série temporelle (heures)",
                -24.0, 24.0, 0.0, 0.5,
                key=f"shift_{selected_sensor}",
            )
            show_expert = st.checkbox(
                "Afficher les données expertes",
                value=False,
                key="show_expert_C",
            )

        if custom_occ:
            q1, q2, q3 = st.columns(3)
            room_start = q1.number_input(
                "Début occupation de la pièce",
                0.0, 23.5, float(standard_start), 0.5,
                key=f"room_start_{selected_sensor}",
            )
            room_end = q2.number_input(
                "Fin occupation de la pièce",
                0.0, 24.0, float(standard_end), 0.5,
                key=f"room_end_{selected_sensor}",
            )
            room_weekend = q3.checkbox(
                "Pièce occupée le week-end",
                value=weekend_occupied_std,
                key=f"room_weekend_{selected_sensor}",
            )
        else:
            room_start = standard_start
            room_end = standard_end
            room_weekend = weekend_occupied_std

        dfC_all = raw_dfC.copy()
        dfC_all["Date_originale"] = dfC_all["Date"]
        dfC_all["Date"] = (
            dfC_all["Date"] + pd.to_timedelta(time_shift, unit="h")
        )
        dfC_all = prepare_section_c(dfC_all)
        dfC_all = add_occupancy_status(
            dfC_all,
            room_start,
            room_end,
            room_weekend,
            weekend_start_std,
            weekend_end_std,
        )

        # Ajout facultatif de la température extérieure.
        dfC_all = merge_external_temperature(
            dfC_all,
            external_temperature,
            tolerance="90min",
        )
        if "temperature_exterieure" in dfC_all.columns:
            dfC_all["delta_temperature_int_ext"] = (
                dfC_all["temperature"]
                - dfC_all["temperature_exterieure"]
            )

        cmin_ts = dfC_all["Date"].min()
        cmax_ts = dfC_all["Date"].max()

        # --- Filtre date + heure de la section C, dans la colonne de gauche ---
        with st.sidebar:
            st.markdown("---")
            st.subheader("🕓 Section C — période à visualiser")
            st.caption(
                f"Capteur « {selected_sensor} » — données disponibles du "
                f"{cmin_ts:%d/%m/%Y %H:%M} au {cmax_ts:%d/%m/%Y %H:%M}."
            )
            c_period_start = datetime_selector(
                "début affichage",
                cmin_ts, cmin_ts, cmax_ts,
                key=f"c_period_start_{selected_sensor}",
            )
            c_period_end = datetime_selector(
                "fin affichage",
                cmax_ts, cmin_ts, cmax_ts,
                key=f"c_period_end_{selected_sensor}",
            )
            if c_period_start > c_period_end:
                st.error("La date/heure de début doit précéder la fin.")
                c_period_start, c_period_end = c_period_end, c_period_start

        f2, f3 = st.columns(2)
        with f2:
            occupancy_filter = st.multiselect(
                "Occupation prise en compte",
                ["Occupé", "Inoccupé"],
                default=["Occupé", "Inoccupé"],
                key="occ_filter_C",
            )
        with f3:
            special_period = st.radio(
                "Période particulière",
                ["Toutes", "Uniquement la nuit", "Uniquement les week-ends"],
                key="special_period_C",
            )

        dfC_period = dfC_all.copy()
        dfC_period = dfC_period[
            (dfC_period["Date"] >= c_period_start)
            & (dfC_period["Date"] <= c_period_end)
        ]

        if special_period == "Uniquement la nuit":
            dfC_period = dfC_period[dfC_period["Est_nuit"]]
        elif special_period == "Uniquement les week-ends":
            dfC_period = dfC_period[dfC_period["Est_weekend"]]

        # Le camembert utilise toute la période choisie, avant le filtre
        # Occupé/Inoccupé, afin de conserver une répartition informative.
        occupancy_counts = (
            dfC_period["Occupation"]
            .value_counts()
            .rename_axis("Occupation")
            .reset_index(name="Mesures")
        )

        dfC = dfC_period[
            dfC_period["Occupation"].isin(occupancy_filter)
        ].copy()

        reliability = sensor_reliability_summary(
            selected_sensor,
            ranking,
            ranking_A,
            ranking_E,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fiabilité A + E", fmt_score(reliability["score_AE"]))
        r2.metric("Fiabilité en A", fmt_score(reliability["score_A"]))
        r3.metric("Fiabilité en E", fmt_score(reliability["score_E"]))
        r4.metric("Évolution A → E", reliability["evolution"])

        if reliability["niveau"] == "Bonne":
            st.success(
                f"Fiabilité {reliability['niveau']} — "
                f"{reliability['message']}"
            )
        elif reliability["niveau"] == "Moyenne":
            st.warning(
                f"Fiabilité {reliability['niveau']} — "
                f"{reliability['message']}"
            )
        else:
            st.error(
                f"Fiabilité {reliability['niveau']} — "
                f"{reliability['message']}"
            )

        if dfC.empty:
            st.warning(
                "Aucune donnée ne correspond aux filtres temporels "
                "et d’occupation."
            )
        else:
            n = len(dfC)

            # Confort thermique : plage fixe (par défaut) ou diagramme de
            # Givoni (si sélectionné dans la barre latérale). Les deux
            # méthodes alimentent ensuite les mêmes indicateurs / le même
            # score, pour rester comparables.
            confort_override = None
            givoni_zones = build_givoni_zones(g_t_min, g_w_min, g_w_max, g_t_max, g_reduction)
            if methode_confort == "Diagramme de Givoni (bioclimatique)":
                dfC = dfC.copy()
                dfC["w_givoni"] = humidity_ratio_from_rh(dfC["temperature"], dfC["humidite"])
                dfC["Zone_Givoni_minimale"] = classify_givoni_comfort(
                    dfC["temperature"], dfC["humidite"], givoni_zones
                )
                confort_override = pd.Series(
                    point_in_convex_polygon(
                        dfC["temperature"].values, dfC["w_givoni"].values,
                        givoni_zones[vitesse_air],
                    ),
                    index=dfC.index,
                )

            alerts, alert_summary = compute_c_alerts(
                dfC,
                temp_min,
                temp_max,
                hum_min,
                hum_max,
                co2_warn,
                co2_crit,
                confort_override=confort_override,
            )
            iaq_score, details = comfort_score_section_c(
                dfC,
                temp_min,
                temp_max,
                hum_min,
                hum_max,
                co2_warn,
                co2_crit,
                confort_override=confort_override,
            )

            st.markdown("###  Statistiques globales")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric(
                " Temp. moy.",
                f"{dfC['temperature'].mean():.1f} °C",
                f"σ = {dfC['temperature'].std():.1f}",
            )
            c2.metric(
                " Humidité moy.",
                f"{dfC['humidite'].mean():.1f} %",
                f"σ = {dfC['humidite'].std():.1f}",
            )
            c3.metric(
                "💨 CO₂ moyen",
                f"{dfC['co2'].mean():.0f} ppm",
                f"Max : {dfC['co2'].max():.0f} ppm",
            )
            c4.metric(
                " CO₂ alerte",
                f"{alerts['CO₂ alerte']} mesures",
                f"{100 * alerts['CO₂ alerte'] / n:.1f} %",
            )
            c5.metric(
                " CO₂ critique",
                f"{alerts['CO₂ critique']} mesures",
                f"{100 * alerts['CO₂ critique'] / n:.1f} %",
            )
            c6.metric(
                "Score qualité ambiante",
                f"{iaq_score:.0f}/100",
                f"{dfC['Date_only'].nunique()} jours",
            )

            st.markdown("### Répartition de l’occupation")
            occ_pie_col, temp_pie_col = st.columns(2)

            with occ_pie_col:
                if not occupancy_counts.empty:
                    fig_pie = px.pie(
                        occupancy_counts,
                        names="Occupation",
                        values="Mesures",
                        hole=0.35,
                        title="Part des mesures occupées / inoccupées",
                    )
                    st.plotly_chart(fig_pie, width="stretch")

            with temp_pie_col:
                occupied_df = dfC_period[
                    dfC_period["Occupation"] == "Occupé"
                ].copy()
                if occupied_df.empty:
                    st.info(
                        "Aucune mesure occupée sur la période sélectionnée."
                    )
                else:
                    st.markdown("#### Classification indicative de la température")
                    st.caption(
                        "Lecture simple de la température en quatre niveaux. "
                        "Les seuils de 25 °C et 30 °C correspondent aux seuils "
                        "climatologiques de chaleur et de forte chaleur utilisés "
                        "par Météo-France pour les températures maximales extérieures. "
                        "Cette classification ne constitue pas, à elle seule, une "
                        "évaluation du confort thermique intérieur."
                    )

                    temp_labels = [
                        "Frais (< 20 °C)",
                        "Doux (20 à < 25 °C)",
                        "Chaud (25 à < 30 °C)",
                        "Forte chaleur (≥ 30 °C)",
                    ]

                    temperatures = pd.to_numeric(
                        occupied_df["temperature"],
                        errors="coerce"
                    )

                    conditions = [
                        temperatures < 20.0,
                        (temperatures >= 20.0) & (temperatures < 25.0),
                        (temperatures >= 25.0) & (temperatures < 30.0),
                        temperatures >= 30.0,
                    ]

                    occupied_df["Categorie_temperature"] = np.select(
                        conditions,
                        temp_labels,
                        default=None,
                    )

                    occupied_df["Categorie_temperature"] = pd.Categorical(
                        occupied_df["Categorie_temperature"],
                        categories=temp_labels,
                        ordered=True,
                    )

                    color_map = {
                        temp_labels[0]: "#42a5f5",
                        temp_labels[1]: "#1b5e20",
                        temp_labels[2]: "#ef6c00",
                        temp_labels[3]: "#b71c1c",
                    }

                    temp_cat_counts = (
                        occupied_df["Categorie_temperature"]
                        .value_counts(sort=False)
                        .reindex(temp_labels, fill_value=0)
                        .rename_axis("Catégorie")
                        .reset_index(name="Mesures")
                    )

                    fig_temp_pie = px.pie(
                        temp_cat_counts,
                        names="Catégorie",
                        values="Mesures",
                        hole=0.35,
                        title="Répartition indicative de la température pendant les heures occupées",
                        color="Catégorie",
                        color_discrete_map=color_map,
                        category_orders={"Catégorie": temp_labels},
                    )

                    fig_temp_pie.update_traces(
                        textinfo="percent+label"
                    )

                    st.plotly_chart(fig_temp_pie, width="stretch")

            if methode_confort == "Diagramme de Givoni (bioclimatique)":
                st.markdown("---")
                st.subheader("🌡️💧 Diagramme de Givoni")
                st.caption(
                    "Les zones colorées représentent les plages de confort "
                    "thermique pour différentes vitesses d'air (ventilation). "
                    "Chaque point est une mesure du logement sélectionné, "
                    "placée selon sa température et son humidité absolue."
                )
                scatter_givoni = dfC[["temperature", "w_givoni"]].rename(
                    columns={"w_givoni": "w"}
                ).dropna()
                fig_givoni = make_givoni_figure(
                    givoni_zones, GIVONI_SPEED_ORDER, scatter_df=scatter_givoni
                )
                st.plotly_chart(fig_givoni, width="stretch")

                gv1, gv2 = st.columns(2)
                gv1.metric(
                    f"Conforme à {vitesse_air}",
                    f"{confort_override.mean() * 100:.1f} %",
                )
                zone_counts = (
                    dfC["Zone_Givoni_minimale"]
                    .value_counts()
                    .reindex(GIVONI_SPEED_ORDER + ["Hors confort"], fill_value=0)
                    .rename_axis("Vitesse d'air minimale requise")
                    .reset_index(name="Mesures")
                )
                with gv2:
                    st.dataframe(zone_counts, width="stretch", hide_index=True)

            st.markdown("---")
            st.subheader("📈 Évolution temporelle")
            et1, et2, et3, et4, et5 = st.tabs([
                "Toutes les mesures",
                "Moyennes journalières",
                "Superposition",
                "Résolution horaire",
                "Comparaison entre capteurs",
            ])

            with et1:
                var_ts = st.selectbox(
                    "Variable",
                    list(VARIABLES),
                    format_func=lambda x: VARIABLES[x]["label"],
                    key="full_ts_var_C",
                )
                fig = px.line(
                    dfC,
                    x="Date",
                    y=var_ts,
                    color="Occupation",
                    title=f"{VARIABLES[var_ts]['label']} — {selected_sensor}",
                )
                fig.update_traces(line_width=0.8)
                if var_ts == "temperature":
                    fig.add_hline(
                        y=temp_max,
                        line_dash="dash",
                        annotation_text=f"Max {temp_max} °C",
                    )
                    fig.add_hline(
                        y=temp_min,
                        line_dash="dash",
                        annotation_text=f"Min {temp_min} °C",
                    )
                elif var_ts == "humidite":
                    fig.add_hline(
                        y=hum_max,
                        line_dash="dash",
                        annotation_text=f"Max {hum_max} %",
                    )
                    fig.add_hline(
                        y=hum_min,
                        line_dash="dash",
                        annotation_text=f"Min {hum_min} %",
                    )
                else:
                    fig.add_hline(
                        y=co2_crit,
                        line_dash="dash",
                        annotation_text=f"Critique {co2_crit} ppm",
                    )
                    fig.add_hline(
                        y=co2_warn,
                        line_dash="dash",
                        annotation_text=f"Alerte {co2_warn} ppm",
                    )
                fig.update_layout(height=380)
                st.plotly_chart(fig, width="stretch")

            with et2:
                daily = (
                    dfC.groupby("Date_only")[
                        ["temperature", "humidite", "co2"]
                    ]
                    .agg(["mean", "min", "max"])
                    .reset_index()
                )
                daily.columns = [
                    "Date",
                    "temperature_mean",
                    "temperature_min",
                    "temperature_max",
                    "humidite_mean",
                    "humidite_min",
                    "humidite_max",
                    "co2_mean",
                    "co2_min",
                    "co2_max",
                ]
                var_daily = st.selectbox(
                    "Variable journalière",
                    list(VARIABLES),
                    format_func=lambda x: VARIABLES[x]["label"],
                    key="daily_var_C_v3",
                )
                fig_daily = go.Figure([
                    go.Scatter(
                        x=daily["Date"],
                        y=daily[f"{var_daily}_max"],
                        mode="lines",
                        name="Maximum",
                        line=dict(width=1, dash="dot"),
                    ),
                    go.Scatter(
                        x=daily["Date"],
                        y=daily[f"{var_daily}_mean"],
                        mode="lines",
                        name="Moyenne",
                        line=dict(width=2),
                    ),
                    go.Scatter(
                        x=daily["Date"],
                        y=daily[f"{var_daily}_min"],
                        mode="lines",
                        name="Minimum",
                        line=dict(width=1, dash="dot"),
                    ),
                ])
                fig_daily.update_layout(
                    height=380,
                    title=f"Minimum, moyenne et maximum — "
                          f"{VARIABLES[var_daily]['label']}",
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_daily, width="stretch")

            with et3:
                fig_overlay = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=(
                        "Température (°C)",
                        "Humidité (%HR)",
                        "CO₂ (ppm)",
                    ),
                    vertical_spacing=0.08,
                )
                fig_overlay.add_trace(
                    go.Scatter(
                        x=dfC["Date"],
                        y=dfC["temperature"],
                        mode="lines",
                        name="Température",
                        line=dict(width=0.8),
                    ),
                    row=1, col=1,
                )
                fig_overlay.add_trace(
                    go.Scatter(
                        x=dfC["Date"],
                        y=dfC["humidite"],
                        mode="lines",
                        name="Humidité",
                        line=dict(width=0.8),
                    ),
                    row=2, col=1,
                )
                fig_overlay.add_trace(
                    go.Scatter(
                        x=dfC["Date"],
                        y=dfC["co2"],
                        mode="lines",
                        name="CO₂",
                        line=dict(width=0.8),
                    ),
                    row=3, col=1,
                )
                fig_overlay.add_hline(
                    y=co2_crit,
                    line_dash="dash",
                    row=3,
                    col=1,
                )
                fig_overlay.update_layout(
                    height=600,
                    showlegend=False,
                    title="Superposition des trois variables",
                )
                st.plotly_chart(fig_overlay, width="stretch")

            with et4:
                for var_heat, heat_title in [
                    ("co2", "CO₂ (ppm)"),
                    ("temperature", "Température (°C)"),
                    ("humidite", "Humidité (%HR)"),
                ]:
                    pivot = dfC.pivot_table(
                        index="Hour",
                        columns="Date_only",
                        values=var_heat,
                        aggfunc="mean",
                    )
                    if pivot.empty:
                        continue
                    pivot.columns = pivot.columns.astype(str)
                    sample_step = max(1, len(pivot.columns) // 30)
                    pivot_sample = pivot[pivot.columns[::sample_step]]

                    data_min = float(np.nanmin(pivot_sample.values))
                    data_max = float(np.nanmax(pivot_sample.values))

                    if var_heat == "co2":
                        heat_colorscale = AIR_QUALITY_COLORSCALE
                        zmin = min(400.0, data_min)
                        zmax = max(float(co2_crit) * 1.2, data_max)
                    elif var_heat == "temperature":
                        heat_colorscale = COMFORT_COLORSCALE
                        zmin, zmax = build_comfort_zrange(
                            (temp_min + temp_max) / 2,
                            data_min, data_max,
                            (temp_max - temp_min) / 2,
                        )
                    else:  # humidite
                        heat_colorscale = COMFORT_COLORSCALE
                        zmin, zmax = build_comfort_zrange(
                            (hum_min + hum_max) / 2,
                            data_min, data_max,
                            (hum_max - hum_min) / 2,
                        )

                    fig_heat = px.imshow(
                        pivot_sample,
                        aspect="auto",
                        color_continuous_scale=heat_colorscale,
                        zmin=zmin,
                        zmax=zmax,
                        labels={
                            "x": "Date",
                            "y": "Heure",
                            "color": heat_title,
                        },
                        title=f"Heatmap heure × jour — {heat_title}",
                    )
                    fig_heat.update_layout(height=320)
                    st.plotly_chart(fig_heat, width="stretch")

            with et5:
                st.caption(
                    "Superpose les courbes de **tous les capteurs** sur la "
                    "section C, pour la variable et la période choisies "
                    "dans la colonne de gauche."
                )
                var_multi = st.selectbox(
                    "Variable",
                    list(VARIABLES),
                    index=0,
                    format_func=lambda x: VARIABLES[x]["label"],
                    key="multi_sensor_var_C",
                )

                # c_period_start/end sont exprimés sur la date décalée du
                # capteur sélectionné : on retire ce décalage pour comparer
                # tous les capteurs sur leurs dates d'origine.
                shift_delta = pd.to_timedelta(time_shift, unit="h")
                abs_period_start = c_period_start - shift_delta
                abs_period_end = c_period_end - shift_delta

                multi_parts = []
                for sensor_name, sensor_df in assigned.items():
                    part = sensor_df[
                        (sensor_df["section"] == "C")
                        & (sensor_df["Date"] >= abs_period_start)
                        & (sensor_df["Date"] <= abs_period_end)
                    ][["Date", "capteur", var_multi]]
                    if not part.empty:
                        multi_parts.append(part)

                if not multi_parts:
                    st.warning(
                        "Aucune donnée pour la période sélectionnée, pour "
                        "l'ensemble des capteurs."
                    )
                else:
                    multi_df = pd.concat(multi_parts, ignore_index=True)
                    fig_multi = px.line(
                        multi_df,
                        x="Date",
                        y=var_multi,
                        color="capteur",
                        title=(
                            f"{VARIABLES[var_multi]['label']} — "
                            "tous les capteurs (section C)"
                        ),
                    )
                    fig_multi.update_traces(line_width=1)

                    if var_multi == "temperature":
                        fig_multi.add_hline(
                            y=temp_max, line_dash="dash",
                            annotation_text=f"Max {temp_max} °C",
                        )
                        fig_multi.add_hline(
                            y=temp_min, line_dash="dash",
                            annotation_text=f"Min {temp_min} °C",
                        )
                    elif var_multi == "humidite":
                        fig_multi.add_hline(
                            y=hum_max, line_dash="dash",
                            annotation_text=f"Max {hum_max} %",
                        )
                        fig_multi.add_hline(
                            y=hum_min, line_dash="dash",
                            annotation_text=f"Min {hum_min} %",
                        )
                    else:
                        fig_multi.add_hline(
                            y=co2_crit, line_dash="dash",
                            annotation_text=f"Critique {co2_crit} ppm",
                        )
                        fig_multi.add_hline(
                            y=co2_warn, line_dash="dash",
                            annotation_text=f"Alerte {co2_warn} ppm",
                        )

                    fig_multi.update_layout(
                        height=480,
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig_multi, width="stretch")

            st.markdown("---")
            st.subheader("🔄 Profils et patterns")
            pc1, pc2 = st.columns(2)

            with pc1:
                st.markdown("**Profil horaire moyen combiné**")
                hourly_all = (
                    dfC.groupby("Hour")[
                        ["temperature", "humidite", "co2"]
                    ]
                    .mean()
                    .reset_index()
                )
                fig_hour = make_subplots(
                    specs=[[{"secondary_y": True}]]
                )
                fig_hour.add_trace(
                    go.Scatter(
                        x=hourly_all["Hour"],
                        y=hourly_all["temperature"],
                        name="Température °C",
                    ),
                    secondary_y=False,
                )
                fig_hour.add_trace(
                    go.Scatter(
                        x=hourly_all["Hour"],
                        y=hourly_all["humidite"],
                        name="Humidité %",
                    ),
                    secondary_y=False,
                )
                fig_hour.add_trace(
                    go.Bar(
                        x=hourly_all["Hour"],
                        y=hourly_all["co2"],
                        name="CO₂ ppm",
                        opacity=0.35,
                    ),
                    secondary_y=True,
                )
                fig_hour.add_hline(
                    y=co2_crit,
                    line_dash="dash",
                    secondary_y=True,
                )
                fig_hour.update_xaxes(
                    title_text="Heure",
                    tickvals=list(range(0, 24, 2)),
                )
                fig_hour.update_yaxes(
                    title_text="Température / humidité",
                    secondary_y=False,
                )
                fig_hour.update_yaxes(
                    title_text="CO₂ (ppm)",
                    secondary_y=True,
                )
                fig_hour.update_layout(
                    height=380,
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_hour, width="stretch")

            with pc2:
                st.markdown("**Profil par jour de la semaine**")
                day_order = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday", "Sunday",
                ]
                day_labels = [
                    "Lun", "Mar", "Mer", "Jeu",
                    "Ven", "Sam", "Dim",
                ]
                weekly = (
                    dfC.groupby("DayName")[
                        ["temperature", "humidite", "co2"]
                    ]
                    .mean()
                    .reindex(day_order)
                    .reset_index()
                )
                weekly["Jour"] = day_labels
                var_week = st.radio(
                    "Variable",
                    list(VARIABLES),
                    horizontal=True,
                    format_func=lambda x: VARIABLES[x]["label"],
                    key="weekly_var_C_v3",
                )
                fig_week = px.bar(
                    weekly,
                    x="Jour",
                    y=var_week,
                    title=f"Profil hebdomadaire — "
                          f"{VARIABLES[var_week]['label']}",
                )
                fig_week.update_layout(height=330)
                st.plotly_chart(fig_week, width="stretch")

            if (
                "temperature_exterieure" in dfC.columns
                and dfC["temperature_exterieure"].notna().any()
            ):
                st.markdown("---")
                st.subheader("🌤️ Influence de la température extérieure")

                valid_ext = dfC.dropna(
                    subset=[
                        "temperature",
                        "temperature_exterieure",
                        "delta_temperature_int_ext",
                    ]
                ).copy()

                if valid_ext.empty:
                    st.info(
                        "Aucun point extérieur ne peut être aligné sur la "
                        "période actuellement filtrée."
                    )
                else:
                    ex1, ex2, ex3, ex4 = st.columns(4)
                    ex1.metric(
                        "Température extérieure moy.",
                        f"{valid_ext['temperature_exterieure'].mean():.1f} °C",
                    )
                    ex2.metric(
                        "Écart intérieur − extérieur",
                        f"{valid_ext['delta_temperature_int_ext'].mean():+.1f} °C",
                    )
                    ex3.metric(
                        "Écart minimal",
                        f"{valid_ext['delta_temperature_int_ext'].min():+.1f} °C",
                    )
                    ex4.metric(
                        "Écart maximal",
                        f"{valid_ext['delta_temperature_int_ext'].max():+.1f} °C",
                    )

                    ext_tab1, ext_tab2, ext_tab3 = st.tabs([
                        "Intérieur / extérieur",
                        "Écart thermique",
                        "Relation intérieur–extérieur",
                    ])

                    with ext_tab1:
                        long_temp = valid_ext[
                            [
                                "Date",
                                "temperature",
                                "temperature_exterieure",
                            ]
                        ].melt(
                            id_vars="Date",
                            var_name="Série",
                            value_name="Température",
                        )
                        long_temp["Série"] = long_temp["Série"].replace({
                            "temperature": "Température intérieure",
                            "temperature_exterieure":
                                "Température extérieure",
                        })
                        fig_ext_series = px.line(
                            long_temp,
                            x="Date",
                            y="Température",
                            color="Série",
                            title=(
                                "Comparaison de la température intérieure "
                                "et extérieure"
                            ),
                        )
                        fig_ext_series.update_layout(height=400)
                        st.plotly_chart(
                            fig_ext_series,
                            width="stretch",
                        )

                    with ext_tab2:
                        fig_delta = px.line(
                            valid_ext,
                            x="Date",
                            y="delta_temperature_int_ext",
                            color="Occupation",
                            title=(
                                "Écart thermique : température intérieure "
                                "moins température extérieure"
                            ),
                            labels={
                                "delta_temperature_int_ext":
                                    "Écart intérieur − extérieur (°C)"
                            },
                        )
                        fig_delta.add_hline(
                            y=0,
                            line_dash="dash",
                            annotation_text="Égalité intérieur / extérieur",
                        )
                        fig_delta.update_layout(height=400)
                        st.plotly_chart(
                            fig_delta,
                            width="stretch",
                        )

                    with ext_tab3:
                        sample_ext = valid_ext.sample(
                            min(1000, len(valid_ext)),
                            random_state=42,
                        )
                        scatter_args = dict(
                            data_frame=sample_ext,
                            x="temperature_exterieure",
                            y="temperature",
                            color="Occupation",
                            opacity=0.5,
                            title=(
                                "Relation entre température extérieure "
                                "et température intérieure"
                            ),
                            labels={
                                "temperature_exterieure":
                                    "Température extérieure (°C)",
                                "temperature":
                                    "Température intérieure (°C)",
                            },
                        )
                        if show_expert and len(sample_ext) >= 3:
                            scatter_args["trendline"] = "ols"

                        fig_ext_scatter = px.scatter(**scatter_args)
                        fig_ext_scatter.update_layout(height=400)
                        st.plotly_chart(
                            fig_ext_scatter,
                            width="stretch",
                        )

                        corr_ext = valid_ext[
                            ["temperature_exterieure", "temperature"]
                        ].corr().iloc[0, 1]
                        st.caption(
                            f"Corrélation intérieur–extérieur : "
                            f"r = {corr_ext:.3f}. "
                            "Une corrélation élevée décrit une évolution "
                            "similaire, mais ne prouve pas une relation causale."
                        )

            st.markdown("---")
            st.subheader("📉 Distributions")
            d1, d2, d3 = st.columns(3)

            with d1:
                fig_dt = px.histogram(
                    dfC,
                    x="temperature",
                    nbins=40,
                    color="Occupation",
                    title="Température (°C)",
                )
                fig_dt.add_vline(x=temp_min, line_dash="dash")
                fig_dt.add_vline(x=temp_max, line_dash="dash")
                fig_dt.update_layout(height=300)
                st.plotly_chart(fig_dt, width="stretch")

            with d2:
                fig_dh = px.histogram(
                    dfC,
                    x="humidite",
                    nbins=40,
                    color="Occupation",
                    title="Humidité (%HR)",
                )
                fig_dh.add_vline(x=hum_min, line_dash="dash")
                fig_dh.add_vline(x=hum_max, line_dash="dash")
                fig_dh.update_layout(height=300)
                st.plotly_chart(fig_dh, width="stretch")

            with d3:
                co2_bins = [
                    -np.inf,
                    400,
                    co2_warn,
                    co2_crit,
                    1500,
                    np.inf,
                ]
                co2_labels = [
                    "< 400",
                    f"400–{co2_warn}",
                    f"{co2_warn}–{co2_crit}",
                    f"{co2_crit}–1500",
                    "> 1500",
                ]
                co2_zone = pd.cut(
                    dfC["co2"],
                    bins=co2_bins,
                    labels=co2_labels,
                    right=False,
                )
                co2_dist = (
                    co2_zone.value_counts(sort=False)
                    .rename_axis("Zone")
                    .reset_index(name="Mesures")
                )
                fig_co2 = px.bar(
                    co2_dist,
                    x="Zone",
                    y="Mesures",
                    color="Zone",
                    title="Distribution par zones CO₂",
                )
                fig_co2.update_layout(
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(fig_co2, width="stretch")

            st.markdown("---")
            st.subheader("🔗 Corrélations entre variables")
            corr_matrix = (
                dfC[["temperature", "humidite", "co2"]]
                .corr()
                .round(3)
            )

            if show_expert:
                cr1, cr2, cr3 = st.columns(3)
                scatter_specs = [
                    (
                        cr1,
                        "temperature",
                        "co2",
                        "Température vs CO₂",
                    ),
                    (
                        cr2,
                        "temperature",
                        "humidite",
                        "Température vs humidité",
                    ),
                    (
                        cr3,
                        "humidite",
                        "co2",
                        "Humidité vs CO₂",
                    ),
                ]
                for target_col, xvar, yvar, title in scatter_specs:
                    valid = dfC[[xvar, yvar]].dropna()
                    if len(valid) >= 3:
                        corr_value = valid.corr().iloc[0, 1]
                        sample = valid.sample(
                            min(500, len(valid)),
                            random_state=42,
                        )
                        fig_sc = px.scatter(
                            sample,
                            x=xvar,
                            y=yvar,
                            opacity=0.4,
                            trendline="ols",
                            title=f"{title} (r={corr_value:.3f})",
                        )
                        fig_sc.update_layout(height=300)
                        target_col.plotly_chart(
                            fig_sc,
                            width="stretch",
                        )

            fig_cm = px.imshow(
                corr_matrix,
                text_auto=True,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Matrice de corrélation",
            )
            fig_cm.update_layout(height=310)
            st.plotly_chart(fig_cm, width="stretch")

            st.markdown("---")
            st.subheader("🚨 Journal des alertes")
            a1, a2 = st.columns(2)

            with a1:
                st.markdown("**Synthèse des dépassements**")
                st.dataframe(
                    alert_summary.round(2),
                    width="stretch",
                    hide_index=True,
                )

            with a2:
                st.markdown("**Détail des dépassements CO₂**")
                co2_events = dfC[
                    dfC["co2"] >= co2_warn
                ][
                    [
                        "Date",
                        "Occupation",
                        "temperature",
                        "humidite",
                        "co2",
                    ]
                ].copy()
                co2_events = (
                    co2_events
                    .sort_values("co2", ascending=False)
                    .head(20)
                )
                co2_events["Niveau"] = np.where(
                    co2_events["co2"] >= co2_crit,
                    "🔴 Critique",
                    "🟡 Alerte",
                )
                st.dataframe(
                    co2_events.reset_index(drop=True),
                    width="stretch",
                    hide_index=True,
                )

            with st.expander(
                "📋 Statistiques descriptives complètes"
            ):
                st.dataframe(
                    dfC[
                        ["temperature", "humidite", "co2"]
                    ].describe().round(2),
                    width="stretch",
                )

            if show_expert:
                st.markdown("### Données expertes")
                comfort_df = pd.DataFrame([{
                    "Température conforme (%)":
                        details["temp_ok_%"],
                    "Humidité conforme (%)":
                        details["humidite_ok_%"],
                    "Confort thermique conjoint (%)":
                        details.get("confort_thermique_%", np.nan),
                    "CO₂ sous alerte (%)":
                        details["co2_ok_%"],
                    "CO₂ critique (%)":
                        details["co2_critique_%"],
                }]).round(2)
                st.dataframe(
                    comfort_df,
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    dfC.head(500),
                    width="stretch",
                )

            st.markdown("---")
            st.subheader("💾 Export des données")
            ec1, ec2, ec3 = st.columns(3)

            with ec1:
                st.download_button(
                    "⬇️ Données filtrées (CSV)",
                    to_csv(dfC),
                    f"section_C_{selected_sensor}.csv",
                    "text/csv",
                )

            with ec2:
                daily_export = (
                    dfC.groupby("Date_only")[
                        ["temperature", "humidite", "co2"]
                    ]
                    .agg(["mean", "min", "max"])
                    .round(2)
                )
                daily_export.columns = [
                    "_".join(col)
                    for col in daily_export.columns
                ]
                st.download_button(
                    "⬇️ Moyennes journalières (CSV)",
                    daily_export.reset_index()
                    .to_csv(index=False)
                    .encode("utf-8-sig"),
                    f"moyennes_journalieres_{selected_sensor}.csv",
                    "text/csv",
                )

            with ec3:
                st.download_button(
                    "⬇️ Rapport alertes (CSV)",
                    alert_summary.to_csv(index=False)
                    .encode("utf-8-sig"),
                    f"rapport_alertes_{selected_sensor}.csv",
                    "text/csv",
                )

            report_c = make_section_c_report(
                dfC,
                selected_sensor,
                temp_min,
                temp_max,
                hum_min,
                hum_max,
                co2_warn,
                co2_crit,
                iaq_score,
                methode_confort=methode_confort,
                confort_details=details,
            )
            if (
                "temperature_exterieure" in dfC.columns
                and dfC["temperature_exterieure"].notna().any()
            ):
                ext_valid_report = dfC.dropna(
                    subset=[
                        "temperature_exterieure",
                        "delta_temperature_int_ext",
                    ]
                )
                report_c += (
                    "\nTempérature extérieure moyenne : "
                    f"{ext_valid_report['temperature_exterieure'].mean():.1f} °C\n"
                    "Écart intérieur - extérieur moyen : "
                    f"{ext_valid_report['delta_temperature_int_ext'].mean():+.1f} °C\n"
                )

            report_c += (
                f"\nFiabilité A+E : "
                f"{fmt_score(reliability['score_AE'])}\n"
                f"Fiabilité A : "
                f"{fmt_score(reliability['score_A'])}\n"
                f"Fiabilité E : "
                f"{fmt_score(reliability['score_E'])}\n"
                f"Évolution : {reliability['evolution']}\n"
                f"Avis : {reliability['message']}\n"
            )
            st.download_button(
                "Télécharger le rapport de la section C",
                report_c.encode("utf-8"),
                f"rapport_section_C_{selected_sensor}.txt",
                "text/plain",
            )

with tabs[7]:
    st.subheader("Rapport synthétique")
    if ranking.empty:
        st.warning("Rapport indisponible.")
    else:
        best, worst = ranking.iloc[0], ranking.iloc[-1]
        st.markdown(f"""
### Synthèse automatique

Capteur le plus cohérent : **{best['capteur']}**  
Score : **{best['score_global']:.1f}/100**

Capteur le plus atypique : **{worst['capteur']}**  
Score : **{worst['score_global']:.1f}/100**


Un capteur avec un biais important mais une bonne dynamique peut être recalibré.  
Un capteur avec un mauvais z-score et de mauvaises variations est plus suspect.
""")
        st.download_button("Exporter le rapport", to_csv(ranking), "rapport_synthetique.csv", "text/csv")
