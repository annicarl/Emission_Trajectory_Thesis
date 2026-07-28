"""Historische Entwicklung und Bestandsbilanz der deutschen Pkw-Flotte.

Das Skript liest Bestand, Neuzulassungen (NZL) und Außerbetriebsetzungen
(ABS), erzeugt sieben PNG-Grafiken und exportiert die Bestandsbilanz als CSV.

Zeitlogik (entscheidend): Ein Bestandswert mit dem Stichtag 01.01.Y ist der
Anfangsbestand des Kalenderjahres Y. Die Bewegungen des Kalenderjahres Y
werden daher mit 01.01.Y und 01.01.(Y+1) bilanziert:

  Reentries_Y = Bestand_01.01.(Y+1) - Bestand_01.01.Y
                + Exits_Y - Neuzulassungen_Y

Damit wird kein Jahresversatz stillschweigend angenommen. Sollte eine andere
Datei statt Stichtagen nur Bestandsjahre enthalten, steuert
STOCK_YEAR_IS_END_YEAR unten deren Interpretation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Konfiguration: Hier Dateinamen und bei Bedarf Spaltennamen anpassen.
# Pfade sind relativ zum Ordner, in dem dieses Skript liegt.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "output_historische_flottendaten"

FILES = {
    "stock": DATA_DIR / "321-140-17133-26-Bestand.csv",
    "registrations": DATA_DIR / "321-140-17133-26-NZL.csv",
    "exits": DATA_DIR / "321-140-17133-26-ABS.csv",
}

COLUMNS = {
    "stock": {
        "time": "Berichtsjahr",
        "drive": "Antriebsart",
        "segment": "Segment",
        "value": "Anzahl",
    },
    "registrations": {
        "time": "Berichtsjahr",
        "drive": "Antriebsart",
        "segment": "Segment",
        "value": "Anzahl",
    },
    "exits": {
        "time": "Berichtsjahr",
        "drive": "Antriebsart",
        "segment": "Segment",
        "value": "Anzahl",
    },
}

# Nur relevant, wenn die Bestandsspalte bloß vierstellige Jahre (keine Daten)
# enthält. False: "2018" = Stichtag 01.01.2018. True: "Bestand 2018" =
# Stichtag 01.01.2019 (Bestand am Ende des Berichtsjahres 2018).
STOCK_YEAR_IS_END_YEAR = False

CSV_SEPARATOR = ";"
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin1")
UNUSUAL_GAP_PERCENT = 5.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Lese eine CSV mit mehreren für deutsche Behördendaten üblichen Encodings."""
    if not path.exists():
        raise FileNotFoundError(f"Eingabedatei fehlt: {path}")
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, sep=CSV_SEPARATOR, encoding=encoding, dtype=str)
            LOG.info("Gelesen: %s (%s Zeilen, Encoding %s)", path.name, len(df), encoding)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Keine passende Zeichenkodierung für {path}: {last_error}")


def parse_german_number(series: pd.Series) -> pd.Series:
    """Konvertiere Zahlen robust (auch '1.234', '1 234' oder '1.234,5')."""
    text = series.astype("string").str.strip().str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    # Nur wenn ein Komma vorkommt, Punkte als Tausendertrenner interpretieren.
    has_comma = text.str.contains(",", na=False)
    text.loc[has_comma] = (
        text.loc[has_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def prepare_dataset(raw: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Spalten vereinheitlichen, Werte bereinigen und Zeitinformation parsen."""
    mapping = COLUMNS[kind]
    missing_columns = set(mapping.values()) - set(raw.columns)
    if missing_columns:
        raise KeyError(f"In {FILES[kind].name} fehlen Spalten: {sorted(missing_columns)}")

    # Nur Duplikate über ALLE Rohdatenspalten sind echte identische
    # Beobachtungen. Mehrere Zeilen mit gleicher Zeit/Antrieb/Segment-Kombination
    # können sich z. B. im Jahr der Erstzulassung unterscheiden und sind legitim.
    full_duplicates = int(raw.duplicated().sum())
    if full_duplicates:
        LOG.warning("%s: %s vollständig identische Rohdatenzeilen; sie werden mitaggregiert.",
                    kind, full_duplicates)

    df = raw.rename(columns={v: k for k, v in mapping.items()})[
        ["time", "drive", "segment", "value"]
    ].copy()

    df["value"] = parse_german_number(df["value"])
    invalid_values = int(df["value"].isna().sum())
    if invalid_values:
        LOG.warning("%s: %s Zeilen ohne gültige Anzahl werden verworfen.", kind, invalid_values)
        df = df.dropna(subset=["value"])

    for category in ("drive", "segment"):
        missing = int(df[category].isna().sum() + df[category].astype("string").str.strip().eq("").sum())
        if missing:
            LOG.warning("%s: %s fehlende Werte in %s werden 'Unbekannt'.", kind, missing, category)
        df[category] = df[category].fillna("Unbekannt").astype(str).str.strip().replace("", "Unbekannt")

    if kind == "stock":
        raw_time = df["time"].astype("string").str.strip()
        # Zuerst echte Daten wie 01.01.2017 parsen. dayfirst verhindert
        # Verwechslungen zwischen deutschem und US-amerikanischem Datumsformat.
        dates = pd.to_datetime(raw_time, dayfirst=True, errors="coerce")
        year_only = raw_time.str.fullmatch(r"\d{4}", na=False)
        numeric_year = pd.to_numeric(raw_time.where(year_only), errors="coerce")
        dates.loc[year_only] = pd.to_datetime(
            {"year": numeric_year.loc[year_only], "month": 1, "day": 1}, errors="coerce"
        )
        if STOCK_YEAR_IS_END_YEAR:
            dates.loc[year_only] = dates.loc[year_only] + pd.DateOffset(years=1)
        df["stock_date"] = dates
        df["year"] = dates.dt.year.astype("Int64")
    else:
        # Extrahiere genau ein vierstelliges Jahr; so funktionieren auch Texte
        # wie "Berichtsjahr 2018", ohne Dezimalwerte als Jahre zu akzeptieren.
        extracted = df["time"].astype("string").str.extract(r"(?<!\d)((?:19|20)\d{2})(?!\d)")[0]
        df["year"] = pd.to_numeric(extracted, errors="coerce").astype("Int64")

    invalid_years = int(df["year"].isna().sum())
    if invalid_years:
        LOG.warning("%s: %s Zeilen ohne interpretierbares Jahr werden verworfen.", kind, invalid_years)
        df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def aggregate(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Summiere alle weiteren Dimensionen (z. B. Erstzulassungsjahr) korrekt auf."""
    return (
        df.groupby(["year", category], as_index=False, dropna=False)["value"]
        .sum()
        .sort_values(["year", category])
    )


def millions_formatter(value: float, _position: int) -> str:
    return f"{value / 1_000_000:.1f} Mio."


def save_stacked_plot(
    df: pd.DataFrame, category: str, title: str, ylabel: str, filename: str
) -> None:
    pivot = aggregate(df, category).pivot(index="year", columns=category, values="value").fillna(0)
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(13, 7), width=0.82, colormap="tab20")
    ax.set_title(title)
    ax.set_xlabel("Bestandsjahr (Stichtag 1. Januar)" if "Bestand" in title else "Kalenderjahr")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.legend(title="Antriebsart" if category == "drive" else "Fahrzeugsegment",
              bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def totals_by_year(df: pd.DataFrame) -> pd.Series:
    return df.groupby("year", dropna=False)["value"].sum().sort_index()


def build_balance(stock: pd.DataFrame, registrations: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    """Erzeuge eine explizite Kalenderjahresbilanz zwischen zwei 1.-Januar-Ständen."""
    stock_totals = totals_by_year(stock)
    reg_totals = totals_by_year(registrations)
    exit_totals = totals_by_year(exits)

    # Nur Kalenderjahre mit Anfangs- UND Folgebestand sind bilanzierbar.
    candidate_years = sorted(set(stock_totals.index) | set(reg_totals.index) | set(exit_totals.index))
    rows: list[dict[str, object]] = []
    for year in candidate_years:
        missing: list[str] = []
        if year not in stock_totals.index:
            missing.append(f"Bestand 01.01.{year}")
        if year + 1 not in stock_totals.index:
            missing.append(f"Bestand 01.01.{year + 1}")
        if year not in reg_totals.index:
            missing.append(f"Neuzulassungen {year}")
        if year not in exit_totals.index:
            missing.append(f"Exits {year}")
        if missing:
            LOG.warning("Bezugsjahr %s nicht vollständig bilanzierbar: %s", year, ", ".join(missing))
            continue

        previous = float(stock_totals.loc[year])
        new = float(reg_totals.loc[year])
        dereg = float(exit_totals.loc[year])
        actual = float(stock_totals.loc[year + 1])
        expected = previous + new - dereg
        gap = actual - expected
        rows.append({
            "Bezugsjahr": year,
            "Datum des Vorjahresbestands": f"01.01.{year}",
            "Datum des tatsächlichen Bestands": f"01.01.{year + 1}",
            "Bestand des Vorjahres": previous,
            "Neuzulassungen im Bezugsjahr": new,
            "Exits im Bezugsjahr": dereg,
            "Erwarteter Bestand": expected,
            "Tatsächlicher Bestand": actual,
            "Bestandslücke bzw. Reentries": gap,
            "Bestandslücke in % des tatsächlichen Bestands": 100 * gap / actual if actual else np.nan,
        })
    return pd.DataFrame(rows)


def save_balance_plot(balance: pd.DataFrame) -> None:
    """Zwei Panels verhindern, dass große Bestände die Bewegungen unlesbar machen."""
    years = balance["Bezugsjahr"].astype(str)
    x = np.arange(len(balance))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    width = 0.36
    ax1.bar(
        x - width / 2,
        balance["Erwarteter Bestand"],
        width,
        label="Erwarteter Folgejahresbestand (berechnet)",
        color="#f2cf5b",
    )
    ax1.bar(
        x + width / 2,
        balance["Tatsächlicher Bestand"],
        width,
        label="Tatsächlicher Folgejahresbestand (Originaldaten)",
        color="#4c78a8",
    )
    ax1.set_ylabel("Fahrzeuge")
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax1.set_title("Folgejahresbestand: berechneter Wert und tatsächliche Originaldaten")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    movement_columns = ["Neuzulassungen im Bezugsjahr", "Exits im Bezugsjahr", "Bestandslücke bzw. Reentries"]
    colors = ["#4c78a8", "#e45756", "#72b7b2"]
    movement_width = 0.25
    for i, (column, color) in enumerate(zip(movement_columns, colors)):
        positions = x + (i - 1) * movement_width
        bars = ax2.bar(positions, balance[column], movement_width, label=column, color=color)
        if column == "Bestandslücke bzw. Reentries":
            labels = [f"{v / 1_000:.0f} Tsd." for v in balance[column]]
            ax2.bar_label(bars, labels=labels, padding=3, fontsize=8, rotation=90)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Fahrzeuge")
    ax2.set_xlabel("Bezugsjahr der Neuzulassungen, Exits und Reentries")
    ax2.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax2.set_xticks(x, years)
    ax2.set_title("Jährliche Bewegungen (separate Skalierung)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_bestandsbilanz.png", dpi=200, bbox_inches="tight")
    plt.close()


def warn_missing_years(name: str, years: Iterable[int]) -> None:
    years = sorted(set(years))
    if not years:
        LOG.warning("%s enthält keine Jahre.", name)
        return
    absent = sorted(set(range(years[0], years[-1] + 1)) - set(years))
    if absent:
        LOG.warning("%s: Lücken in der Jahresreihe: %s", name, absent)


def plausibility_checks(
    datasets: dict[str, pd.DataFrame], balance: pd.DataFrame, raw_numeric_sums: dict[str, float]
) -> None:
    LOG.info("--- Plausibilitätsprüfungen ---")
    for kind, df in datasets.items():
        aggregated_sum = float(totals_by_year(df).sum())
        if np.isclose(aggregated_sum, raw_numeric_sums[kind], rtol=0, atol=0.01):
            LOG.info("%s: Aggregatsumme stimmt mit bereinigter Rohdatensumme überein.", kind)
        else:
            LOG.warning("%s: Aggregatsumme %.0f weicht von Rohdatensumme %.0f ab.",
                        kind, aggregated_sum, raw_numeric_sums[kind])
        warn_missing_years(kind, df["year"])

    if balance.empty:
        LOG.warning("Keine vollständigen, zeitlich abgestimmten Jahre für die Bestandsbilanz.")
        return
    LOG.info("Zeitabstimmung erfolgreich für Bezugsjahre %s bis %s.",
             balance["Bezugsjahr"].min(), balance["Bezugsjahr"].max())
    negatives = balance.loc[balance["Bestandslücke bzw. Reentries"] < 0, "Bezugsjahr"].tolist()
    if negatives:
        LOG.warning("Negative Reentries (unverändert beibehalten) in: %s", negatives)
    unusual = balance.loc[
        balance["Bestandslücke in % des tatsächlichen Bestands"].abs() > UNUSUAL_GAP_PERCENT,
        ["Bezugsjahr", "Bestandslücke in % des tatsächlichen Bestands"],
    ]
    if not unusual.empty:
        LOG.warning("Ungewöhnliche Bestandslücken (> %.1f%%):\n%s",
                    UNUSUAL_GAP_PERCENT, unusual.to_string(index=False))
    else:
        LOG.info("Keine Bestandslücke über dem Schwellenwert von %.1f%%.", UNUSUAL_GAP_PERCENT)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {kind: read_csv_robust(path) for kind, path in FILES.items()}
    data = {kind: prepare_dataset(frame, kind) for kind, frame in raw.items()}
    raw_numeric_sums = {kind: float(df["value"].sum()) for kind, df in data.items()}

    plot_specs = [
        ("stock", "drive", "Entwicklung des Pkw-Bestands nach Antriebsart", "Fahrzeugbestand", "01_bestand_antriebsart.png"),
        ("stock", "segment", "Entwicklung des Pkw-Bestands nach Fahrzeugsegment", "Fahrzeugbestand", "02_bestand_segment.png"),
        ("registrations", "drive", "Jährliche Pkw-Neuzulassungen nach Antriebsart", "Neuzulassungen", "03_neuzulassungen_antriebsart.png"),
        ("registrations", "segment", "Jährliche Pkw-Neuzulassungen nach Fahrzeugsegment", "Neuzulassungen", "04_neuzulassungen_segment.png"),
        ("exits", "drive", "Jährliche Pkw-Außerbetriebsetzungen nach Antriebsart", "Außerbetriebsetzungen", "05_exits_antriebsart.png"),
        ("exits", "segment", "Jährliche Pkw-Außerbetriebsetzungen nach Fahrzeugsegment", "Außerbetriebsetzungen", "06_exits_segment.png"),
    ]
    for kind, category, title, ylabel, filename in plot_specs:
        save_stacked_plot(data[kind], category, title, ylabel, filename)

    balance = build_balance(data["stock"], data["registrations"], data["exits"])
    if balance.empty:
        LOG.warning("Bestandsbilanz ist leer; CSV wird exportiert, Bilanzgrafik entfällt.")
    else:
        save_balance_plot(balance)

    csv_path = OUTPUT_DIR / "bestandsbilanz_reentries.csv"
    balance.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8-sig")

    print("\nBESTANDSBILANZ (Bewegungen beziehen sich jeweils auf das Bezugsjahr)\n")
    if balance.empty:
        print("Keine vollständig bilanzierbaren Jahre vorhanden.")
    else:
        display = balance.copy()
        number_columns = display.select_dtypes(include="number").columns.difference(["Bezugsjahr"])
        formatters = {col: (lambda value: f"{value:,.0f}".replace(",", ".")) for col in number_columns}
        percent_col = "Bestandslücke in % des tatsächlichen Bestands"
        formatters[percent_col] = lambda value: f"{value:.3f} %".replace(".", ",")
        print(display.to_string(index=False, formatters=formatters))

    plausibility_checks(data, balance, raw_numeric_sums)
    LOG.info("Ergebnisse gespeichert in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
