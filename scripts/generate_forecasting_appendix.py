#!/usr/bin/env python3
"""Generate forecasting result artifacts for DMKD v4.

The aggregation follows the manuscript protocol: cumulative forecast scores are
averaged over rolling origins and horizons within a reconstruction realization,
then over the ten realizations within each experimental condition. Forecasts
from the complete original history are retained as a descriptive benchmark and
are never inserted into the ranks or inferential cells for the 17 reconstructors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "prediction_experiment_results"
    / "prediction_results_rolling_origins_merged_20261010_202224.csv"
)
OUTPUT = (
    ROOT
    / "LG"
    / "DataMiningAndKnowledgeDiscovery_v4"
    / "tables"
    / "appendix_forecasting_108_conditions.tex"
)
TABLE_OUTPUT = OUTPUT.with_name("forecast_matrix.tex")
FIGURE_DIR = OUTPUT.parents[1] / "figures" / "results"

REALIZATION_KEYS = [
    "prediction_model",
    "reconstruction_model",
    "dataset_name",
    "technique",
    "structure",
    "rate_percent",
    "reconstruction_iteration",
]
CELL_KEYS = REALIZATION_KEYS[:-1]
USECOLS = [
    "source_type",
    "metric_scope",
    "smape",
    "forecast_horizon",
    *REALIZATION_KEYS,
]

DATASETS = {
    "boiler_outlet_temp_univ": "Boiler",
    "pump_sensor_28_univ": "Pump",
    "vibration_sensor_S1": "Vibration",
}
FORECASTERS = {
    "holt_winters": "Holt--Winters",
    "prophet": "Prophet",
    "sarimax": "ARIMA",
    "xgboost": "XGBoost",
}
METHODS = {
    "original": "Original",
    "impute_bfill": "Backward fill",
    "interpolate_cubic": "Cubic",
    "impute_ffill": "Forward fill",
    "knn": "KNN",
    "interpolate_linear": "Linear",
    "impute_mean": "Mean",
    "impute_median": "Median",
    "interpolate_pchip": "PCHIP",
    "sarimax": "ARIMA",
    "stable_diffusion_2_gaf": "GAF",
    "stable_diffusion_2_gaf_finetuned": "GAF-FT",
    "stable_diffusion_2_mtf": "MTF",
    "stable_diffusion_2_mtf_finetuned": "MTF-FT",
    "stable_diffusion_2_rp": "RP",
    "stable_diffusion_2_rp_finetuned": "RP-FT",
    "stable_diffusion_2_spec": "SPEC",
    "stable_diffusion_2_spec_finetuned": "SPEC-FT",
}
CLASSICAL = [
    "impute_bfill",
    "interpolate_cubic",
    "impute_ffill",
    "knn",
    "interpolate_linear",
    "impute_mean",
    "impute_median",
    "interpolate_pchip",
    "sarimax",
]
SD2 = [
    "stable_diffusion_2_gaf",
    "stable_diffusion_2_gaf_finetuned",
    "stable_diffusion_2_mtf",
    "stable_diffusion_2_mtf_finetuned",
    "stable_diffusion_2_rp",
    "stable_diffusion_2_rp_finetuned",
    "stable_diffusion_2_spec",
    "stable_diffusion_2_spec_finetuned",
]


def load_balanced_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partials: list[pd.DataFrame] = []
    horizon_partials: list[pd.DataFrame] = []
    original_partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(SOURCE, usecols=USECOLS, chunksize=200_000):
        selected = chunk[
            chunk["source_type"].eq("reconstructed")
            & chunk["metric_scope"].eq("cumulative")
        ]
        grouped = selected.groupby(REALIZATION_KEYS, as_index=False, dropna=False).agg(
            smape_sum=("smape", "sum"), smape_count=("smape", "count")
        )
        partials.append(grouped)

        horizon_keys = [*REALIZATION_KEYS, "forecast_horizon"]
        horizon_partials.append(
            selected.groupby(horizon_keys, as_index=False, dropna=False).agg(
                smape_sum=("smape", "sum"), smape_count=("smape", "count")
            )
        )

        original = chunk[
            chunk["source_type"].eq("original")
            & chunk["metric_scope"].eq("cumulative")
        ]
        original_partials.append(
            original.groupby(
                ["prediction_model", "dataset_name", "forecast_horizon"],
                as_index=False,
                dropna=False,
            ).agg(smape_sum=("smape", "sum"), smape_count=("smape", "count"))
        )

    combined = pd.concat(partials, ignore_index=True)
    realization = combined.groupby(REALIZATION_KEYS, as_index=False, dropna=False).agg(
        smape_sum=("smape_sum", "sum"), smape_count=("smape_count", "sum")
    )
    realization["smape"] = realization["smape_sum"] / realization["smape_count"]
    cells = realization.groupby(CELL_KEYS, as_index=False, dropna=False).agg(
        smape=("smape", "mean"), realizations=("reconstruction_iteration", "nunique")
    )

    expected_realizations = 4 * 17 * 108 * 10
    expected_cells = 4 * 17 * 108
    if len(realization) != expected_realizations:
        raise ValueError(
            f"Expected {expected_realizations:,} realization rows, got {len(realization):,}."
        )
    if len(cells) != expected_cells:
        raise ValueError(f"Expected {expected_cells:,} cells, got {len(cells):,}.")
    if not cells["realizations"].eq(10).all():
        counts = cells["realizations"].value_counts().sort_index().to_dict()
        raise ValueError(f"Expected ten realizations per cell, got {counts}.")
    if set(cells["dataset_name"]) != set(DATASETS):
        raise ValueError("Unexpected dataset grid.")
    if set(cells["prediction_model"]) != set(FORECASTERS):
        raise ValueError("Unexpected forecasting-model grid.")
    expected_methods = set(METHODS) - {"original"}
    if set(cells["reconstruction_model"]) != expected_methods:
        raise ValueError("Unexpected reconstruction-method grid.")

    horizon_keys = [*REALIZATION_KEYS, "forecast_horizon"]
    horizon_combined = pd.concat(horizon_partials, ignore_index=True)
    horizon_realization = horizon_combined.groupby(
        horizon_keys, as_index=False, dropna=False
    ).agg(smape_sum=("smape_sum", "sum"), smape_count=("smape_count", "sum"))
    horizon_realization["smape"] = (
        horizon_realization["smape_sum"] / horizon_realization["smape_count"]
    )
    horizon_cell_keys = [*CELL_KEYS, "forecast_horizon"]
    horizon_cells = horizon_realization.groupby(
        horizon_cell_keys, as_index=False, dropna=False
    ).agg(smape=("smape", "mean"))

    original_combined = pd.concat(original_partials, ignore_index=True)
    original_horizon = original_combined.groupby(
        ["prediction_model", "dataset_name", "forecast_horizon"], as_index=False
    ).agg(smape_sum=("smape_sum", "sum"), smape_count=("smape_count", "sum"))
    original_horizon["smape"] = (
        original_horizon["smape_sum"] / original_horizon["smape_count"]
    )
    original_horizon = original_horizon.drop(columns=["smape_sum", "smape_count"])
    if (
        len(original_horizon.groupby(["prediction_model", "dataset_name"])) != 12
        or set(original_horizon["prediction_model"]) != set(FORECASTERS)
        or set(original_horizon["dataset_name"]) != set(DATASETS)
    ):
        raise ValueError("Incomplete original-history benchmark.")
    return cells, horizon_cells, original_horizon


def render_table(
    pivot: pd.DataFrame,
    forecaster: str,
    family: str,
    selected: list[str],
) -> list[str]:
    forecaster_label = FORECASTERS[forecaster]
    label_forecaster = forecaster.replace("_", "-")
    label_family = family.lower().replace(" ", "-").replace("~", "")
    header = "Dataset & Mechanism & Structure & Rate & " + " & ".join(
        METHODS[method] for method in selected
    ) + r" \\"
    reference_note = (
        " The Original column is the complete-history benchmark and is "
        "invariant across missingness conditions."
        if "original" in selected
        else ""
    )
    lines = [
        r"\begin{longtable}{llll" + "r" * len(selected) + "}",
        (
            rf"\caption{{Complete forecasting sMAPE matrix for {forecaster_label}: "
            rf"{family} reconstruction methods.{reference_note}}}"
            rf"\label{{tab:appendix-forecasting-108-{label_forecaster}-{label_family}}}\\"
        ),
        r"\toprule",
        header,
        r"\midrule\endfirsthead",
        r"\toprule",
        header,
        r"\midrule\endhead",
    ]
    for index, row in pivot[selected].iterrows():
        dataset, mechanism, structure, rate = index
        values = " & ".join(f"{value:.2f}" for value in row)
        lines.append(
            f"{DATASETS[dataset]} & {mechanism} & {structure} & {int(rate)}\\% & {values} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return lines


def write_appendix(cells: pd.DataFrame, original_horizon: pd.DataFrame) -> None:
    lines = [
        r"\ifdefined\pdfpageattr",
        r"  \newtoks\DMKDforecastsavedpageattr",
        r"  \global\DMKDforecastsavedpageattr=\expandafter{\the\pdfpageattr}%",
        r"\fi",
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
    ]
    for forecaster in FORECASTERS:
        subset = cells[cells["prediction_model"].eq(forecaster)]
        pivot = subset.pivot(
            index=["dataset_name", "technique", "structure", "rate_percent"],
            columns="reconstruction_model",
            values="smape",
        ).sort_index()
        if pivot.shape != (108, 17) or pivot.isna().any().any():
            raise ValueError(f"Incomplete matrix for {forecaster}: {pivot.shape}.")
        reference = (
            original_horizon[original_horizon["prediction_model"].eq(forecaster)]
            .groupby("dataset_name")["smape"]
            .mean()
        )
        pivot.insert(0, "original", [reference.loc[index[0]] for index in pivot.index])
        lines.extend(
            render_table(
                pivot,
                forecaster,
                "classical",
                ["original", *CLASSICAL],
            )
        )
        lines.append(r"\clearpage")
        lines.extend(render_table(pivot, forecaster, "Stable Diffusion~2", SD2))
        if forecaster != list(FORECASTERS)[-1]:
            lines.append(r"\clearpage")
    lines.extend(
        [
            r"\end{landscape}",
            r"\ifdefined\pdfpageattr",
            r"  \global\pdfpageattr=\expandafter{\the\DMKDforecastsavedpageattr}%",
            r"\fi",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_table(cells: pd.DataFrame, original_horizon: pd.DataFrame) -> None:
    ranking = cells.groupby(
        ["prediction_model", "reconstruction_model"], as_index=False
    ).agg(smape=("smape", "mean"))
    ranking["rank"] = ranking.groupby("prediction_model")["smape"].rank(
        method="min"
    ).astype(int)
    pivot = ranking.pivot(
        index="reconstruction_model", columns="prediction_model", values="smape"
    )
    ranks = ranking.pivot(
        index="reconstruction_model", columns="prediction_model", values="rank"
    )
    order = ranking.groupby("reconstruction_model")["smape"].mean().sort_values().index
    original = (
        original_horizon.groupby(["prediction_model", "dataset_name"], as_index=False)
        .agg(smape=("smape", "mean"))
        .groupby("prediction_model")["smape"]
        .mean()
    )
    forecast_order = ["holt_winters", "prophet", "sarimax", "xgboost"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Balanced forecasting sMAPE (\%) by forecaster and training-history variant. Parentheses give ranks among the 17 reconstructed histories; the complete original history is an unranked reference.}",
        r"\label{tab:forecast-matrix}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Training history & Holt--Winters & Prophet & ARIMA & XGBoost \\",
        r"\midrule",
        "Original (no reconstruction) & "
        + " & ".join(f"{original[model]:.3f} (ref.)" for model in forecast_order)
        + r" \\",
        r"\midrule",
    ]
    for model in order:
        values = [
            f"{pivot.loc[model, forecaster]:.3f} ({int(ranks.loc[model, forecaster])})"
            for forecaster in forecast_order
        ]
        lines.append(f"{METHODS[model]} & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\footnotesize Reconstructed histories are balanced over 108 missingness cells after averaging origins, cumulative horizons, and ten realizations. The original-history reference is first averaged over origins and horizons within each dataset and then equally over the three datasets; it is not replicated across missingness cells.",
            r"\end{table}",
        ]
    )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(
    cells: pd.DataFrame,
    horizon_cells: pd.DataFrame,
    original_horizon: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    forecast_order = ["holt_winters", "prophet", "sarimax", "xgboost"]

    ranking = cells.groupby(
        ["prediction_model", "reconstruction_model"], as_index=False
    ).agg(smape=("smape", "mean"))
    pivot = ranking.pivot(
        index="reconstruction_model", columns="prediction_model", values="smape"
    )
    order = ranking.groupby("reconstruction_model")["smape"].mean().sort_values().index
    pivot = pivot.loc[order, forecast_order]
    original = (
        original_horizon.groupby(["prediction_model", "dataset_name"], as_index=False)
        .agg(smape=("smape", "mean"))
        .groupby("prediction_model")["smape"]
        .mean()
    )
    pivot = pd.concat(
        [pd.DataFrame([original], index=["original"])[forecast_order], pivot]
    )
    colors = np.log10(pivot.to_numpy() + 0.01)
    fig, ax = plt.subplots(figsize=(7.4, 7.3))
    image = ax.imshow(colors, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(pivot)), [METHODS[model] for model in pivot.index])
    ax.set_xticks(
        range(4), ["Holt--Winters", "Prophet", "ARIMA", "XGBoost"],
        rotation=20, ha="right"
    )
    ax.axhline(0.5, color="white", linewidth=2)
    midpoint = (colors.min() + colors.max()) / 2
    for row in range(len(pivot)):
        for column in range(4):
            ax.text(
                column,
                row,
                f"{pivot.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if colors[row, column] > midpoint else "black",
            )
    ax.set_title("Forecast performance by training-history variant\n(cell values are sMAPE [%])")
    fig.colorbar(image, ax=ax, label="log10(sMAPE + 0.01)")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "06_reconstruction_by_forecasting_method.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    horizon = horizon_cells.groupby(
        ["dataset_name", "forecast_horizon", "prediction_model", "reconstruction_model"],
        as_index=False,
    ).agg(smape=("smape", "mean"))
    horizon["family"] = np.where(
        horizon["reconstruction_model"].str.startswith("stable_diffusion"),
        "best SD2",
        "best classical",
    )
    best = horizon.groupby(
        ["dataset_name", "forecast_horizon", "prediction_model", "family"],
        as_index=False,
    )["smape"].min()
    original_plot = original_horizon.copy()
    original_plot["family"] = "original history"
    plot_data = pd.concat(
        [
            best[["dataset_name", "forecast_horizon", "prediction_model", "family", "smape"]],
            original_plot[["dataset_name", "forecast_horizon", "prediction_model", "family", "smape"]],
        ],
        ignore_index=True,
    )
    dataset_names = {
        "boiler_outlet_temp_univ": "Boiler outlet temperature",
        "pump_sensor_28_univ": "Pump sensor 28",
        "vibration_sensor_S1": "Vibration sensor S1",
    }
    styles = {
        "best classical": ("o", "-"),
        "best SD2": ("o", "-"),
        "original history": ("s", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (dataset, title) in zip(axes, dataset_names.items()):
        part = plot_data[plot_data["dataset_name"].eq(dataset)].groupby(
            ["forecast_horizon", "family"], as_index=False
        )["smape"].mean()
        for family in ["best classical", "best SD2", "original history"]:
            values = part[part["family"].eq(family)].sort_values("forecast_horizon")
            marker, linestyle = styles[family]
            ax.plot(
                values["forecast_horizon"].astype(str),
                values["smape"],
                marker=marker,
                linestyle=linestyle,
                label=family,
            )
        ax.set_title(title)
        ax.set_xlabel("horizon [samples]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("sMAPE [%], mean across forecasters")
    axes[-1].legend(frameon=False)
    fig.suptitle("Forecast performance by horizon, dataset, and training-history family", fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "09_horizon_classical_vs_sd2.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    cells, horizon_cells, original_horizon = load_balanced_results()
    write_appendix(cells, original_horizon)
    write_main_table(cells, original_horizon)
    write_figures(cells, horizon_cells, original_horizon)
    print(
        "Wrote forecasting appendix, main table, and figures "
        f"({len(cells):,} reconstructed cells; original history retained as reference)."
    )


if __name__ == "__main__":
    main()
