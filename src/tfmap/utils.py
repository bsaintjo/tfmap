import numpy as np
from pybaselines import Baseline
import polars as pl
import polars.selectors as cs
from scipy.signal import savgol_filter


def mask_wavenumbers(df: pl.DataFrame, ranges: list[tuple[float, float]]):
    metadata_cols = df.select(~cs.starts_with("wavenumber")).columns
    masked_spectra_cols = [
        col
        for col in df.select(cs.contains("wavenumber")).columns
        if not any(
            low <= float(col.split("_")[-1]) <= high for low, high in ranges
        )
    ]
    return df.select(masked_spectra_cols + metadata_cols)


def clip_wavenumbers(
    df: pl.DataFrame, low: float = 0.0, high: float = 5000.0
) -> pl.DataFrame:
    metadata_cols = df.select(~cs.starts_with("wavenumber")).columns
    clipped_spectra_cols = [
        col
        for col in df.select(cs.contains("wavenumber")).columns
        if low <= float(col.split("_")[-1]) <= high
    ]
    return df.select(clipped_spectra_cols + metadata_cols)


def rubberband_correction(df: pl.DataFrame):
    selector = cs.starts_with("wavenumber")
    spectra_df = df.select(selector)
    metadata_df = df.select(~selector)
    wavenumbers = [float(col.split("_")[-1]) for col in spectra_df.columns]

    baseline_fitter = Baseline(x_data=wavenumbers)
    acc = []
    for spectra in spectra_df.to_numpy():
        res = baseline_fitter.rubberband(spectra, segments=[900, 1900, 2250])[
            0
        ]
        acc.append(spectra - res)
    return pl.concat(
        [pl.DataFrame(np.array(acc), schema=spectra_df.columns), metadata_df],
        how="horizontal",
    )


def savgol(df: pl.DataFrame, savgol_params: dict) -> pl.DataFrame:
    selector = cs.starts_with("wavenumber")
    spectra_df = df.select(selector)
    metadata_df = df.select(~selector)
    acc = []
    for spectra in spectra_df.to_numpy():
        res = savgol_filter(spectra, **savgol_params)
        acc.append(res)
    return pl.concat(
        [pl.DataFrame(np.array(acc), schema=spectra_df.columns), metadata_df],
        how="horizontal",
    )


def _select_spectra(df: pl.DataFrame, low: float, high: float) -> list[str]:
    cols = [
        col
        for col in df.select(cs.contains("wavenumber")).columns
        if low <= float(col.split("_")[1]) <= high
    ]
    return cols


def _add_peak_loc_cols(
    df: pl.DataFrame, name: str, cols: list[str]
) -> pl.DataFrame:
    peak_name = name + "_peak"
    loc_name = name + "_loc"
    col_name = loc_name + "_col"
    return df.with_columns(
        pl.col(name)
        .map_elements(lambda xs: [abs(x) for x in xs])
        .list.max()
        .alias(peak_name),
        pl.col(name)
        .map_elements(lambda xs: [abs(x) for x in xs])
        .list.arg_max()
        .alias(col_name)
        .map_elements(
            lambda x: cols[x], return_dtype=pl.String, returns_scalar=True
        ),
    ).with_columns(
        pl.col(col_name)
        .str.split("_")
        .list.last()
        .cast(pl.Float64)
        .alias(loc_name)
    )


def _build_amide_df(df: pl.DataFrame) -> pl.DataFrame:
    amide_i_cols = _select_spectra(df, low=1600, high=1700)
    amide_ii_cols = _select_spectra(df, low=1510, high=1580)
    res = df.with_columns(
        pl.concat_list(amide_i_cols).alias("amide_i"),
        pl.concat_list(amide_ii_cols).alias("amide_ii"),
    )
    res = _add_peak_loc_cols(res, "amide_i", amide_i_cols)
    res = _add_peak_loc_cols(res, "amide_ii", amide_ii_cols)
    return res


def norm_by_amide_ii(df: pl.DataFrame) -> pl.DataFrame:
    amide_df = _build_amide_df(df)
    return amide_df.with_columns(
        cs.contains("wavenumber") / pl.col("amide_ii_peak")
    )
