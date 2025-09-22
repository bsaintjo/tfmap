import polars as pl
import polars.selectors as cs
from sklearn.cluster import KMeans
import umap
from typing import Optional


def umap_to_kmean(
    df: pl.DataFrame,
    n_neighbors=15,
    masks: Optional[list[tuple[float, float]]] = None,
):
    if masks is None:
        masks = [(0.0, 5000.0)]

    cols = [
        col
        for col in df.select(cs.starts_with("wavenumber")).columns
        if any(low <= float(col.split("_")[-1]) <= high for low, high in masks)
    ]
    X = df.select(cols).to_numpy()
    components = umap.UMAP(
        densmap=True, n_neighbors=n_neighbors
    ).fit_transform(X)
    split = KMeans(n_clusters=2).fit_predict(components)
    return df.with_columns(
        pc1=components[:, 0], pc2=components[:, 1], split=split
    )
