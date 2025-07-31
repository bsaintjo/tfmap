"""
.. include:: ../../README.md
"""

import math
from pathlib import Path
from typing import Any, Callable, Optional, Self
import re
import io
import struct
import logging
import sys
import warnings

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import polars as pl

_LOGGER = logging.getLogger(__name__)

_IMAGE_METADATA_SECTION_MARKER = b"\xff\xff\x7f\x7f\xff\xff\x7f\xff"
_IMAGE_SECTION_MARKER = (
    bytes.fromhex(
        "b56d00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    )
    * 2
)
# N_SPECTRA = b"".join([b"\xff"] * 16 + [b"\x00"] * 2)
_N_WAVENUMBERS_SECTION_MARKER = re.compile(
    b"".join(
        [rb"[^\x00]"] * 3
        + [b"\x00"] * 5
        + [rb"[^\x00]"] * 3
        + [b"\x00"] * 5
        + [bytes.fromhex("6f12833a")]
    )
)


def _parse_bmp(xs: bytes) -> tuple[Image.Image, bytes]:
    file_size, xs = _parse_file_size(xs)
    bmp_image = Image.open(io.BytesIO(xs[:file_size]))
    return bmp_image, xs[file_size:]


def _parse_jpeg_bmp_image_pair(
    xs: bytes, skip=False, log=False
) -> tuple[list[Image.Image], bytes]:
    if log:
        _LOGGER.debug([hex(x) for x in xs[:20]])
    if skip:
        xs = xs[2:]
    file_size, xs = _parse_file_size(xs)

    jpeg_image = Image.open(io.BytesIO(xs[:file_size]))
    xs = xs[file_size:]
    file_size, xs = _parse_file_size(xs)
    bmp_image = Image.open(io.BytesIO(xs[:file_size]))
    return [jpeg_image, bmp_image], xs[file_size:]


def _parse_images_from_map_bytes(file: bytes) -> list[Image.Image]:
    images = []
    image_idx = file.find(_IMAGE_SECTION_MARKER)
    image_bytes = file[image_idx + len(_IMAGE_SECTION_MARKER) :]
    try:
        parsed_images, next_image_bytes = _parse_jpeg_bmp_image_pair(
            image_bytes, skip=False
        )
        image_bytes = next_image_bytes
        images.append(parsed_images[0])
    except Exception as e:
        _LOGGER.debug(f"Exception: {e}", image_bytes[:20])
        _LOGGER.debug("trying bmp")
        bmp_image, image_bytes = _parse_bmp(image_bytes)
        images.append(bmp_image)
    while True:
        try:
            parsed_images, image_bytes = _parse_jpeg_bmp_image_pair(
                image_bytes, skip=True, log=True
            )
            # display(parsed_images[0])
            # display(parsed_images[1])
            _LOGGER.debug("Info: ", parsed_images[0].info)
            _LOGGER.debug("Projection", parsed_images[0].getprojection())
            _LOGGER.debug("Bands", parsed_images[0].getbands())
            _LOGGER.debug("bbox", parsed_images[0].getbbox())
            images.append(parsed_images[0])
        except Exception as e:
            _LOGGER.debug(e)
            _LOGGER.debug("trying bmp")
            try:
                bmp_image, image_bytes = _parse_bmp(image_bytes)
                images.append(bmp_image)
            except Exception as e:
                _LOGGER.debug("No image: ", e)
                break
    _LOGGER.info(f"num images {len(images)}")
    return images


def _parse_position(xs: bytes) -> tuple[tuple[float, float], bytes]:
    x = struct.unpack("f", xs[:4])[0]
    y = struct.unpack("f", xs[4:8])[0]
    # x, y = np.frombuffer(xs[:8], dtype=np.float32, count=2)
    return (x, y), xs[8:]


def _peek_file_size(xs: bytes) -> int:
    return struct.unpack("I", xs[:4])[0]


def _parse_file_size(xs: bytes) -> tuple[int, bytes]:
    file_size = struct.unpack("I", xs[:4])[0]
    return file_size, xs[4:]


def _parse_spectra_frame(
    xs: bytes, n_wavenumbers: int
) -> tuple[list[float], bytes]:
    xs = xs[84:]  # Skip 84 bytes of metadata
    end_position = n_wavenumbers * 4
    acc = np.frombuffer(
        xs[:end_position], dtype=np.float32, count=n_wavenumbers
    )
    xs = xs[end_position:]
    # acc = []
    # for _ in range(n_wavenumbers):
    #     spec = struct.unpack("f", xs[:4])[0]
    #     acc.append(spec)
    #     xs = xs[4:]
    return acc, xs


class Atlus(object):
    def __init__(
        self,
        images: list[Image.Image],
        image_coords: dict[int, list[float]],
        pixels: dict[int, tuple[float, float]],
        spectra_dict: dict[int, list[float]],
        filepath: str,
    ):
        """@private"""
        self.images = images
        """@private"""
        self.image_coords = image_coords
        """@private"""
        self.pixels = pixels
        """@private"""
        self.spectra_dict = spectra_dict
        """@private"""
        self.filepath = filepath
        """@private"""

    def spectra(self) -> dict[int, ArrayLike]:
        """
        Returns a dictionary with the key as the index of the spectra and the values are numpy arrays of float values representing the spectra

        The index is the same as the index returned in Atlus.spectra.
        """
        return self.spectra_dict

    def spectra_coordinates(self) -> dict[int, tuple[float, float]]:
        """
        Returns a dictionary with the key as the index of the spectra and the values are tuples representing
        the coordinates of the spectra.

        Useful for comparing the spectra spatially.
        The coordinates are on the same coordinate system used for the embedded image.
        The index is the same as the index returned in Atlus.spectra.
        """
        return self.pixels

    @staticmethod
    def from_map_filepath(
        filepath: str | Path, parse_spectra: bool = True
    ) -> Self:
        """
        Parse a ThermoFisher Omnic Atlus file, and returns an Atlus object.

        By default, this method will extract both the image and the spectra.
        Extracting the spectra can take significantly longer, and if its not
        needed, set parse_spectra to False.

        When parse_spectra is False, both
        """
        pixels = dict()
        with open(filepath, "rb") as full_file:
            full_file = full_file.read()

        # N spectra
        spectra_frame_idx = full_file.find(b"Spectrum 1 of ")
        _LOGGER.debug(f"Start of frame: {spectra_frame_idx}")
        spectra_frame_end = full_file[spectra_frame_idx:].find(b"\x00")
        _LOGGER.debug(f"Start of title: {spectra_frame_end}")

        first_spec_title = full_file[
            spectra_frame_idx : spectra_frame_idx + spectra_frame_end
        ]
        n_spectra = int(first_spec_title.split(b" ")[-1])
        _LOGGER.info(f"Num. of spectra: {n_spectra}")

        # Image metadata
        atlus_idx = full_file.find(_IMAGE_METADATA_SECTION_MARKER)
        slop = 8  # Extra information we don't care about at the moment

        file = full_file[
            atlus_idx + len(_IMAGE_METADATA_SECTION_MARKER) + slop :
        ]
        # Number of spectra * 4 (number of bytes in 32-bit float) * 2 (pairs of positions)
        end_position = n_spectra * 4 * 2
        positions = np.frombuffer(file[:end_position], dtype=np.float32)
        for spectra_idx, coord in enumerate(np.split(positions, n_spectra)):
            pixels[spectra_idx] = (coord[0], coord[1])
        file = file[end_position:]

        image_coord_idx = 0
        image_coords = dict()
        file = file[8:]  # Skip image dimension section marker
        while file[:4] != b"\x00\x00\xff\xff":
            bottom_left, file = _parse_position(file)
            top_right, file = _parse_position(file)
            parsed_extent = [
                bottom_left[0],
                top_right[0],
                bottom_left[1],
                top_right[1],
            ]
            if any(math.isnan(x) for x in parsed_extent):
                _LOGGER.warning("WARNING: Encountered NaN pixel positions")
                break

            image_coords[image_coord_idx] = parsed_extent
            image_coord_idx += 1

        # Embedded image
        images = _parse_images_from_map_bytes(full_file)

        # N Wavelengths
        n_wavelengths_section = re.search(
            _N_WAVENUMBERS_SECTION_MARKER, full_file
        ).end()
        n_wavenumbers = struct.unpack(
            "H", full_file[n_wavelengths_section : n_wavelengths_section + 2]
        )[0]
        _LOGGER.info(f"n wavenumbers: {n_wavenumbers}")

        # Spectra
        spectra_dict: dict[int, list[float]] = dict()

        if parse_spectra:
            data = full_file[spectra_frame_idx:]

            for idx in tqdm.tqdm(range(n_spectra), desc="Parsing spectra"):
                try:
                    parsed_spectra, data = _parse_spectra_frame(
                        data, n_wavenumbers
                    )
                except ValueError:
                    warnings.warn(
                        f"""Failed to complete parsing spectra expected number of spectra\n
                        Expected {n_wavenumbers} spectra\n
                        but failed on spectra {idx}\n
                        data is possibly truncated."""
                    )
                    break
                next_frame_idx = data.find(b"Spectrum ")
                data = data[next_frame_idx:]
                spectra_dict[idx] = parsed_spectra

        return Atlus(
            images=images,
            image_coords=image_coords,
            pixels=pixels,
            spectra_dict=spectra_dict,
            filepath=str(filepath),
        )

    def _export_npz(self, filepath, wavenumbers: Optional[np.ndarray]):
        """Export Atlus data with image, coordinates, and spectra to an NPZ file."""
        if wavenumbers is None:
            wavenumbers = np.linspace(650, 4000, 3475)

        np.savez(
            file=filepath,
            rgb_image=np.array(self._map_image()),
            rgb_image_coordinates=np.array(self.image_extent()),
            spectra=np.array(
                [
                    x[1]
                    for x in sorted(
                        list(self.spectra_dict.items()), key=lambda x: x[0]
                    )
                ]
            ),
            wavenumbers=np.linspace(650, 4000, 3475),
            spectra_coordinates=np.array(
                [
                    x[1]
                    for x in sorted(
                        list(self.pixels.items()), key=lambda x: x[0]
                    )
                ]
            ),
        )

    def image_extent(self) -> list[float, float, float, float]:
        """
        Return the extent representing the maximum and minimum coordinates of the MAP file image.

        ```text
        [3] +-------+
            |       |
            |       |
            |       |
        [2] +-------+
           [0]     [1]
        ```
        """
        x_axis = np.array(
            [coord[:2] for coord in self.image_coords.values()]
        ).ravel()
        y_axis = np.array(
            [coord[2:] for coord in self.image_coords.values()]
        ).ravel()
        return [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]

    def cell_segmented(self, seg_image) -> dict[int, bool]:
        seg_rows, seg_cols = seg_image.shape
        [x_min, x_max, y_min, y_max] = self.image_extent()
        acc = dict()
        for idx, (x, y) in self.pixels.items():
            proj_col = round(((x - x_min) / (x_max - x_min)) * seg_cols)
            proj_row = round(
                seg_rows - ((y - y_min) / (y_max - y_min)) * seg_rows
            )
            acc[idx] = seg_image[proj_row][proj_col]
        return acc

    def _pixels_projected(
        self, rows: int, cols: int
    ) -> dict[int, tuple[int, int]]:
        seg_rows, seg_cols = rows, cols
        [x_min, x_max, y_min, y_max] = self.image_extent()
        acc = dict()
        for idx, (x, y) in self.pixels.items():
            proj_col = round(((x - x_min) / (x_max - x_min)) * seg_cols)
            proj_row = round(
                seg_rows - ((y - y_min) / (y_max - y_min)) * seg_rows
            )
            acc[idx] = (proj_row, proj_col)
        return acc

    def _pixels_projected_extent(self, rows: int, cols: int) -> list[int]:
        xs = []
        ys = []
        for x, y in self._pixels_projected(rows, cols).values():
            xs.append(x)
            ys.append(y)
        return [min(xs), max(xs), min(ys), max(ys)]

    def _pixels_projected_height_width(
        self, rows: int, cols: int
    ) -> tuple[int, int]:
        [xmin, xmax, ymin, ymax] = self._pixels_projected_extent(rows, cols)
        return (abs(ymax - ymin), abs(xmax - xmin))

    def _map_image_bytes(self) -> io.BytesIO:
        bs = io.BytesIO()
        fig, ax = plt.subplots(1, 1)
        ax.set_axis_off()
        for idx, image in enumerate(self.images):
            ax.imshow(image, extent=self.image_coords[idx])
        extent = self.image_extent()
        ax.set_xlim(*extent[:2])
        ax.set_ylim(*extent[2:])
        extent = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        fig.savefig(
            bs,
            format="png",
            #    bbox_inches=extent,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
        plt.close()
        return bs

    def _map_image(self) -> Image.Image:
        return Image.open(self._map_image_bytes())

    def plot_rgb_image(self, ax: Any):
        """
        Plot the embedded image on a matplotlib Axes ax.
        """
        for idx, image in enumerate(self.images):
            ax.imshow(image, extent=self.image_coords[idx])
        extent = self.image_extent()
        ax.set_xlim(*extent[:2])
        ax.set_ylim(*extent[2:])
        ax.set_ylabel("Position (um)")
        ax.set_xlabel("Position (um)")

    def _pixel_xs_ys(self) -> tuple[list[float], list[float]]:
        pixel_xs, pixel_ys = list(zip(*self.pixels.values()))
        return pixel_xs, pixel_ys

    def _plot_with_spectra(
        self,
        prep_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        ylim: Optional[tuple[float, float]] = None,
    ) -> Optional[tuple[Figure, Any]]:
        """
        Returns an interactive plot showing the image, location of pixels, and corresponding spectra when clicked on.

        Tested in a Jupyter Notebook. Install ipympl if using in VS Code.

        Requires mplcursors, will return None and print warning to stderr if not found.
        """
        try:
            import mplcursors
        except ImportError:
            print(
                "Module 'mplcursors' not found: pip install tfmap[full] to use this method",
                file=sys.stderr,
            )
            return None

        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(14, 4),
            gridspec_kw=dict(width_ratios=[4, 6], wspace=0.1),
        )
        for idx, image in enumerate(self.images):
            ax[0].imshow(image, extent=self.image_coords[idx])
        ax[0].set_ylabel("Position (um)")
        ax[0].set_xlabel("Position (um)")
        pixel_xs, pixel_ys = list(zip(*self.pixels.values()))
        scatter = ax[0].scatter(
            pixel_xs, pixel_ys, c="red", marker="x", linewidths=0.5
        )

        cursor = mplcursors.cursor(scatter)

        def plot_spectra(sel):
            spectra = self.spectra_dict[sel.index]
            if prep_fn is not None:
                # spectra = signal.savgol_filter(all_spectra[sel.index], window_length=21, polyorder=3, deriv=2)
                spectra = prep_fn(spectra)
            ax[1].clear()
            ax[1].plot(np.linspace(650, 4000, len(spectra)), spectra)
            ax[1].invert_xaxis()
            if ylim is not None:
                ax[1].set_ylim(*ylim)

        cursor.connect("add", plot_spectra)
        return fig, ax

    def _repr_jpeg_(self):
        """Helper function used to plot image in Jupyter notebooks."""
        fig, ax = plt.subplots(1, 1)
        for idx, image in enumerate(self.images):
            ax.imshow(image, extent=self.image_coords[idx])
        extent = self.image_extent()
        ax.set_xlim(*extent[:2])
        ax.set_ylim(*extent[2:])
        ax.set_ylabel("Position (um)")
        ax.set_xlabel("Position (um)")
        bs = io.BytesIO()
        fig.savefig(bs, format="JPEG")
        plt.close()

        return Image.open(bs)._repr_jpeg_()

    def _atlus_to_df(self) -> pl.DataFrame:
        pixel_idx, pixel_pos = list(zip(*self.pixels.items()))
        pixel_x, pixel_y = list(zip(*pixel_pos))
        pixel_df = pl.DataFrame(
            dict(idx=pixel_idx, pixel_x=pixel_x, pixel_y=pixel_y)
        )

        spectra_idx, spectra = list(zip(*self.spectra_dict.items()))
        spectra_df = pl.DataFrame(np.array(spectra))
        spectra_df = spectra_df.with_columns(
            pl.Series(name="idx", values=spectra_idx)
        )

        return pixel_df.join(spectra_df, on="idx").drop("idx")
