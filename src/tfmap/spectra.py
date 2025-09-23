import re
import struct
from typing import Self

import numpy as np
import tqdm

from tfmap import (
    _IMAGE_METADATA_SECTION_MARKER,
    _LOGGER,
    _N_WAVENUMBERS_SECTION_MARKER,
    _parse_spectra_frame,
)
from numpy.typing import ArrayLike


class AtlusSpectra(object):
    def __init__(
        self,
        pixels: dict[int, tuple[float, float]],
        spectra_dict: dict[int, ArrayLike],
    ):
        self.pixels = pixels
        self.spectra_dict = spectra_dict

    @classmethod
    def from_bytes(cls, file_bytes: bytes) -> Self:
        # N spectra
        spectra_frame_idx = file_bytes.find(b"Spectrum 1 of ")
        _LOGGER.debug(f"Start of frame: {spectra_frame_idx}")
        spectra_frame_end = file_bytes[spectra_frame_idx:].find(b"\x00")
        _LOGGER.debug(f"Start of title: {spectra_frame_end}")

        first_spec_title = file_bytes[
            spectra_frame_idx : spectra_frame_idx + spectra_frame_end
        ]
        n_spectra = int(first_spec_title.split(b" ")[-1])
        _LOGGER.info(f"Num. of spectra: {n_spectra}")

        atlus_idx = file_bytes.find(_IMAGE_METADATA_SECTION_MARKER)
        slop = 8  # Extra information we don't care about at the moment
        file = file_bytes[
            atlus_idx + len(_IMAGE_METADATA_SECTION_MARKER) + slop :
        ]

        # Number of spectra * 4 (number of bytes in 32-bit float) * 2 (pairs of positions)
        pixels = dict()
        end_position = n_spectra * 4 * 2
        positions = np.frombuffer(file[:end_position], dtype=np.float32)
        for spectra_idx, coord in enumerate(np.split(positions, n_spectra)):
            pixels[spectra_idx] = (coord[0], coord[1])
        file = file[end_position:]

        # N Wavenumbers
        n_wavenumbers_section = re.search(
            _N_WAVENUMBERS_SECTION_MARKER, file_bytes
        ).end()
        n_wavenumbers = struct.unpack(
            "H", file_bytes[n_wavenumbers_section : n_wavenumbers_section + 2]
        )[0]
        _LOGGER.info(f"n wavenumbers: {n_wavenumbers}")

        # Spectra
        spectra_dict: dict[int, ArrayLike] = dict()

        data = file_bytes[spectra_frame_idx:]

        for idx in tqdm.tqdm(range(n_spectra), desc="Parsing spectra"):
            try:
                parsed_spectra, data = _parse_spectra_frame(
                    data, n_wavenumbers
                )
            except ValueError:
                _LOGGER.warning(
                    f"""Failed to complete parsing spectra expected number of spectra\n
                    Expected {n_wavenumbers} spectra\n
                    but failed on spectra {idx}\n
                    data is possibly truncated."""
                )
                break
            next_frame_idx = data.find(b"Spectrum ")
            data = data[next_frame_idx:]
            spectra_dict[idx] = parsed_spectra
        return cls(pixels, spectra_dict)
