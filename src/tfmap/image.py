import math
from typing import Self

from tfmap import (
    _IMAGE_METADATA_SECTION_MARKER,
    _LOGGER,
    _parse_images_from_map_bytes,
    _parse_position,
)


class AtlusImage(object):
    def __init__(self, images, image_coords):
        self.images = images
        self.image_coords = image_coords

    @classmethod
    def from_bytes(cls, file_bytes: bytes) -> Self:
        atlus_idx = file_bytes.find(_IMAGE_METADATA_SECTION_MARKER)
        slop = 8  # Extra information we don't care about at the moment

        file = file_bytes[
            atlus_idx + len(_IMAGE_METADATA_SECTION_MARKER) + slop :
        ]
        # # Number of spectra * 4 (number of bytes in 32-bit float) * 2 (pairs of positions)
        # end_position = n_spectra * 4 * 2
        # positions = np.frombuffer(file[:end_position], dtype=np.float32)
        # for spectra_idx, coord in enumerate(np.split(positions, n_spectra)):
        #     pixels[spectra_idx] = (coord[0], coord[1])
        # file = file[end_position:]

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
                _LOGGER.warning(
                    "WARNING: Encountered NaN pixel positions parsing image coordinates"
                )
                break

            image_coords[image_coord_idx] = parsed_extent
            image_coord_idx += 1

        # Embedded image
        images = _parse_images_from_map_bytes(file_bytes)
        return cls(images, image_coords)
