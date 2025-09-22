import argparse
from pathlib import Path
import sys
from importlib.metadata import version
import warnings

from tfmap import Atlus
from PIL.PngImagePlugin import PngInfo

try:
    __version__ = version("tfmap")
except Exception:
    __version__ = "0.0.0"


def export_image(input_file: str, output_file: str):
    atlus = Atlus.from_map_filepath(input_file, parse_spectra=False)
    metadata = PngInfo()
    extent_str = ",".join([str(x) for x in atlus.image_extent()])
    metadata.add_text("extent", extent_str)
    image = atlus._map_image()
    image.save(output_file, pnginfo=metadata, format="PNG")


def export_spectra(input_file: str, output_file: str):
    atlus = Atlus.from_map_filepath(input_file, parse_spectra=True)
    df = atlus.to_polars()
    df.write_csv(output_file)


def export_spectra_npz(input_file: str, output_file: str):
    atlus = Atlus.from_map_filepath(input_file, parse_spectra=True)
    atlus._export_npz(output_file)
    # df.write_csv(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interacting with ThermoFisher Omnic Atlus MAP files"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Subcommand to run"
    )

    image_parser = subparsers.add_parser(
        "export-image",
        help="Export an image from ThermoFisher Omnic Atlus MAP file",
        description="""
Extracts the embedded image from within a ThermoFisher Omnic Atlus file.
The output image is saved as a PNG file.
Image coordinates are saved as metadata within the file with the extent key
and matches the usage as the Atlus.image_extent method.
""",
    )
    image_parser.add_argument(
        "-i", "--input", required=True, help="Path to Atlus MAP file"
    )
    image_parser.add_argument(
        "-o", "--output", required=True, help="Output file path as PNG"
    )

    spectra_parser = subparsers.add_parser(
        "export-spectra",
        help="Export spectra data from ThermoFisher Omnic Atlus MAP file",
    )

    spectra_parser.add_argument(
        "-i", "--input", required=True, help="Path to Atlus MAP file"
    )
    spectra_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path, use npz/csv for filetype",
    )
    spectra_parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Don't print progress bar during parsing",
    )

    args = parser.parse_args()

    if args.command == "export-image":
        export_image(args.input, args.output)
    elif args.command == "export-spectra":
        match ext := Path(args.output).suffix:
            case ".csv":
                export_spectra(args.input, args.output)
            case ".npz":
                export_spectra_npz(args.input, args.output)
            case _:
                warnings.warn(
                    f"Unrecognized file extension: '{ext}', must be .csv or .npz"
                )
                sys.exit(2)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
