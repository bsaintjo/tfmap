import argparse
import sys
from importlib.metadata import version
from tfmap import Atlus

try:
    __version__ = version("tfmap")
except Exception:
    __version__ = "0.0.0"


def export_image(input_file: str, output_file: str):
    atlus = Atlus.from_map_filepath(input_file, parse_spectra=False)
    print(f"This is where we'd process {input_file} and save an image to {output_file}")


def export_spectra(input_file: str, output_file: str):
    atlus = Atlus.from_map_filepath(input_file, parse_spectra=True)
    df = atlus._atlus_to_df()
    print(f"Exporting spectra from {input_file} to {output_file}")
    df.write_csv(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interacting with ThermoFisher Omnic Atlus MAP files"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    image_parser = subparsers.add_parser(
        "export-image", help="Export an image from ThermoFisher Omnic Atlus MAP file"
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
        "-o", "--output", required=True, help="Output file path as csv"
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
        export_spectra(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
