# Analyze ThermoFisher Omnic Atlus MAP files with Python

- [Analyze ThermoFisher Omnic Atlus MAP files with Python](#analyze-thermofisher-omnic-atlus-map-files-with-python)
  - [Usage](#usage)
    - [From Omnic](#from-omnic)
    - [Command line](#command-line)
  - [Installation](#installation)
    - [via `pip`](#via-pip)
    - [via `docker`](#via-docker)
  - [Why use this?](#why-use-this)
  - [Adapting to other versions](#adapting-to-other-versions)

## Usage

### From Omnic

### Command line

```bash
# Export the embedded image as a PNG file
tfmap export-image -i data/test.map -o image.png
# Export spectra
tfmap export-spectra -i data/test.map -o spectra.csv
```

## Installation

### via `pip`

```bash
git clone https://www.github.com/bsaintjo/tfmap
cd tfmap
pip install -e .
```

### via `docker`

```bash
docker build -t tfmap .
docker run -it --rm tfmap --help
```

## Why use this?

This Python package allows for reading spectra and image data from MAP files produced by the Omnic Atlus software from ThermoFisher.

The main advantage of this package is to allow access of the embedded bright-field images taken prior to the hyperspectral image acquisition. This is particularly useful in cases like live cell FTIR microspectroscopy, where it can be valuable to integrate the bright-field image, such as knowing the boundaries of the cells, with the hyperspectral data.

## Adapting to other versions

The current version of `tfmap` has been developed for Omnic version 9.something. It uses very rough heuristics based on searching for section markers (a unique, specific sets of bytes) that are positioned relative to desired data. For example, to find the part of the file containing image metadata, we search for the pattern "0xffff7f7fffff7ff", and take byte offsets relative to this.

If you want to change the pattern used for specific sections, there are configs for different versions of omnic files.
TODO configuration toml explain
regex

- Image Metadata
  - TODO
- Image
  - Image magic numbers
- Spectra and number of spectra
  - Search for `Spectrum`, ~84 bytes pass the end, there should be an array of 32-bit floats.