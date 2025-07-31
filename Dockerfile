# Use the official Python image from the Debian slim variant
FROM python:3.11-slim

# Set environment variables for non-interactive installation and locale
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set the working directory
WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"
ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml .
RUN uv pip install -r pyproject.toml --all-extras

# Copy the rest of the application code
COPY . .

# Install the package and its dependencies
RUN uv pip install -e ".[full]"

ENTRYPOINT ["tfmap"]