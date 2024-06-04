# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create necessary directories
RUN mkdir -p data/co2 data/sai data/initial_1950_100 data/regional data/f_no_sai

# Copy only the specific files
COPY launch.py .
COPY planet_parasol_demo/ ./planet_parasol_demo/
COPY requirements.txt .
COPY fair/ ./fair/
COPY data/co2/co2_img.png ./data/co2/co2_img.png
COPY data/sai/sai_img.png ./data/sai/sai_img.png
COPY data/initial_1950_100/ ./data/initial_1950_100/
COPY data/regional/ ./data/regional/
COPY data/f_no_sai/ ./data/f_no_sai/
COPY data/temp/ ./data/temp/
COPY data/species_configs_properties_calibration1.2.0.csv ./data/species_configs_properties_calibration1.2.0.csv
COPY img/logo.svg ./img/logo.svg

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variables
ENV NAME World
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run app.py when the container launches
CMD ["python", "launch.py"]
