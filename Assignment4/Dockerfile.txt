# Use the official Python image as the base image
FROM python

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from the host to the working directory in the container
COPY requirements.txt .

# Install Python dependencies listed in requirements.txt using pip, with no cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python application files (app.py, score.py, and model.pkl) from the host to the working directory in the container
COPY app.py .
COPY score.py .
COPY model.pkl .

# Expose port 5000 to allow communication to the application running inside the container
EXPOSE 5000

# Specify the command to run when the container starts, running the Python application (app.py) using Python interpreter
CMD ["python", "app.py"]
