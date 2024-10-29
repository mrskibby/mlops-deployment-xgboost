# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 to match Cloud Run's expected port
EXPOSE 8080

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Set environment variable for the port
ENV PORT=8080
CMD ["python", "app.py"]
# Run the application and ensure Flask listens on port 8080
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
