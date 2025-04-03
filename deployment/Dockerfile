# Use Python as base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy necessary files
COPY api/ /app/api
COPY model/ /app/model
COPY deployment/requirements.txt /app/requirements.txt
COPY api/templates/ /app/templates/
COPY api/static/ /app/static/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python", "api/app.py"]
