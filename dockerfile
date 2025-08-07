# 1. Start with an official Python 3.11 "slim" base image
FROM python:3.11-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install all the Python libraries listed in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your other project files (app.py, stock_agent.py) into the container
COPY . .

# 6. Expose port 8501, which is the default port for Streamlit
EXPOSE 8501

# 7. The command to run when the container starts
# This starts your Streamlit app and makes it accessible
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]