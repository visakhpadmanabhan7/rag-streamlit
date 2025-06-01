FROM python:3.10-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the app from the correct subfolder
CMD ["streamlit", "run", "basic_rag/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
