# Sentiment Analyzer Web Application

This is a Python-based web application for sentiment analysis. It uses a Flask web framework and comes with a Docker image for easy deployment. You can run the application either by using Docker or by setting up the project locally. Follow the instructions below for both approaches.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Method 1: Run with Docker](#method-1-run-with-docker)
- [Method 2: Run Locally (Without Docker)](#method-2-run-locally-without-docker)
- [How to Contribute](#how-to-contribute)

---

## Prerequisites

Before you start, make sure you have the following installed:

- **[Docker](https://www.docker.com/get-started)** (for running the application in a containerized environment).
- **[Python 3.12+](https://www.python.org/downloads/)** and **pip** (for running the application without Docker).
- If you choose to run the application **locally**, a basic understanding of Python and Flask is recommended.

---

## Method 1: Run with Docker

If you prefer to run the web application using Docker, follow these steps:

### 1. Install Docker

Make sure you have Docker installed. If not, [download and install Docker](https://www.docker.com/get-started).

After installation, make sure Docker is running. You can check with:

```bash
docker --version
```

### 2. Log in to Docker Hub (if necessary)

You need a Docker Hub account to pull and run the image. If you don't have one, you can create an account at Docker Hub.

```bash
docker login
```
Enter your username and password when prompted.

### 3. Pull the Docker Image

```bash
docker pull ehteesham/sentimentanalyzer:latest
```
This will download the Docker image to your local machine.

### 4. Run the Docker Container

Once the image is pulled, you can run the Docker container:

```bash
docker run -p 5000:5000 yourusername/sentimentanalyzer
```
This command will run the Flask application inside the container and expose it on port 5000.

### 5. Access the Application
After the container is running, open a browser and go to:
```bash
http://127.0.0.1:5000
```
You should now see the sentiment analysis web app running.


## Method 2: Run Locally (Without Docker)

If you prefer to run the project locally without Docker, follow these steps.

### 1. Clone the Project Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Ehteesham/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Create a Virtual Environment (Optional but Recommended)

It’s a good practice to create a virtual environment to manage dependencies. You can do this with the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Or on macOS/Linux
source venv/bin/activate
```

### 3. Install the Required Dependencies

Install the necessary dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources (For Tokenization and Stopwords)

Some resources from NLTK are required for text processing. In your terminal, run the following commands:

```bash
python3 -m nltk.downloader punkt_tab stopwords
```
This ensures that the application can tokenize text and filter out stopwords.

### 5. Run the Flask Application

Once the dependencies are installed, you can run the Flask app with:

```bash
python app.py
```

### 6. Access the Application

After the Flask app starts, you can open your browser and go to:
```bash
http://127.0.0.1:5000
```
This will open the sentiment analysis web app.

## How to Contribute
If you'd like to contribute to this project, follow these steps:

1. Fork the repository and create a new branch for your changes.
2. Clone your forked repository to your local machine.
3. Make changes and ensure everything works locally.
4. Commit your changes and push them to your fork.
5. Create a pull request from your branch to the main repository.

If you encounter any issues, feel free to open an issue on GitHub or contact the maintainer.

## Contact
* Maintainer: Ansari Ehteesham Aqeel
* Email: an.ehteesham@gmail.com