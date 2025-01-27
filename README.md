# Fotodokumentation-Tool

This is a Fotodokumentation-Tool built with FastAPI. It allows users to upload images, preview them, and generate a PDF document with metadata and captions.

## Features

- Upload multiple images
- Preview uploaded images
- Rotate images before generating PDF
- Generate PDF with image metadata and captions
- Download generated PDF
- Create another document

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/andreas294/fotodokumentation-tool.git
    cd fotodokumentation-tool
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:

    ```sh
    uvicorn app:app --reload
    ```

2. Open your browser and navigate to `http://127.0.0.1:8000` to access the application.

## Deployment

To deploy this application to a platform that uses a `Procfile` (e.g., Heroku), follow these steps:

1. Ensure you have a `Procfile` in the root directory of your project with the following content:

    ```plaintext
    web: uvicorn app:app --host 0.0.0.0 --port $PORT
    ```

2. Ensure you have a `runtime.txt` file specifying the Python version:

    ```plaintext
    python-3.9.7
    ```

3. Deploy the application using the platform's deployment instructions.

## Endpoints

- `GET /`: Home page to upload images
- `POST /upload`: Endpoint to upload images
- `POST /generate_pdf`: Endpoint to generate PDF from uploaded images
- `POST /create_another_document`: Endpoint to reset the application for creating another document
- `POST /close_window`: Endpoint to close the application window

## Directory Structure

```
fotodokumentation-tool/
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── templates/
│   └── index.html
└── uploads/
```

## License

This project is licensed under the MIT License.