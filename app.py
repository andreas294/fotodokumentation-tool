import os
import sys
import webbrowser
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
from datetime import datetime
import pillow_heif  # type: ignore  # HEIC-Unterstützung
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import exifread
from typing import List
from jinja2 import Environment, select_autoescape
from pydantic import BaseModel, Field
import re
import tempfile
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import json
import shutil

# HEIC-Unterstützung registrieren
pillow_heif.register_heif_opener()

app = FastAPI()

# Ensure the uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount(f"/{UPLOAD_DIR}", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Ensure the templates directory exists
TEMPLATES_DIR = "c:/Users/angerner/templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Add custom zip filter to Jinja2 environment
def zip_filter(a, b):
    return zip(a, b)

templates.env.filters['zip'] = zip_filter

def sort_images_by_metadata(image_files, metadata):
    """Sortiert die Bilder nach Aufnahmedatum. Falls kein Aufnahmedatum vorhanden ist, nach Dateiname."""
    def get_sort_key(meta):
        date = meta.get('Aufnahmedatum', '0000-00-00 00:00:00')
        return date if date and date != 'Unbekannt' else '0000-00-00 00:00:00'

    sorted_images = sorted(
        zip(image_files, metadata),
        key=lambda x: get_sort_key(x[1]),
        reverse=False
    )
    
    if any(meta['Aufnahmedatum'] == 'Unbekannt' for _, meta in sorted_images):
        sorted_images = sorted(
            sorted_images,
            key=lambda x: x[0].lower()
        )
    
    return [img for img, _ in sorted_images], [meta for _, meta in sorted_images]

def convert_to_degrees(value):
    """Updated convert_to_degrees function from V1.1"""
    if isinstance(value, tuple):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)
    return value

def get_gps_info(exif_data):
    gps_info = {}
    for key, value in exif_data.items():
        decode = TAGS.get(key, key)
        if decode == "GPSInfo" and isinstance(value, dict):
            for t, val in value.items():
                sub_decoded = GPSTAGS.get(t, t)
                gps_info[sub_decoded] = val
    return gps_info

def inspect_image_metadata(image_path):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return {
            "Bildbezeichnung": os.path.basename(image_path),
            "Aufnahmedatum": "Unbekannt",
            "GPS-Koordinaten": "Nicht verfügbar",
            "Blickrichtung": "Nicht verfügbar",
            "img": image_path  # Ensure the filename is included
        }
    
    exif_data = img.getexif()
    if not exif_data:
        print(f"No EXIF data found for image {image_path}")
        return {
            "Bildbezeichnung": os.path.basename(image_path),
            "Aufnahmedatum": "Unbekannt",
            "GPS-Koordinaten": "Nicht verfügbar",
            "Blickrichtung": "Nicht verfügbar",
            "img": image_path  # Ensure the filename is included
        }

    metadata = {
        "Bildbezeichnung": os.path.basename(image_path),
        "img": image_path  # Ensure the filename is included
    }
    date_taken = None
    gps_info = get_gps_info(exif_data)
    direction = None

    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)

        if tag_name == 'DateTimeOriginal':
            try:
                date_taken = datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%d.%m.%Y %H:%M:%S")
            except ValueError:
                date_taken = "Unbekannt"
        elif tag_name == 'DateTime':
            try:
                date_taken = datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%d.%m.%Y %H:%M:%S")
            except ValueError:
                date_taken = "Unbekannt"

    if gps_info:
        print(f"GPS Info for image {image_path}: {gps_info}")  # Debugging statement
        if gps_info.get('GPSLatitude') and gps_info.get('GPSLatitudeRef') and gps_info.get('GPSLongitude') and gps_info.get('GPSLongitudeRef'):
            latitude = gps_info['GPSLatitude']
            latitude_ref = gps_info['GPSLatitudeRef']
            longitude = gps_info['GPSLongitude']
            longitude_ref = gps_info['GPSLongitudeRef']

            lat = convert_to_degrees(latitude)
            lon = convert_to_degrees(longitude)
            if latitude_ref != "N":
                lat = -lat
            if longitude_ref != "E":
                lon = -lon

            metadata['GPS-Koordinaten'] = f"{lat:.6f}, {lon:.6f}"
        else:
            metadata['GPS-Koordinaten'] = "Nicht verfügbar"

        if 'GPSImgDirection' in gps_info:
            direction = gps_info['GPSImgDirection']
            if isinstance(direction, tuple):
                direction = direction[0] / direction[1]
            metadata['Blickrichtung'] = f"{direction:.2f}°"
        else:
            metadata['Blickrichtung'] = "Nicht verfügbar"
    else:
        metadata['GPS-Koordinaten'] = "Nicht verfügbar"
        metadata['Blickrichtung'] = "Nicht verfügbar"

    metadata['Aufnahmedatum'] = date_taken if date_taken else "Unbekannt"
    metadata.setdefault('GPS-Koordinaten', "Nicht verfügbar")
    metadata.setdefault('Blickrichtung', "Nicht verfügbar")
    return metadata

def inspect_image_metadata2(image_path):
    """Updated metadata inspection function from V1.1"""
    f = open(image_path, 'rb')
    tags = exifread.process_file(f)

    metadata = {
        "Blickrichtung": None,
        "GPS-Koordinaten": None,
        "Aufnahmedatum": None,
        "img": image_path  # Ensure the filename is included
    }
    gps_long = None
    gps_lat = None
    for tag in tags.keys():
        if tag == 'GPS GPSImgDirection':
            metadata["Blickrichtung"] = [round(float(v), 1) for v in tags['GPS GPSImgDirection'].values][0]

        if tag == 'GPS GPSLatitude':
            gps_lat = convert_to_degrees(tuple(float(v) for v in tags[tag].values))

        if tag == 'GPS GPSLongitude':
            gps_long = convert_to_degrees(tuple(float(v) for v in tags[tag].values))

        if tag == 'EXIF DateTimeOriginal':
            try:
                metadata["Aufnahmedatum"] = tags[tag].values
            except ValueError:
                metadata["Aufnahmedatum"] = "Unbekannt"

    if gps_long and gps_lat:
        metadata["GPS-Koordinaten"] = f"{round(gps_lat, 5)}; {round(gps_long, 5)}"
    else:
        metadata["GPS-Koordinaten"] = "Unbekannt"
    return metadata

def rotate_image(image, angle):
    """Simple image rotation function"""
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image

def create_pdf_with_captions(title, date, images, metadata):
    """Erstellt ein PDF mit Bildern und Bildunterschriften (mit Rotation)."""
    pdf_io = io.BytesIO()
    pdf = canvas.Canvas(pdf_io, pagesize=A4)
    page_width, page_height = A4
    margin = 30
    header_height = 50
    footer_margin = 30
    image_spacing = 15
    caption_spacing = 20
    available_height = page_height - margin - header_height - footer_margin

    def set_header(is_first_page):
        if is_first_page:
            pdf.setFont("Helvetica-Bold", 20)
            pdf.drawString(margin, page_height - margin - header_height + 35, f"Fotodokumentation: {title}")
            pdf.setFont("Helvetica", 14)
            pdf.drawString(margin, page_height - margin - header_height + 20, f"Datum der Erkundung: {date}")

    is_first_page = True
    current_y = page_height - margin - header_height

    for i, img in enumerate(images):
        try:
            img = img.convert('RGB')  # Ensure it is a PIL Image

            aspect_ratio = img.width / img.height
            target_width = page_width - 2 * margin
            target_height = available_height / 2 - image_spacing - caption_spacing

            if aspect_ratio > 1:
                if target_width / aspect_ratio > target_height:
                    target_width = target_height * aspect_ratio
                else:
                    target_height = target_width / aspect_ratio
            else:
                if target_height * aspect_ratio > target_width:
                    target_height = target_width / aspect_ratio
                else:
                    target_width = target_height * aspect_ratio

            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=95)
            img_io.seek(0)
            img_reader = ImageReader(img_io)

            if is_first_page:
                set_header(is_first_page)
                is_first_page = False

            if i % 2 == 0:
                position_y = current_y - target_height
                pdf.drawImage(img_reader, margin, position_y, width=target_width, height=target_height)

                caption = (f"Aufnahmedatum: {metadata[i]['Aufnahmedatum']}, "
                           f"GPS: {metadata[i]['GPS-Koordinaten']}, "
                           f"Blickrichtung: {metadata[i]['Blickrichtung']}")
                pdf.setFont("Helvetica", 8)
                pdf.drawString(margin, position_y - 10, caption)

                current_y = position_y - caption_spacing - image_spacing
            else:
                position_y = current_y - target_height
                pdf.drawImage(img_reader, margin, position_y, width=target_width, height=target_height)

                caption = (f"Aufnahmedatum: {metadata[i]['Aufnahmedatum']}, "
                           f"GPS: {metadata[i]['GPS-Koordinaten']}, "
                           f"Blickrichtung: {metadata[i]['Blickrichtung']}")
                pdf.setFont("Helvetica", 8)
                pdf.drawString(margin, position_y - 10, caption)

                if (i + 1) % 2 == 0 and i + 1 < len(images):
                    pdf.showPage()
                    current_y = page_height - margin
                    set_header(False)

        except Exception as e:
            logging.error(f"Error processing image {i + 1}: {e}")
            continue

    pdf.save()
    pdf_io.seek(0)
    return pdf_io

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class UploadImagesResponse(BaseModel):
    images: List[str]
    metadata: List[dict]

class GeneratePDFResponse(BaseModel):
    pdf_url: str

def clear_uploads_folder(exclude_files=None):
    """Clears the uploads folder, excluding specified files."""
    try:
        exclude_files = exclude_files or []
        for filename in os.listdir(UPLOAD_DIR):
            if filename not in exclude_files:
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    except Exception as e:
        logging.error(f"Failed to clear uploads folder: {e}")

@app.post("/upload", response_model=UploadImagesResponse)
async def upload_images(request: Request, title: str = Form(...), date: str = Form(...), images: List[UploadFile] = File(...)):
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No files uploaded")

        image_files = []
        metadata = []
        
        for file in images:
            # Clean the filename to remove any path components
            safe_filename = os.path.basename(file.filename)
            img_path = os.path.join(UPLOAD_DIR, safe_filename)
            
            # Save the uploaded file
            with open(img_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Inspect metadata before converting HEIC/HEIF images to JPEG
            meta = inspect_image_metadata2(img_path)
            
            # Handle HEIC/HEIF conversion
            if safe_filename.lower().endswith(('.heic', '.heif')):
                try:
                    heif_file = pillow_heif.read_heif(img_path)
                    img = Image.frombytes(
                        heif_file.mode, 
                        heif_file.size, 
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    )
                    img_path_jpeg = os.path.splitext(img_path)[0] + '.jpg'
                    img.save(img_path_jpeg, format="JPEG")
                    img_path = img_path_jpeg
                except Exception as e:
                    print(f"Error converting HEIC file {safe_filename}: {e}")
                    continue
            
            # Convert other image formats to JPEG
            if not safe_filename.lower().endswith('.jpg'):
                try:
                    with Image.open(img_path) as img:
                        img_path_jpeg = os.path.splitext(img_path)[0] + '.jpg'
                        img.save(img_path_jpeg, format="JPEG")
                        img_path = img_path_jpeg
                except Exception as e:
                    print(f"Error converting image {safe_filename} to JPEG: {e}")
                    continue
            
            image_files.append(img_path)
            metadata.append(meta)

        # Sort images based on metadata
        sorted_images, sorted_metadata = sort_images_by_metadata(image_files, metadata)

        # Generate URLs for the images to preview
        image_urls = [f"/{UPLOAD_DIR}/{os.path.basename(img)}" for img in sorted_images]

        # Clear the uploads folder, excluding the sorted images
        clear_uploads_folder(exclude_files=[os.path.basename(img) for img in sorted_images])

        return {"images": image_urls, "metadata": sorted_metadata}
        
    except Exception as e:
        print(f"Error in upload_images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SavePDFRequest(BaseModel):
    filename: str

def sanitize_filename(filename):
    """Remove or replace invalid filename characters."""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove any non-printable characters
    filename = ''.join(char for char in filename if char.isprintable())
    # Trim spaces and dots at the end
    filename = filename.strip('. ')
    # Limit length (Windows has 255 char limit)
    if len(filename) > 200:
        filename = filename[:200]
    return filename or 'Fotodokumentation'  # Default name if everything was removed

# Allow CORS for all origins (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files from the downloads directory
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")

class GeneratePDFRequest(BaseModel):
    title: str
    date: str
    images: List[str]
    rotations: List[int]
    metadata: List[str]
    include: List[bool]  # Add include field

@app.post("/generate_pdf", response_model=GeneratePDFResponse)
async def generate_pdf(form_data: GeneratePDFRequest):
    try:
        logging.debug(f"Received form data: {form_data}")
        
        # Print the list of images in the terminal
        print("Images:", form_data.images)
        print("Rotations:", form_data.rotations)
        print("Metadata:", form_data.metadata)
        
        # Ensure images list is not empty
        if not form_data.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        formatted_date = datetime.strptime(form_data.date, '%Y-%m-%d').strftime('%d.%m.%Y')
        safe_title = sanitize_filename(form_data.title)
        
        image_objects = []
        decoded_metadata = []

        # Process images (unchanged)
        processed_images = set()
        
        # Print the processed_images set in the terminal
        print("Processed Images:", processed_images)
        
        for img_path, rotation, include, meta in zip(form_data.images, form_data.rotations, form_data.include, form_data.metadata):
            if not include:
                continue
            try:
                # Decode the URL-encoded image path
                filename = os.path.basename(img_path.replace('/', os.sep))
                filename = re.sub(r'%20', ' ', filename)  # Replace URL-encoded spaces
                abs_img_path = os.path.join(UPLOAD_DIR, filename)
                
                if not os.path.exists(abs_img_path):
                    logging.error(f"Image not found at {abs_img_path}")
                    continue

                if abs_img_path in processed_images:
                    logging.warning(f"Image {abs_img_path} already processed, skipping.")
                    continue

                with Image.open(abs_img_path) as img:
                    processed_img = img.convert('RGB')
                    if rotation and int(rotation) != 0:
                        logging.debug(f"Rotating image {filename} by {rotation} degrees")
                        processed_img = rotate_image(processed_img, int(rotation))
                    image_objects.append(processed_img.copy())
                    decoded_metadata.append(json.loads(meta))
                    processed_images.add(abs_img_path)
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}")
                continue

        if not image_objects:
            raise HTTPException(status_code=400, detail="No valid images to process")

        # Create PDF in memory
        try:
            pdf_io = create_pdf_with_captions(form_data.title, formatted_date, image_objects, decoded_metadata)
        except Exception as e:
            logging.error(f"Error creating PDF: {e}")
            raise HTTPException(status_code=500, detail="Error creating PDF")

        # Save the PDF to the uploads directory
        filename = f"{safe_title}_Fotodokumentation.pdf"
        save_path = os.path.join(UPLOAD_DIR, filename)
        
        # Ensure the directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save the PDF
        try:
            with open(save_path, 'wb') as f:
                pdf_io.seek(0)
                f.write(pdf_io.read())
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            raise HTTPException(status_code=500, detail="Error saving PDF")
        
        logging.debug(f"PDF saved at {save_path}")
        
        # Return the PDF URL as a response
        return {"pdf_url": f"/{UPLOAD_DIR}/{filename}"}
        
    except Exception as e:
        logging.error(f"Error in generate_pdf: {e}")
        logging.error(f"Form data: {form_data}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.post("/create_another_document", response_model=dict)
async def create_another_document():
    # Logic to create another document
    clear_uploads_folder()
    return {"message": "Ready to create another document"}

@app.post("/close_window")
async def close_window():
    try:
        # Send a response that will trigger window.close()
        clear_uploads_folder()
        return {"status": "closing"}
    except Exception as e:
        print(f"Error closing window: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed port to 8002
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
