from pathlib import Path
import base64
import mimetypes
from urllib.parse import urlparse
from PIL import ImageGrab  # for clipboard
import requests

def prepare_image_url(image_path: str | Path) -> str:
    """
    Convert image path/URL into a format suitable for API calls.
    Returns base64 data URI for local files, or original URL for web URLs.
    Special value '^V' reads from clipboard.
    """
    if image_path == '^V':
        # Get image from clipboard
        try:
            clipboard_img = ImageGrab.grabclipboard()
            if clipboard_img is None:
                raise ValueError("No image found in clipboard")
            
            # Convert PIL Image to base64
            import io
            buf = io.BytesIO()
            clipboard_img.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_b64}"
            
        except Exception as e:
            raise ValueError(f"Failed to read image from clipboard: {e}")

    # Convert Path to string if needed
    image_path = str(image_path)

    # Check if it's a URL
    try:
        result = urlparse(image_path)
        if all([result.scheme, result.netloc]):
            # Validate URL points to an image
            response = requests.head(image_path)
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image: {content_type}")
            return image_path
    except Exception:
        pass

    # Handle local file
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get mime type
    mime_type = mimetypes.guess_type(path)[0]
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"File does not appear to be an image: {image_path}")

    # Read and encode file
    with open(path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        return f"data:{mime_type};base64,{img_b64}"
    
def prepare_image_message(message: str, image_path: str | Path) -> list[dict[str, str]]:
    """
    Embed an image into a message using Markdown syntax.
    Args:
        message (str): text component of message.
        image_path (str | Path): Path to image file, URL, or '^V' to read from clipboard.
    """
    image_url = prepare_image_url(image_path)
    return [
        {"type": "text", "text": message},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]