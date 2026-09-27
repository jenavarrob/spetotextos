from pathlib import Path
from io import BytesIO

try:
    from PIL import Image
    import pytesseract
except ImportError: 
    Image = None
    pytesseract = None

class TesseractService:
    """Local image-to-text service backed by the Tesseract executable."""

    def __init__(self, command: str | None = None, language: str = "eng"):
        self.command = command
        self.language = language

    def _configure(self) -> None:
        if self.command:
            pytesseract.pytesseract.tesseract_cmd = self.command

    def check_connection(self) -> dict[str, object]:
        """Return a non-throwing health result for the local OCR executable."""
        result = {
            "connected": False,
            "command": self.command or "PATH",
            "language": self.language,
            "version": None,
            "error": None,
        }
        if Image is None or pytesseract is None:
            result["error"] = (
                "Local OCR dependencies are missing. Install them with "
                "'python -m pip install Pillow pytesseract'."
            )
            return result

        try:
            self._configure()
            result["version"] = str(pytesseract.get_tesseract_version()).strip()
            result["connected"] = True
        except Exception as exc:
            result["error"] = str(exc)
        return result

    def extract_text(self, contents: bytes) -> str:
        if Image is None or pytesseract is None:
            raise RuntimeError(
                "Local OCR dependencies are missing. Install them with "
                "'python -m pip install Pillow pytesseract'."
            )

        self._configure()

        with Image.open(BytesIO(contents)) as image:
            text = pytesseract.image_to_string(image, lang=self.language)
        return text.strip()
