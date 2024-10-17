import tempfile
from PyPDF2 import PdfReader

class PDFService:
    @staticmethod
    def process_pdf(uploaded_file):
        """Process the uploaded PDF and extract text."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        reader = PdfReader(temp_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()

        return pdf_text