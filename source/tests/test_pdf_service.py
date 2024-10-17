import unittest
from services.pdf_service import PDFService
from io import BytesIO
from PyPDF2 import PdfWriter

class TestPDFService(unittest.TestCase):

    def test_process_pdf_valid(self):
        pdf_stream = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(pdf_stream)
        pdf_stream.seek(0)

        class MockFile:
            def read(self):
                return pdf_stream.read()

        mock_file = MockFile()
        result = PDFService.process_pdf(mock_file)
        self.assertIsInstance(result, str)


    def test_process_pdf_empty_file(self):
        # Simula un archivo PDF vacío
        class MockFile:
            def read(self):
                return b''

        mock_file = MockFile()
        result = PDFService.process_pdf(mock_file)
        self.assertEqual(result, '')

if __name__ == '__main__':
    unittest.main()