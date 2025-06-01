from PyPDF2 import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    return "".join(page.extract_text() for page in reader.pages)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
