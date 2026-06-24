from pypdf import PdfReader

def load_pdf(pdf_path):
    reader = PdfReader("Data/rag_oops.pdf")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text