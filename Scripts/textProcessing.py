'''
This script can be used to read entire pdf, convert that into block of texts, preprocess text etc.
'''

# Import libraries
import PyPDF2

# Function to read pdf
def read_pdf(filename:str) -> PyPDF2._reader.PdfReader:
    reader = PyPDF2.PdfReader('../PDFs/'+filename)
    return reader

# Function that returns only content of pdf as string
def get_pdf_content(filename:str) -> str:
    reader = read_pdf(filename=filename)
    pdf_content = ''
    for page in reader.pages:
        pdf_content += page.extract_text()
    
    return pdf_content


# Function that takes large block of text and return list of smaller segments 
def content_to_segments(content:str, max_lines:int = 30)->list[str]:
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_chunk_len = 0

    for line in lines:
        current_chunk.append(line)
        current_chunk_len += 1
        if(current_chunk_len >= max_lines):
            chunks.append('\n'.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks
