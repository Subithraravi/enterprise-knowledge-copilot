from app.services.document_processor import (
    extract_text,
    split_into_chunks
)

file_path = "sample.txt"

# Step 1: Extract document text
text = extract_text(file_path)

print("EXTRACTED TEXT:")
print(text)

# Step 2: Split extracted text into chunks
chunks = split_into_chunks(
    text,
    chunk_size=100,
    overlap=20
)

print("\nTOTAL CHUNKS:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\nCHUNK {index}:")
    print(chunk)