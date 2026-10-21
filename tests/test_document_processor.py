from app.services.document_processor import split_into_chunks


def test_empty_text():
    assert split_into_chunks("") == []


def test_short_text():
    text = "Hello, Enterprise Knowledge Copilot!"

    chunks = split_into_chunks(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_overlap():
    text = "A" * 1200

    chunks = split_into_chunks(
        text,
        chunk_size=500,
        overlap=50
    )

    assert len(chunks) == 3
    assert len(chunks[0]) == 500

    # First chunk's final 50 characters should
    # match the second chunk's first 50 characters.
    assert chunks[0][-50:] == chunks[1][:50]