def chunk(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    out, i = [], 0
    text = text.strip()
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        i += size - overlap
    return out
