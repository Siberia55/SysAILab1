from pathlib import Path

import pymupdf


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_document(file_path: Path) -> str:
    """
    Загружает текст из одного Markdown-, TXT- или PDF-файла.

    Возвращает извлечённый текст в виде строки.
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат файла: {file_path.name}"
        )

    if suffix == ".pdf":
        with pymupdf.open(file_path) as document:
            pages = [
                page.get_text()
                for page in document
            ]

        return "\n".join(pages).strip()

    return file_path.read_text(encoding="utf-8").strip()


def load_documents(directory: Path) -> list[dict[str, str]]:
    """
    Загружает все поддерживаемые документы из папки.

    Для каждого документа возвращает:
    - source: имя файла;
    - text: извлечённый текст.
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"Папка с документами не найдена: {directory}"
        )

    documents = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = load_document(file_path)

        if not text:
            print(f"Предупреждение: файл пустой — {file_path.name}")
            continue

        documents.append({
            "source": file_path.name,
            "text": text,
        })

    return documents
