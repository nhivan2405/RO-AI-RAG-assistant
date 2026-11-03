import tempfile
import unittest
from pathlib import Path

from docx import Document

from src.chunking.chunker import chunk_documents, make_chunks
from src.ingestion.loader import load_documents


class IngestionAndChunkingTest(unittest.TestCase):
    def test_loads_text_and_docx_with_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "notes.txt").write_text("Nội dung TXT", encoding="utf-8")
            document = Document()
            document.add_paragraph("Nội dung DOCX")
            document.save(folder / "notes.docx")

            records = load_documents(str(folder))

        self.assertEqual(
            records,
            [
                ("notes.docx", None, "Nội dung DOCX"),
                ("notes.txt", None, "Nội dung TXT"),
            ],
        )

    def test_fixed_chunking_preserves_metadata_and_overlap(self):
        chunks = chunk_documents(
            [("source.txt", 2, "abcdefghij")],
            chunk_size=4,
            chunk_overlap=1,
            strategy="fixed",
        )

        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "defg", "ghij", "j"])
        self.assertEqual([chunk.chunk_id for chunk in chunks], [
            "source.txt:p2:c0",
            "source.txt:p2:c1",
            "source.txt:p2:c2",
            "source.txt:p2:c3",
        ])
        self.assertTrue(all(chunk.source == "source.txt" and chunk.page == 2 for chunk in chunks))

    def test_make_chunks_defaults_to_recursive_strategy(self):
        documents = [("source.txt", None, "Một. Hai. Ba.")]
        expected = chunk_documents(documents, 7, 0, strategy="recursive")

        self.assertEqual(make_chunks(documents, 7, 0), expected)

    def test_rejects_invalid_chunk_options(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_documents([], chunk_size=100, chunk_overlap=100)
