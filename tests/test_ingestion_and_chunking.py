import tempfile
import unittest
from pathlib import Path

from docx import Document

from src.chunking.chunker import chunk_documents, make_chunks
from src.ingestion.loader import load_documents


# Bảo vệ metadata và các trường hợp cấu trúc học liệu dễ bị tách sai.
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

    def test_preserves_docx_heading_table_and_body_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.docx"
            document = Document()
            document.add_heading("Network", level=1)
            document.add_paragraph("IPv4 uses 32 bits.")
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Protocol"
            table.cell(0, 1).text = "Port"
            table.cell(1, 0).text = "HTTPS"
            table.cell(1, 1).text = "443"
            document.add_paragraph("IPv6 uses 128 bits.")
            document.save(path)

            records = load_documents(directory)

        text = records[0][2]
        self.assertLess(text.index("# Network"), text.index("IPv4 uses 32 bits."))
        self.assertLess(text.index("IPv4 uses 32 bits."), text.index("| Protocol | Port |"))
        self.assertLess(text.index("| HTTPS | 443 |"), text.index("IPv6 uses 128 bits."))

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

    def test_structure_aware_keeps_study_facts_with_their_context(self):
        documents = [
            (
                "network.txt",
                7,
                "# Network\n\nIPv4 uses 32 bits; IPv6 uses 128 bits. HTTPS uses port 443.\n\n"
                "# Calculation\n\nFormula: subnet_count = 2^borrowed_bits.\n\n"
                "Where borrowed_bits is the number of host bits used by the subnet mask.\n\n"
                "Step 1: borrowed_bits = 3. Step 2: subnet_count = 2^3 = 8.\n\n"
                "# Ports\n\n| Protocol | Port |\n| HTTPS | 443 |\n| DNS | 53 |",
            )
        ]

        chunks = chunk_documents(
            documents,
            chunk_size=180,
            chunk_overlap=20,
            strategy="structure_aware",
        )

        chunk_texts = [chunk.text for chunk in chunks]
        self.assertTrue(
            any("IPv4 uses 32 bits; IPv6 uses 128 bits" in text for text in chunk_texts)
        )
        self.assertTrue(any("HTTPS uses port 443" in text for text in chunk_texts))
        self.assertTrue(
            any(
                "subnet_count = 2^borrowed_bits" in text
                and "Where borrowed_bits" in text
                for text in chunk_texts
            )
        )
        self.assertTrue(
            any("| Protocol | Port |" in text and "| HTTPS | 443 |" in text for text in chunk_texts)
        )
        self.assertTrue(
            any("Step 1:" in text and "Step 2:" in text for text in chunk_texts)
        )
        self.assertTrue(
            all(chunk.source == "network.txt" and chunk.page == 7 for chunk in chunks)
        )
        self.assertEqual({chunk.section for chunk in chunks}, {"# Network", "# Calculation", "# Ports"})

    def test_structure_aware_reports_oversized_section_splits(self):
        long_paragraph = "value = " + "x" * 300
        chunks = chunk_documents(
            [("formula.txt", None, "# Formula\n\n" + long_paragraph)],
            chunk_size=100,
            chunk_overlap=10,
            strategy="structure_aware",
        )

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.section == "# Formula" for chunk in chunks))
        split_chunks = [chunk for chunk in chunks if chunk.split_reason is not None]
        self.assertTrue(split_chunks)
        self.assertTrue(
            all(chunk.split_reason == "oversized_structure" for chunk in split_chunks)
        )
        self.assertTrue(all(chunk.split_reason is not None for chunk in chunks))

    def test_structure_aware_repeats_heading_and_table_header_after_split(self):
        table_rows = "\n".join(f"| Row {index} | {index} |" for index in range(10))
        text = "# Ports\n\n| Name | Value |\n| --- | --- |\n" + table_rows
        chunks = chunk_documents(
            [("ports.txt", 3, text)],
            chunk_size=90,
            chunk_overlap=10,
            strategy="structure_aware",
        )

        table_chunks = [chunk for chunk in chunks if chunk.split_reason == "oversized_table"]
        self.assertGreater(len(table_chunks), 1)
        self.assertTrue(all(chunk.text.startswith("# Ports") for chunk in table_chunks))
        self.assertTrue(all("| Name | Value |" in chunk.text for chunk in table_chunks))
        self.assertTrue(all(chunk.source == "ports.txt" and chunk.page == 3 for chunk in chunks))

    def test_structure_aware_uses_overlap_and_stable_chunk_ids(self):
        text = "# Section\n\nFirst paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        first = chunk_documents(
            [("notes.txt", None, text)],
            chunk_size=60,
            chunk_overlap=20,
            strategy="structure_aware",
        )
        second = chunk_documents(
            [("notes.txt", None, text)],
            chunk_size=60,
            chunk_overlap=20,
            strategy="structure_aware",
        )

        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertTrue(all(chunk.section == "# Section" for chunk in first))
        self.assertIn("Second paragraph.", first[1].text)
