import json
import re
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

from embedding_chunks import (
    DEFAULTS, LlamaTokenizer, chunk_documents, document_tokens,
    format_document, format_query, load_embedding_config, split_text,
)


class ByteTokenizer:
    """Adversarial tokenizer: numeric and Unicode text cannot hide in char estimates."""
    def count(self, text):
        return len(text.encode("utf-8")) + 2


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def assert_bounded(self, chunks, limit):
        self.assertTrue(chunks)
        self.assertTrue(all(document_tokens(c, self.tokenizer) <= limit for c in chunks))

    def test_short_form_stays_intact(self):
        text = "Form W-2 2025\nBox 1: 85,000\nBox 2: 12,000"
        docs = chunk_documents([Document(text, {"document_type": "tax_form"})], tokenizer=self.tokenizer)
        self.assertEqual([d.page_content for d in docs], [text])

    def test_oversized_unbroken_unicode_is_lossless(self):
        text = "漢字1234567890" * 100
        chunks = split_text(text, limit=160, overlap=0, tokenizer=self.tokenizer)
        self.assert_bounded(chunks, 160)
        self.assertEqual("".join(chunks), text)

    def test_overlap_keeps_whole_values_and_progresses(self):
        values = [f"row{i:04d}=12345.67" for i in range(100)]
        chunks = split_text("\n".join(values), limit=240, overlap=48, tokenizer=self.tokenizer)
        self.assert_bounded(chunks, 240)
        self.assertTrue(all(any(v in c for c in chunks) for v in values))
        self.assertLess(len(chunks), 100)
        self.assertTrue(any(set(a.splitlines()) & set(b.splitlines()) for a, b in zip(chunks, chunks[1:])))

    def test_split_forms_repeat_source_year_and_table_headers(self):
        header = "| Item | Amount |\n| --- | --- |\n"
        rows = [f"| row{i:04d} | {i}.00 |\n" for i in range(120)]
        text = "Tax year: 2025\nAccount: Demo\n\n" + header + "".join(rows)
        original = Document(text, {"source_file": "statement.pdf", "document_type": "tax_form", "doc_id": "form"})
        settings = {**DEFAULTS, "form_chunk_tokens": 512}
        chunks = chunk_documents([original], settings=settings, tokenizer=self.tokenizer)
        self.assertGreater(len(chunks), 1)
        self.assert_bounded([d.page_content for d in chunks], 512)
        for chunk in chunks:
            self.assertIn("statement.pdf", chunk.page_content)
            self.assertIn("2025", chunk.page_content)
            self.assertIn(header.strip(), chunk.page_content)
        self.assertTrue(all(any(row.strip() in c.page_content for c in chunks) for row in rows))
        self.assertEqual(len({c.metadata['doc_id'] for c in chunks}), len(chunks))
        self.assertEqual(original.metadata['doc_id'], 'form')
        again = chunk_documents([original], settings=settings, tokenizer=self.tokenizer)
        self.assertEqual(chunks, again)

    def test_general_and_form_policies_differ(self):
        text = "A complete sentence with related facts. " * 100
        general = chunk_documents([Document(text)], tokenizer=self.tokenizer)
        forms = chunk_documents([Document(text, {"document_type": "tax_form"})], tokenizer=self.tokenizer)
        self.assert_bounded([c.page_content for c in general], 384)
        self.assert_bounded([c.page_content for c in forms], 1024)
        self.assertGreater(len(general), len(forms))

    def test_legacy_tax_code_is_rechunked_with_unique_ids(self):
        text = "search_document: " + "A statutory provision. " * 200
        original = Document(text, {"doc_id": "tax42", "path": "26 USC 42"})
        docs = chunk_documents([original], kind="tax_code", tokenizer=self.tokenizer)
        self.assert_bounded([d.page_content for d in docs], 512)
        self.assertTrue(all("search_document:" not in d.page_content for d in docs))
        self.assertTrue(all(d.metadata['token_count'] == document_tokens(d.page_content, self.tokenizer) for d in docs))
        self.assertEqual(len({d.metadata['doc_id'] for d in docs}), len(docs))

    def test_paragraphs_are_not_lost_at_boundaries(self):
        text = "\n\n".join(f"Paragraph {i}: " + "details " * 20 for i in range(20))
        chunks = split_text(text, limit=260, overlap=0, tokenizer=self.tokenizer)
        normalize = lambda value: re.sub(r"\s+", " ", value).strip()
        self.assertEqual(normalize(" ".join(chunks)), normalize(text))

    def test_default_config_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'llm.conf'
            self.assertEqual(load_embedding_config(path)['general_chunk_tokens'], 384)
            path.write_text(json.dumps({'embeddings': {'general_overlap_tokens': 384}}))
            with self.assertRaises(ValueError):
                load_embedding_config(path)

    def test_ingestion_and_queries_select_separate_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'llm.conf'
            # The defaults also work when the config is missing.
            self.assertEqual(load_embedding_config(path)['model'], 'embeddinggemma-2-F16')
            self.assertEqual(load_embedding_config(path, purpose='ingestion')['model'], 'embeddinggemma-2-gpu')
            path.write_text(json.dumps({'embeddings': {
                'model': 'cpu-alias', 'ingestion_model': 'gpu-alias',
                'api_base': 'http://localhost:8080', 'batch_size': 16,
            }}))
            query = load_embedding_config(path)
            ingestion = load_embedding_config(path, purpose='ingestion')
            self.assertEqual(query['model'], 'cpu-alias')
            self.assertEqual(ingestion['model'], 'gpu-alias')
            self.assertEqual({k: v for k, v in query.items() if k != 'model'},
                             {k: v for k, v in ingestion.items() if k != 'model'})
            with self.assertRaises(ValueError):
                load_embedding_config(path, purpose='typo')

    def test_query_endpoint_override_does_not_change_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'llm.conf'
            path.write_text(json.dumps({'embeddings': {
                'api_base': 'http://localhost:8080',
                'query_api_base': 'http://localhost:8081',
            }}))
            query = load_embedding_config(path)
            ingestion = load_embedding_config(path, purpose='ingestion')
            self.assertEqual(query['api_base'], 'http://localhost:8081/v1')
            self.assertEqual(ingestion['api_base'], 'http://localhost:8080/v1')
            self.assertEqual(query['model'], 'embeddinggemma-2-F16')
            self.assertEqual(ingestion['model'], 'embeddinggemma-2-gpu')

    def test_clients_and_tokenizers_use_their_purpose_endpoint(self):
        import llm_client
        for purpose, port in [('query', 8081), ('ingestion', 8080)]:
            with self.subTest(purpose=purpose), \
                 patch('llm_client.OpenAIEmbeddings') as client, \
                 patch('llm_client.get_tokenizer') as tokenizer:
                llm_client.create_embeddings(purpose=purpose)
                expected = f'http://127.0.0.1:{port}/v1'
                self.assertEqual(client.call_args.kwargs['base_url'], expected)
                self.assertEqual(tokenizer.call_args.args[0]['api_base'], expected)

    def test_chunking_uses_ingestion_tokenizer(self):
        with patch('embedding_chunks.get_tokenizer', return_value=self.tokenizer) as tokenizer:
            chunk_documents([Document('A short passage')])
            self.assertEqual(tokenizer.call_args.args[0]['model'], 'embeddinggemma-2-gpu')

    def test_formatting_strips_old_prefix_once(self):
        self.assertEqual(format_document('search_document: passage'), 'title: none | text: passage')
        self.assertEqual(format_query('a question'), 'task: search result | query: a question')

    def test_server_tokenizer_matches_embedding_special_token_options(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"tokens": [1, 2, 3]}'
        tokenizer = LlamaTokenizer('http://127.0.0.1:8080/v1', 'embeddinggemma-2-F16')
        with patch('embedding_chunks.urlopen', return_value=Response()) as request:
            self.assertEqual(tokenizer.count('private input'), 3)
            self.assertEqual(tokenizer.count('private input'), 3)
            request.assert_called_once()
            req = request.call_args.args[0]
            self.assertTrue(req.full_url.endswith('/tokenize'))
            payload = json.loads(req.data)
            self.assertTrue(payload['add_special'])
            self.assertTrue(payload['parse_special'])
            self.assertEqual(payload['model'], 'embeddinggemma-2-F16')

    def test_tokenizer_failure_does_not_silently_estimate(self):
        with patch('embedding_chunks.urlopen', side_effect=OSError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'character estimates are not used'):
                LlamaTokenizer('http://localhost:8080/v1', 'test').count('text')


if __name__ == '__main__':
    unittest.main()
