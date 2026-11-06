import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf
from langchain_core.documents import Document
import ingest_user
import ingest_background
from pdf_vision import PDFVisionExtractor, load_vision_config
from test_pdf_vision import Response


class FreshIngestionTests(unittest.TestCase):
    def test_both_scripts_accept_fresh_and_no_cache(self):
        for module in (ingest_user, ingest_background):
            self.assertFalse(module.parse_args([]).fresh)
            self.assertTrue(module.parse_args(['--fresh']).fresh)
            self.assertTrue(module.parse_args(['--no-cache']).fresh)

    def test_fresh_extraction_bypasses_and_refreshes_cache(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            settings = load_vision_config()
            settings['cache_directory'] = directory
            normal = PDFVisionExtractor(settings)
            fresh = PDFVisionExtractor(settings, use_cache=False)
            with patch('pdf_vision.urlopen', side_effect=[Response(text='old amount'), Response(text='new amount')]) as post:
                self.assertEqual(normal.extract_page(page), 'old amount')
                self.assertEqual(fresh.extract_page(page), 'new amount')
                self.assertEqual(normal.extract_page(page), 'new amount')
                self.assertEqual(post.call_count, 2)
                self.assertFalse(json.loads(post.call_args.args[0].data)['cache_prompt'])
            with patch('pdf_vision.urlopen', side_effect=OSError('offline')):
                with self.assertRaises(RuntimeError): fresh.extract_page(page)
            # Failure did not overwrite the last successful extraction.
            with patch('pdf_vision.urlopen', side_effect=AssertionError('should reuse cache')):
                self.assertEqual(normal.extract_page(page), 'new amount')

    def test_fresh_triage_bypasses_and_refreshes_cache(self):
        def decision(action, page_type):
            return json.dumps(dict(action=action, page_type=page_type, uncertain=False,
                                   regions=[], reason='test'))
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            settings = load_vision_config()
            settings['cache_directory'] = directory
            normal = PDFVisionExtractor(settings)
            fresh = PDFVisionExtractor(settings, use_cache=False)
            responses = [Response(text=decision('skip','instructions')),
                         Response(text=decision('extract_page','records'))]
            with patch('pdf_vision.urlopen', side_effect=responses) as post:
                self.assertEqual(normal.triage_page(page)['action'], 'skip')
                self.assertEqual(fresh.triage_page(page)['action'], 'extract_page')
                self.assertEqual(normal.triage_page(page)['action'], 'extract_page')
                self.assertEqual(post.call_count, 2)

    def test_background_fresh_reembeds_existing_ids_and_removes_stale_records(self):
        for fresh in (False, True):
            with self.subTest(fresh=fresh), tempfile.TemporaryDirectory() as directory:
                filename = Path(directory)/'source.jsonl'
                filename.write_text('{}\n')
                records = {'same':'old', 'stale':'removed source'}
                store = Mock()
                store.get.side_effect = lambda **kwargs: {'metadatas':[{'doc_id':key} for key in records]}
                store.reset_collection.side_effect = records.clear
                def upsert(*, ids, documents, **kwargs): records.update(zip(ids, documents))
                store._collection.upsert.side_effect = upsert
                docs = [Document(page_content='updated', metadata={'doc_id':'same'}),
                        Document(page_content='added', metadata={'doc_id':'new'})]
                def batches(embeddings, batches, **kwargs):
                    for batch in batches: yield batch, [[1.] for _ in batch]
                with patch.object(ingest_background, 'JSONL_FILE', str(filename)), \
                     patch.object(ingest_background, 'records_to_documents', return_value=docs), \
                     patch.object(ingest_background, 'create_embeddings'), \
                     patch.object(ingest_background, 'Chroma', return_value=store), \
                     patch.object(ingest_background, 'embed_document_batches', side_effect=batches), \
                     redirect_stdout(io.StringIO()):
                    ingest_background.ingest(fresh=fresh)
                if fresh:
                    store.reset_collection.assert_called_once()
                    self.assertEqual(records, {'same':'updated','new':'added'})
                else:
                    store.reset_collection.assert_not_called()
                    self.assertEqual(records, {'same':'old','stale':'removed source','new':'added'})

    def test_empty_background_input_preserves_store_even_when_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/'empty.jsonl'
            filename.write_text('')
            with patch.object(ingest_background, 'JSONL_FILE', str(filename)), \
                 patch.object(ingest_background, 'Chroma') as store, redirect_stdout(io.StringIO()):
                ingest_background.ingest(fresh=True)
            store.assert_not_called()


if __name__ == '__main__': unittest.main()
