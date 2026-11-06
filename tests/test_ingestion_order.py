"""Verify phase boundaries without invoking model servers or altering indexes."""
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ingest_user
from langchain_core.documents import Document


class IngestionOrderTests(unittest.TestCase):
    def run_ingestion(self, *, fail_images=False, fail_triage=False, fail_extraction=False, fresh=False, clear_on_start=True):
        events = []
        paths = [Path('first.pdf'), Path('second.pdf')]
        profiles = [SimpleNamespace(path=p, strategy=SimpleNamespace(value='form'),
                    confidence='high', reason='test', page_count=2,
                    native_page_ratio=0, has_form_fields=False) for p in paths]

        def images(*args):
            for pdf in paths:
                for page in (1, 2):
                    events.append(('image', pdf.name, page))
            if fail_images: raise RuntimeError('image failure')
            return [Mock() for _ in range(4)]

        def triage(profile, extractor):
            decisions = []
            for page in (1, 2):
                events.append(('triage', profile.path.name, page))
                if fail_triage: raise RuntimeError('triage failure')
                decisions.append({'action': 'extract_page'})
            return decisions

        def extract(profile, extractor, decisions):
            docs = []
            for page in (1, 2):
                events.append(('extract', profile.path.name, page))
                if fail_extraction: raise RuntimeError('extraction failure')
                docs.append(Document(page_content=f'{profile.path.name} page {page}',
                            metadata={'source_file': profile.path.name, 'page': page}))
            return docs

        def chunks(docs):
            events.append(('tokenize',))
            return docs

        def embed_batches(embeddings, batches, **kwargs):
            for docs in batches:
                for doc in docs:
                    events.append(('text', doc.metadata['source_file'], doc.metadata['page']))
                yield docs, [[1.0] for _ in docs]

        store = Mock()
        store.get.return_value = {'metadatas': []}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            # Failures must preserve this existing store and never create a Chroma client.
            sentinel = Path(directory) / 'existing-index'
            sentinel.write_text('preserve')
            replacements = dict(discover_pdfs=Mock(return_value=paths),
                                analyze_pdf=Mock(side_effect=profiles),
                                prepare_pdf_images=images, triage_pdf=triage, process_pdf=extract,
                                PDFVisionExtractor=Mock(), attach_pdf_images=Mock(),
                                chunk_documents=chunks, create_embeddings=Mock(),
                                embed_document_batches=embed_batches,
                                Chroma=Mock(return_value=store), store_pdf_images=Mock(),
                                STORE_DIR=directory, CLEAR_ON_START=clear_on_start, BATCH_SIZE=1)
            for name, value in replacements.items():
                stack.enter_context(patch.object(ingest_user, name, value))
            clear = stack.enter_context(patch.object(ingest_user.shutil, 'rmtree'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if fail_images or fail_triage or fail_extraction:
                with self.assertRaises(RuntimeError): ingest_user.ingest(fresh=fresh)
                clear.assert_not_called()
                replacements['Chroma'].assert_not_called()
                replacements['create_embeddings'].assert_not_called()
                self.assertEqual(sentinel.read_text(), 'preserve')
            else:
                ingest_user.ingest(fresh=fresh)
                if clear_on_start or fresh:
                    clear.assert_called_once_with(directory)
                else:
                    clear.assert_not_called()
                replacements['PDFVisionExtractor'].assert_called_once_with(use_cache=not fresh)
                replacements['store_pdf_images'].assert_called_once()
        return events

    def test_all_images_then_all_extraction_then_all_text_embeddings(self):
        events = self.run_ingestion()
        self.assertEqual([event[0] for event in events],
                         ['image'] * 4 + ['triage'] * 4 + ['extract'] * 4 + ['tokenize'] + ['text'] * 4)

    def test_fresh_bypasses_vision_cache_and_forces_rebuild(self):
        events = self.run_ingestion(fresh=True, clear_on_start=False)
        self.assertEqual([event[0] for event in events].count('text'), 4)

    def test_image_failure_prevents_extraction_and_store_changes(self):
        events = self.run_ingestion(fail_images=True)
        self.assertTrue(all(event[0] == 'image' for event in events))

    def test_triage_failure_prevents_extraction_and_store_changes(self):
        events = self.run_ingestion(fail_triage=True)
        self.assertEqual([event[0] for event in events], ['image'] * 4 + ['triage'])

    def test_extraction_failure_prevents_text_embeddings_and_store_changes(self):
        events = self.run_ingestion(fail_extraction=True)
        self.assertEqual([event[0] for event in events], ['image'] * 4 + ['triage'] * 4 + ['extract'])


if __name__ == '__main__':
    unittest.main()
