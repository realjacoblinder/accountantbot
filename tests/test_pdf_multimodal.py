import ast
import base64
import json
import math
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from query_prompts import RETRIEVED_IMAGE_PROMPT, FINAL_ANSWER_PROMPT
from embedding_chunks import DEFAULTS
from pdf_multimodal import (
    PDFImageEmbedder, attach_pdf_images, discover_pdfs, page_text,
    prepare_pdf_images, store_pdf_images,
)
from retrieval_content import keyword_documents, retrieval_content

try:
    import fitz
except ImportError:
    fitz = None


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.payload).encode()


class ImageClientTests(unittest.TestCase):
    def test_image_payload_uses_gpu_alias_without_text_prefix_or_pdf_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'page.png'
            path.write_bytes(b'png-bytes')
            settings = {**DEFAULTS, 'model': 'embeddinggemma-2-gpu'}
            client = PDFImageEmbedder(settings)
            replies = [Response({'input_tokens': 280}), Response({'data': [{'index': 0, 'embedding': [1.] * 768}]})]
            with patch('pdf_multimodal.urlopen', side_effect=replies) as post:
                vector, tokens = client.embed_page(path)
            self.assertEqual(tokens, 280)
            self.assertAlmostEqual(sum(v * v for v in vector), 1)
            requests = [call.args[0] for call in post.call_args_list]
            count, embed = [json.loads(req.data) for req in requests]
            self.assertTrue(requests[0].full_url.endswith('/chat/completions/input_tokens'))
            self.assertTrue(requests[1].full_url.endswith('/embeddings'))
            self.assertEqual(embed['model'], 'embeddinggemma-2-gpu')
            content = embed['input'][0]['content']
            self.assertEqual(content, count['messages'][0]['content'])
            self.assertEqual(len(content), 1)
            self.assertEqual(content[0]['type'], 'image_url')
            uri = content[0]['image_url']['url']
            self.assertEqual(base64.b64decode(uri.split(',', 1)[1]), b'png-bytes')

    def test_image_budget_is_checked_before_embedding(self):
        client = PDFImageEmbedder(DEFAULTS)
        with patch('pdf_multimodal.image_data_url', return_value='data:image/png;base64,AA=='):
            with patch.object(client, '_post', return_value={'input_tokens': 2049}) as post:
                with self.assertRaisesRegex(ValueError, 'PDF page needs 2049 tokens'):
                    client.embed_page('unused.png')
                self.assertEqual(post.call_count, 1)

    def test_bad_vectors_are_rejected(self):
        for vector in ([0.] * 768, [math.nan] * 768, [1.] * 512):
            client = PDFImageEmbedder(DEFAULTS)
            replies = [{'input_tokens': 280}, {'data': [{'index': 0, 'embedding': vector}]}]
            with patch('pdf_multimodal.image_data_url', return_value='data:image/png;base64,AA=='):
                with patch.object(client, '_post', side_effect=replies):
                    with self.assertRaises(ValueError): client.embed_page('unused.png')

    def test_missing_projector_or_server_is_not_silently_text_only(self):
        with patch('pdf_multimodal.image_data_url', return_value='data:image/png;base64,AA=='):
            with patch('pdf_multimodal.urlopen', side_effect=OSError('unavailable')):
                with self.assertRaisesRegex(RuntimeError, '--mmproj'):
                    PDFImageEmbedder(DEFAULTS).embed_page('unused.png')


@unittest.skipUnless(fitz, 'PyMuPDF is required for actual PDF rendering tests')
class PDFRenderingTests(unittest.TestCase):
    def test_native_scanned_and_fillable_pages_are_all_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ('first', 'second'): (root / folder).mkdir()
            filename = root / 'first' / 'statement.PDF'
            with fitz.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 72), 'Tax year 2025. W-2 wage statement.')
                widget = fitz.Widget()
                widget.field_name, widget.field_value = 'Box 1 wages', '85000.00'
                widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
                widget.rect = fitz.Rect(72, 100, 220, 125)
                page.add_widget(widget)
                # The second page has pixels only, as a scanned PDF would.
                scan = page.get_pixmap().tobytes('png')
                page = pdf.new_page()
                page.insert_image(page.rect, stream=scan)
                pdf.save(filename)
            (root / 'second' / 'statement.pdf').write_bytes(filename.read_bytes())
            files = discover_pdfs(root)
            self.assertEqual(len(files), 2)
            embedder = SimpleNamespace(settings={'model': 'embeddinggemma-2-gpu'},
                                      embed_page=Mock(return_value=([1.] + [0.] * 767, 280)))
            settings = {'render_dpi': 150, 'max_image_edge': 1200, 'image_directory': root / 'images'}
            pages = prepare_pdf_images(files, root, embedder=embedder, pdf_settings=settings)
            self.assertEqual(len(pages), 4)
            self.assertEqual(embedder.embed_page.call_count, 4)
            self.assertEqual(len({p.doc_id for p in pages}), 4)
            self.assertEqual(len({p.metadata['page_image_path'] for p in pages}), 4)
            self.assertIn('Box 1 wages: 85000.00', pages[0].text)
            self.assertEqual(pages[1].text, 'PDF page 2 of first/statement.PDF')
            for page in pages:
                pix = fitz.Pixmap(page.metadata['page_image_path'])
                self.assertLessEqual(max(pix.width, pix.height), 1201)
            doc = Document('text', {'source_pdf_path': str(filename.resolve()), 'page': 1})
            attach_pdf_images([doc], pages)
            self.assertEqual(doc.metadata['page_image_path'], pages[0].metadata['page_image_path'])
            store = SimpleNamespace(_collection=Mock())
            store_pdf_images(store, pages)
            self.assertEqual(store._collection.upsert.call_count, 4)
            call = store._collection.upsert.call_args_list[0].kwargs
            self.assertEqual(call['embeddings'], [pages[0].embedding])
            self.assertEqual(call['metadatas'][0]['embedding_type'], 'page_image')
            # IDs are stable across reruns even for equal basenames in subfolders.
            again = prepare_pdf_images(files, root, embedder=embedder, pdf_settings=settings)
            self.assertEqual([p.doc_id for p in again], [p.doc_id for p in pages])

    def test_failed_visual_preparation_never_writes_a_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = root / 'input.pdf'
            with fitz.open() as pdf:
                pdf.new_page(); pdf.save(filename)
            embedder = SimpleNamespace(settings={'model': 'gpu'}, embed_page=Mock(side_effect=RuntimeError('no projector')))
            with self.assertRaisesRegex(RuntimeError, 'no projector'):
                prepare_pdf_images([filename], root, embedder=embedder, pdf_settings={
                    'render_dpi': 150, 'max_image_edge': 1200, 'image_directory': root / 'images'})
            self.assertTrue(filename.exists())


class RetrievalTests(unittest.TestCase):
    def test_images_are_deduplicated_bounded_and_cited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / f'{i}.png' for i in range(3)]
            for image in images: image.write_bytes(b'image')
            docs = [Document(f'Page {i}', {'source_file': 'a.pdf', 'page': i + 1, 'page_image_path': str(image)})
                    for i, image in enumerate(images)]
            result = retrieval_content([docs[0], *docs], settings={'image_directory': root, 'max_retrieved_pages': 2})
            self.assertEqual(len([c for c in result if c['type'] == 'image_url']), 2)
            self.assertIn('[SOURCE: a.pdf, page 1]', result[0]['text'])
            bad = Document('text', {'page_image_path': str(root.parent / 'other.png')})
            with self.assertRaises(ValueError):
                retrieval_content([bad], settings={'image_directory': root, 'max_retrieved_pages': 2})

    def test_keyword_search_uses_text_but_image_only_stores_work(self):
        text, image = Document('words'), Document('page', {'embedding_type': 'page_image'})
        self.assertEqual(keyword_documents([text, image]), [text])
        self.assertEqual(keyword_documents([image]), [image])

    def test_agent_supplies_images_after_all_tool_results(self):
        # Exercise the real agent loop without loading its optional vector/LLM dependencies.
        tree = ast.parse(Path('agentic_rag_query.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_query')
        class Message:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Human(Message): pass
        class Tool(Message): pass
        calls = [{'name': 'user', 'args': {}, 'id': '1'}, {'name': 'tax', 'args': {}, 'id': '2'}]
        responses = [SimpleNamespace(tool_calls=calls, content=''), SimpleNamespace(tool_calls=[], content='answer')]
        snapshots = []
        def invoke(messages):
            snapshots.append(list(messages))
            return responses.pop(0)
        model = SimpleNamespace(invoke=Mock(side_effect=invoke))
        tools = {'user': SimpleNamespace(invoke=lambda args: [
            {'type': 'text', 'text': '[SOURCE: a.pdf, page 1]\ncontent'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]),
                 'tax': SimpleNamespace(invoke=lambda args: 'statute')}
        scope = {'HumanMessage': Human, 'ToolMessage': Tool, 'SystemMessage': Message,
                 'SYSTEM_PROMPT': 'instructions', 'MAX_ITERATIONS': 5,
                 'RETRIEVED_IMAGE_PROMPT': RETRIEVED_IMAGE_PROMPT,
                 'FINAL_ANSWER_PROMPT': FINAL_ANSWER_PROMPT}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<agent-loop>', 'exec'), scope)
        self.assertEqual(scope['run_query'](model, tools, 'question', verbose=False), 'answer')
        messages = snapshots[-1]
        self.assertIsInstance(messages[-1], Human)
        self.assertEqual([m.tool_call_id for m in messages if isinstance(m, Tool)], ['1', '2'])
        self.assertTrue(any(c['type'] == 'image_url' for c in messages[-1].content))
        self.assertTrue(any('a.pdf, page 1' in c.get('text', '') for c in messages[-1].content))


if __name__ == '__main__':
    unittest.main()
