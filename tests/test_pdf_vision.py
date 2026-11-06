import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf

from pdf_vision import DEFAULTS, PDFVisionExtractor, load_vision_config, render_regions


class Response:
    def __init__(self, reason='stop', text='| Box | Amount |\n| --- | --- |\n| 1 | 123.45 |'):
        self.payload = {'choices': [{'finish_reason': reason, 'message': {'content': text}}]}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.payload).encode()


class VisionTests(unittest.TestCase):
    def settings(self, directory):
        return {**DEFAULTS, 'cache_directory': Path(directory),
                'api_base': 'http://localhost:8080/v1', 'api_key': 'test', 'timeout': 600}

    def test_config_inherits_service_but_uses_separate_extraction_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'llm': {'model': 'answer-model', 'api_base': 'http://localhost:9999',
                                             'api_key': 'local-key', 'timeout': 60}}))
            settings = load_vision_config(path)
            self.assertEqual(settings['model'], 'gemma-4-E4B')
            self.assertEqual(settings['api_base'], 'http://localhost:9999/v1')
            self.assertEqual(settings['api_key'], 'local-key')
            for grid in (0, 5, True):
                path.write_text(json.dumps({'pdf_extraction': {'grid_size': grid}}))
                with self.assertRaises(ValueError): load_vision_config(path)

    def test_real_rendering_preserves_overview_and_four_quadrants(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=144, height=144)
            for xy, text in [((10, 20), 'A'), ((80, 20), 'B'), ((10, 90), 'C'), ((80, 90), 'D')]:
                page.insert_text(xy, text)
            images = render_regions(page, {**DEFAULTS, 'dpi': 300})
            sizes = [(pymupdf.Pixmap(image).width, pymupdf.Pixmap(image).height) for image in images]
            self.assertEqual(sizes, [(600, 600)] + [(300, 300)] * 4)
            self.assertEqual(len(render_regions(page, {**DEFAULTS, 'grid_size': 1})), 1)

    def test_payload_cache_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(directory)
            with patch('pdf_vision.render_regions', return_value=[b'overview', b'crop']), \
                 patch('pdf_vision.urlopen', return_value=Response()) as post:
                extractor = PDFVisionExtractor(settings)
                text = extractor.extract_page(None)
                self.assertIn('123.45', text)
                self.assertEqual(extractor.extract_page(None), text)
                self.assertEqual(post.call_count, 1)
                request = post.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(payload['model'], 'gemma-4-E4B')
                self.assertEqual(payload['reasoning_effort'], 'none')
                self.assertIs(payload['chat_template_kwargs']['enable_thinking'], False)
                self.assertEqual([p['type'] for p in payload['messages'][0]['content']],
                                 ['image_url', 'image_url', 'text'])
                settings['model'] = 'other-model'
                extractor.extract_page(None)
                self.assertEqual(post.call_count, 2)
            self.assertEqual(len(list(Path(directory).rglob('*.json'))), 2)

    def test_failures_never_cache_incomplete_results(self):
        for response in (Response('length'), Response(text=''), Response('content_filter')):
            with self.subTest(response=response.payload), tempfile.TemporaryDirectory() as directory:
                with patch('pdf_vision.render_regions', return_value=[b'png']), \
                     patch('pdf_vision.urlopen', return_value=response):
                    with self.assertRaises(RuntimeError):
                        PDFVisionExtractor(self.settings(directory)).extract_page(None)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_network_failure_does_not_fall_back_to_native_or_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('pdf_vision.render_regions', return_value=[b'png']), \
                 patch('pdf_vision.urlopen', side_effect=OSError('unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'vision projector'):
                    PDFVisionExtractor(self.settings(directory)).extract_page(None)

    def test_ingestion_transcribes_native_and_blank_pages_with_provenance(self):
        import ingest_user
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'forms.pdf'
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((72, 72), 'Existing native text')
                pdf.new_page()
                pdf.save(filename)
            profile = ingest_user.analyze_pdf(filename)
            extractor = Mock(settings={'model': 'gemma-4-E4B'})
            extractor.extract_page.side_effect = ['# Form\nBox 1: 123.45', '[blank page]']
            with patch.object(ingest_user, 'PDF_DIR', directory):
                docs = ingest_user.process_pdf(profile, extractor)
            self.assertEqual(extractor.extract_page.call_count, 2)
            self.assertEqual([d.metadata['page'] for d in docs], [1, 2])
            self.assertTrue(all(d.metadata['extraction_method'] == 'gemma_vision' for d in docs))
            self.assertEqual(docs[0].page_content, '# Form\nBox 1: 123.45')


if __name__ == '__main__':
    unittest.main()
