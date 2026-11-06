import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import pymupdf

from pdf_vision import PDFVisionExtractor, load_vision_config, bounded_boxes, ExtractionTimeout
from test_pdf_vision import Response


class DenseExtractionTests(unittest.TestCase):
    def extractor(self, directory):
        settings = load_vision_config()
        settings.update(cache_directory=directory, max_region_area=0.25, crop_max_tokens=4096)
        return PDFVisionExtractor(settings)

    def test_whole_page_and_review_are_split_before_model_requests(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            for action in ('extract_page', 'review'):
                with self.subTest(action=action), \
                     patch.object(extractor, 'extract_page', side_effect=AssertionError('Unbounded page request')), \
                     patch.object(extractor, 'extract_crop', return_value='Box 1: 123.45') as crop:
                    result = extractor.extract_selected(page, {'action':action})
                    self.assertEqual(crop.call_count, 4)
                    boxes = [call.args[1] for call in crop.call_args_list]
                    for box in boxes:
                        self.assertLessEqual((box[2]-box[0])*(box[3]-box[1]), 0.27)
                    for x in (0,0.25,0.5,0.75,1):
                        for y in (0,0.25,0.5,0.75,1):
                            self.assertTrue(any(b[0]<=x<=b[2] and b[1]<=y<=b[3] for b in boxes))
                    self.assertIn('123.45', result)

    def test_small_selection_is_not_expanded_outside_original_bounds(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            box = [0.1,0.2,0.4,0.5]
            self.assertEqual(bounded_boxes(page, box, 0.25), [box])

    def test_crop_budget_is_capped_without_changing_global_limit(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            extractor.settings['max_tokens'] = 65536
            with patch('pdf_vision.urlopen', return_value=Response()) as post:
                extractor.extract_crop(page, [0,0,0.5,0.5])
                import json
                self.assertEqual(json.loads(post.call_args.args[0].data)['max_tokens'], 4096)
            self.assertEqual(extractor.settings['max_tokens'], 65536)

    def test_network_read_timeout_splits_crop_and_caches_only_successes(self):
        for timeout in (TimeoutError('read timed out'), URLError(TimeoutError('read timed out'))):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
                page = pdf.new_page()
                extractor = self.extractor(directory)
                with patch('pdf_vision.urlopen', side_effect=[timeout, Response(text='upper'), Response(text='lower')]) as post:
                    result = extractor.extract_crop(page, [0,0,0.5,0.5])
                self.assertEqual(post.call_count, 3)
                self.assertIn('upper', result)
                self.assertIn('lower', result)
                from pathlib import Path
                self.assertEqual(len(list((Path(directory)/'text').glob('*.json'))), 2)

    def test_timeout_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            extractor.settings['max_region_retries'] = 1
            with patch('pdf_vision.urlopen', side_effect=TimeoutError('read timed out')) as post:
                with self.assertRaises(ExtractionTimeout):
                    extractor.extract_crop(page, [0,0,0.5,0.5])
                self.assertEqual(post.call_count, 2)


if __name__ == '__main__': unittest.main()
