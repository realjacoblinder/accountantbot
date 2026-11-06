import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from pdf_triage import selected_boxes, validate_decision, TRIAGE_PROMPT, TRIAGE_SCHEMA
from pdf_vision import atomic_json
from pdf_vision import PDFVisionExtractor, load_vision_config, TruncatedExtraction
from test_pdf_vision import Response


def decision(action='extract_page', page_type='records', regions=None, uncertain=False):
    return dict(action=action, page_type=page_type, regions=regions or [],
                uncertain=uncertain, reason='Synthetic test content')


class TriageTests(unittest.TestCase):
    def extractor(self, directory):
        settings = load_vision_config()
        settings['cache_directory'] = Path(directory)
        return PDFVisionExtractor(settings)

    def test_bad_boxes_and_unsafe_skips_are_not_silently_accepted(self):
        for box in [[0,0,2,1], [0.5,0,0.1,1], [0,0,float('nan'),1], [False,0,1,1]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_decision(decision('extract_regions', regions=[{'bbox':box,'description':'field'}]))
        self.assertEqual(validate_decision(decision('skip'))['action'], 'review')
        self.assertEqual(validate_decision(decision('skip', 'instructions', uncertain=True))['action'], 'review')
        self.assertEqual(validate_decision(decision('skip', 'instructions'))['action'], 'skip')

    def test_whole_page_regions_are_ignored_without_mutating_input(self):
        for regions in ([{'bbox':[0,0,1,1], 'description':'all fields'}],
                        [{'bbox':[0,0,2,1]}], 'invalid extra regions'):
            with self.subTest(regions=regions):
                raw = decision('extract_page')
                raw['regions'] = regions
                normalized = validate_decision(raw)
                self.assertEqual(normalized['action'], 'extract_page')
                self.assertEqual(normalized['regions'], [])
                self.assertEqual(raw['regions'], regions)

    def test_skip_with_regions_and_explicit_review_both_retain_page(self):
        region = [{'bbox':[0,0,0.5,0.5], 'description':'possible records'}]
        self.assertEqual(validate_decision(decision('skip', 'instructions', region))['action'], 'review')
        self.assertEqual(validate_decision(decision('review', 'mixed', region))['action'], 'review')

    def test_old_fallback_cache_is_bypassed_and_normalized_result_reused(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            image = page.get_pixmap(dpi=extractor.settings['triage_dpi'], colorspace=pymupdf.csRGB,
                                   alpha=False, annots=True).tobytes('png')
            # An unversioned cache entry must not override a fresh decision.
            legacy = extractor.cache_path([image], TRIAGE_PROMPT, 'triage')
            atomic_json(legacy, {'decision': decision('review', 'uncertain', uncertain=True)})
            raw = decision('extract_page', regions=[{'bbox':[0,0,1,1], 'description':'form'}])
            with patch('pdf_vision.urlopen', return_value=Response(text=json.dumps(raw))) as post:
                result = extractor.triage_page(page)
                self.assertEqual(result['action'], 'extract_page')
                self.assertEqual(result['regions'], [])
                self.assertEqual(extractor.triage_page(page), result)
                post.assert_called_once()

    def test_missing_or_invalid_audit_reason_does_not_change_valid_actions(self):
        for action, page_type in [('extract_page', 'records'), ('extract_regions', 'mixed'),
                                  ('skip', 'instructions'), ('review', 'uncertain')]:
            for reason in (None, '', '   ', [], 42):
                with self.subTest(action=action, reason=reason):
                    raw = decision(action, page_type)
                    if action == 'extract_regions':
                        raw['regions'] = [{'bbox':[0,0,0.5,0.5]}]
                    raw['reason'] = reason
                    result = validate_decision(raw)
                    self.assertEqual(result['action'], action)
                    self.assertIn('no explanation supplied', result['reason'])
                    if action == 'extract_regions':
                        self.assertIn('no description supplied', result['regions'][0]['description'])
                    self.assertEqual(raw['reason'], reason)
        raw = decision()
        del raw['reason']
        self.assertEqual(validate_decision(raw)['action'], 'extract_page')

    def test_missing_reason_does_not_bypass_skip_or_coordinate_safeguards(self):
        raw = decision('skip', 'records')
        del raw['reason']
        self.assertEqual(validate_decision(raw)['action'], 'review')
        raw = decision('skip', 'instructions', [{'bbox':[0,0,1,1]}])
        del raw['reason']
        self.assertEqual(validate_decision(raw)['action'], 'review')
        raw = decision('extract_regions', 'mixed', [{'bbox':[0,0,2,1]}])
        del raw['reason']
        with self.assertRaisesRegex(ValueError, 'bounding box'): validate_decision(raw)

    def test_raw_missing_reason_response_is_saved_and_normalized_cache_reused(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            raw = decision('extract_regions', 'mixed', [{'bbox':[0,0,0.5,0.5]}])
            del raw['reason']
            with patch('pdf_vision.urlopen', return_value=Response(text=json.dumps(raw))) as post:
                result = extractor.triage_page(page)
                self.assertEqual(result['action'], 'extract_regions')
                self.assertEqual(extractor.triage_page(page), result)
                post.assert_called_once()
            saved = json.loads(next((Path(directory)/'triage').glob('*.json')).read_text())
            self.assertEqual(json.loads(saved['raw_response']), raw)
            self.assertIsNone(saved['validation_error'])

    def test_overlapping_regions_merge_and_padding_stays_on_page(self):
        regions = [{'bbox':[0,0,0.4,0.4], 'description':'one'},
                   {'bbox':[0.3,0.3,0.6,0.6], 'description':'two'},
                   {'bbox':[0.8,0.8,1,1], 'description':'three'}]
        boxes = selected_boxes(decision('extract_regions', regions=regions), 0.02)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], [0,0,0.62,0.62])
        self.assertEqual(boxes[1], [0.78,0.78,1,1])

    def test_triage_uses_one_overview_json_small_budget_and_cache(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            with patch('pdf_vision.urlopen', return_value=Response(text=json.dumps(decision()))) as post:
                self.assertEqual(extractor.triage_page(page)['action'], 'extract_page')
                extractor.triage_page(page)
                self.assertEqual(post.call_count, 1)
                payload = json.loads(post.call_args.args[0].data)
                self.assertEqual(payload['max_tokens'], extractor.settings['triage_max_tokens'])
                self.assertEqual(payload['reasoning_effort'], 'none')
                self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
                self.assertEqual(payload['response_format']['type'], 'json_schema')
                self.assertEqual(payload['response_format']['json_schema']['schema'], TRIAGE_SCHEMA)
                self.assertEqual(len(payload['messages'][0]['content']), 2)
            self.assertEqual(len(list((Path(directory)/'triage').glob('*.json'))), 1)

    def test_invalid_or_truncated_triage_retains_page(self):
        for response in [Response(text='not json'), Response('length'), Response(text='{}')]:
            with self.subTest(response=response.payload), tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
                page = pdf.new_page()
                with patch('pdf_vision.urlopen', return_value=response):
                    result = self.extractor(directory).triage_page(page)
                self.assertEqual(result['action'], 'review')

    def test_filled_fields_override_skip(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            widget = pymupdf.Widget()
            widget.field_name, widget.field_value = 'Box 1', '123.45'
            widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            widget.rect = pymupdf.Rect(20,20,200,50)
            page.add_widget(widget)
            with patch('pdf_vision.urlopen', return_value=Response(text=json.dumps(decision('skip','instructions')))):
                result = self.extractor(directory).triage_page(page)
            self.assertEqual(result['action'], 'review')

    def test_skip_makes_no_extraction_call_and_selected_crop_is_300dpi(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page(width=144, height=144)
            extractor = self.extractor(directory)
            with patch.object(extractor, 'request', return_value='Box 1: 123.45') as request:
                self.assertEqual(extractor.extract_selected(page, decision('skip','instructions')), '')
                request.assert_not_called()
                d = decision('extract_regions', 'mixed', regions=[{'bbox':[0,0,0.5,0.5],'description':'values'}])
                result = extractor.extract_selected(page, d)
                self.assertIn('123.45', result)
                images = request.call_args.args[0]
                self.assertEqual(len(images), 1)
                self.assertEqual(pymupdf.Pixmap(images[0]).width, 312)  # half-page + 2% margin

    def test_truncated_crop_splits_and_never_caches_incomplete_text(self):
        with tempfile.TemporaryDirectory() as directory, pymupdf.open() as pdf:
            page = pdf.new_page()
            extractor = self.extractor(directory)
            with patch.object(extractor, 'request', side_effect=[TruncatedExtraction('limit'), 'upper', 'lower']) as request:
                result = extractor.extract_crop(page, [0,0,1,1])
                self.assertEqual(request.call_count, 3)
                self.assertIn('upper', result)
                self.assertIn('lower', result)
            self.assertEqual(len(list((Path(directory)/'text').glob('*.json'))), 2)
            extractor.settings['max_region_retries'] = 0
            with patch.object(extractor, 'request', side_effect=TruncatedExtraction('limit')):
                with self.assertRaises(TruncatedExtraction): extractor.extract_crop(page, [0,0,1,1])

    def test_report_and_filtered_ingestion_keep_page_provenance(self):
        import ingest_user
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/'forms.pdf'
            with pymupdf.open() as pdf:
                pdf.new_page(); pdf.new_page(); pdf.save(filename)
            extractor = self.extractor(Path(directory)/'cache')
            profile = ingest_user.analyze_pdf(filename)
            decisions = [decision('skip','instructions'), decision()]
            with patch.object(ingest_user, 'PDF_DIR', directory), \
                 patch.object(extractor,'triage_page',side_effect=decisions), \
                 patch.object(extractor,'extract_selected',return_value='Box 1: 123.45') as extract:
                selected = ingest_user.triage_pdf(profile, extractor)
                docs = ingest_user.process_pdf(profile, extractor, selected)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].metadata['page'], 2)
            report = json.loads(next((Path(directory)/'cache'/'reports').glob('*.json')).read_text())
            self.assertEqual(report['pages'][0]['action'], 'skip')
            self.assertEqual(report['pages'][1]['page'], 2)
            self.assertEqual(report['source_file'], 'forms.pdf')


if __name__ == '__main__': unittest.main()
