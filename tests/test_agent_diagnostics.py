import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from agent_diagnostics import LOGGER, QueryTrace, preview, endpoint_name
from agentic_rag_query import run_query, parse_args


class DiagnosticTests(unittest.TestCase):
    def test_streamed_tool_arguments_are_assembled_before_execution(self):
        snapshots = []
        class Model:
            def stream(self, messages, **kwargs):
                self_test.assertTrue(kwargs['stream_usage'])
                snapshots.append(list(messages))
                if len(snapshots) == 1:
                    yield AIMessageChunk(content='', tool_call_chunks=[{
                        'name':'search_tax_code', 'args':'{"query": "tax', 'id':'call-1', 'index':0,
                    }])
                    yield AIMessageChunk(content='', tool_call_chunks=[{
                        'name':None, 'args':' year"}', 'id':None, 'index':0,
                    }], response_metadata={'finish_reason':'tool_calls'})
                else:
                    yield AIMessageChunk(content='Done.', response_metadata={'finish_reason':'stop'},
                                         usage_metadata={'input_tokens':12,'output_tokens':3,'total_tokens':15})
        self_test = self
        tool = Mock(invoke=Mock(return_value='PRIVATE_DOCUMENT_CONTENT'))
        with self.assertLogs(LOGGER, level='INFO') as captured:
            answer = run_query(Model(), {'search_tax_code':tool}, 'PRIVATE_QUESTION', verbose=True)
        self.assertEqual(answer, 'Done.')
        tool.invoke.assert_called_once_with({'query':'tax year'})
        self.assertEqual(len(snapshots), 2)
        results = [m for m in snapshots[1] if isinstance(m, ToolMessage)]
        self.assertEqual(results[0].tool_call_id, 'call-1')
        self.assertEqual(results[0].content, 'PRIVATE_DOCUMENT_CONTENT')
        logs = '\n'.join(captured.output)
        self.assertIn('stream_finished', logs)
        self.assertIn('output_tokens', logs)
        self.assertIn('tool=search_tax_code', logs)
        self.assertNotIn('PRIVATE_DOCUMENT_CONTENT', logs)
        self.assertNotIn('PRIVATE_QUESTION', logs)
        self.assertNotIn('tax year', logs)

    def test_real_chat_client_stream_includes_usage_and_preserves_query_settings(self):
        import json
        import httpx
        from unittest.mock import patch
        from langchain_openai import ChatOpenAI
        import llm_client
        requests = []
        base = {'id':'synthetic','object':'chat.completion.chunk','created':0,'model':'gemma-4-26B-A4B'}
        chunks = [
            {**base, 'choices':[{'index':0,'delta':{'role':'assistant','content':'Done.'},'finish_reason':None}]},
            {**base, 'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},
            {**base, 'choices':[], 'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}},
        ]
        wire = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={'content-type':'text/event-stream'}, content=wire.encode())
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with patch('llm_client.ChatOpenAI', side_effect=lambda **kwargs: ChatOpenAI(**kwargs, http_client=client)):
                model = llm_client.create_chat_model()
            with self.assertLogs(LOGGER, level='INFO') as captured:
                result = QueryTrace().model_call(model, [HumanMessage(content='test')], '1')
        self.assertEqual(result.content, 'Done.')
        self.assertEqual(result.usage_metadata['output_tokens'], 2)
        self.assertTrue(requests[0]['stream'])
        self.assertTrue(requests[0]['stream_options']['include_usage'])
        self.assertEqual(requests[0]['max_completion_tokens'], 8192)
        self.assertEqual(requests[0]['reasoning_effort'], 'medium')
        self.assertIn('finish_reason=stop', '\n'.join(captured.output))

    def test_heartbeat_and_failure_stop_cleanly_without_logging_error_content(self):
        trace = QueryTrace(heartbeat_seconds=0.01)
        heartbeat = threading.Event()
        original = trace.info
        def info(message, *args):
            if 'still waiting' in message: heartbeat.set()
            return original(message, *args)
        trace.info = info
        with self.assertLogs(LOGGER, level='INFO') as captured:
            with self.assertRaises(TimeoutError):
                with trace.phase('llm=1'):
                    self.assertTrue(heartbeat.wait(1))
                    raise TimeoutError('PRIVATE_REQUEST_CONTENT')
        logs = '\n'.join(captured.output)
        self.assertIn('still waiting', logs)
        self.assertIn('error_type=TimeoutError', logs)
        self.assertNotIn('PRIVATE_REQUEST_CONTENT', logs)
        self.assertFalse(any(t.name == 'agent-heartbeat-' + trace.run_id for t in threading.enumerate()))

    def test_blocking_mode_warns_on_limit_and_invalid_calls(self):
        response = AIMessage(content='', invalid_tool_calls=[{
            'name':'search_tax_code','args':'{bad','id':'1','error':'invalid JSON',
        }], response_metadata={'finish_reason':'length'})
        model = SimpleNamespace(invoke=Mock(return_value=response))
        with self.assertLogs(LOGGER, level='INFO') as captured:
            result = QueryTrace(stream=False).model_call(model, [HumanMessage(content='question')], '1')
        self.assertIs(result, response)
        logs = '\n'.join(captured.output)
        self.assertIn('output token limit', logs)
        self.assertIn('could not parse 1 tool calls', logs)
        self.assertIn('no answer text', logs)

    def test_trace_preview_never_contains_image_payload(self):
        value = 'data:image/png;base64,VEhJU0lTQU5JTUFHRQ=='
        self.assertNotIn('VEhJU0lT', preview(value))
        self.assertIn('image data omitted', preview(value))
        self.assertEqual(endpoint_name('http://user:password@localhost:8080/v1?token=secret'),
                         'http://localhost:8080/v1')
        with self.assertLogs(LOGGER, level='INFO') as captured:
            trace = QueryTrace(trace_content=True)
            trace.tool_call({'search':SimpleNamespace(invoke=lambda args:[
                {'type':'text','text':'a bounded excerpt'},
                {'type':'image_url','image_url':{'url':value}},
            ])}, {'name':'search','args':{},'id':'1'}, '1')
        self.assertNotIn('VEhJU0lT', '\n'.join(captured.output))
        self.assertIn('a bounded excerpt', '\n'.join(captured.output))

    def test_cli_supports_content_trace_file_and_blocking_mode(self):
        args = parse_args(['--debug','--trace-content','--no-stream','--log-file','trace.log'])
        self.assertTrue(args.debug)
        self.assertTrue(args.trace_content)
        self.assertTrue(args.no_stream)
        self.assertEqual(args.log_file, 'trace.log')


if __name__ == '__main__': unittest.main()
