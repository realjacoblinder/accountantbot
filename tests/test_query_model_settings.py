import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
import llm_client


class QueryModelSettingsTests(unittest.TestCase):
    def test_token_cap_and_medium_effort_reach_tool_bound_http_request(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                'id':'synthetic', 'object':'chat.completion', 'created':0,
                'model':'gemma-4-26B-A4B', 'choices':[{
                    'index':0, 'finish_reason':'stop',
                    'message':{'role':'assistant','content':'Supported answer'},
                }],
            })
        with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
            def construct(**kwargs): return ChatOpenAI(**kwargs, http_client=http_client)
            with patch('llm_client.ChatOpenAI', side_effect=construct):
                model = llm_client.create_chat_model()
            tools = [{'type':'function','function':{
                'name':'search_tax_code','description':'Search excerpts',
                'parameters':{'type':'object','properties':{'query':{'type':'string'}}},
            }}]
            result = model.bind_tools(tools).invoke('Test question')
        self.assertEqual(result.content, 'Supported answer')
        self.assertEqual(len(requests), 1)
        # LangChain serializes max_tokens as OpenAI's max_completion_tokens;
        # llama-server accepts this alias for its generation limit.
        self.assertEqual(requests[0]['max_completion_tokens'], 8192)
        self.assertEqual(requests[0]['reasoning_effort'], 'medium')
        self.assertEqual(requests[0]['tools'][0]['function']['name'], 'search_tax_code')

    def test_missing_config_retains_defaults_and_invalid_cap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'llm.conf'
            config = llm_client.load_llm_config(path)
            self.assertEqual(config['max_tokens'], 8192)
            self.assertEqual(config['reasoning_effort'], 'medium')
            for value in (0, -1, True, '8192'):
                path.write_text(json.dumps({'llm':{'max_tokens':value}}))
                with self.assertRaises(ValueError): llm_client.load_llm_config(path)


if __name__ == '__main__': unittest.main()
