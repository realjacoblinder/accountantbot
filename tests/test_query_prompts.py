"""Exercise the actual agent loops at the retrieval limit without model calls."""
import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from query_prompts import (
    AGENT_SYSTEM_PROMPT, TAX_ONLY_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT, RETRIEVED_IMAGE_PROMPT,
)


class Message:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


class QueryLoopTests(unittest.TestCase):
    def test_both_agents_deliver_partial_answer_policy_at_retrieval_limit(self):
        for filename, system in [('agentic_rag_query.py', AGENT_SYSTEM_PROMPT),
                                 ('agentic_rag_query_tax_only.py', TAX_ONLY_SYSTEM_PROMPT)]:
            with self.subTest(filename=filename):
                tree = ast.parse(Path(filename).read_text())
                function = next(node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name == 'run_query')
                scope = dict(SystemMessage=Message, HumanMessage=Message, ToolMessage=Message,
                             SYSTEM_PROMPT=system, MAX_ITERATIONS=2,
                             RETRIEVED_IMAGE_PROMPT=RETRIEVED_IMAGE_PROMPT,
                             FINAL_ANSWER_PROMPT=FINAL_ANSWER_PROMPT)
                exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), scope)
                snapshots = []
                def invoke(messages):
                    snapshots.append(list(messages))
                    if len(snapshots) <= 2:
                        return SimpleNamespace(content='', tool_calls=[{
                            'name': 'search_tax_code', 'args': {'query': 'requested-year rule'},
                            'id': str(len(snapshots)),
                        }])
                    return SimpleNamespace(content='Partial answer: annual authority unavailable.', tool_calls=[])
                tool = Mock(invoke=Mock(return_value='[SOURCE: 26 USC 63]\nStatutory excerpt'))
                model = SimpleNamespace(invoke=invoke)
                with redirect_stdout(io.StringIO()):
                    result = scope['run_query'](model, {'search_tax_code': tool}, 'Question', verbose=False)
                self.assertEqual(len(snapshots), 3)
                self.assertEqual(snapshots[-1][0].content, system)
                self.assertEqual(snapshots[-1][-1].content, FINAL_ANSWER_PROMPT)
                self.assertEqual(tool.invoke.call_count, 2)
                self.assertEqual(result, 'Partial answer: annual authority unavailable.')
                self.assertEqual([m.content for m in snapshots[-1]
                                  if hasattr(m, 'tool_call_id')],
                                 ['[SOURCE: 26 USC 63]\nStatutory excerpt'] * 2)


if __name__ == '__main__':
    unittest.main()
