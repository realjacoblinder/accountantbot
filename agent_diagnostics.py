"""Agent-only tracing with timed stages, streaming progress, and bounded previews."""
import json
import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

from langchain_core.messages import message_chunk_to_message
from langchain_core.retrievers import BaseRetriever

LOGGER = logging.getLogger('accountantbot.agent')
LOGGER.propagate = False


def configure_logging(*, debug=False, log_file=None):
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOGGER.addHandler(console)
    if log_file:
        file = logging.FileHandler(log_file, encoding='utf-8')
        file.setFormatter(formatter)
        LOGGER.addHandler(file)
    LOGGER.setLevel(logging.DEBUG if debug else logging.INFO)


def preview(value, limit=500):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = re.sub(r'data:[^\s"\']*;base64,[A-Za-z0-9+/=_-]+', '[image data omitted]', text)
    return text[:limit].replace('\n', '\\n') + ('…' if len(text) > limit else '')


def content_stats(content):
    if isinstance(content, str):
        return len(content), 0
    chars, images = 0, 0
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str): chars += len(part)
            elif isinstance(part, dict):
                if part.get('type') in ('image_url', 'image'): images += 1
                elif isinstance(part.get('text'), str): chars += len(part['text'])
    return chars, images


def endpoint_name(url):
    value = urlsplit(str(url))
    host = value.hostname or ''
    return f'{value.scheme}://{host}' + (f':{value.port}' if value.port else '') + value.path


class TracedRetriever(BaseRetriever):
    retriever: BaseRetriever
    label: str

    def _get_relevant_documents(self, query, *, run_manager):
        start = time.monotonic()
        LOGGER.info('retrieval=%s started query_chars=%d', self.label, len(query))
        try:
            docs = self.retriever.invoke(query)
        except BaseException as error:
            LOGGER.error('retrieval=%s failed elapsed=%.2fs error_type=%s',
                         self.label, time.monotonic()-start, type(error).__name__)
            raise
        LOGGER.info('retrieval=%s completed elapsed=%.2fs records=%d text_chars=%d distinct_sources=%d',
                    self.label, time.monotonic()-start, len(docs),
                    sum(len(d.page_content) for d in docs),
                    len({d.metadata.get('source_file') or d.metadata.get('path') for d in docs}))
        for rank, document in enumerate(docs, 1):
            LOGGER.debug('retrieval=%s rank=%d page=%s representation=%s chunk_part=%s text_chars=%d',
                         self.label, rank, document.metadata.get('page'),
                         document.metadata.get('embedding_type', 'text'),
                         document.metadata.get('chunk_part'), len(document.page_content))
        return docs


class QueryTrace:
    def __init__(self, *, enabled=True, trace_content=False, stream=True, heartbeat_seconds=10):
        if heartbeat_seconds <= 0:
            raise ValueError('heartbeat_seconds must be positive')
        self.enabled, self.trace_content, self.stream = enabled, trace_content, stream
        self.heartbeat_seconds = heartbeat_seconds
        self.run_id = uuid.uuid4().hex[:8]
        self.started = time.monotonic()
        if enabled and not LOGGER.handlers:
            configure_logging()

    def info(self, message, *args):
        if self.enabled: LOGGER.info('run=%s ' + message, self.run_id, *args)

    def warning(self, message, *args):
        if self.enabled: LOGGER.warning('run=%s ' + message, self.run_id, *args)

    @contextmanager
    def phase(self, label):
        if not self.enabled:
            yield
            return
        start, stop = time.monotonic(), threading.Event()
        self.info('%s started', label)
        def heartbeat():
            while not stop.wait(self.heartbeat_seconds):
                self.info('%s still waiting elapsed=%.1fs (not a server token count)',
                          label, time.monotonic()-start)
        thread = threading.Thread(target=heartbeat, daemon=True, name="agent-heartbeat-" + self.run_id)
        thread.start()
        try:
            yield
        except BaseException as error:
            LOGGER.error('run=%s %s failed elapsed=%.2fs error_type=%s',
                         self.run_id, label, time.monotonic()-start, type(error).__name__)
            if self.trace_content:
                self.info('%s error_detail=%s', label, preview(str(error)))
            raise
        else:
            self.info('%s completed elapsed=%.2fs', label, time.monotonic()-start)
        finally:
            stop.set()
            thread.join(timeout=1)

    def model_call(self, model, messages, round_label):
        if not self.enabled:
            return model.invoke(messages)
        chars, images = 0, 0
        for message in messages:
            count, attached = content_stats(message.content)
            chars += count; images += attached
        self.info('llm=%s request messages=%d text_chars=%d images=%d streaming=%s',
                  round_label, len(messages), chars, images, self.stream)
        for index, message in enumerate(messages):
            count, attached = content_stats(message.content)
            LOGGER.debug('run=%s llm=%s message_index=%d role=%s text_chars=%d images=%d',
                         self.run_id, round_label, index, getattr(message, 'type', type(message).__name__),
                         count, attached)
        bound = getattr(model, 'bound', model)
        self.info('llm=%s settings model=%s max_tokens=%s reasoning_effort=%s timeout=%s',
                  round_label, getattr(bound, 'model_name', 'unknown'),
                  getattr(bound, 'max_tokens', 'unknown'), getattr(bound, 'reasoning_effort', 'unknown'),
                  getattr(bound, 'request_timeout', 'unknown'))
        with self.phase('llm=' + round_label):
            if self.stream and callable(getattr(model, 'stream', None)):
                total = None
                chunks, text_chars, reasoning_chars, tool_fragments = 0, 0, 0, 0
                started, last_progress = time.monotonic(), time.monotonic()
                for chunk in model.stream(messages, stream_usage=True):
                    chunks += 1
                    if chunks == 1:
                        self.info('llm=%s first_chunk elapsed=%.2fs', round_label, time.monotonic()-started)
                    text_chars += content_stats(chunk.content)[0]
                    reasoning = (getattr(chunk, 'additional_kwargs', {}) or {}).get('reasoning_content', '')
                    reasoning_chars += len(reasoning) if isinstance(reasoning, str) else 0
                    tool_fragments += len(getattr(chunk, 'tool_call_chunks', []) or [])
                    if self.trace_content and (chunk.content or reasoning):
                        self.info('llm=%s chunk_preview=%s', round_label, preview(chunk.content or reasoning))
                    total = chunk if total is None else total + chunk
                    if time.monotonic()-last_progress >= 5:
                        self.info('llm=%s received_chunks=%d text_chars=%d reasoning_chars_exposed=%d tool_fragments=%d',
                                  round_label, chunks, text_chars, reasoning_chars, tool_fragments)
                        last_progress = time.monotonic()
                if total is None:
                    raise RuntimeError('LLM stream ended without a response')
                response = message_chunk_to_message(total)
                self.info('llm=%s stream_finished chunks=%d text_chars=%d reasoning_chars_exposed=%d tool_fragments=%d',
                          round_label, chunks, text_chars, reasoning_chars, tool_fragments)
            else:
                response = model.invoke(messages)
        metadata = getattr(response, 'response_metadata', {}) or {}
        usage = getattr(response, 'usage_metadata', None) or metadata.get('token_usage')
        calls = response.tool_calls
        invalid = getattr(response, 'invalid_tool_calls', []) or []
        self.info('llm=%s response finish_reason=%s tool_calls=%d invalid_tool_calls=%d text_chars=%d usage=%s',
                  round_label, metadata.get('finish_reason', 'unknown'), len(calls), len(invalid),
                  content_stats(response.content)[0], json.dumps(usage, default=str))
        if metadata.get('finish_reason') == 'length':
            self.warning('llm=%s hit the output token limit; response/tool arguments may be incomplete', round_label)
        if invalid:
            self.warning('llm=%s could not parse %d tool calls', round_label, len(invalid))
        if not calls and not response.content:
            self.warning('llm=%s returned no answer text and no valid tool calls', round_label)
        if self.trace_content and response.content and not self.stream:
            self.info('llm=%s response_preview=%s', round_label, preview(response.content))
        return response

    def tool_call(self, tool_map, call, round_label):
        label = f'tool={call["name"]} round={round_label} call_id={call["id"]}'
        self.info('%s argument_keys=%s argument_chars=%d', label,
                  sorted(call['args']), len(json.dumps(call['args'], default=str)))
        if self.trace_content:
            self.info('%s arguments=%s', label, preview(call['args']))
        with self.phase(label):
            result = tool_map[call['name']].invoke(call['args'])
        chars, images = content_stats(result)
        self.info('%s result text_chars=%d images=%d', label, chars, images)
        if self.trace_content:
            text = result if isinstance(result, str) else [p for p in result if p.get('type') == 'text']
            self.info('%s result_preview=%s', label, preview(text))
        return result

    def finish(self, answer):
        self.info('query completed elapsed=%.2fs answer_chars=%d',
                  time.monotonic()-self.started, content_stats(answer)[0])
        return answer
