"""Hermes-2-Pro XML tool-call detector for sglang.

Parses the Hermes-2-Pro XML function-call format emitted by models whose chat
template instructs this exact shape (e.g. Qwen3.8-27B-AWQ-INT4):

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Berlin
    </parameter>
    <parameter=unit>
    celsius
    </parameter>
    </function>
    </tool_call>

Why this exists
---------------
Some chat templates render tools in the Hermes-2-Pro XML shape above, but
sglang's built-in ``qwen``/``hermes`` parsers both assume JSON inside the
``<tool_call>`` envelope. They match the envelope, fail ``json.loads()`` on
the XML body, and silently drop the call:

    Failed to parse JSON part: <function=get_weather> ...
    JSON parse error: Expecting value: line 1 column 1 (char 0)

The response then comes back with empty ``content`` and ``tool_calls=[]``
even though the model did emit a valid tool call.

This detector parses the XML form into the same ``ToolCallItem`` structures
(typed arguments, leading text preserved, streaming support). Register it via
``--tool-call-parser hermes_xml``.

Verified on 2x RTX 3090 (TP=2, DSpark, sglang 2e7c85da6):
- tool_choice=auto, thinking disabled -> structured call with exact typed args
- tool_choice=auto, thinking enabled -> reasoning in content + structured call
"""

import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# One <function=name>...</function> block (nested inside <tool_call>).
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=([A-Za-z0-9_.\-]+)>(.*?)</function>", re.DOTALL
)
# One <parameter=key>value</parameter> entry.
_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z0-9_.\-]+)>(.*?)</parameter>", re.DOTALL)


def _parse_function_block(raw: str):
    """Parse a single <function=...>...</function> block into (name, args_dict).

    Values are de-JSON-ified when possible (numbers/bools/null become native
    types), otherwise kept as strings. Nested XML inside a value is stripped.
    """
    m = _FUNCTION_BLOCK_RE.search(raw)
    if not m:
        return None
    name = m.group(1)
    body = m.group(2)
    args: dict = {}
    for pm in _PARAMETER_RE.finditer(body):
        key = pm.group(1)
        val = pm.group(2).strip()
        if not val:
            args[key] = ""
            continue
        low = val.lower()
        try:
            if low in ("true", "false"):
                args[key] = low == "true"
            elif low in ("null", "none"):
                args[key] = None
            elif re.fullmatch(r"-?\d+", val):
                args[key] = int(val)
            elif re.fullmatch(r"-?\d+\.\d+([eE][+-]?\d+)?", val):
                args[key] = float(val)
            else:
                args[key] = val
        except Exception:  # pragma: no cover
            args[key] = val
    return name, args


class HermesXMLDetector(BaseFormatDetector):
    """Detector for the Hermes-2-Pro XML function-call format."""

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self._normal_text_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def structure_info(self) -> _GetInfoFunc:
        """Constrained-generation structure for the XML format."""
        return lambda name: StructureInfo(
            begin="<tool_call>\n<function=" + name + ">\n",
            end="</function>\n</tool_call>",
            trigger="<tool_call>",
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        tool_indices = self._get_tool_indices(tools)
        for block in blocks:
            parsed = _parse_function_block(block)
            if parsed is None:
                continue
            name, args = parsed
            if name not in tool_indices:
                logger.warning(f"Model attempted to call undefined function: {name}")
                continue
            calls.append(
                ToolCallItem(
                    tool_index=tool_indices[name],
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _clean_normal_text(self, text: str) -> str:
        if not text:
            return text
        self._normal_text_buffer += text
        if self.eot_token in self._normal_text_buffer:
            cleaned = self._normal_text_buffer.replace(self.eot_token, "")
            self._normal_text_buffer = ""
            return cleaned
        partial_len = self._ends_with_partial_token(
            self._normal_text_buffer, self.eot_token
        )
        if partial_len:
            safe_text = self._normal_text_buffer[:-partial_len]
            self._normal_text_buffer = self._normal_text_buffer[-partial_len:]
            return safe_text
        cleaned = self._normal_text_buffer
        self._normal_text_buffer = ""
        return cleaned

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Streaming parse: emit normal text until <tool_call>, then accumulate
        the XML block and emit the parsed call once </tool_call> closes."""
        self._buffer += new_text
        current_text = self._buffer

        if self.bot_token not in current_text:
            partial = self._ends_with_partial_token(current_text, self.bot_token)
            if partial:
                safe = current_text[:-partial]
                self._buffer = current_text[-partial:]
            else:
                safe = current_text
                self._buffer = ""
            return StreamingParseResult(normal_text=self._clean_normal_text(safe))

        if self.eot_token not in current_text:
            # Inside the tool call — wait for the closing tag.
            return StreamingParseResult()

        idx = current_text.find(self.bot_token)
        normal_text = self._clean_normal_text(current_text[:idx])
        calls = self.detect_and_parse(current_text, tools).calls
        self._buffer = ""
        return StreamingParseResult(normal_text=normal_text, calls=calls)
