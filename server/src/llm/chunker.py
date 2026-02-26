"""Sentence boundary detection for streaming LLM→TTS pipeline.

Accumulates LLM tokens and yields complete sentences as soon as a
sentence boundary is detected. This allows TTS to start synthesizing
each sentence immediately without waiting for the full LLM response.
"""

import logging
import re

logger = logging.getLogger("avatar.llm.chunker")

# Sentence-ending punctuation patterns
# Matches: . ! ? … and their combinations, followed by whitespace or end of string
_SENTENCE_END = re.compile(
    r"[.!?…]"       # sentence-ending punctuation
    r"[\"\'\)\]]*"   # optional closing quotes/brackets
    r"(?:\s|$)"      # followed by whitespace or end of string
)

# Minimum characters before we look for a sentence boundary.
# Avoids splitting on abbreviations like "Dr." or "e.g."
_MIN_CHUNK_LENGTH = 20

# Maximum characters to accumulate before forcing a flush,
# even without a sentence boundary (safety valve for long streams).
_MAX_CHUNK_LENGTH = 500


class SentenceChunker:
    """Accumulates streaming tokens and yields complete sentences.

    Usage:
        chunker = SentenceChunker()
        for token in llm_tokens:
            for sentence in chunker.add(token):
                await tts.synthesize(sentence)
        # Flush any remaining text
        final = chunker.flush()
        if final:
            await tts.synthesize(final)
    """

    def __init__(
        self,
        min_length: int = _MIN_CHUNK_LENGTH,
        max_length: int = _MAX_CHUNK_LENGTH,
    ) -> None:
        """Initialize chunker.

        Args:
            min_length: Minimum chars before checking for sentence boundaries.
            max_length: Force flush after this many chars.
        """
        self._buffer = ""
        self._min_length = min_length
        self._max_length = max_length

    def add(self, token: str) -> list[str]:
        """Add a token and return any complete sentences.

        Args:
            token: A text token from the LLM stream.

        Returns:
            List of complete sentences (may be empty if no boundary found yet).
        """
        self._buffer += token
        sentences: list[str] = []

        while True:
            # Don't look for boundaries in very short buffers
            if len(self._buffer) < self._min_length:
                break

            # Force flush if buffer is too long
            if len(self._buffer) >= self._max_length:
                sentences.append(self._buffer.strip())
                self._buffer = ""
                break

            # Search for sentence boundary
            match = _SENTENCE_END.search(self._buffer, self._min_length - 1)
            if match:
                end_pos = match.end()
                sentence = self._buffer[:end_pos].strip()
                self._buffer = self._buffer[end_pos:]
                if sentence:
                    sentences.append(sentence)
            else:
                break

        return sentences

    def flush(self) -> str | None:
        """Flush any remaining text in the buffer.

        Call this after the LLM stream ends to get the final chunk.

        Returns:
            Remaining text, or None if buffer is empty.
        """
        text = self._buffer.strip()
        self._buffer = ""
        return text if text else None

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer = ""
