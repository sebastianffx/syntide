"""Store and replay language-model responses used to reword questions.

A language model can reword a question or explanation but cannot choose its
answer. Every result is checked against the original evidence. Requests record
the provider, model, prompt version, parameters, and prompt. Stored response
checksums allow the same task to be reconstructed without another model call.

Replay mode requires a stored response. Record mode reuses stored responses and
requests a new response only when needed. Fixed templates provide an offline
default; the optional Anthropic provider requires credentials and may incur
usage charges.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from syntide.explore.plans import content_id

QUESTION_PROMPT = "question-paraphrase@1"
RATIONALE_PROMPT = "rationale-rewrite@1"
DEFAULT_MODEL = "claude-opus-5-5"


class TeacherError(RuntimeError):
    """A stored response is missing or fails verification."""


@dataclass(frozen=True)
class TeacherRequest:
    provider: str
    model: str
    prompt_version: str
    params: dict[str, Any]
    prompt: str

    @property
    def key(self) -> str:
        return "tr-" + content_id(asdict(self))[:24]


@dataclass(frozen=True)
class TeacherResponse:
    key: str
    text: str
    ok: bool  # False for refusals and empty replies
    meta: dict[str, Any] = field(
        default_factory=dict
    )  # served model, stop reason, request id, usage

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def ref(self) -> dict[str, Any]:
        return {"request_key": self.key, "response_sha256": self.sha256}


class Provider(Protocol):
    name: str

    def complete(self, request: TeacherRequest) -> TeacherResponse: ...


class TeacherStore:
    """Append-only response storage with one response per request identifier."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, TeacherResponse] = {}
        if self.path.exists():
            for line_no, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                item = json.loads(line)
                response = TeacherResponse(
                    item["key"], item["text"], item["ok"], item.get("meta", {})
                )
                if response.sha256 != item["response_sha256"]:
                    raise TeacherError(
                        f"{self.path}:{line_no}: the response does "
                        f"not match its sha256"
                    )
                if (
                    item["key"] in self._records
                    and self._records[item["key"]] != response
                ):
                    raise TeacherError(
                        f"{self.path}:{line_no}: request "
                        f"{item['key']} has two responses"
                    )
                self._records[item["key"]] = response

    def get(self, key: str) -> TeacherResponse | None:
        return self._records.get(key)

    def put(self, request: TeacherRequest, response: TeacherResponse) -> None:
        if response.key in self._records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "key": response.key,
            "request": asdict(request),
            "text": response.text,
            "ok": response.ok,
            "meta": response.meta,
            "response_sha256": response.sha256,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._records[response.key] = response

    def sha256(self) -> str | None:
        """Content checksum of responses, independent of file order."""
        if not self._records:
            return None
        return content_id(
            sorted([k, r.sha256, r.ok] for k, r in self._records.items())
        )


class Teacher:
    """Language-model provider with response storage and replay."""

    def __init__(
        self,
        provider: Provider,
        store: TeacherStore,
        mode: str = "record",
        model: str = DEFAULT_MODEL,
        params: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("replay", "record"):
            raise ValueError("teacher mode must be replay or record")
        self.provider, self.store, self.mode = provider, store, mode
        self.model = model
        self.params = dict(params or {})

    def request(self, prompt_version: str, prompt: str) -> TeacherRequest:
        return TeacherRequest(
            self.provider.name, self.model, prompt_version, self.params, prompt
        )

    def ask(self, prompt_version: str, prompt: str) -> TeacherResponse:
        request = self.request(prompt_version, prompt)
        stored = self.store.get(request.key)
        if stored is not None:
            return stored
        if self.mode == "replay":
            raise TeacherError(
                f"no stored response for {request.key} "
                f"({prompt_version}); replay never calls a provider"
            )
        response = self.provider.complete(request)
        self.store.put(request, response)
        return response

    def provenance(self) -> dict[str, Any]:
        return {
            "provider": self.provider.name,
            "model": self.model,
            "params": self.params,
            "mode": self.mode,
            "store": str(self.store.path),
            "store_sha256": self.store.sha256(),
        }


def question_prompt(question: str, options: list[str]) -> str:
    listed = (
        (
            f"\nAnswer options (keep them exactly as written if you "
            f"mention them): {', '.join(options)}"
        )
        if options
        else ""
    )
    return (
        "Reword this question about an ECG recording. Keep its "
        "meaning, every number, every lead name and "
        "every unit. Do not answer it, add facts or change what is "
        "asked. Reply with the reworded question "
        f"only, ending with a question mark.\n\nQuestion: {question}{listed}"
    )


def rationale_prompt(question: str, answer: str, claims: list[str]) -> str:
    facts = "\n".join(f"- {claim}" for claim in claims)
    return (
        "Write a short explanation (at most three sentences) for the "
        "answer below, using only these verified "
        "facts. Keep every number exactly as written; add no other "
        "number, finding or diagnosis; "
        "do not write "
        f"an 'Answer:' line.\n\nQuestion: {question}\nAnswer: "
        f"{answer}\nVerified facts:\n{facts}"
    )


class TemplateProvider:
    """Reword questions offline using fixed, reproducible patterns."""

    name = "syntide.template@1"
    _QUESTION = (
        "Based on this recording, {q}",
        "Looking at the ECG, {q}",
        "From the waveform alone, {q}",
    )
    _RATIONALE = ("{r}", "In short: {r}")

    def complete(self, request: TeacherRequest) -> TeacherResponse:
        source = request.prompt.rsplit("Question: ", 1)[-1]
        pick = int(hashlib.sha256(request.prompt.encode()).hexdigest()[:8], 16)
        if request.prompt_version == QUESTION_PROMPT:
            question = source.split("\nAnswer options", 1)[0].strip()
            text = self._QUESTION[pick % len(self._QUESTION)].format(
                q=question[:1].lower() + question[1:]
            )
        elif request.prompt_version == RATIONALE_PROMPT:
            facts = request.prompt.split("Verified facts:\n", 1)[-1]
            sentences = " ".join(
                line[2:] for line in facts.splitlines() if line.startswith("- ")
            )
            text = self._RATIONALE[pick % len(self._RATIONALE)].format(
                r=sentences
            )
        else:
            raise TeacherError(
                f"{self.name} has no pattern for {request.prompt_version}"
            )
        return TeacherResponse(request.key, text, True, {"provider": self.name})


class AnthropicProvider:
    """Paid provider requiring the Anthropic package and credentials."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 1024,
    ) -> None:
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def complete(self, request: TeacherRequest) -> TeacherResponse:
        try:
            import anthropic
        except ImportError as error:
            raise ImportError(
                "the anthropic teacher needs the SDK: uv run --with "
                "anthropic ..."
            ) from error
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=request.model,
            max_tokens=int(request.params.get("max_tokens", self.max_tokens)),
            output_config={"effort": request.params.get("effort", self.effort)},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": request.prompt}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        ok = response.stop_reason == "end_turn" and bool(text)
        meta = {
            "served_model": response.model,
            "stop_reason": response.stop_reason,
            "request_id": getattr(response, "_request_id", None),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }
        return TeacherResponse(request.key, text, ok, meta)


def make_teacher(config: dict[str, Any] | None) -> Teacher | None:
    """Build a rewording provider when its settings are supplied."""
    if not config or not config.get("provider"):
        return None
    store = TeacherStore(config["store"])
    if config["provider"] == "template":
        provider: Provider = TemplateProvider()
        model = "syntide.template@1"
        params: dict[str, Any] = {}
    elif config["provider"] == "anthropic":
        model = config.get("model", DEFAULT_MODEL)
        params = {
            "effort": config.get("effort", "medium"),
            "max_tokens": int(config.get("max_tokens", 1024)),
        }
        provider = AnthropicProvider(
            model, params["effort"], params["max_tokens"]
        )
    else:
        raise ValueError(
            f"unknown teacher provider {config['provider']!r}; choose "
            f"template or anthropic"
        )
    return Teacher(provider, store, config.get("mode", "record"), model, params)
