"""Pluggable VLM backends for constraint generation.

ReKep originally hard-coded an OpenAI client. This module keeps that path
byte-compatible (`backend: openai`) and adds two ways to drive a local
Qwen-VL instead:

  backend: openai         chatgpt-4o-latest etc. via api.openai.com (default)
  backend: openai_compat  any OpenAI-compatible server (vLLM/SGLang serving Qwen-VL)
  backend: qwen_local     in-process transformers, no server

All backends take OpenAI-style `messages` so `ConstraintGenerator` does not
need to know which one is active. The local backend re-parses the data URI
back into a PIL image, since Qwen's chat template wants the image inline
rather than as a URL.
"""

import base64
import io
import os
import re
import time


def _extract_text_and_images(messages):
    """Pull plain text and PIL images out of OpenAI-style messages."""
    from PIL import Image

    texts, images = [], []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            if part.get("type") == "text":
                texts.append(part["text"])
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                match = re.match(r"data:image/\w+;base64,(.*)", url, re.DOTALL)
                if match is None:
                    raise ValueError(f"Only base64 data URIs are supported, got: {url[:40]}...")
                images.append(Image.open(io.BytesIO(base64.b64decode(match.group(1)))).convert("RGB"))
    return "\n".join(texts), images


def _extract_raw_images(messages):
    """Pull (media_type, base64) pairs and text out of OpenAI-style messages.

    Anthropic takes base64 directly, so unlike the local backend there is no
    need to decode into PIL first.
    """
    texts, images = [], []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            if part.get("type") == "text":
                texts.append(part["text"])
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                match = re.match(r"data:(image/\w+);base64,(.*)", url, re.DOTALL)
                if match is None:
                    raise ValueError(f"Only base64 data URIs are supported, got: {url[:40]}...")
                images.append((match.group(1), match.group(2)))
    return "\n".join(texts), images


METADATA_KEYS = ("num_stages", "grasp_keypoints", "release_keypoints")


def sanitize_metadata_lines(output):
    """Strip inline comments from ReKep's three metadata assignment lines.

    Upstream parses these with a bare `int(x.strip())` over a comma split, so a
    model that writes `grasp_keypoints = [0  # the cube]` raises
    "invalid literal for int()" — observed from qwen3-vl. Only the metadata
    lines are touched: comments inside the generated constraint functions are
    legitimate Python and must survive.
    """
    cleaned = []
    for line in output.split("\n"):
        stripped = line.lstrip()
        if any(stripped.startswith(f"{key} ") or stripped.startswith(f"{key}=")
               for key in METADATA_KEYS) and "#" in line:
            line = line.split("#", 1)[0].rstrip()
        cleaned.append(line)
    return "\n".join(cleaned)


def find_degenerate_constraints(output):
    """Names of subgoal constraints whose cost ignores `end_effector`.

    The VLM's output IS the objective function and nothing type-checks it. A
    constraint like `norm(keypoints[0] - (keypoints[0] + [0,0,0.2]))` is valid
    Python, returns a constant, and gives the solver zero gradient — the run
    looks healthy and silently optimises nothing (observed from claude-opus-5
    on the Lift task). Evaluating each with two different gripper poses catches
    it in milliseconds.

    Path constraints that delegate to get_grasping_cost_by_keypoint_idx are
    legitimately ee-independent and are skipped.
    """
    import numpy as np

    # Extract just the function blocks, exactly as upstream's
    # _parse_and_save_constraints does. The raw answer also contains prose and
    # markdown fences, so exec()ing it wholesale throws and would make this
    # check silently pass everything.
    lines, blocks, start = output.split("\n"), [], None
    for i, line in enumerate(lines):
        if line.startswith("def "):
            start = i
        if line.startswith("    return ") and start is not None:
            blocks.append("\n".join(lines[start:i + 1]))
            start = None
    # Every early return here means "no degenerate constraints found", which is
    # indistinguishable from "the validator worked and everything is fine". That
    # is the same silent-degradation trap that let LIBERO's init-state load fail
    # unnoticed for an afternoon (see NOTES.md 1), so each failure says so.
    if not blocks:
        print("degeneracy check: no parseable constraint functions in the VLM "
              "output — NOT validated")
        return []

    scope = {"np": np, "numpy": np,
             "get_grasping_cost_by_keypoint_idx": lambda idx: 0.0}
    try:
        exec("\n\n".join(blocks), scope)  # noqa: S102 - generated constraints, same as upstream
    except Exception as exc:  # noqa: BLE001 - a syntax error is the parser's problem, not ours
        print(f"degeneracy check: constraints did not execute "
              f"({type(exc).__name__}: {str(exc)[:80]}) — NOT validated")
        return []

    # Size the dummy keypoint array to what the constraints actually INDEX, not
    # to a guess. This was hardcoded to two keypoints, so on any task with more
    # -- libero_goal/0 proposes 18 -- every constraint raised IndexError and the
    # check silently validated nothing. It only surfaced because the handler now
    # says when it skipped something.
    referenced = [int(m) for m in re.findall(r"keypoints\[\s*(\d+)\s*\]", output)]
    count = max(referenced, default=1) + 1
    # spread them so a constraint that genuinely depends on keypoint geometry
    # produces different costs at different ee poses
    keypoints = np.stack([
        np.array([0.05 * (i % 5), 0.05 * (i // 5), 0.83 + 0.02 * (i % 3)])
        for i in range(count)
    ])
    # Which keypoint each stage grabs. ReKep's solver MOVES a grasped keypoint
    # with the candidate ee pose (subgoal_solver.transform_keypoints), so a
    # constraint written purely between keypoints is still a function of the ee
    # -- indirectly, through the one being carried.
    #
    # This check holds keypoints FIXED, so it cannot see that dependence.
    # Measured on libero_goal/0, Claude wrote exactly that shape and it was
    # CORRECT:
    #     offsetted = keypoints[6] + [0.25, 0, 0]
    #     cost = abs(keypoints[2][0] - offsetted[0])
    # with grasp_keypoints = [2, -1], i.e. keypoint 2 is in the gripper during
    # stage 2. Flagging it would reject a valid plan and force a pointless
    # retry -- a worse failure than the silent no-op this check used to be.
    grasp_kp = []
    if m := re.search(r"grasp_keypoints\s*=\s*\[([^\]]*)\]", output):
        grasp_kp = [int(x) for x in re.findall(r"-?\d+", m.group(1))]

    def carried_by(stage):
        """Keypoints already in the gripper when `stage` runs (1-indexed)."""
        return {k for k in grasp_kp[: stage - 1] if k >= 0}

    degenerate = []
    for name, fn in scope.items():
        if not (callable(fn) and name.startswith("stage") and "subgoal" in name):
            continue
        if "get_grasping_cost_by_keypoint_idx" in getattr(fn, "__code__", type("", (), {"co_names": ()})).co_names:
            continue
        stage = int(re.match(r"stage(\d+)", name).group(1))
        source = "\n".join(b for b in blocks if b.lstrip().startswith(f"def {name}"))
        uses = {int(i) for i in re.findall(r"keypoints\[\s*(\d+)\s*\]", source)}
        if uses & carried_by(stage):
            continue        # ee-dependent through a carried keypoint
        try:
            costs = [float(fn(np.array([0.0, 0.0, z]), keypoints)) for z in (0.83, 1.30)]
        except Exception as exc:  # noqa: BLE001 - surfaces later with a real traceback
            # not degenerate, but not checked either — say which
            print(f"degeneracy check: {name} raised "
                  f"({type(exc).__name__}: {str(exc)[:60]}) — NOT validated")
            continue
        if abs(costs[0] - costs[1]) < 1e-9:
            degenerate.append(name)
    return degenerate


class RetryingBackend:
    """Shared retry-until-the-answer-has-the-required-markers loop.

    A local 8B model is markedly less reliable than a frontier model at holding
    a long output format. ReKep's parser needs `num_stages =`,
    `grasp_keypoints =` and `release_keypoints =` verbatim and raises an
    unhelpful "num_stages not found in output" when they are absent, so it is
    cheaper to detect that here and re-ask with an explicit reminder.
    """

    def query(self, messages):
        markers = self.config.get("required_markers") or []
        attempts = int(self.config.get("max_retries", 0)) + 1
        for attempt in range(attempts):
            output = sanitize_metadata_lines(self._query_once(messages, attempt))
            missing = [m for m in markers if m not in output]
            degenerate = [] if missing else find_degenerate_constraints(output)
            if not missing and not degenerate:
                return output
            if missing:
                print(f"  [attempt {attempt+1}/{attempts}] missing {missing} — retrying with a format reminder")
                reminder = (
                    "\n\nYour previous answer was missing these required lines: "
                    + ", ".join(f"`{m} ...`" for m in missing)
                    + ". Reply again with the COMPLETE answer, including every "
                    "constraint function AND those exact lines verbatim."
                )
            else:
                print(f"  [attempt {attempt+1}/{attempts}] degenerate: {degenerate} — cost ignores end_effector")
                reminder = (
                    "\n\nThese subgoal constraints return the same cost no matter where "
                    "the gripper is, so the solver has no gradient and the stage cannot "
                    "be achieved: " + ", ".join(f"`{n}`" for n in degenerate) + ". Every "
                    "subgoal constraint must measure a distance FROM `end_effector` TO the "
                    "target — e.g. `np.linalg.norm(end_effector - target_point)`, not "
                    "`np.linalg.norm(keypoints[0] - target_point)`. Reply again with the "
                    "COMPLETE corrected answer, including all metadata lines."
                )
            messages = list(messages) + [
                {"role": "assistant", "content": output[-2000:]},
                {"role": "user", "content": reminder},
            ]
        return output


class AnthropicBackend(RetryingBackend):
    """Claude via the official Anthropic SDK.

    Two things differ from the OpenAI-compatible path and both are load-bearing:

      * No assistant prefill. The `prefill` trick that bypasses qwen3-vl's
        forced thinking returns a 400 on Claude Opus 5 / 4.8 / 4.7 and Sonnet
        4.6 — the field is deliberately ignored here rather than passed through.
      * Thinking is on by default on Opus 5 and its raw text is never returned,
        so there is no reasoning budget to manage: the answer arrives as
        ordinary text blocks and cannot be starved by a thinking overrun.
    """

    def __init__(self, config):
        import anthropic

        self.config = config
        self.client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile

    def _query_once(self, messages, attempt=0):
        text, images = _extract_raw_images(messages)
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}}
            for mt, b64 in images
        ]
        content.append({"type": "text", "text": text})

        model = self.config.get("model", "claude-opus-5")
        kwargs = {
            "model": model,
            "max_tokens": self.config["max_tokens"],
            "messages": [{"role": "user", "content": content}],
        }
        effort = self.config.get("effort")
        if effort:
            kwargs["output_config"] = {"effort": effort}

        start = time.time()
        print(f"Querying {model}...", end="\r")
        # Stream: ReKep's answer runs 1-2k tokens and non-streaming requests at
        # this max_tokens risk an SDK HTTP timeout.
        with self.client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            raise RuntimeError(f"{model} refused: {getattr(message, 'stop_details', None)}")
        output = "".join(b.text for b in message.content if b.type == "text")
        print(f"[{time.time()-start:.1f}s] Querying {model}...Done "
              f"({message.usage.input_tokens} in / {message.usage.output_tokens} out)")
        return output


class OpenAIBackend(RetryingBackend):
    """Streams from api.openai.com. This is ReKep's original behaviour."""

    def __init__(self, config):
        from openai import OpenAI

        self.config = config
        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY", "")}
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        self.client = OpenAI(**kwargs)

    def _query_once(self, messages, attempt=0):
        # Reasoning models that offer no server-side off switch can still be
        # short-circuited by prefilling an already-closed thinking block as the
        # start of the assistant turn: the model resumes after `</think>` and
        # goes straight to the answer. Needed for qwen3-vl on ollama, where
        # both `think: false` and `/no_think` are silently ignored.
        prefill = self.config.get("prefill")
        if prefill:
            messages = list(messages) + [{"role": "assistant", "content": prefill}]

        stream = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            stream=True,
        )
        output, reasoning = "", ""
        start = time.time()
        last_print = 0.0
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                output += delta.content
            # Reasoning models (Qwen3-VL via ollama, o-series) stream their chain of
            # thought in a separate field. It must not reach the constraint parser,
            # but we keep it so an empty answer can be diagnosed.
            reasoning += getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or ""
            now = time.time() - start
            if now - last_print > 1.0:
                print(f"[{now:.0f}s] Querying {self.config['model']}... "
                      f"{len(reasoning)} reasoning / {len(output)} answer chars", end="\r")
                last_print = now
        print(f"[{time.time()-start:.1f}s] Querying {self.config['model']}...Done "
              f"({len(reasoning)} reasoning / {len(output)} answer chars)")
        if not output.strip():
            raise RuntimeError(
                f"{self.config['model']} returned an empty answer after "
                f"{len(reasoning)} chars of reasoning. The token budget was likely spent "
                f"thinking — raise constraint_generator.max_tokens (currently "
                f"{self.config['max_tokens']})."
            )
        return output


class OpenAICompatBackend(OpenAIBackend):
    """vLLM/SGLang serving Qwen-VL behind an OpenAI-compatible route.

    Identical wire format to OpenAIBackend; only the endpoint and the dummy
    api key differ, so serving Qwen-VL this way needs no code change here.
    """

    def __init__(self, config):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=os.environ.get("VLM_API_KEY", "EMPTY"),
            base_url=config.get("base_url", "http://localhost:8000/v1"),
        )


class QwenLocalBackend:
    """In-process Qwen-VL via transformers. No server, no API key.

    Loaded lazily so that importing this module stays cheap for the other
    backends — the weights are several GB.
    """

    def __init__(self, config):
        self.config = config
        self.model_path = config.get("model_path", config["model"])
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading {self.model_path} (first call only)...")
        start = time.time()
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.config.get("device_map", "auto"),
        ).eval()
        print(f"Loaded in {time.time()-start:.1f}s")

    def __call__(self, messages):
        """`ground()` and ReKep both call a backend directly.

        Every other backend inherits `__call__` from `RetryingBackend`; this
        one does not inherit and only had `query`, so it raised
        "'QwenLocalBackend' object is not callable" at the point of use rather
        than at construction. Aliasing keeps the interface uniform.
        """
        return self.query(messages)

    def query(self, messages):
        import torch

        self._load()
        text, images = _extract_text_and_images(messages)

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[prompt], images=images or None, return_tensors="pt"
        ).to(self._model.device)

        # temperature 0 means greedy; passing it alongside do_sample=True warns and misbehaves.
        temperature = self.config["temperature"]
        gen_kwargs = {"max_new_tokens": self.config["max_tokens"]}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        start = time.time()
        print(f"Querying {self.model_path}...", end="\r")
        with torch.no_grad():
            generated = self._model.generate(**inputs, **gen_kwargs)
        # strip the prompt tokens; only the completion is parsed downstream
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        output = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        print(f"[{time.time()-start:.2f}s] Querying {self.model_path}...Done")
        return output


BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "openai_compat": OpenAICompatBackend,
    "qwen_local": QwenLocalBackend,
}


def make_backend(config):
    name = config.get("backend", "openai")
    if name not in BACKENDS:
        raise ValueError(f"Unknown VLM backend '{name}'. Options: {sorted(BACKENDS)}")
    return BACKENDS[name](config)
