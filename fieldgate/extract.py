"""Field extraction with a local causal language model, one field at a time.

The JSON skeleton is written by this code and the model only fills in each value, so the output
always parses and every value has its own token probabilities. For each field the model writes
either `null` or a quoted string copied from the receipt.

Three things are recorded per field:
  the greedy value and the log-probabilities of the tokens that spelled it,
  optional sampled values (temperature 0.7) for self-consistency,
  the model's own stated confidence, asked in a follow-up turn (the usual "LLM confidence" baseline).

Only `transformers` and `torch` are needed, on CPU. Nothing leaves the machine.
"""
from __future__ import annotations

import math
import re
import time

from .schema import FIELDS

SYSTEM = ("You extract fields from the OCR text of a shop or restaurant receipt. The OCR may contain mistakes. "
          "Copy each value exactly as it is printed, including its separators. Answer null for a field that is "
          "not printed on the receipt. Do not calculate or infer a value that is not printed.")

CONFIDENCE = ("For each field, how likely is it that your answer is exactly right? "
              "Reply with one JSON object with the same keys and an integer from 0 to 100 for each.")

MAX_CHARS = 6000
MAX_VALUE_TOKENS = 20
_BARE = re.compile(r"-?[\d.,]+")


def user_prompt(text: str) -> str:
    fields = "\n".join(f"- {f.name}: {f.describe}" for f in FIELDS)
    return (f"Receipt text:\n<<<\n{text[:MAX_CHARS]}\n>>>\n\nFields:\n{fields}\n\n"
            "Reply with one JSON object with exactly these keys. Each value is a string copied from the receipt, or null.")


def parse_value(text: str) -> tuple[bool, str | None, int, bool]:
    """Read a JSON scalar from the start of `text`.

    Returns (finished, value, end, malformed): `end` is how many characters of `text` belong to the
    value, and `malformed` is True when the model wrote something that is neither null nor a string.
    """
    body = text.lstrip()
    lead = len(text) - len(body)
    if body == "":
        return False, None, 0, False
    if body.startswith('"'):
        close = body.find('"', 1)
        if close == -1:
            return False, None, 0, False
        return True, body[1:close], lead + close + 1, False
    if "null".startswith(body[:4]):
        if len(body) < 4:
            return False, None, 0, False
        return True, None, lead + 4, False
    bare = _BARE.match(body)
    if bare:
        if bare.end() == len(body):                 # the number may not be finished
            return False, None, 0, False
        number = bare.group(0).rstrip(".,")
        return True, number, lead + len(number), True
    return True, None, 0, True


class Extractor:
    def __init__(self, model_dir: str, threads: int | None = None, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype))
        self.model.eval()
        self.name = model_dir.rstrip("/").split("/")[-1]

    # -- low level ---------------------------------------------------------------------------

    def _ids(self, text: str):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

    def _feed(self, ids, past):
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def _chat(self, messages: list[dict], prefix: str) -> str:
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + prefix

    def _value(self, logits, past, temperature: float, generator):
        """Generate one JSON scalar. Returns (value, stats, text_to_feed_before_the_next_key, past)."""
        torch = self.torch
        fed: list[int] = []
        logps: list[float] = []
        p_null = 0.0
        value, malformed, remainder = None, False, " null"
        for step in range(MAX_VALUE_TOKENS + 1):
            logp = torch.log_softmax(logits, dim=-1)
            if step == 0:                      # how much the model leaned towards "not on the receipt"
                top = torch.topk(logp, 20)
                p_null = sum(math.exp(lp) for lp, t in zip(top.values.tolist(), top.indices.tolist())
                             if self.tokenizer.decode([t]).strip().startswith("n"))
            if temperature > 0:
                token = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=generator))
            else:
                token = int(torch.argmax(logits))
            before = self.tokenizer.decode(fed)
            after = self.tokenizer.decode(fed + [token])
            done, value, end, malformed = parse_value(after)
            if done or step == MAX_VALUE_TOKENS:
                logps.append(float(logp[token]))
                if not done:                   # ran out of room inside a string: close it
                    value, malformed = after.lstrip().lstrip('"'), True
                    remainder = after[len(before):] + '"'
                elif end == 0:                 # unreadable: record null and write null into the context
                    remainder = " null" if not fed else ""
                else:
                    remainder = after[len(before):end]
                break
            logps.append(float(logp[token]))
            fed.append(token)
            logits, past = self._feed(torch.tensor([[token]]), past)
        stats = {"tokens": len(logps), "sum_logp": sum(logps), "min_logp": min(logps), "first_logp": logps[0],
                 "p_null": p_null, "malformed": malformed}
        return value, stats, remainder, past

    def _fill(self, logits0, past, prompt_len: int, rows: int, temperature: float, generator) -> list[dict]:
        """Fill every field for `rows` copies of the same prompt in one batch.

        Row 0 is greedy; the others are sampled at `temperature`. On a CPU the cost of a step is
        dominated by reading the weights once, so extra rows are nearly free. Rows finish a value
        at different steps; a row that is ahead feeds padding that the others never attend to, and
        position ids skip the padding, so each row computes exactly what it would alone.
        """
        torch = self.torch
        tok = self.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if rows > 1:
            past.batch_repeat_interleave(rows)
        logits = logits0.unsqueeze(0).repeat(rows, 1)
        mask = torch.ones(rows, prompt_len, dtype=torch.long)
        position = [prompt_len] * rows
        out: list[dict] = [{} for _ in range(rows)]
        for i, f in enumerate(FIELDS):
            closing = f', "{FIELDS[i + 1].name}":' if i + 1 < len(FIELDS) else "}"
            state = [{"fed": [], "logps": [], "queue": None, "p_null": 0.0} for _ in range(rows)]
            while True:
                feed, real = [], []
                for b, s in enumerate(state):
                    if s["queue"] is None:                         # still writing the value
                        logp = torch.log_softmax(logits[b], dim=-1)
                        if not s["logps"]:      # how much the model leaned towards "not on the receipt"
                            top = torch.topk(logp, 20)
                            s["p_null"] = sum(math.exp(lp) for lp, t in zip(top.values.tolist(), top.indices.tolist())
                                              if tok.decode([t]).strip().startswith("n"))
                        if b > 0 and temperature > 0:
                            token = int(torch.multinomial(torch.softmax(logits[b] / temperature, dim=-1), 1,
                                                          generator=generator))
                        else:
                            token = int(torch.argmax(logits[b]))
                        s["logps"].append(float(logp[token]))
                        before, after = tok.decode(s["fed"]), tok.decode(s["fed"] + [token])
                        done, value, end, malformed = parse_value(after)
                        if not done and len(s["logps"]) > MAX_VALUE_TOKENS:      # no room left: close the string
                            done, value, malformed = True, after.lstrip().lstrip('"'), True
                            remainder = after[len(before):] + '"'
                        elif done and end == 0:                                  # unreadable: write null
                            remainder = "" if s["fed"] else " null"
                        elif done:
                            remainder = after[len(before):end]
                        if done:
                            s["queue"] = self._ids(remainder + closing)[0].tolist()
                            lp = s["logps"]
                            out[b][f.name] = {"value": value, "tokens": len(lp), "sum_logp": sum(lp), "min_logp": min(lp),
                                              "first_logp": lp[0], "p_null": s["p_null"], "malformed": malformed}
                        else:
                            s["fed"].append(token)
                            feed.append(token)
                            real.append(1)
                            continue
                    if s["queue"]:
                        feed.append(s["queue"][0])
                        real.append(1)
                    else:
                        feed.append(pad)
                        real.append(0)
                if all(s["queue"] is not None for s in state):
                    # Every row has its value: feed what is left of each row's closing text in one step.
                    width = max(len(s["queue"]) for s in state)
                    if width:
                        ids = [s["queue"] + [pad] * (width - len(s["queue"])) for s in state]
                        keep = [[1] * len(s["queue"]) + [0] * (width - len(s["queue"])) for s in state]
                        where = [[position[b] + min(j, max(len(s["queue"]) - 1, 0)) for j in range(width)]
                                 for b, s in enumerate(state)]
                        mask = torch.cat([mask, torch.tensor(keep)], dim=1)
                        result = self.model(input_ids=torch.tensor(ids), past_key_values=past, use_cache=True,
                                            attention_mask=mask, position_ids=torch.tensor(where))
                        past = result.past_key_values
                        for b, s in enumerate(state):
                            if s["queue"]:
                                logits[b] = result.logits[b, len(s["queue"]) - 1].float()
                                position[b] += len(s["queue"])
                    break
                for s, used in zip(state, real):
                    if used and s["queue"]:
                        s["queue"].pop(0)
                mask = torch.cat([mask, torch.tensor(real).unsqueeze(1)], dim=1)
                result = self.model(input_ids=torch.tensor(feed).unsqueeze(1), past_key_values=past, use_cache=True,
                                    attention_mask=mask, position_ids=torch.tensor(position).unsqueeze(1))
                past = result.past_key_values
                fresh = result.logits[:, -1].float()
                for b in range(rows):
                    if real[b]:
                        logits[b] = fresh[b]
                        position[b] += 1
        return out

    # -- public ------------------------------------------------------------------------------

    def extract(self, text: str, samples: int = 0, temperature: float = 0.7, seed: int = 0,
                ask_confidence: bool = True) -> dict:
        torch = self.torch
        started = time.time()
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(text)}]
        first_key = f'{{"{FIELDS[0].name}":'
        with torch.inference_mode():
            prompt_ids = self._ids(self._chat(messages, first_key))
            logits0, past = self._feed(prompt_ids, None)
            prompt_len = prompt_ids.shape[1]

            generator = torch.Generator().manual_seed(seed)
            filled = self._fill(logits0, past, prompt_len, 1 + samples, temperature, generator)
            fields = filled[0]
            drawn = [{k: v["value"] for k, v in row.items()} for row in filled[1:]]
            del past

            stated: dict = {}
            if ask_confidence:
                answer = "{" + ", ".join(f'"{k}": ' + ("null" if v["value"] is None else '"' + v["value"] + '"')
                                         for k, v in fields.items()) + "}"
                follow = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": CONFIDENCE}]
                logits, cpast = self._feed(self._ids(self._chat(follow, first_key)), None)
                for i, f in enumerate(FIELDS):
                    value, _, remainder, cpast = self._value(logits, cpast, 0.0, None)
                    number = re.search(r"\d+", value or "")
                    stated[f.name] = min(100, int(number.group(0))) if number else None
                    closing = f', "{FIELDS[i + 1].name}":' if i + 1 < len(FIELDS) else "}"
                    logits, cpast = self._feed(self._ids(remainder + closing), cpast)
                del cpast
        return {"model": self.name, "fields": fields, "stated": stated, "samples": drawn,
                "prompt_tokens": int(prompt_len), "seconds": round(time.time() - started, 2)}
