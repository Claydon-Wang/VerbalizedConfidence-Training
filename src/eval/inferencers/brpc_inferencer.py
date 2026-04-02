import logging
import os
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.eval.inferencers.verbalized_confidence_inferencer import VerbalizedConfidenceInferencer


class BayesianResidualProbe(nn.Module):
    def __init__(self, hidden_size: int, probe_hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden_size),
            nn.ReLU(),
            nn.Linear(probe_hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)


class BRPCInferencer(VerbalizedConfidenceInferencer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self._posthoc_confidences = None
        self._raw_confidences = None
        self._brpc_deltas = None
        self._brpc_model = None
        self._brpc_probe = None

    @staticmethod
    def _confidence_char_span(completion_text: str):
        matches = list(re.finditer(r"<confidence>.*?</confidence>", completion_text, re.DOTALL | re.MULTILINE))
        if not matches:
            return None
        last_match = matches[-1]
        return last_match.start(), last_match.end()

    def _feature_char_span(self, completion_text: str):
        confidence_span = self._confidence_char_span(completion_text)
        if confidence_span is None:
            return 0, len(completion_text)
        return 0, confidence_span[0]

    def _build_segment_mask_for_completion(self, completion_ids, completion_text: str, span):
        active_len = len(completion_ids)
        segment_mask = torch.zeros(active_len, dtype=torch.bool)
        if span is None:
            return segment_mask

        span_start, span_end = span
        try:
            encoded = self.model.tokenizer(
                completion_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            encoded_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]
            if encoded_ids == completion_ids:
                for token_position, (token_start, token_end) in enumerate(offsets):
                    if token_end > span_start and token_start < span_end:
                        segment_mask[token_position] = True
                return segment_mask
        except Exception:
            pass

        prefix_start = 0
        completion_ids_tensor = torch.tensor(completion_ids, dtype=torch.long)
        for idx in range(active_len):
            token_span_end = len(
                self.model.tokenizer.decode(
                    completion_ids_tensor[: idx + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            if token_span_end > span_start and prefix_start < span_end:
                segment_mask[idx] = True
            prefix_start = token_span_end
        return segment_mask

    def _load_probe_state_dict(self, checkpoint_path: str):
        probe_state = {}
        if not os.path.isdir(checkpoint_path):
            raise ValueError(f"BRPC inferencer requires a local checkpoint directory, got: {checkpoint_path}")

        safetensor_files = [name for name in os.listdir(checkpoint_path) if name.endswith(".safetensors")]
        for file_name in safetensor_files:
            from safetensors import safe_open

            file_path = os.path.join(checkpoint_path, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("brpc_probe."):
                        probe_state[key.replace("brpc_probe.", "", 1)] = f.get_tensor(key)

        if not probe_state:
            bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.isfile(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")
                for key, value in state_dict.items():
                    if key.startswith("brpc_probe."):
                        probe_state[key.replace("brpc_probe.", "", 1)] = value

        if not probe_state:
            raise ValueError(f"No brpc_probe weights found under checkpoint: {checkpoint_path}")
        return probe_state

    def _load_brpc_modules(self):
        if self._brpc_model is not None and self._brpc_probe is not None:
            return

        checkpoint_path = self.config.checkpoint_name or self.config.model_name_or_path
        self._brpc_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self._brpc_model.eval()

        hidden_size = None
        for attr in ("hidden_size", "d_model", "n_embd"):
            hidden_size = getattr(self._brpc_model.config, attr, None)
            if hidden_size is not None:
                break
        if hidden_size is None:
            raise ValueError("Unable to infer hidden size for BRPC eval probe.")

        self._brpc_probe = BayesianResidualProbe(
            hidden_size=int(hidden_size),
            probe_hidden_size=int(self.config.brpc_probe_hidden_size),
        )
        probe_state = self._load_probe_state_dict(checkpoint_path)
        self._brpc_probe.load_state_dict(probe_state, strict=True)
        self._brpc_probe.eval()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self._brpc_model.to(device)
            self._brpc_probe.to(device)
        else:
            device = torch.device("cpu")
        self._brpc_device = device

    def _compute_brpc_confidences(self, texts, outputs):
        self.model.close()
        self._load_brpc_modules()

        flat_items = []
        raw_confidences = []
        for prompt_text, output in zip(texts, outputs):
            for generation in output.outputs:
                conf_pattern = r"<confidence>(.*?)</confidence>"
                conf_matches = re.findall(conf_pattern, generation.text, re.DOTALL | re.MULTILINE)
                conf_text = conf_matches[-1].strip() if conf_matches else ""
                _, raw_conf = self.confidence_extractor(conf_text)
                raw_confidences.append(raw_conf)

                prompt_ids = self.model.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                completion_ids = self.model.tokenizer(generation.text, add_special_tokens=False)["input_ids"]
                feature_mask = self._build_segment_mask_for_completion(
                    completion_ids,
                    generation.text,
                    self._feature_char_span(generation.text),
                )
                full_ids = prompt_ids + completion_ids
                flat_items.append(
                    {
                        "input_ids": full_ids,
                        "prompt_len": len(prompt_ids),
                        "completion_len": len(completion_ids),
                        "feature_mask": feature_mask,
                    }
                )

        batch_size = int(getattr(self.config, "brpc_posthoc_batch_size", 4))
        delta_raw_list = []
        with torch.no_grad():
            for start in range(0, len(flat_items), batch_size):
                batch_items = flat_items[start : start + batch_size]
                max_len = max(len(item["input_ids"]) for item in batch_items)
                pad_id = self.model.tokenizer.pad_token_id or self.model.tokenizer.eos_token_id

                batch_input_ids = []
                batch_attention_mask = []
                for item in batch_items:
                    pad_len = max_len - len(item["input_ids"])
                    batch_input_ids.append(item["input_ids"] + [pad_id] * pad_len)
                    batch_attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)

                input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=self._brpc_device)
                attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=self._brpc_device)
                outputs_hf = self._brpc_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden = outputs_hf.hidden_states[-1]

                for batch_idx, item in enumerate(batch_items):
                    prompt_len = item["prompt_len"]
                    completion_len = item["completion_len"]
                    completion_hidden = last_hidden[batch_idx, prompt_len : prompt_len + completion_len, :]
                    feature_mask = item["feature_mask"].to(device=completion_hidden.device, dtype=completion_hidden.dtype)
                    denom = feature_mask.sum().clamp(min=1.0)
                    pooled_hidden = (completion_hidden * feature_mask.unsqueeze(-1)).sum(dim=0) / denom
                    probe_dtype = next(self._brpc_probe.parameters()).dtype
                    pooled_hidden = pooled_hidden.to(dtype=probe_dtype)
                    delta_raw = self._brpc_probe(pooled_hidden.unsqueeze(0)).squeeze(0)
                    delta_raw_list.append(delta_raw)

        delta_raw_tensor = torch.stack(delta_raw_list).float().cpu()
        raw_conf_tensor = torch.tensor(raw_confidences, dtype=torch.float32).clamp(
            min=self.config.brpc_eps, max=1.0 - self.config.brpc_eps
        )
        delta_tensor = delta_raw_tensor
        calibrated_conf_tensor = torch.sigmoid(torch.logit(raw_conf_tensor) + delta_tensor)

        self._raw_confidences = raw_conf_tensor.view(len(outputs), self.config.num_generations).tolist()
        self._brpc_deltas = delta_tensor.view(len(outputs), self.config.num_generations).tolist()
        self._posthoc_confidences = calibrated_conf_tensor.view(len(outputs), self.config.num_generations).tolist()
        logging.info(
            "Config %s BRPC diagnostics: raw_conf_mean=%.4f calibrated_conf_mean=%.4f delta_mean=%.4f delta_abs_mean=%.4f",
            self.config.name,
            raw_conf_tensor.mean().item(),
            calibrated_conf_tensor.mean().item(),
            delta_tensor.mean().item(),
            delta_tensor.abs().mean().item(),
        )

    def estimate_confidence(self, texts, outputs):
        outputs = super().estimate_confidence(texts, outputs)
        self._compute_brpc_confidences(texts, outputs)
        return outputs

    def extract_output_columns(self, outputs):
        output_columns = super().extract_output_columns(outputs)
        if self._posthoc_confidences is not None:
            if getattr(self.config, "save_brpc_diagnostics", False):
                output_columns["raw_confidences"] = output_columns["confidences"]
                output_columns["calibrated_confidences"] = self._posthoc_confidences
                output_columns["brpc_deltas"] = self._brpc_deltas
            output_columns["confidences"] = self._posthoc_confidences
        return output_columns
