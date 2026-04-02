import gc
import importlib
from contextlib import contextmanager

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model_path_for_generation = self._resolve_generation_model_path(config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path_for_generation, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_path_for_generation,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
        )

    def _resolve_generation_model_path(self, model_name_or_path: str) -> str:
        return model_name_or_path

    def build_generation_inputs(self, prompts):
        prompt_ids = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True)
        texts = [self.tokenizer.decode(x) for x in prompt_ids]
        return texts, prompt_ids

    @contextmanager
    def override_vllm_progress_desc(self, progress_desc):
        if progress_desc is None:
            yield
            return

        llm_module = importlib.import_module("vllm.entrypoints.llm")
        original_tqdm = llm_module.tqdm

        def wrapped_tqdm(*args, **kwargs):
            kwargs["desc"] = progress_desc
            return original_tqdm(*args, **kwargs)

        llm_module.tqdm = wrapped_tqdm
        try:
            yield
        finally:
            llm_module.tqdm = original_tqdm

    def generate(self, texts, n=None, temperature=None, max_tokens=None, seed=None, logprobs=None, progress_desc=None):
        sampling_params = SamplingParams(
            n=self.config.num_generations if n is None else n,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            seed=self.config.seed if seed is None else seed,
            logprobs=logprobs,
        )
        with self.override_vllm_progress_desc(progress_desc):
            return self.llm.generate(texts, sampling_params=sampling_params)

    def close(self):
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
