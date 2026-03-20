import logging
import math

from vllm import SamplingParams

from src.eval.inferencers.base_inferencer import BaseInferencer


class PTrueInferencer(BaseInferencer):
    P_TRUE_INST = "\n\nIs the above answer correct? Respond with Yes or No only.\nResponse:"
    YES_VARIANTS = ["Yes", " Yes"]
    NO_VARIANTS = ["No", " No"]

    def requires_model_for_confidence_estimation(self) -> bool:
        return True

    @staticmethod
    def _logsumexp(values):
        if not values:
            return None
        max_val = max(values)
        return max_val + math.log(sum(math.exp(value - max_val) for value in values))

    @staticmethod
    def _extract_yes_no_probability(yes_logprobs, no_logprobs):
        yes_score = PTrueInferencer._logsumexp(yes_logprobs)
        no_score = PTrueInferencer._logsumexp(no_logprobs)
        if yes_score is None or no_score is None:
            return None

        max_score = max(yes_score, no_score)
        normalizer = max_score + math.log(math.exp(yes_score - max_score) + math.exp(no_score - max_score))
        return math.exp(yes_score - normalizer)

    def _collect_variant_token_ids(self, variants):
        token_ids = []
        for variant in variants:
            encoded = self.model.tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids.append(encoded[0])
        return list(dict.fromkeys(token_ids))

    def _build_sampling_params(self):
        yes_token_ids = self._collect_variant_token_ids(self.YES_VARIANTS)
        no_token_ids = self._collect_variant_token_ids(self.NO_VARIANTS)
        allowed_token_ids = yes_token_ids + no_token_ids
        if not yes_token_ids or not no_token_ids:
            raise ValueError(
                "p_true inferencer requires single-token Yes/No variants for the current tokenizer. "
                f"Got yes_token_ids={yes_token_ids}, no_token_ids={no_token_ids}"
            )

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            max_tokens=1,
            logprobs=len(allowed_token_ids),
            allowed_token_ids=allowed_token_ids,
        )
        return sampling_params, set(yes_token_ids), set(no_token_ids)

    def estimate_confidence(self, texts, outputs):
        prompts = []
        prompt_indices = []
        for output_idx, (text, output) in enumerate(zip(texts, outputs)):
            for sample_idx in range(self.config.num_generations):
                prompts.append(text + output.outputs[sample_idx].text + self.P_TRUE_INST)
                prompt_indices.append((output_idx, sample_idx))

        sampling_params, yes_token_ids, no_token_ids = self._build_sampling_params()
        with self.model.override_vllm_progress_desc("Estimating p_true"):
            ptrue_outputs = self.model.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        invalid_count = 0
        for (output_idx, sample_idx), ptrue_output in zip(prompt_indices, ptrue_outputs):
            logprob_dict = ptrue_output.outputs[0].logprobs[0] if ptrue_output.outputs[0].logprobs else {}
            yes_logprobs = []
            no_logprobs = []
            for token_id, candidate in logprob_dict.items():
                if token_id in yes_token_ids:
                    yes_logprobs.append(candidate.logprob)
                elif token_id in no_token_ids:
                    no_logprobs.append(candidate.logprob)
            yes_probability = self._extract_yes_no_probability(yes_logprobs, no_logprobs)
            if yes_probability is None:
                yes_probability = 0.5
                invalid_count += 1
            outputs[output_idx].outputs[sample_idx].text += f"<confidence> {yes_probability} </confidence>"

        total_outputs = self.config.num_generations * len(outputs)
        logging.info(
            "Config %s: estimated p_true confidence for %d/%d outputs; %d fell back to 0.5",
            self.config.name,
            total_outputs - invalid_count,
            total_outputs,
            invalid_count,
        )
        return outputs
