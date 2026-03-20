import logging

import numpy as np

from src.eval.inferencers.base_inferencer import BaseInferencer


class AnswerSequenceLikelihoodInferencer(BaseInferencer):
    def generate_outputs(self, texts):
        return self.model.generate(texts, logprobs=1)

    def estimate_confidence(self, texts, outputs):
        invalid_count = 0
        for output in outputs:
            for i in range(self.config.num_generations):
                picked = output.outputs[i]
                len_gen = len(picked.logprobs)
                tokens = []
                probs = []
                for j in range(len_gen):
                    lp_val = next(iter(picked.logprobs[j].values())).logprob
                    token = next(iter(picked.logprobs[j].values())).decoded_token
                    probs.append(np.exp(lp_val))
                    tokens.append(token)
                answer_indices = [idx for idx, token in enumerate(tokens) if token == "answer"]
                end_index = answer_indices[-1] if len(answer_indices) >= 1 else None
                start_index = answer_indices[-2] if len(answer_indices) >= 2 else None
                if start_index is None or end_index is None or end_index - start_index >= 30:
                    output.outputs[i].text = output.outputs[i].text + "<confidence> 0.5 </confidence>"
                    invalid_count += 1
                else:
                    # selected_probs = probs[start_index:end_index]
                    # avg_prob = sum(selected_probs) / len(selected_probs)
                    # output.outputs[i].text = output.outputs[i].text + f"<confidence> {avg_prob} </confidence>"
                    selected_probs = probs[start_index:end_index]

                    # 防止 log(0)
                    eps = 1e-6
                    selected_probs = [min(max(p, eps), 1.0) for p in selected_probs]

                    log_probs = [max(np.log(p), -20) for p in selected_probs]
                    avg_log_prob = sum(log_probs) / len(log_probs)

                    geom_mean = np.exp(avg_log_prob)   # 这就是 TPU 的 confidence

                    output.outputs[i].text = (
                        output.outputs[i].text + f"<confidence> {geom_mean} </confidence>"
                    )

        total_outputs = self.config.num_generations * len(outputs)
        logging.info(
            "Config %s: defaulted confidence to 0.5 for %d/%d outputs due to invalid answer-prob spans",
            self.config.name,
            invalid_count,
            total_outputs,
        )
        return outputs
