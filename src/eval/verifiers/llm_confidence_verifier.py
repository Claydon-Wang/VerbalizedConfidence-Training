import gc

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.eval.inferencers.base_inferencer import BaseInferencer


def llm_confidence_verifier(
    local_dataset,
    config,
    format_fn="confidence_format",
    **kwargs,
):
    label_dict = {"is_correct": []}
    is_correct = []
    generation_len = []
    confidence = []
    is_conf_legal = []
    n = config.num_generations

    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for answer_text in local_dataset[i]["answers"]:
            q_spec_ans.append(answer_text if answer_text else "I don't know")
        extracted_answers.append(q_spec_ans)

    sys_prompt = """
    You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers. 
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not. 
    Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
    """

    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(config.judge_model_name_or_path, trust_remote_code=True)
    for i in range(len(local_dataset)):
        for j in range(n):
            prompt = f"""
            Question: {local_dataset[i]["question"]}
            Ground Truth Answers: {local_dataset[i]["answer"]}
            Model Generated Answer: {extracted_answers[i][j]}
            """
            processed_prompt = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
            tokenized_prompt = tokenizer.apply_chat_template(processed_prompt, truncation=False, add_generation_prompt=True)
            prompts.append(tokenizer.decode(tokenized_prompt))

    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=config.judge_max_tokens)
    llm = LLM(
        model=config.judge_model_name_or_path,
        gpu_memory_utilization=config.judge_gpu_memory_utilization,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    responses = []
    for output in outputs:
        responses.append(1 if "yes" in output.outputs[0].text.lower() else 0)

    agg_responses = []
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i : i + n])

    for i in range(len(local_dataset)):
        correctness_list, generation_len_list, confidence_list, conf_legal_list = [], [], [], []
        generations = local_dataset[i]["generations"]
        confidences = local_dataset[i]["confidences"]
        for j, (generation_text, confidence_text) in enumerate(zip(generations, confidences)):
            actual_correctness = agg_responses[i][j]
            conf_format, conf_level = BaseInferencer.confidence_extractor(confidence_text)
            conf_legal_list.append(conf_format)
            generation_len_list.append(len(generation_text))
            confidence_list.append(conf_level)
            correctness_list.append(1 if actual_correctness == 1 else 0)

        is_correct.append(correctness_list)
        generation_len.append(generation_len_list)
        confidence.append(confidence_list)
        is_conf_legal.append(conf_legal_list)

    label_dict["is_correct"] = is_correct
    label_dict["generation_len"] = generation_len
    label_dict["confidence"] = confidence
    label_dict["is_conf_legal"] = is_conf_legal

    del llm
    gc.collect()
    return label_dict
