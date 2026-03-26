import gc

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_llm_judge(config):
    tokenizer = AutoTokenizer.from_pretrained(config.judge_model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=config.judge_model_name_or_path,
        gpu_memory_utilization=config.judge_gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.judge_max_model_len,
    )
    return tokenizer, llm


def close_llm_judge(llm):
    del llm
    gc.collect()


def llm_answers_equivalent_batch(questions, answer_pairs, config, tokenizer=None, llm=None):
    if not answer_pairs:
        return []

    sys_prompt = """
    You are a judge that will be given a question and two candidate answers to that question.
    Determine whether the two answers express the same final answer.
    Ignore differences in wording, formatting, and explanation style.
    Your response should be a single word. 'YES' if they are equivalent and 'NO' if they are not.
    """

    owns_judge = tokenizer is None or llm is None
    if owns_judge:
        tokenizer, llm = build_llm_judge(config)

    prompts = []
    for question, (answer_a, answer_b) in zip(questions, answer_pairs):
        prompt = f"""
        Question: {question}
        Answer A: {answer_a}
        Answer B: {answer_b}
        """
        processed_prompt = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        tokenized_prompt = tokenizer.apply_chat_template(processed_prompt, truncation=False, add_generation_prompt=True)
        prompts.append(tokenizer.decode(tokenized_prompt))

    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=config.judge_max_tokens)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    responses = [1 if "yes" in output.outputs[0].text.lower() else 0 for output in outputs]

    if owns_judge:
        close_llm_judge(llm)
    return responses


def llm_verifier(
    local_dataset,
    config,
    format_fn="confidence_format",
    **kwargs,
):
    label_dict = {"is_correct": []}
    is_correct = []
    n = config.num_generations

    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for answer_text in local_dataset[i]["predictions"]:
            q_spec_ans.append(answer_text if answer_text else "I don't know")
        extracted_answers.append(q_spec_ans)

    sys_prompt = """
    You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers. 
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not. 
    Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
    """

    prompts = []
    tokenizer, llm = build_llm_judge(config)
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
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    responses = []
    for output in outputs:
        responses.append(1 if "yes" in output.outputs[0].text.lower() else 0)

    agg_responses = []
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i : i + n])

    for i in range(len(local_dataset)):
        correctness_list = []
        for j in range(n):
            actual_correctness = agg_responses[i][j]
            correctness_list.append(1 if actual_correctness == 1 else 0)

        is_correct.append(correctness_list)

    label_dict["is_correct"] = is_correct

    close_llm_judge(llm)
    return label_dict
