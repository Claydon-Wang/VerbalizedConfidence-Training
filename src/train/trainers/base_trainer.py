from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def validate_reward_specs(self, optimization_rewards, monitoring_rewards):
        return None

    def get_reward_names(self, optimization_rewards, monitoring_rewards):
        return list(optimization_rewards.keys()), list(monitoring_rewards)

    @abstractmethod
    def prepare_prompts(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def generate_completions(self, prompt_bundle):
        raise NotImplementedError

    @abstractmethod
    def compute_rewards(self, generation_outputs, inputs):
        raise NotImplementedError

    @abstractmethod
    def compute_advantages(self, rewards):
        raise NotImplementedError

    @abstractmethod
    def compute_policy_loss(self, per_token_logps, old_per_token_logps, advantages, completion_mask):
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, model, inputs):
        raise NotImplementedError
