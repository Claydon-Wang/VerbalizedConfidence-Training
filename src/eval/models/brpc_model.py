import json
import os
import shutil

from src.eval.models.base_model import BaseModel


class BRPCModel(BaseModel):
    def __init__(self, config):
        self.stripped_model_cache_root = os.path.join("/tmp", "rlcr_vllm_stripped")
        super().__init__(config)

    @staticmethod
    def _has_brpc_probe_weights(model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            return False

        safetensor_files = [name for name in os.listdir(model_dir) if name.endswith(".safetensors")]
        for file_name in safetensor_files:
            from safetensors import safe_open

            with safe_open(os.path.join(model_dir, file_name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("brpc_probe."):
                        return True

        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            import torch

            state_dict = torch.load(bin_path, map_location="cpu")
            return any(key.startswith("brpc_probe.") for key in state_dict)
        return False

    def _build_stripped_checkpoint(self, model_dir: str) -> str:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch

        os.makedirs(self.stripped_model_cache_root, exist_ok=True)
        stripped_dir = os.path.join(self.stripped_model_cache_root, os.path.basename(os.path.abspath(model_dir)))
        marker_path = os.path.join(stripped_dir, ".brpc_stripped")

        if os.path.isfile(marker_path):
            return stripped_dir

        os.makedirs(stripped_dir, exist_ok=True)
        for file_name in os.listdir(model_dir):
            src = os.path.join(model_dir, file_name)
            dst = os.path.join(stripped_dir, file_name)
            if file_name.endswith(".safetensors") or file_name == "pytorch_model.bin":
                continue
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        safetensor_files = [name for name in os.listdir(model_dir) if name.endswith(".safetensors")]
        for file_name in safetensor_files:
            src = os.path.join(model_dir, file_name)
            dst = os.path.join(stripped_dir, file_name)
            tensors = {}
            metadata = None
            with safe_open(src, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for key in f.keys():
                    if key.startswith("brpc_probe."):
                        continue
                    tensors[key] = f.get_tensor(key)
            save_file(tensors, dst, metadata=metadata)

        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            state_dict = {key: value for key, value in state_dict.items() if not key.startswith("brpc_probe.")}
            torch.save(state_dict, os.path.join(stripped_dir, "pytorch_model.bin"))

        index_files = [name for name in os.listdir(model_dir) if name.endswith(".index.json")]
        for file_name in index_files:
            src = os.path.join(model_dir, file_name)
            dst = os.path.join(stripped_dir, file_name)
            with open(src, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            if "weight_map" in index_data:
                index_data["weight_map"] = {
                    key: value for key, value in index_data["weight_map"].items() if not key.startswith("brpc_probe.")
                }
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(index_data, f)

        with open(marker_path, "w", encoding="utf-8") as f:
            f.write("ok\n")
        return stripped_dir

    def _resolve_generation_model_path(self, model_name_or_path: str) -> str:
        if not os.path.isdir(model_name_or_path):
            return model_name_or_path
        if not self._has_brpc_probe_weights(model_name_or_path):
            return model_name_or_path
        return self._build_stripped_checkpoint(model_name_or_path)

