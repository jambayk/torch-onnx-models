from torch_onnx_models.models.base import CausalLMModel


class Phi3CausalLMModel(CausalLMModel):
    def preprocess_weights(self, state_dict):
        # not fully sure if we should split the qkv and gate_up projections here or just create a model class with unsplit projections
        state_dict = super().preprocess_weights(state_dict)
        q_size = self.config.num_attention_heads * self.config.head_dim
        kv_size = self.config.num_key_value_heads * self.config.head_dim
        for key in list(state_dict.keys()):
            if "qkv_proj" in key:
                state_dict[key.replace("qkv_proj", "q_proj")] = state_dict[key][:q_size]
                state_dict[key.replace("qkv_proj", "k_proj")] = state_dict[key][q_size : q_size + kv_size]
                state_dict[key.replace("qkv_proj", "v_proj")] = state_dict[key][q_size + kv_size :]
                del state_dict[key]
            elif "gate_up_proj" in key:
                state_dict[key.replace("gate_up_proj", "gate_proj")] = state_dict[key][: self.config.intermediate_size]
                state_dict[key.replace("gate_up_proj", "up_proj")] = state_dict[key][self.config.intermediate_size :]
                del state_dict[key]
        return state_dict
