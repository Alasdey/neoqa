from data_gen.llm.wrapper.models.claude_wrapper import ClaudeWrapper
from data_gen.llm.wrapper.models.gpt_wrapper import GPTWrapper
from data_gen.llm.wrapper.models.vllm_wrapper import VLLMWrapper
from data_gen.llm.wrapper.models.openrouter_wrapper import OPENROUTERWrapper


def get_llm(model_name, temperature: float = 0.0, max_tokens: int=512):
    if model_name == 'claude-35':
        return ClaudeWrapper(model_version='3.5', temperature=temperature, max_tokens=max_tokens)
    elif model_name == 'gpt4-turbo':
        max_tokens = min((max_tokens, 40960))
        return GPTWrapper('gpt-4-turbo-2024-04-09', temperature=temperature, max_tokens=max_tokens)
    elif model_name == 'gpt4-o':
        max_tokens = min((max_tokens, 40960))
        return GPTWrapper('gpt-4o-2024-11-20', temperature=temperature, max_tokens=max_tokens)
    elif "vllm/" in model_name:
        max_tokens = min((max_tokens, 40960))
        return VLLMWrapper(model_name[5:], temperature=temperature, max_tokens=max_tokens)
    elif "openrouter/" in model_name:
        max_tokens = min((max_tokens, 40960))
        return OPENROUTERWrapper(model_name[11:], temperature=temperature, max_tokens=max_tokens)
    else:
        raise NotImplementedError
