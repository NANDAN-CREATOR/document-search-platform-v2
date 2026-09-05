import logging
from pathlib import Path
from typing import Dict
import yaml
from config.settings import settings

logger = logging.getLogger(__name__)
_prompt_cache: Dict[str, Dict] = {}

def load_prompts(prompt_file: str = "system_prompt.yaml") -> Dict:
    global _prompt_cache
    if prompt_file in _prompt_cache:
        return _prompt_cache[prompt_file]
    prompt_path = Path(settings.prompts_dir) / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r") as f:
        prompts = yaml.safe_load(f)
    _prompt_cache[prompt_file] = prompts
    logger.info(f"Loaded prompts from: {prompt_path}")
    return prompts

def get_prompt(key: str, prompt_file: str = "system_prompt.yaml", **kwargs) -> str:
    prompts = load_prompts(prompt_file)
    if key not in prompts:
        raise KeyError(f"Prompt key '{key}' not found in {prompt_file}")
    template = prompts[key]
    return template.format(**kwargs) if kwargs else template
