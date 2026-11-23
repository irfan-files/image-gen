from abc import ABC, abstractmethod
from PIL import Image


class BaseGenerator(ABC):
    def __init__(self, model_id: str, backend: str, cache_dir: str):
        self.model_id = model_id
        self.backend = backend
        self.cache_dir = cache_dir
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str, width: int, height: int,
                 seed: int, steps: int, guidance: float, scheduler: str) -> Image.Image:
        raise NotImplementedError

    def unload(self) -> None:
        self.loaded = False
