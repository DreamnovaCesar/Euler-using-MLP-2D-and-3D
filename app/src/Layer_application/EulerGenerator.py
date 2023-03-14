from abc import ABC
from abc import abstractmethod

from .Decorators.Timer import Timer
from .Decorators.Singleton import Singleton

@Singleton.singleton
class EulerGenerator(ABC):

    @Timer.timer
    @abstractmethod
    def generate_euler_samples_random():
        pass
    
    @Timer.timer
    @abstractmethod
    def generate_euler_samples_settings():
        pass