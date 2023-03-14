from typing import List, Tuple
from typing import Union, Any
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

from .Model import Model
from .ModelBuilderMLP import ModelBuilderMLP
from ..Json.JsonFileHander import JsonFileHandler
from .Optimizer.AdadeltaOptimizer import AdadeltaOptimizer
from .Optimizer.AdagradOptimizer import AdagradOptimizer
from .Optimizer.AdamaxOptimizer import AdamaxOptimizer
from .Optimizer.AdamOptimizer import AdamOptimizer
from .Optimizer.FTRLOptimizer import FtrlOptimizer
from .Optimizer.NadamOptimizer import NadamOptimizer
from .Optimizer.RMSpropOptimizer import RMSpropOptimizer
from .Optimizer.SGDOptimizer import SGDOptimizer
from .Optimizer.Optimizer import Optimizer

from typing import Union

class MLP(Model):

    def __init__(self, input_shape: Tuple[int, ...],
                    output_shape : Tuple[int, ...],
                    JSON_Handler : JsonFileHandler,
                    JSON_file : str,
                    Optimizer : Optimizer,                   
                    Model : ModelBuilderMLP,
                    epochs : int):
        
        self.input_shape = input_shape;
        self.output_shape = output_shape;

        self.MLP_hp = JSON_Handler.read_json_file(JSON_file);

        self.dense_1 = self.MLP_hp['dense_1'];
        self.output = self.MLP_hp['output'];
        self.activation_1 = self.MLP_hp['activation_1'];
        self.activation_output = self.MLP_hp['activation_output'];

        self.optimizer = Optimizer.get_optimizer(0.0000001);
        self.loss = self.MLP_hp['loss'];
        self.metrics = self.MLP_hp['metrics'];

        print(self.input_shape)
        print(self.output_shape)
        print(self.optimizer)

        print(self.dense_1)
        print(self.output)
        print(self.activation_1)
        print(self.activation_output)
        print(self.loss)
        print(self.metrics)

        self.model = Model.build_model(self.input_shape,
                                       self.dense_1,
                                       self.output,
                                       self.activation_1,
                                       self.activation_output);
        self.epochs = epochs;

    def compile_model(self):
        self.model.compile(optimizer = self.optimizer, loss = self.loss, 
                            metrics = [self.metrics])

    def fit_model(self):

        Hist_data = self.model.fit(self.input_shape, self.output_shape, epochs = self.epochs, verbose = True)

        return self.model, Hist_data
    
    def predict_model(self, Data) -> Union[None, Any]:
        return self.model.predict(Data)
    