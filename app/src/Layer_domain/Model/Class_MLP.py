from typing import List, Tuple
from typing import Union, Any
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

from .Class_Model import Model
from .Class_MLPModelBuilder import MLPModelBuilder
from ..Class_JsonFileHander import JsonFileHandler
from ..Model.Optimizer.Class_AdadeltaOptimizer import AdadeltaOptimizer
from ..Model.Optimizer.Class_AdagradOptimizer import AdagradOptimizer
from ..Model.Optimizer.Class_AdamaxOptimizer import AdamaxOptimizer
from ..Model.Optimizer.Class_AdamOptimizer import AdamOptimizer
from ..Model.Optimizer.Class_FTRLOptimizer import FtrlOptimizer
from ..Model.Optimizer.Class_NadamOptimizer import NadamOptimizer
from ..Model.Optimizer.Class_RMSpropOptimizer import RMSpropOptimizer
from ..Model.Optimizer.Class_SGDOptimizer import SGDOptimizer

from typing import Union

class MLP(Model):

    def __init__(self, input_shape: Tuple[int, ...],
                    output_shape : Tuple[int, ...],
                    JSON_Handler : JsonFileHandler,
                    JSON_file : str,
                    Optimizer : Union[AdadeltaOptimizer, 
                                      AdagradOptimizer,
                                      AdamaxOptimizer,
                                      AdamOptimizer,
                                      FtrlOptimizer,
                                      NadamOptimizer,
                                      RMSpropOptimizer,
                                      SGDOptimizer],                   
                    Model : MLPModelBuilder,
                    epochs : int):
        
        self.input_shape = input_shape;
        self.output_shape = output_shape;

        self.MLP_hp = JSON_Handler.read_json_file(JSON_file)

        self.dense_1 = self.MLP_hp['dense_1'];
        self.output = self.MLP_hp['output'];
        self.activation_1 = self.MLP_hp['activation_1'];
        self.activation_output = self.MLP_hp['activation_output'];

        self.optimizer = Optimizer.get_optimizer(0.0000001)
        self.loss = self.MLP_hp['loss'];
        self.metrics = self.MLP_hp['metrics'];

        self.model = Model.build_model(self.input_shape,
                                       self.dense_1,
                                       self.output,
                                       self.activation_1,
                                       self.activation_output)
        self.epochs = epochs;

    def compile_model(self):
        self.model.compile(optimizer = self.optimizer, loss = self.loss, 
                            metrics = self.metrics)

    def fit(self):
        return self.model.fit(self.input_shape, self.output_shape, 
                                epochs = self.epochs, verbose = True)
    
    def predict(self, Data) -> Union[None, Any]:
        return self.model.predict(Data)
    