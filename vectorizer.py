import torch
from torch import nn



class VectorDynamicTanh(nn.Module):
    def __init__(self, input_shape):
    
        super().__init__()
        
           
        self.alpha = nn.Parameter(torch.randn(input_shape))
       

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x



class HyperVectorDynamicTanh(nn.Module):
    def __init__(self):
    
        super().__init__()
       
           
    def forward(self, x, alpha):
        x = torch.tanh(alpha * x)
        return x


      
             	   
   

class LocalMappingUnit(nn.Module):
    def __init__(self,dim):
        super().__init__()
        
           
        self.token_vdyt = VectorDynamicTanh(dim)
      
      
             	   
    def forward(self, x):
    
        x = self.token_vdyt(x)    	
      
        
        return x
    	

class GlobalMappingUnit(nn.Module):
    def __init__(self,dim,num_tokens):
        super().__init__()
        
             
        self.state_vdyt = VectorDynamicTanh(dim)
        self.probe_vdyt = VectorDynamicTanh(dim)
        self.learner_vdyt = VectorDynamicTanh(dim * num_tokens) 
        self.readout_hvdyt = HyperVectorDynamicTanh()
              
                                      	   
    def forward(self, x):
    
       
        state = self.state_vdyt(x)
        probe = self.probe_vdyt(x)
        
        dim0 = state.shape[0]
        dim1 = state.shape[1]
        dim2 = state.shape[2]
        state = state.reshape([dim0,dim1*dim2])
        probe = probe.reshape([dim0,dim1*dim2])
        
        alpha = self.learner_vdyt(state)
       
        readout = self.readout_hvdyt(probe, alpha)
        
        

        readout = readout.reshape([dim0,dim1,dim2])

        return readout          

class VectorizerBlock(nn.Module):
    def __init__(self, d_model, num_tokens):
        super().__init__()
       
         
        self.local_mapping = LocalMappingUnit(d_model)
        self.global_mapping = GlobalMappingUnit(d_model, num_tokens)
        
    
        
        
        
    def forward(self, x):
                  
        residual = x
        
        x = self.global_mapping(x)
    
        x = x + residual
        
        residual = x
        
        x = self.local_mapping(x)
        
                                          
        out = x + residual
        
        
        return out



class Vectorizer(nn.Module):
    def __init__(self, d_model,num_tokens, num_layers):
        super().__init__()
        
        self.model = nn.Sequential(
            *[VectorizerBlock(d_model,num_tokens) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








