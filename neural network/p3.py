
import numpy as np

layer_outputs=[[4.8,1.21,2.835],
               [8.9,-1.81,0.2],
               [1.41,1.051,0.026]]


exp_values=np.exp(layer_outputs)


print(np.sum(layer_outputs, axis=1))
'''
norm_values=exp_values/np.sum(exp_values)


print(norm_values)
print(sum(norm_values))'''