from lxmls.readers import galton
from matplotlib import pyplot as plt
import numpy as np

galton_data = galton.load()

# Get data.
use_bias = False #True
y = galton_data[:,0] 
if use_bias:
    x = np.vstack( [galton_data[:,1], np.ones(galton_data.shape[0])] )  
else:
    x = np.vstack( [galton_data[:,1], np.zeros(galton_data.shape[0])] )
    
# derivative of the error function e
def get_e_dev(w, x, y): # y, x, 
    error_i = np.matmul(w, x) - y
    derro_dw = np.matmul(2*x, error_i) / len(y)
    # print(derro_dw, np.multiply(error_i,error_i).sum())
    return derro_dw

# Initialize w.
w = np.array([1,0])

# Initialize w.
w = np.array([0.5,50.0])

def grad_desc(start_w, eps, prec, x,y): #gradient=get_e_dev
    '''
    runs the gradient descent algorithm and returns the list of estimates
    example of use grad_desc(start_x=3.9, eps=0.01, prec=0.00001)
    '''
    w_new = start_w
    w_old = start_w + prec * 2
    res = [w_new]
    mses = []
    while abs(w_old-w_new).sum() > prec:
        w_old = w_new
        w_new = w_old - eps * get_e_dev(w_new, x, y)
        res.append(w_new)
        
        error_i = np.matmul(w_new, x) - y
        mse = np.multiply(error_i, error_i).mean()
        mses.append(mse)
        print(w_new, mse)
    return np.array(res), np.array(mses)

res, mses = grad_desc(w, 0.0002, 0.00001, x, y)
w_sgd = res[-1]
w_sgd
#res

plt.plot(res[:, 0], res[:, 1], '*')
plt.show()
