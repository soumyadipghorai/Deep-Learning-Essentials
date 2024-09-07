import numpy as np 

X= [0.5, 2.5]
y= [0.2, 0.9]

def f(x, w, b) : 
    return 1/(1+np.exp(-(w*x+b)))

def error(w, b) : 
    err = 0.0 
    for x, y in zip(X, y) : 
        fx = f(x, w, b)
        err += (fx-y)**2 
    
    return 0.5*err 

def grad_b(x, w, b, y) : 
    fx = f(x, w, b)
    return (fx-y)*fx*(1-fx)

def grad_w(x, w, b, y) : 
    fx = f(x, w, b)
    return (fx-y)*fx*(1-fx)*x 

def gradient_descent() : 
    w, b, eta, max_epochs = -2, -2, 1.0, 1000 

    for i in range(max_epochs) :
        dw, db = 0, 0 
        for x, y in zip(X, y) :
            dw += grad_w(x, w, b, y)
            dw += grad_b(x, w, b, y)

        w -= eta*dw
        b -= eta*db