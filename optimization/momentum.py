import numpy as np 
from gradient_descent import *

def stochastic_gradient_descent() : 
    w, b, eta, max_epochs = -2, -2, 1.0, 1000 

    for i in range(max_epochs) :
        dw, db = 0, 0 
        for x, y in zip(X, y) :
            dw += grad_w(x, w, b, y)
            dw += grad_b(x, w, b, y)

            w -= eta*dw
            b -= eta*db

def mini_batch_gradient_descent() : 
    w, b, eta, max_epochs = -2, -2, 1.0, 1000 
    min_batch_size = 25

    for i in range(max_epochs) :
        dw, db, num_points_seen = 0, 0, 0
        for x, y in zip(X, y) :
            dw += grad_w(x, w, b, y)
            dw += grad_b(x, w, b, y) 
            num_points_seen += 1
            if num_points_seen%min_batch_size == 0 :
                w -= eta*dw
                b -= eta*db
                dw, db = 0, 0

def do_mgd(max_epochs) : 
    w, b, eta = -2, -2, 1.0 
    prev_vw, prev_vb, beta = 0, 0, 0.9 

    for i in range(max_epochs) : 
        dw, db = 0, 0 
        for x, y in zip(X, y) : 
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        vw = beta * prev_vw + eta*dw 
        vb = beta * prev_vb + eta*db 
        w -= vw 
        b -= vb 
        prev_vw = vw
        prev_vb = vb 

def do_nag(max_epochs) : 
    w, b, eta = -2, -2, 1.0 
    prev_vw, prev_vb, beta = 0, 0, 0.9 

    for i in range(max_epochs) : 
        dw, db = 0, 0 
        v_w = beta * prev_vw
        v_b = beta * prev_vb
        for x, y in zip(X, y) : 
            dw += grad_w(x, w - v_w, b - v_b, y)
            db += grad_b(x, w - v_w, b - v_b, y)

        vw = beta * prev_vw + eta*dw 
        vb = beta * prev_vb + eta*db 
        w -= vw 
        b -= vb 
        prev_vw = vw
        prev_vb = vb 

def do_line_search_gradient_descent(max_epochs) : 
    w, b, etas = -2, -2, [0.1, 0.5, 1, 2, 10]

    for i in range(max_epochs) :
        dw, db = 0, 0 
        for x, y in zip(X, y) :
            dw += grad_w(x, w, b, y)
            dw += grad_b(x, w, b, y)

        min_error = 10000 
        for eta in etas : 
            temp_w = w - eta * dw 
            temp_b = b - eta * db
            if error(temp_w, temp_b) < min_error : 
                best_w = temp_w
                best_b = temp_b
                min_error = error(best_w, best_b)

        w = best_w
        b = best_b

def do_adagrad(max_epochs) : 
    w, b, eta = -2, -2, 1.0 
    v_w, v_b, eps = 0, 0, 1e-8

    for i in range(max_epochs) : 
        dw, db = 0, 0  
        for x, y in zip(X, y) : 
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        v_w = v_w + dw**2 
        v_b = v_b + db**2 
        w -= eta*dw/(np.sqrt(v_w)+eps)
        b -= eta*db/(np.sqrt(v_b)+eps)

def do_rmsprop(max_epochs) : 
    w, b, eta = -4, -4, 1.0 
    beta = 0.5
    v_w, v_b, eps = 0, 0, 1e-4

    for i in range(max_epochs) : 
        dw, db = 0, 0  
        for x, y in zip(X, y) : 
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        v_w = v_w*beta + (1-beta)*dw**2 
        v_b = v_b*beta + (1-beta)*db**2
        w -= eta*dw/(np.sqrt(v_w)+eps)
        b -= eta*db/(np.sqrt(v_b)+eps)

def do_adadelta(max_epochs) : 
    w, b = -4, -4
    beta = 0.99
    v_w, v_b, eps = 0, 0, 1e-4
    u_w, u_b = 0, 0

    for i in range(max_epochs) : 
        dw, db = 0, 0  
        for x, y in zip(X, y) : 
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        v_w = v_w*beta + (1-beta)*dw**2 
        v_b = v_b*beta + (1-beta)*db**2
        delta_w = dw*np.sqrt(u_w+eps)/(np.sqrt(v_w+eps))
        delta_b = dw*np.sqrt(u_b+eps)/(np.sqrt(v_b+eps))
        u_w = beta*u_w + (1-beta)*delta_w**2
        u_b = beta*u_b + (1-beta)*delta_b**2
        w -= delta_w
        b -= delta_b

def do_adam_sgd(max_epochs) : 
    w, b, eta = -4, -4, 0.1
    beta1, beta2 = 0.9, 0.999
    m_w, m_b, v_w, v_b = 0, 0, 0, 0

    for i in range(max_epochs) : 
        dw, db, eps = 0, 0, 1e-10
        for x, y in zip(X, y) : 
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        m_w = beta1*m_w + (1-beta1)*dw
        m_b = beta1*m_b + (1-beta1)*db
        v_w = beta2*v_w + (1-beta2)*dw**2
        v_b = beta2*v_b + (1-beta2)*db**2

        m_w_hat = m_w/(1-np.power(beta1, i+1))
        m_b_hat = m_b/(1-np.power(beta1, i+1))
        v_w_hat = v_w/(1-np.power(beta2, i+1))
        v_b_hat = v_b/(1-np.power(beta2, i+1))

        w -= eta*m_w_hat/(np.sqrt(v_w_hat)+eps)
        b -= eta*m_b_hat/(np.sqrt(v_b_hat)+eps)

def do_nadam_sgd(max_epochs) : 
    w, b, eta = -4, -4, 0.1
    beta1, beta2 = 0.9, 0.99
    m_w_hat, m_b_hat, v_w_hat, v_b_hat = 0, 0, 0, 0

    for i in range(max_epochs) : 
        dw, db, eps = 0, 0, 1e-10
        for x, y in zip(X, y) : 
            dw += grad_w(x, w, b, y)
            db += grad_b(x, w, b, y)

        m_w = beta1*m_w + (1-beta1)*dw
        m_b = beta1*m_b + (1-beta1)*db
        v_w = beta2*v_w + (1-beta2)*dw**2
        v_b = beta2*v_b + (1-beta2)*db**2

        m_w_hat = m_w/(1-beta1**(i+1))
        m_b_hat = m_b/(1-beta1**(i+1))
        v_w_hat = v_w/(1-beta2**(i+1))
        v_b_hat = v_b/(1-beta2**(i+1))

        w -= (eta/np.sqrt(v_w_hat+eps))*(beta1*m_w_hat+(1-beta1)*dw/(1-beta1**(1+1)))
        b -= (eta/np.sqrt(v_b_hat+eps))*(beta1*m_b_hat+(1-beta1)*db/(1-beta1**(1+1)))

def cyclic_lr(iteration, max_lr, base_lr, step_size) : 
    cycle  = np.floor(1+iteration/(2*step_size))
    x = np.abs(iteration/step_size - 2*cycle+1)
    lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))
    return lr 

def do_gradient_descent_clr(max_epochs) : 
    w, b = -2, 0.0001 
    for i in range(max_epochs) : 
        dw, db = 0, 0
        dw = grad_w(w, b)
        db = grad_b(w, b)

        w -= cyclic_lr(i, max_lr=0.1, base_lr=0.001, step_size=30)*dw 
        b -= cyclic_lr(i, max_lr=0.1, base_lr=0.001, step_size=30)*db
 