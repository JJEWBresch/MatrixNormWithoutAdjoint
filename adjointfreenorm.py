import numpy as np
import matplotlib.pyplot as plt
import scipy

'''
    smapling and help functions
'''
def samp_initial(d):

    for i in range(100):

        v = np.zeros(d)

        while np.linalg.norm(v) < 1e-4:
            v = np.random.randn(d)
        
        v = v/np.linalg.norm(v)

    return v

def samp_orthogonal(v,nor=1):

    d = np.size(v)

    x = np.random.randn(d)

    #x = x - (np.sum(v*x))/(np.sum(v*v))*v
    x = x - (np.sum(v*x))*v
    if nor == 1:
        x = x/np.linalg.norm(x)

    return x

def update_step(v, tau, x):
    s = tau * x
    u = v + s
    return u / np.linalg.norm(u)

def first_diff_A(a,b,tau):

    return -a * tau**2 + b * tau + a

def first_diff_AV(a,b,c,d,tau,sig):

    return b + sig*d - tau*(a + sig*c), c + tau*d - sig*(a + tau*b)

def second_diff_A(a,b,tau):

    return b - 2*tau*a

def second_diff_AV(a,b,c,d,tau,sig):

    q11 = -(a + sig*c)*(1 - 2*tau**2) - (b + sig*d)*3*tau
    q12 = d - tau*c - (b - tau*a)*sig
    q21 = d - sig*b - (c - sig*a)*tau
    q22 = -(a + tau*b)*(1 - 2*sig**2) - (c + tau*d)*3*sig

    # return q11, q22, q11*q22 - q12**2

    if q11 - 1e-10 < 0 and q11*q22 - q12**2 + 1e-10 > 0:
        return True
    if q11 + 1e-10 > 0 and q11*q22 - q21**2 + 1e-10 > 0:
        return True
    else:
        return False

def funcA(A, v):

    k = np.linalg.norm(np.dot(A, v))**2

    return k

def gen_a_A(A, v, x):

    av = np.dot(A,v)
    ax = np.dot(A,x)

    a = np.sum(av*ax)

    return a

def gen_b_A(A, v, x):

    av = np.dot(A,v)
    ax = np.dot(A,x)

    b = np.linalg.norm(ax)**2 - np.linalg.norm(av)**2

    return b

def stepsize_A(a, b, x=1):

    nx = np.linalg.norm(x)**2

    tau = np.sign(a) * (b/2/np.abs(a)/nx + np.sqrt(b**2/4/a**2/nx**2 + 1/nx))

    return tau

def funcAV(A, V, v, u):

    k = np.sum(u*np.dot(A,v)) - np.sum(v*np.dot(V,u))

    return k

def gen_a_AV(A, V, u, w, v, x):
    
    ax = np.dot(A,x)
    av = np.dot(A,v)
    vu = np.dot(V,u)
    vw = np.dot(V,w)

    a = np.sum(u * ax) - np.sum(x * vu) + np.sum(w * av) - np.sum(v * vw)

    return a

def gen_b_AV(A, V, u, w, v, x):
    
    ax = np.dot(A,x)
    av = np.dot(A,v)
    vu = np.dot(V,u)
    vw = np.dot(V,w)

    b = 2 * (np.sum(w * ax) - np.sum(x * vw) - np.sum(u * av) + np.sum(v * vu))

    return b

def stepsize_AV(a, b):
    if a > 0:
        tau = b/2/a + np.sqrt(b**2/4/a**2 + 1)
    if a < 0:
        tau = b/2/a - np.sqrt(b**2/4/a**2 + 1)
    
    return tau

def gen_ad_AV(A, V, u, w, v, x):

    ax = np.dot(A,x)
    av = np.dot(A,v)
    vu = np.dot(V,u)
    vw = np.dot(V,w)

    a = np.sum(u * av) - np.sum(vu * v)
    d = np.sum(w * ax) - np.sum(vw * x)

    return a, d

def gen_bc_AV(A, V, u, w, v, x):

    ax = np.dot(A,x)
    av = np.dot(A,v)
    vu = np.dot(V,u)
    vw = np.dot(V,w)

    b = np.sum(w * av) - np.sum(vw * v)
    c = np.sum(u * ax) - np.sum(vu * x)

    return b, c

def stepsize_AV_double(a, b, c, d):
    if a*b + c*d > 0:
        tau = -(a**2 - b**2 + c**2 - d**2) / 2 / (a*b + c*d) + np.sqrt((a**2 - b**2 + c**2 - d**2)**2 / 4 / (a*b + c*d)**2 + 1)
        sig = (c + tau*d) / (a + tau*b)
    else:
        tau = -(a**2 - b**2 + c**2 - d**2) / 2 / (a*b + c*d) - np.sqrt((a**2 - b**2 + c**2 - d**2)**2 / 4 / (a*b + c*d)**2 + 1)
        sig = (c + tau*d) / (a + tau*b)

    return tau, sig

'''
    main algorithms
'''


def MatFreeAdjNorm(A, iter, eps, nor=1, show=0):

    d,d = np.shape(A)

    # valsol = np.max(np.linalg.eigh(np.transpose(A)@A)[0])
    if d == 1:
        print('||A|| = ', np.linalg.norm(A))
        return A / np.linalg.norm(A), A / np.linalg.norm(A), np.linalg.norm(A), 0, 0, 0
    if 1001 > d > 10:
        valsol = scipy.sparse.linalg.eigs(np.transpose(np.conjugate(A))@A, k=1, which='LM')[0]
    elif d > 1000:
        valsol = 0
    else:
        valsol = np.max(np.linalg.eig(np.transpose(np.conjugate(A))@A)[0])

    funcval = []
    listtau = []
    listerror = []
    lista = []
    k = 1
    
    if show == 1:
        print('iter. \t| func-value \t| res.  \t| tau  \t\t| alpha \t| beta  \t| h^(1)(tau) \t| h^(2)(tau) \t| update \t| error')
        print('------------------------------------------------------------------------------------------------------------------------------------------------------')

    '''
        Initialisation
    '''
    v = samp_initial(d)
    funcval = np.append(funcval, funcA(A, v))
    optv = v

    if show == 1:
        print(0, '\t|', "%10.3e"%(funcA(A, v)), '\t|', "%10.3e"%(np.abs(funcA(A, v) - valsol)), '\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---")

    v1 = v.copy()
    x = samp_orthogonal(v.copy(),nor)
    a = gen_a_A(A, v, x)

    while np.abs(a) > eps and k < iter + 1:

        v1 = v.copy()

        b = gen_b_A(A, v, x)
        if nor == 0:
            b = gen_b_A(A, np.linalg.norm(x)**2 * v, x)
            # b = np.linalg.norm(np.dot(A, x.copy()))**2 - np.linalg.norm(x)**2*np.linalg.norm(np.dot(A, v.copy()))**2
        
        tau = stepsize_A(a, b)
        if nor == 0:
            tau = stepsize_A(a, b, x)

        v = update_step(v, tau, x)

        lista = np.append(lista, a)
        listtau = np.append(listtau, tau)
        listerror = np.append(listerror, np.linalg.norm(v - v1))
        funcval = np.append(funcval,funcA(A,v))
        funcopt = np.max(funcval)
        
        if funcval[k] == funcopt:
            optv = v.copy()
            up = 1

        if np.mod(k,1000) == 0 and show == 1:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(funcval[k] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(first_diff_A(a,b,tau)), '\t|', "%10.3e"%(second_diff_A(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v)))

        k += 1
        x = samp_orthogonal(v.copy(),nor)
        a = gen_a_A(A,v,x)
            
    if k == 1:
        print('A is orthogonal/unitar with ||A|| = ', "%10.7e"%np.sqrt(funcA(A, v)), '\t| a_0 = ', "%10.7e"%a)
    else:
        if show == 0:
            print('iter. \t| func-value \t| residuum  \t| sing-vec-error')
            print(k-1, '\t|', "%10.3e"%(funcval[k-1]), '\t|', "%10.3e"%(np.abs(funcval[k-1] - valsol)), '\t|', "%10.3e"%(np.abs(1 - funcA(A, v)/valsol)))
        print('||A|| = ', np.sqrt(funcA(A, v)))


    return v, optv, valsol, funcval, listtau, lista, listerror




def MatFreeAdjOpNorm(A, V=np.zeros(1), iter=10000, eps=1e-7, show=0):

    if np.all(V==0):
        V = np.zeros(np.shape(A.T))

    d,p = np.shape(A)
    if np.shape(A) != np.shape(V.T):
        print('dimensions does not fit!')

    valsol = np.sqrt(scipy.sparse.linalg.eigs(np.transpose(np.conjugate(A - V.T))@(A - V.T), k=1, which='LM')[0])

    funcval = []
    listtau = []
    listerror = []
    lista = []
    k = 1
    
    if show == 1:
        print('iter. \t| func-value \t| res.  \t| tau  \t\t| alpha \t| beta  \t| h^(1)(tau) \t| h^(2)(tau) \t| sample \t| error \t| sing-vec')
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    '''
        Initzilisation
    '''
    v = samp_initial(p)
    u = samp_initial(d)
    funcval = np.append(funcval, funcAV(A, V, v, u))
    
    if show == 1:
        print(0, '\t|', "%10.3e"%(funcval[0]), '\t|', "%10.3e"%(np.abs(funcval[0] - valsol)), '\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---")


    v1 = v.copy()
    u1 = u.copy()
    x = samp_orthogonal(v.copy())
    w = samp_orthogonal(u.copy())
    a = gen_a_AV(A, V, u, w, v, x)
    # a = np.sum(u*np.dot(A,x)) - np.sum(x*np.dot(V,u)) + np.sum(w*np.dot(A,v)) - np.sum(v*np.dot(V,w))
    tau = 0

    while np.abs(a) > eps and k < iter + 1:

        v1 = v.copy()
        u1 = u.copy()

        b = gen_b_AV(A, V, u, w, v, x)

        tau = stepsize_AV(a, b)

        v = update_step(v, tau, x)
        u = update_step(u, tau, w)

        lista = np.append(lista, a)
        listtau = np.append(listtau, tau)
        listerror = np.append(listerror, np.linalg.norm(v - v1) + np.linalg.norm(u - u1))
        funcval = np.append(funcval, funcAV(A, V, v, u))
        funcopt = np.max(funcval)
        up = 0
        
        if funcval[k] == funcopt:
            optv = v.copy()
            optu = u.copy()
            up = 1


        if np.mod(k,1000) == 0 and show == 1:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(funcval[k] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(first_diff_A(a,b,tau)), '\t|', "%10.3e"%(second_diff_A(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v) + np.linalg.norm(u1 - u)), '\t|', "%10.3e"%np.minimum(np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))), 2-np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v)))))

        k += 1
        x = samp_orthogonal(v.copy())
        w = samp_orthogonal(u.copy())
        a = gen_a_AV(A, V, u, w, v, x)

    if k == 1:
        optu = u.copy()
        optv = v.copy()
        print('A = V with ||A - V|| = ', "%10.7e"%np.abs(funcAV(A, V, v, u)), '\t| with a_0 = ', "%10.7e"%a)
    else:
        if show == 0:
            print('iter \t| func-value \t| residuum  \t| sing-vec-error')
            print(k-1, '\t|', "%10.3e"%(funcval[k-1]), '\t|', "%10.3e"%(np.abs(funcval[k-1] - valsol)), '\t|', "%10.3e"%np.minimum(np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))), 2-np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v)))))
        print('||A|| = ', funcAV(A, V, v, u))

    return u, v, valsol, optu, optv, funcval, listtau, lista, listerror

def MatFreeAdjOpNormDouble(A, V=np.zeros(1), iter=10000, eps=1e-7, show=0):

    if np.all(V==0):
        V = np.zeros(np.shape(A.T))

    d,p = np.shape(A)
    if np.shape(A) != np.shape(V.T):
        print('dimensions does not fit!')

    valsol = np.sqrt(scipy.sparse.linalg.eigs(np.transpose(np.conjugate(A - V.T))@(A - V.T), k=1, which='LM')[0])

    funcval = []
    listtau = []
    listerror = []
    listabcd = []
    listbc = []
    approx = 0
    listapprox1 = []
    listapprox2 = []
    listapprox2min = []
    listapprox3 = []
    k = 1

    if d <= 100 and p <= 100:
        approx = 1
        AVop = (A - V.T).T@(A - V.T)

    if show == 1:
        print('iter. \t| func-value \t| res.  \t| tau+sig  \t| alpha \t| beta \t\t| gamma \t| delta \t| q^(1)(tau,sig) \t| q^(2)(tau, sig) \t| sample \t| error \t| sing-vec')
        print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    '''
        Initzilisation
    '''
    v = samp_initial(p)
    u = samp_initial(d)
    funcval = np.append(funcval, funcAV(A, V, v, u))
    
    if show == 1:
        print(0, '\t|', "%10.3e"%(funcval[0]), '\t|', "%10.3e"%(np.abs(funcval[0] - valsol)), '\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t\t|', "---", '\t\t\t|', "---", '\t\t|', "---", '\t\t|', "---")

    x = samp_orthogonal(v.copy())
    w = samp_orthogonal(u.copy())
    b, c = gen_bc_AV(A, V, u, w, v, x)

    while np.abs(np.abs(b) + np.abs(c)) > eps and k < iter+1:

        a, d = gen_ad_AV(A, V, u, w, v, x)
        
        if approx == 1:
            listapprox1 = np.append(listapprox1, np.linalg.norm(AVop@v - np.linalg.norm((A - V.T)@v)*np.linalg.norm((A.T - V)@u)*v))

        v1 = v.copy()
        u1 = u.copy()

        if approx == 1:
            listapprox2 = np.append(listapprox2, np.linalg.norm(AVop@v - a**2*v))
            listapprox2min = np.append(listapprox2min, np.min(listapprox2))
            listapprox3 = np.append(listapprox3, np.linalg.norm(AVop@v))

        tau, sig = stepsize_AV_double(a, b, c, d)

        v = update_step(v, sig, x)
        u = update_step(u, tau, w)

        listabcd = np.append(listabcd, a*b + c*d)
        listbc = np.append(listbc, np.abs(b) + np.abs(c))
        listtau = np.append(listtau, tau)
        listerror = np.append(listerror, np.linalg.norm(v - v1) + np.linalg.norm(u - u1))
        funcval = np.append(funcval, np.abs(funcAV(A, V, v, u)))
        funcopt = np.max(funcval)
        up = 0
        
        if funcval[k] == funcopt:
            optu = u.copy()
            optv = v.copy()
            up = 1
        
        if (np.mod(k,1000) == 0 or 1 <= k <= 10) and show == 1:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(np.abs(funcval[k]) - valsol)), '\t|', "%10.3e"%(tau+sig), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|',  "%10.3e"%(c), '\t|',  "%10.3e"%(d), '\t|', "%10.3e"%(np.linalg.norm(first_diff_AV(a,b,c,d,tau,sig))), '\t\t|', "%10.3e"%(second_diff_AV(a,b,c,d,tau,sig)), '\t\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v) + np.linalg.norm(u1 - u)), '\t|', "%10.3e"%np.minimum(np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))), 2-np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v)))))

        k += 1
        x = samp_orthogonal(v.copy())
        w = samp_orthogonal(u.copy())

        b, c = gen_bc_AV(A, V, u, w, v, x)

    if k == 1:
        optu = u.copy()
        optv = v.copy()
        print('A = V with ||A - V|| = ', "%10.7e"%np.abs(funcAV(A, V, v, u)), '\t| with a_0 = ', "%10.7e"%a)
    else:
        if show == 0:
            print('iter. \t| func-value \t| residuum \t| sing-vec-error')
            print(k-1, '\t|', "%10.3e"%(funcval[k-1]), '\t|', "%10.3e"%(np.abs(np.abs(funcval[k-1]) - valsol)), '\t|', "%10.3e"%np.minimum(np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))), 2-np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v)))))
        print('||A|| = ', np.abs(funcAV(A, V, v, u)))

    return u, v, valsol, optu, optv, funcval, listtau, listabcd, listbc, listerror, listapprox1, listapprox2, listapprox2min, listapprox3
