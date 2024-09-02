import numpy as np
import matplotlib.pyplot as plt
import scipy

'''
    smapling functions
'''
def sampv0(d):

    for i in range(100):

        v = np.zeros(d)

        while np.linalg.norm(v) < 1e-4:
            v = np.random.randn(d)
        
        v = v/np.linalg.norm(v)

    return v

def sampx(v,nor):

    d = np.size(v)

    x = np.random.randn(d)

    #x = x - (np.sum(v*x))/(np.sum(v*v))*v
    x = x - (np.sum(v*x))*v
    if nor == 1:
        x = x/np.linalg.norm(x)

    return x

'''
    help functions
'''

def onediff(a,b,tau):

    return -a*tau**2 + b*tau + a

def secdiff(a,b,tau):

    return b - 2*tau*a

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

def update_v(v, tau, x):

    v_up = v + tau*x
    
    return v_up/np.linalg.norm(v_up)

def stepsize(a, b, x=1):

    nx = np.linalg.norm(x)**2

    tau = np.sign(a) * (b/2/np.abs(a)/nx + np.sqrt(b**2/4/a**2/nx**2 + 1/nx))

    return tau

def funcAV(A, V, v, u):

    k = np.sum(u*np.dot(A,v)) - np.sum(v*np.dot(V,u))

    return k

'''
    main algorithm
'''
def MatFreeAdjNormEntire(A, iter, br, eps, nor):

    d,d = np.shape(A)

    # valsol = np.max(np.linalg.eigh(np.transpose(A)@A)[0])
    valsol = scipy.sparse.linalg.eigs(np.transpose(np.conjugate(A))@A, k=1, which='LM')[0]

    funcval = []
    listtau = []
    listerror = []
    lista = []

    '''
        Initialisation
    '''
    v = sampv0(d)
    funcval = np.append(funcval, funcA(A, v))
    optv = v

    print('iter. \t| func-value \t| res.  \t| tau  \t\t| alpha \t| beta  \t| h^(1)(tau) \t| h^(2)(tau) \t| sample \t| error')
    print('------------------------------------------------------------------------------------------------------------------------------------------------------')

    for k in range(iter):

        a = 0
        b = 0
        tau = 0
        if np.mod(k, 1000) == 0:
            dummy = 0

        x = np.random.randn(d)
        smap = 0

        while a == 0 and b <= 0: # or tau == 0: # or b >= 0:

            v1 = v.copy()
            x = sampx(v.copy(),nor)
            smap = smap + 1

            '''
                respampling if it's necessary
            '''

            a = np.sum(np.dot(A,v.copy())*np.dot(A,x.copy()))
            if k == 0 and np.abs(a) < 1e-14:
                print('A is orthogonal and ||A|| = ', np.linalg.norm(np.dot(A,v.copy())), 'a_0 = ', a)
                return v, optv, funcval, listtau, lista, listerror

            b = np.linalg.norm(np.dot(A, x.copy()))**2 - np.linalg.norm(np.dot(A, v.copy()))**2
            if nor == 0:
                b = np.linalg.norm(np.dot(A, x.copy()))**2 - np.linalg.norm(x)**2*np.linalg.norm(np.dot(A, v.copy()))**2
            
            if a > 0:
                tau = b/2/a + np.sqrt(b**2/4/a**2 + 1)
                vg = v + tau*x

            if a < 0:
                '''
                    setting a => 0
                '''
                lista = np.append(lista, a)
                tau = b/2/a - np.sqrt(b**2/4/a**2 + 1)
                vg = v + tau*x
            if a == 0:
                if b > 0:
                    vg = x.copy()

        if nor == 0:
            tau = b/2/a/np.linalg.norm(x)**2 + np.sqrt(b**2/4/a**2/np.linalg.norm(x)**4 + 1/np.linalg.norm(x)**2)

        listtau = np.append(listtau, tau)

        v = vg.copy()/np.linalg.norm(vg.copy())

        listerror = np.append(listerror, np.linalg.norm(v - v1))
        funcval = np.append(funcval, funcA(A,v))
        funcopt = np.max(funcval)
        up = 0
        if funcval[k] == funcopt:
            optv = v
            up = 1

        if np.mod(k,1000) == 0:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(funcval[k] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', smap, '\t|', dummy, '\t|', "%10.3e"%(np.linalg.norm(v1 - v)))
            # plt.show()
            plt.savefig('circle-2-%s.pdf' % k)
        if br == 1:
            #if k > 100 and np.abs(funcval[k-2] - funcval[k-1]) < 10**(-eps):
            #    break
            if k > 100 and tau < 10**(-eps):
                break


    return v, optv, funcval, listtau, lista, listerror


def MatFreeAdjNorm(A, iter, eps, nor=1):

    d,d = np.shape(A)

    # valsol = np.max(np.linalg.eigh(np.transpose(A)@A)[0])
    valsol = scipy.sparse.linalg.eigs(np.transpose(np.conjugate(A))@A, k=1, which='LM')[0]

    funcval = []
    listtau = []
    listerror = []
    lista = []
    k = 1

    print('iter. \t| func-value \t| res.  \t| tau  \t\t| alpha \t| beta  \t| h^(1)(tau) \t| h^(2)(tau) \t| update \t| error')
    print('------------------------------------------------------------------------------------------------------------------------------------------------------')

    '''
        Initialisation
    '''
    v = sampv0(d)
    funcval = np.append(funcval, funcA(A, v))
    optv = v

    print(0, '\t|', "%10.3e"%(funcA(A, v)), '\t|', "%10.3e"%(np.abs(funcA(A, v) - valsol)), '\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---")

    v1 = v.copy()
    x = sampx(v.copy(),nor)
    a = gen_a_A(A, v, x)

    while np.abs(a) > eps:

        v1 = v.copy()

        b = gen_b_A(A, v, x)
        if nor == 0:
            b = gen_b_A(A, np.linalg.norm(x)**2 * v, x)
            # b = np.linalg.norm(np.dot(A, x.copy()))**2 - np.linalg.norm(x)**2*np.linalg.norm(np.dot(A, v.copy()))**2
        
        tau = stepsize(a, b)
        if nor == 0:
            tau = stepsize(a, b, x)

        v = update_v(v, tau, x)

        lista = np.append(lista, a)
        listtau = np.append(listtau, tau)
        listerror = np.append(listerror, np.linalg.norm(v - v1))
        funcval = np.append(funcval,funcA(A,v))
        funcopt = np.max(funcval)
        
        if funcval[k] == funcopt:
            optv = v.copy()
            up = 1

        if np.mod(k,1000) == 0:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(funcval[k] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v)))

        k += 1
        x = sampx(v.copy(),nor)
        a = gen_a_A(A,v,x)
            
    if k == 1:
        print('A is orthogonal/unitar with ||A|| = ', funcA(A, v), 'a_0 = ', a)
    else:
        print(k-1, '\t|', "%10.3e"%(funcval[k-1]), '\t|', "%10.3e"%(np.abs(funcval[k-1] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v)))
        print('||A|| = ', funcA(A, v))


    return v, optv, funcval, listtau, lista, listerror


def MatFreeAdjOpNorm(A, V, iter, eps=1e-7, nor=1):

    d,p = np.shape(A)
    if np.shape(A) != np.shape(V.T):
        print('dimensions does not fit!')

    # valsol = np.max(np.linalg.eigh(np.transpose(A - V.T)@(A - V.T))[0])
    valsol = scipy.sparse.linalg.eigs(np.transpose(np.conjugate(A - V.T))@(A - V.T), k=1, which='LM')[0]

    funcval = []
    listtau = []
    listerror = []
    lista = []
    k = 1

    print('iter. \t| func-value \t| res.  \t| tau  \t\t| alpha \t| beta  \t| h^(1)(tau) \t| h^(2)(tau) \t| sample \t| error \t| sing-vec')
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    '''
        Initzilisation
    '''
    v = sampv0(p)
    u = sampv0(d)
    funcval = np.append(funcval, funcAV(A, V, v, u)**2)
    
    print(0, '\t|', "%10.3e"%(funcAV(A, V, v, u)**2), '\t|', "%10.3e"%(np.abs(funcAV(A, V, v, u)**2 - valsol)), '\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---", '\t\t|', "---")


    v1 = v.copy()
    u1 = u.copy()
    x = sampx(v.copy(),nor=1)
    w = sampx(u.copy(),nor=1)
    a = np.sum(u*np.dot(A,x)) - np.sum(x*np.dot(V,u)) + np.sum(w*np.dot(A,v)) - np.sum(v*np.dot(V,w))

    while np.abs(a) > eps:

        v1 = v.copy()
        u1 = u.copy()

        b = 2*(np.sum(w*np.dot(A,x)) - np.sum(x*np.dot(V,w)) - np.sum(u*np.dot(A,v)) + np.sum(v*np.dot(V,u)))
            
        if a > 0:
                tau = b/2/a + np.sqrt(b**2/4/a**2 + 1)
                vg = v + tau*x
                ug = u + tau*w

        if a < 0:
                tau = b/2/a - np.sqrt(b**2/4/a**2 + 1)
                vg = v + tau*x
                ug = u + tau*w

        v = vg.copy()/np.linalg.norm(vg.copy())
        u = ug.copy()/np.linalg.norm(ug.copy())

        lista = np.append(lista, a)
        listtau = np.append(listtau, tau)
        listerror = np.append(listerror, np.linalg.norm(v - v1) + np.linalg.norm(u - u1))
        funcval = np.append(funcval, funcAV(A, V, v, u)**2)
        funcopt = np.max(funcval)
        up = 0
        
        if funcval[k] == funcopt:
            optv = v.copy()
            optu = u.copy()
            up = 1


        if np.mod(k,1000) == 0:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(funcval[k] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v) + np.linalg.norm(u1 - u)), '\t|', "%10.3e"%np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))))

        k += 1
        x = sampx(v.copy(),nor=1)
        w = sampx(u.copy(),nor=1)
        a = np.sum(u*np.dot(A,x)) - np.sum(x*np.dot(V,u)) + np.sum(w*np.dot(A,v)) - np.sum(v*np.dot(V,w))

    print(k-1, '\t|', "%10.3e"%(funcval[k-1]), '\t|', "%10.3e"%(np.abs(funcval[k-1] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v) + np.linalg.norm(u1 - u)), '\t|', "%10.3e"%np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))))

    if k == 1:
        print('A is orthogonal/unitar with ||A||^2 = ', funcAV(A, V, v, u)**2, 'a_0 = ', a)
    else:
        print(k-1, '\t|', "%10.3e"%(funcval[k-1]), '\t|', "%10.3e"%(np.abs(funcval[k-1] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', up, '\t\t|', "%10.3e"%(np.linalg.norm(v1 - v) + np.linalg.norm(u1 - u)), '\t|', "%10.3e"%np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))))
        print('||A||^2 = ', funcAV(A, V, v, u)**2)

    return v, optv, funcval, listtau, lista, listerror


def MatFreeAdjOpNormEntire(A, V, iter, br, eps, nor):

    d,p = np.shape(A)
    if np.shape(A) != np.shape(V.T):
        print('dimensions does not fit!')

    valsol = np.max(np.linalg.eigh(np.transpose(A - V.T)@(A - V.T))[0])
    # print(valsol)

    v = sampv0(p)
    u = sampv0(d)

    funcval = np.zeros(iter)
    listtau = []
    listerror = []
    lista = []
    optv = v

    print('iter. \t| func-value \t| res.  \t| tau  \t\t| alpha \t| beta  \t| h^(1)(tau) \t| h^(2)(tau) \t| sample \t| error \t| sing-vec')
    print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------')


    for k in range(iter):

        a = 0
        b = 0
        tau = 0
        if np.mod(k, 1000) == 0:
            dummy = 0

        x = np.random.randn(d)
        w = np.random.randn(d)
        smap = 0

        while a == 0 and b <= 0: # or tau == 0: # or b >= 0:

            v1 = v.copy()
            u1 = u.copy()
            x = sampx(v.copy(),nor=1)
            w = sampx(u.copy(),nor=1)
            
            smap = smap + 1

            '''
                respampling if it's necessary
            '''

            a = np.sum(u*np.dot(A,x)) - np.sum(x*np.dot(V,u)) + np.sum(w*np.dot(A,v)) - np.sum(v*np.dot(V,w))

            if k == 0 and np.abs(a) < 1e-14:
                print('A is orthogonal and ||A|| = ', np.linalg.norm(np.dot(A,v.copy())), 'a_0 = ', a)
                return v, optv, funcval, listtau, lista, listerror

            b = 2*(np.sum(w*np.dot(A,x)) - np.sum(x*np.dot(V,w)) - np.sum(u*np.dot(A,v)) + np.sum(v*np.dot(V,u)))
            
            if a > 0:
                lista = np.append(lista, a)
                tau = b/2/a + np.sqrt(b**2/4/a**2 + 1)
                vg = v + tau*x
                ug = u + tau*w

            if a < 0:
                '''
                    setting a => 0
                '''
                lista = np.append(lista, a)
                tau = b/2/a - np.sqrt(b**2/4/a**2 + 1)
                vg = v + tau*x
                ug = u + tau*w
            if a == 0:
                if b > 0:
                    vg = np.sqrt(2)*x.copy()
                    ug = np.sqrt(2)*w.copy()

        listtau = np.append(listtau, tau)

        v = vg.copy()/np.linalg.norm(vg.copy())
        u = ug.copy()/np.linalg.norm(ug.copy())

        listerror = np.append(listerror, np.linalg.norm(v - v1) + np.linalg.norm(u - u1))
        funcval[k] = funcAV(A, V, v, u)**2
        funcopt = np.max(funcval)
        up = 0
        if funcval[k] == funcopt:
            optv = v
            up = 1


        if np.mod(k,1000) == 0:
            print(k, '\t|', "%10.3e"%(funcval[k]), '\t|', "%10.3e"%(np.abs(funcval[k] - valsol)), '\t|', "%10.3e"%(tau), '\t|', "%10.3e"%(a), '\t|', "%10.3e"%(b), '\t|', "%10.3e"%(onediff(a,b,tau)), '\t|', "%10.3e"%(secdiff(a,b,tau)), '\t|', smap, '\t|', dummy, '\t|', "%10.3e"%(np.linalg.norm(v1 - v) + np.linalg.norm(u1 - u)), '\t|', "%10.3e"%np.linalg.norm(u - np.dot(A - V.T, v)/np.linalg.norm(np.dot(A - V.T, v))))

        if br == 1:
            if k > 100 and tau < 10**(-eps):
                break


    return v, optv, funcval, valsol, listtau, lista, listerror