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
def funcA(A,v):

    return np.linalg.norm(A@v)**2/np.linalg.norm(v)**2

def onediff(a,b,tau):

    return -a*tau**2 + b*tau + a

def secdiff(a,b,tau):

    return b - 2*tau*a

'''
    main algorithm
'''
def MatFreeAdjNorm(A, iter, br, eps, nor):

    d,d = np.shape(A)

    valsol = np.max(np.linalg.eigh(np.transpose(A)@A)[0])

    '''
        Initialisation
    '''
    v = sampv0(d)

    funcval = np.zeros(iter)
    listtau = []
    listerror = []
    lista = []
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
        funcval[k] = funcA(A,v)
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