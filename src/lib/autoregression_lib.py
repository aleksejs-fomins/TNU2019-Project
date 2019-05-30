import numpy as np

'''
Algorithm-AR(p)-Multivariate-WithTrials

x(t) = sum_i A(i)x(t-i) + B u(t)



'''

# Construct stacked past measurements
def AR_STACK_LAG(data, p):
    nCh, nTr, nT = data.shape
    
    Y = data[:,:,p:]
    X = np.zeros((nCh*p, nTr, nT-p))
    
    for i in range(p):
        X[nCh*i:nCh*(i+1)] = data[:,:,i:i-p]
    return X, Y

def AR_UNSTACK_A(A, p):
    nCh = A.shape[0]
    return [A[:, nCh*i:nCh*(i+1)] for i in range(p)]

# # Compute L2 norm of the fit
# def AR_L2(data, p, A_LST, U=None, B=None):    
#     ABIG = np.hstack(A_LST)
#     XBIG, Y = AR_STACK_LAG(data, p)
    
#     eps = Y - np.einsum('ai,ijk', ABIG, XBIG)
#     if U is not None:
#         eps -= np.einsum('ai,ijk', B, U)
    
#     return np.linalg.norm(eps)**2

def AR_PREDICT(data, p, A_LST, U=None, B=None):
    ABIG = np.hstack(A_LST)
    XBIG, Y = AR_STACK_LAG(data, p)
    
    rez = np.einsum('ai,ijk', ABIG, XBIG)
    if U is not None:
        rez += np.einsum('ai,ijk', B, U)
    
    return rez

def AR_MLE(data, p, U = None):
    # Construct stacked past measurements
    X, Y = AR_STACK_LAG(data, p)
    
    # Construct linear system for transition matrices
    M11 = np.einsum('ajk,bjk', X, Y)
    M12 = np.einsum('ajk,bjk', X, X)
    
    if U is None:
        REZ_A = np.linalg.inv(M12).dot(M11).T
        return AR_UNSTACK_A(REZ_A, p)
    else:    
        M13 = np.einsum('ajk,bjk', X, U)
        M21 = np.einsum('ajk,bjk', U, Y)
        M22 = C.T  #np.einsum('ajk,bjk', U, X)
        M23 = np.einsum('ajk,bjk', U, U)

        # Solve system
        M12INV = np.linalg.inv(M12)
        M23INV = np.linalg.inv(M23)
        TMP11 = M11 - M13.dot(M23INV.dot(M21))
        TMP12 = M12 - M13.dot(M23INV.dot(M22))
        TMP21 = M21 - M22.dot(M12INV.dot(M11))
        TMP22 = M23 - M22.dot(M12INV.dot(M13))

        REZ_A = np.linalg.inv(TMP12).dot(TMP11).T
        REZ_B = np.linalg.inv(TMP22).dot(TMP21).T

        # Unstack A matrices
        REZ_A_LST = AR_UNSTACK_A(REZ_A, p)

        return REZ_A_LST, REZ_B