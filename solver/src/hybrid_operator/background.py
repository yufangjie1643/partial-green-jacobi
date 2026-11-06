"""Exact uniform Hex8 FI nodal stencil from tensor-product 1D integrals."""
import itertools
import numpy as np

OFFSETS=np.array([d for d in itertools.product((-1,0,1),repeat=3)
                  if d[2]>0 or (d[2]==0 and d[1]>0) or (d[2]==d[1]==0 and d[0]>0)],dtype=np.int32)
VOIGT=np.array([[0,5,4],[5,1,3],[4,3,2]])


def uniform_stencil(C,h):
    """Return 13 antipodal-pair 3x3 blocks; offsets are (x,y,z).

    At offset d, M=(2h/3,h/6), D=(2/h,-1/h) for d=(0,+-1),
    and H=-d/2 = integral N_i' N_j. Mixed derivative integral
    is -H_i H_j times the remaining mass factor. Thus S_d=S_-d.
    Applying pairs to u(x+d)+u(x-d)-2u(x) enforces the zero row sum.
    """
    C=np.asarray(C,dtype=np.float64);h=np.asarray(h,dtype=np.float64)
    if C.shape!=(6,6) or h.shape!=(3,) or np.any(h<=0):raise ValueError("C/h shape")
    result=[]
    for d in OFFSETS:
        M=np.where(d==0,2*h/3,h/6)
        D=np.where(d==0,2/h,-1/h)
        H=-d.astype(float)/2
        G=np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                G[i,j]=D[i]*np.prod(np.delete(M,i)) if i==j else -H[i]*H[j]*M[3-i-j]
        S=np.zeros((3,3))
        for a in range(3):
            for b in range(3):
                S[a,b]=sum(C[VOIGT[a,i],VOIGT[b,j]]*G[i,j] for i in range(3) for j in range(3))
        result.append(S)
    return np.array(result)


def apply_cpu(u,shape,stencil):
    field=np.asarray(u).reshape(*shape,3)
    out=np.zeros_like(field)
    for d,S in zip(OFFSETS,stencil):
        shifted=np.roll(field,tuple(d[::-1]),axis=(0,1,2))+np.roll(field,tuple(-d[::-1]),axis=(0,1,2))-2*field
        out+=np.einsum("ij,...j->...i",S,shifted)
    return out.reshape(-1)
