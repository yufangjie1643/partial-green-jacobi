"""Three rounded, dimensionless, full-grid alpha laws."""
import math
import torch

CONFIGS={'law4':dict(coefficient=.75,complexity_threshold=2.5,exponent=4,contrast_exponent=.5),
         'law8':dict(coefficient=.75,complexity_threshold=2.5,exponent=8,contrast_exponent=.5),
         'law_gate':dict(coefficient=.75,complexity_threshold=4.,exponent='hard',contrast_exponent=.5)}

def extract_law(rho):
    ell=rho.float().log();lo,hi=torch.aminmax(ell);span=hi-lo
    q=(ell-lo)/span.clamp_min(1e-12);n=q.shape[0]
    s1=[];s2=[];active=[]
    for axis in range(3):
        d=(torch.roll(q,-1,axis)-q).abs()
        s1.append(d.mean());s2.append(d.square().mean());active.append((d>1e-6).float().mean())
    total1=sum(s1);total2=sum(s2);a=sum(active)/3
    j=total2/total1.clamp_min(1e-20)
    return torch.stack((a,j,span/math.log(10000.),a*n))

def evaluate(values,config):
    a,j,c,n=map(float,values)
    if n<=1e-12 or c<=1e-12:return 0.
    v=1-config['coefficient']*math.sqrt(max(0.,a*j))/c**config['contrast_exponent']
    if config['exponent']=='hard':v=v if n>=config['complexity_threshold'] else 0.
    else:v-=(config['complexity_threshold']/n)**config['exponent']
    return min(1.,max(0.,v))
