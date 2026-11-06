"""Matched whole-grid / yarn-only rotation and 36 / 21 output controls."""
from physics import *
from femlib.materials import engineering_to_voigt,isotropic_to_voigt

STRONG=dict(matrix=dict(E=3500.,Nu=.35),yarn=dict(E11=135000.,E22=9000.,E33=9000.,
        Nu12=.30,Nu13=.30,Nu23=.35,G12=5000.,G13=5000.,G23=3500.))
NEAR=dict(matrix=dict(E=3500.,Nu=.35),yarn=dict(E11=74000.,E22=74000.,E33=74000.,
        Nu12=.2,Nu13=.2,Nu23=.2,G12=30800.,G13=30800.,G23=30800.))

def construct(data,lo,hi,material,selected,packed):
    m,y=material['matrix'],material['yarn'];B=hi-lo;N=int(np.prod(data['grid_shape']))
    cm=torch.tensor(isotropic_to_voigt(m['E'],m['Nu']),device='cuda',dtype=torch.float64)
    cy=torch.tensor(engineering_to_voigt(y['E11'],y['E22'],y['E33'],y['Nu12'],y['Nu13'],y['Nu23'],y['G12'],y['G13'],y['G23']),device='cuda',dtype=torch.float64)
    masks=np.asarray(data['yarn_id'][lo:hi]>=0).reshape(-1);ids=np.flatnonzero(masks)
    if selected:
        a=np.ascontiguousarray(data['orient1'][lo:hi].reshape(-1,3)[masks])
        b=np.ascontiguousarray(data['orient2'][lo:hi].reshape(-1,3)[masks])
    else:
        a=data['orient1'][lo:hi].reshape(-1,3).copy();b=data['orient2'][lo:hi].reshape(-1,3).copy()
        a[~masks]=[1,0,0];b[~masks]=[0,1,0]
    a=torch.as_tensor(a,device='cuda',dtype=torch.float64);b=torch.as_tensor(b,device='cuda',dtype=torch.float64)
    a=a/torch.linalg.vector_norm(a,dim=-1,keepdim=True).clamp_min(1e-14)
    c=torch.cross(a,b,dim=-1);c=c/torch.linalg.vector_norm(c,dim=-1,keepdim=True).clamp_min(1e-14)
    b=torch.cross(c,a,dim=-1);R=torch.stack([a,b,c],dim=-1)
    i1=torch.tensor([1,0,0],device='cuda');i2=torch.tensor([2,2,1],device='cuda')
    r1=R.index_select(-2,i1);r2=R.index_select(-2,i2)
    T=torch.cat([torch.cat([R*R,2*R.index_select(-1,i1)*R.index_select(-1,i2)],-1),
                 torch.cat([r1*r2,r1.index_select(-1,i1)*r2.index_select(-1,i2)+r1.index_select(-1,i2)*r2.index_select(-1,i1)],-1)],-2)
    del R,a,b,c,r1,r2
    if selected:TC=T@cy
    else:
        local=torch.where(torch.as_tensor(masks,device='cuda')[:,None,None],cy,cm)
        TC=T@local;del local
    if packed:rot=torch.stack([(TC[:,i,:]*T[:,j,:]).sum(-1) for i,j in PAIRS],-1)
    else:rot=(TC@T.transpose(-1,-2)).reshape(-1,36)
    del TC,T
    if selected:
        coeff=torch.stack([cm[i,j] for i,j in PAIRS]) if packed else cm.reshape(-1)
        field=coeff.expand(B*N,-1).clone()
        field.index_copy_(0,torch.as_tensor(ids,device='cuda'),rot)
        del rot
    else:field=rot
    return field.reshape(B,N,21 if packed else 36).contiguous()

def setup(data,lo,hi,material,selected,packed,fft):
    C=construct(data,lo,hi,material,selected,packed)
    nz,ny,nx=data['grid_shape'];voxel=data['domain_size'][lo]/np.array([nx,ny,nz])
    solver=make_solver(C,data['grid_shape'],voxel,fft=False);op=solver.op
    op._pack_real_half_fft_symbol=(fft==2)
    op.enable_fft_precond(ref_strategy='zeman',spectrum_mode='complex_full' if fft==0 else 'real_half',
       validation_mode='vectorized_hermitian',allow_symbol_sharing=False)
    return solver
