"""Independent dense FE and finite-difference checks before application timing."""
import json
import numpy as np
import torch
from scenarios import HERE, CASES, Candidates, ScenarioResident, initial, pilot
from candidates import element_geometry, stiffness

def dense_reference(b,n,a,loads):
    offsets,grad=element_geometry(n)
    Ke=stiffness(grad,b.C0,1/n**3)
    z,y,x=np.meshgrid(*([np.arange(n)]*3),indexing='ij')
    nodes=(((z.ravel()[:,None]+offsets[:,2])%n*n+
            (y.ravel()[:,None]+offsets[:,1])%n)*n+
            (x.ravel()[:,None]+offsets[:,0])%n)
    ids=(3*nodes[:,:,None]+np.arange(3)).reshape(n**3,24)
    K=np.zeros((3*n**3,3*n**3))
    np.add.at(K,(ids[:,:,None],ids[:,None,:]),a.reshape(-1,1,1)*Ke)
    energies=[]; solutions=[]; rhss=[]
    for macro in loads:
        E=np.array([[macro[0],macro[5]/2,macro[4]/2],
                    [macro[5]/2,macro[1],macro[3]/2],
                    [macro[4]/2,macro[3]/2,macro[2]]])
        affine=(offsets/n@E.T).reshape(-1)
        rhs=np.zeros(3*n**3)
        np.add.at(rhs,ids,-a.reshape(-1,1)*(Ke@affine))
        u=np.zeros_like(rhs);u[3:]=np.linalg.solve(K[3:,3:],rhs[3:])
        u=(u.reshape(-1,3)-u.reshape(-1,3).mean(0)).reshape(-1)
        local=u[ids]+affine
        e=np.einsum('ei,ij,ej->e',local,Ke,local)*n**3
        energies.append(e);solutions.append(u);rhss.append(rhs)
    energies=np.array(energies)
    return np.array(solutions),np.array(rhss),(energies*a.reshape(1,-1)).mean(1),energies

def cpu_oc(x,g,dv,beta,volume):
    ratio=np.maximum(g,1e-30)/np.maximum(dv,1e-30)
    lo,hi=0.,ratio.max()*1e4
    for _ in range(45):
        lam=(lo+hi)/2
        new=np.clip(x*np.sqrt(ratio/max(lam,1e-30)),np.maximum(0,x-.1),np.minimum(1,x+.1))
        if pilot.physical(new,beta)[0].mean()>volume:lo=lam
        else:hi=lam
    return new

def main():
    report={'passed':False,'cases':[]}
    with pilot.Guard(),torch.inference_mode():
        b=Candidates();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for case,spec in CASES.items():
                n=4;r=ScenarioResident(b,n,case);r.capture()
                r.material(torch.ones((n,)*3,device='cuda',dtype=torch.float64))
                r.select('0.5');r.solutions.zero_();r.solve(1e-11)
                q,_=r.energy()
                assert torch.allclose(q,r.normalizers,atol=2e-12,rtol=2e-12)
                x=initial(n,20261106,spec['volume'])
                for beta in [1.,8.]:
                    r.outer.set_initial(x);r.outer.evaluate(beta);r.material(r.outer.a)
                    r.select('0.5');r.solutions.zero_();r.solve(1e-11)
                    q,e=r.energy();J=float(r.objective(q))
                    ph,dp=pilot.physical(x,beta);a=pilot.material(ph)
                    u,rhs,qref,eref=dense_reference(b,n,a,spec['loads'])
                    ur=r.solutions.cpu().numpy()
                    rhs_gpu=np.stack([r.op.compute_rhs(m).cpu().numpy().ravel() for m in spec['loads']])
                    coeff=np.array(spec['weights'])/np.einsum('li,ij,lj->l',spec['loads'],b.C0,spec['loads'])
                    elocal=(eref*coeff[:,None]).sum(0).reshape((n,)*3)
                    g=pilot.filt(dp*3*(1-1e-4)*ph**2*elocal)
                    dv=pilot.filt(dp)
                    direction=np.random.default_rng(72).normal(size=x.shape)
                    direction/=np.linalg.norm(direction)
                    h=1e-5;values=[];volumes=[]
                    for sign in [-1,1]:
                        perturbed=x+sign*h*direction
                        php,_=pilot.physical(perturbed,beta)
                        _,_,qp,_=dense_reference(b,n,pilot.material(php),spec['loads'])
                        values.append(float(qp@coeff));volumes.append(float(php.mean()))
                    fd=(values[1]-values[0])/(2*h);analytic=float((g*direction).mean())
                    fdv=(volumes[1]-volumes[0])/(2*h);av=float((dv*direction).mean())
                    r.outer.step(e)
                    oc=cpu_oc(x,g,dv,beta,spec['volume'])
                    errors=dict(energy_relative=float(np.max(np.abs(q.cpu().numpy()-qref)/np.maximum(abs(qref),1e-14))),
                        displacement_relative=float(np.linalg.norm(ur-u)/max(np.linalg.norm(u),1e-14)),
                        rhs_absolute=float(np.max(abs(rhs_gpu-rhs))),
                        local_energy_relative=float(np.linalg.norm(e.cpu().numpy()-elocal)/np.linalg.norm(elocal)),
                        gradient_relative=float(abs(fd-analytic)/max(abs(fd),abs(analytic),1e-10)),
                        volume_gradient_absolute=float(abs(fdv-av)),
                        oc_absolute=float(np.max(abs(r.outer.new.cpu().numpy()-oc))))
                    assert errors['energy_relative']<1e-9 and errors['displacement_relative']<1e-7,errors
                    assert errors['rhs_absolute']<1e-12 and errors['local_energy_relative']<1e-7,errors
                    assert errors['gradient_relative']<2e-5 and errors['volume_gradient_absolute']<1e-9,errors
                    assert errors['oc_absolute']<1e-7,errors
                    report['cases'].append(dict(case=case,beta=beta,J=J,errors=errors))
                del r
            # Serial/parallel comparison, including generalized non-shear affine data.
            report['load_scheduling']=[]
            for case in ['S30','M30']:
                results=[]
                for mode in ['serial','parallel']:
                    r=ScenarioResident(b,8,case,mode);r.capture()
                    r.outer.set_initial(initial(8,20261106,.3));r.outer.evaluate(4.)
                    r.material(r.outer.a);r.select('0.5');r.solve(1e-10)
                    q,e=r.energy();r.outer.step(e)
                    results.append((q.cpu().numpy(),r.outer.new.cpu().numpy()))
                    del r
                errors=[float(np.max(abs(a-c))) for a,c in zip(*results)]
                assert max(errors)<1e-7,errors
                report['load_scheduling'].append(dict(case=case,max_absolute_errors=errors))
            # Existing 30% shear adapter must preserve the archived equations.
            from resident import Resident
            b.loads=np.eye(6)[3:];b.affine.cache_clear()
            r=Resident(b,8,load_mode='parallel');r.capture()
            r.outer.set_initial(initial(8,20261106,.3));r.outer.evaluate(4.)
            r.material(r.outer.a);r.select('0.5');r.solve(1e-10)
            q,e=r.energy();r.outer.step(e)
            old=(q.cpu().numpy(),r.outer.new.cpu().numpy());del r
            r=ScenarioResident(b,8,'S30');r.capture()
            r.outer.set_initial(initial(8,20261106,.3));r.outer.evaluate(4.)
            r.material(r.outer.a);r.select('0.5');r.solve(1e-10)
            q,e=r.energy();r.outer.step(e)
            new=(q.cpu().numpy(),r.outer.new.cpu().numpy())
            report['legacy_max_absolute_errors']=[float(np.max(abs(a-c))) for a,c in zip(old,new)]
            assert max(report['legacy_max_absolute_errors'])<1e-8
    report['passed']=True
    pilot.save(HERE/'results/verification.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
