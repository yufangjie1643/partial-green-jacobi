"""Independent FE, sensitivity, filter, OC and load-scheduling checks."""
import importlib.util
import json
import numpy as np
import torch
from reproduction import HERE,PaperBackend,PaperResident,initial,pilot

spec=importlib.util.spec_from_file_location('scenario_dense_reference',HERE.parent/'asg_scenarios_20261105/verify.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def filter_matrix(n,rmin):
    # Independent all-pairs distance construction, including boundary truncation.
    xyz=np.array([[x,y,z] for z in range(n) for y in range(n) for x in range(n)])
    H=np.maximum(0,rmin-np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=-1))
    return H,H.sum(1)


def physical(x,H,hs,beta,eta,emin):
    t=(H@x.ravel())/hs
    den=np.tanh(beta*eta)+np.tanh(beta*(1-eta));q=np.tanh(beta*(t-eta))
    rho=(np.tanh(beta*eta)+q)/den;dp=beta*(1-q*q)/den
    return rho.reshape(x.shape),dp.reshape(x.shape),(emin+(1-emin)*rho**3).reshape(x.shape)


def cpu_oc(x,df,dg,H,hs,beta,eta,emin,volume):
    lo,hi=0.,1e9;count=0
    while (hi-lo)/(hi+lo)>1e-4 and hi>1e-40:
        mid=.5*(lo+hi)
        new=np.maximum(.001,np.maximum(x-.1,np.minimum(1,np.minimum(x+.1,x*np.maximum(1e-10,-df/dg/mid)**.3))))
        rho,_,_=physical(new,H,hs,beta,eta,emin)
        if rho.mean()>volume:lo=mid
        else:hi=mid
        count+=1
        assert count<180
    return new,count


def main():
    report={'passed':False,'cases':[]}
    with pilot.Guard(),torch.inference_mode():
        b=PaperBackend();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for objective in ['shear','bulk','auxetic']:
                n=4;iteration=7;volume=.3 if objective=='bulk' else .2
                r=PaperResident(b,n,objective,volume,rmin=2.1);r.capture()
                r.material(torch.ones((n,n,n),device='cuda',dtype=torch.float64));r.select('0.5');r.solve(1e-11)
                solid=r.effective_tensor().cpu().numpy()
                assert np.max(abs(solid-b.C0))<1e-11
                H,hs=filter_matrix(n,2.1)
                rng=np.random.default_rng(51);x=rng.uniform(.15,.7,size=(n,n,n))
                r.outer.set_initial(x);r.outer.evaluate();r.material(r.outer.a);r.select('0.5');r.solutions.zero_()
                r.solve(1e-11);f,e,diagonal=r.energy(iteration)
                ph,dp,a=physical(x,H,hs,4.,.5,1e-9)
                loads=np.vstack([np.eye(6),[1,1,1,0,0,0]])
                ur,rhs,qr,er=old.dense_reference(b,n,a,loads)
                weights={'shear':np.array([0,0,0,-1,-1,-1,0]),
                    'bulk':np.array([0,0,0,0,0,0,-1]),
                    'auxetic':np.array([-1-.8**iteration]*3+[0,0,0,1])}[objective]
                eref=(weights[:,None]*er).sum(0).reshape(x.shape)
                df=(H@((dp*3*(1-1e-9)*ph**2*eref).ravel()/hs)/n**3).reshape(x.shape)
                dg=(H@(dp.ravel()/hs)).reshape(x.shape)
                direction=rng.normal(size=x.shape);direction/=np.linalg.norm(direction)
                values=[];volumes=[];h=1e-5
                for sign in [-1,1]:
                    pp,_,aa=physical(x+sign*h*direction,H,hs,4,.5,1e-9)
                    _,_,qq,_=old.dense_reference(b,n,aa,loads)
                    values.append(float(qq@weights));volumes.append(pp.sum())
                fd=(values[1]-values[0])/(2*h);analytic=float((df*direction).sum())
                r.outer.step(e)
                candidate,oc_steps=cpu_oc(x,df,dg,H,hs,4,.5,1e-9,volume)
                errors={'filter_normalizer':float(np.max(abs(r.outer.hs.cpu().numpy().ravel()-hs))),
                    'physical_density':float(np.max(abs(r.outer.ph.cpu().numpy()-ph))),
                    'displacement_relative':float(np.linalg.norm(r.solutions.cpu().numpy()-ur[:6])/np.linalg.norm(ur[:6])),
                    'objective_absolute':abs(float(f)-float(qr@weights)),
                    'local_energy_relative':float(np.linalg.norm(e.cpu().numpy()-eref)/np.linalg.norm(eref)),
                    'sensitivity_relative':abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12),
                    'volume_sensitivity':abs((volumes[1]-volumes[0])/(2*h)-float((dg*direction).sum())),
                    'oc_absolute':float(np.max(abs(r.outer.new.cpu().numpy()-candidate)))}
                assert errors['filter_normalizer']<1e-12 and errors['physical_density']<1e-12,errors
                assert errors['objective_absolute']<1e-11 and errors['local_energy_relative']<1e-8,errors
                assert errors['displacement_relative']<1e-7 and errors['sensitivity_relative']<1e-5,errors
                assert errors['volume_sensitivity']<1e-7 and errors['oc_absolute']<1e-9,errors
                assert int(r.outer.bounds[3])==oc_steps and not bool(r.outer.bounds[2])
                report['cases'].append({'objective':objective,'errors':errors,'oc_bisections':oc_steps})
                del r
            scheduled=[]
            for mode in ['serial','parallel']:
                r=PaperResident(b,6,'auxetic',.2,load_mode=mode);r.capture()
                r.outer.set_initial(initial(6,1));r.outer.evaluate();r.material(r.outer.a);r.select('0.5');r.solve(1e-10)
                f,e,_=r.energy(10);r.outer.step(e)
                scheduled.append((float(f),r.outer.new.cpu().numpy()));del r
            report['scheduling']={'objective_absolute':abs(scheduled[0][0]-scheduled[1][0]),
                                  'density_absolute':float(np.max(abs(scheduled[0][1]-scheduled[1][1])))}
            assert max(report['scheduling'].values())<1e-8
    report['passed']=True
    (HERE/'results/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
