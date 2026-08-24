#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import numpy as np
import run_validation_fast as rv

p=argparse.ArgumentParser(); p.add_argument('--a',required=True); p.add_argument('--b',required=True); p.add_argument('--r',type=float,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--basis',default='minao'); args=p.parse_args()
a,b,r=args.a,args.b,args.r
m=rv.make_mol(a,b,r,args.basis); mf=rv.run_scf(m); S=m.intor_symmetric('int1e_ovlp')
X,oname,oerr=rv.orth_ao(m,S); F=X.T@mf.get_fock()@X; H=X.T@mf.get_hcore()@X
C=np.linalg.solve(X,mf.mo_coeff); nocc=int(np.sum(mf.mo_occ>0)); P=2*C[:,:nocc]@C[:,:nocc].T
idem=float(np.linalg.norm(P@P-2*P)/max(np.linalg.norm(2*P),1e-14))
homo=float(mf.mo_energy[nocc-1]); lumo=float(mf.mo_energy[nocc])
e0=rv.endpoint(m,F,H,P,0,a,b,homo,lumo); e1=rv.endpoint(m,F,H,P,1,b,a,homo,lumo)
payload=dict(pair=f'{a}-{b}',a=a,b=b,r_ang=r,e_tot=float(mf.e_tot),nao=m.nao_nr(),nelectron=m.nelectron,nocc=nocc,nvirt=m.nao_nr()-nocc,homo=homo,lumo=lumo,orthogonalization=oname,orth_error=oerr,density_idempotency_error=idem,endpoints=[e0,e1])
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('wb') as f: pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
print(f"completed {a}-{b}: E={mf.e_tot:.12f} nao={m.nao_nr()} nvirt={m.nao_nr()-nocc}")
