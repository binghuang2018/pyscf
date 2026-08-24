#!/usr/bin/env python3
import argparse, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import run_validation_fast as rv

p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
raw=[]
for fn in sorted(args.input.rglob('*.pkl')):
    with fn.open('rb') as f: raw.append(pickle.load(f))
if len(raw)!=15: raise RuntimeError(f'Expected 15 pair payloads, found {len(raw)}: {[x.name for x in args.input.rglob("*.pkl")]}')
order={e:i for i,e in enumerate(rv.ELS)}; raw.sort(key=lambda x:(order[x['a']],order[x['b']]))
pairs=[]; endpoints=[]
for rr in raw:
    pairs.append(dict(pair=rr['pair'],a=rr['a'],b=rr['b'],r_ang=rr['r_ang'],e_tot=rr['e_tot'],nao=rr['nao'],nelectron=rr['nelectron'],nocc=rr['nocc'],nvirt=rr['nvirt'],homo_ev=rr['homo']*rv.EV,lumo_ev=rr['lumo']*rv.EV,gap_ev=(rr['lumo']-rr['homo'])*rv.EV,orthogonalization=rr['orthogonalization'],orth_error=rr['orth_error'],density_idempotency_error=rr['density_idempotency_error']))
    for ep in rr['endpoints']:
        row=dict(pair=rr['pair'],element=ep['element'],partner=ep['partner'],valence_dim=len(ep['va']),partner_valence_dim=len(ep['vb']),fock_rank_1e2=ep['fm']['rank_1e2'],fock_rank_1e3=ep['fm']['rank_1e3'],fock_first_fraction=ep['fm']['first'],fock_stable_rank=ep['fm']['stable'],hcore_rank_1e2=ep['hm']['rank_1e2'],hcore_first_fraction=ep['hm']['first'],density_rank_1e2=ep['pm']['rank_1e2'],density_rank_1e3=ep['pm']['rank_1e3'],density_first_fraction=ep['pm']['first'],density_stable_rank=ep['pm']['stable'],active_occ=float(ep['occ'][np.argmax(ep['activity'])]),active_f_overlap=ep['active_f_overlap'],active_p_overlap=ep['active_p_overlap'],projector_identity_error=ep['projector_identity_error'],sigma_vacuum_norm=float(np.linalg.norm(ep['sigmas']['vacuum'])),sigma_midgap_norm=float(np.linalg.norm(ep['sigmas']['midgap'])))
        for j,x in enumerate(ep['occ']): row[f'occ{j+1}']=float(x)
        for j,x in enumerate(ep['fm']['s']): row[f'fsv{j+1}']=float(x)
        for j,x in enumerate(ep['pm']['s']): row[f'psv{j+1}']=float(x)
        endpoints.append(row)
partner_rows=[]
for el in rv.ELS:
    items=[]
    for rr in raw:
        for ep in rr['endpoints']:
            if ep['element']==el:
                if rr['a']==rr['b'] and ep['atom']==1: continue
                items.append(ep)
    FJ=np.concatenate([x['Fab'] for x in items],axis=1); PJ=np.concatenate([x['Pab'] for x in items],axis=1)
    fm,pm=rv.metrics(FJ),rv.metrics(PJ); occ=np.array([x['occ'] for x in items])
    aw,AU=rv.projector_coherence([x['active'] for x in items]); fw,FU=rv.projector_coherence([x['fport'] for x in items]); pw,PU=rv.projector_coherence([x['pport'] for x in items])
    Uj,sj,Vj=np.linalg.svd(PJ,full_matrices=False); rport=max(1,pm['rank_1e2']); Uport=Uj[:,:rport]
    Fmean=sum(x['Faa'] for x in items)/len(items); Pmean=sum(x['Paa'] for x in items)/len(items); Q=rv.null_complement(Uport)
    f_full=[]; f_spec=[]; p_full=[]; p_spec=[]; sig_full=[]; sig_res=[]; sigmean=sum(x['sigmas']['vacuum'] for x in items)/len(items)
    for x in items:
        d=len(x['va']); dF=x['Faa']-Fmean; dF=dF-np.trace(dF)/d*np.eye(d); dP=x['Paa']-Pmean; dS=x['sigmas']['vacuum']-sigmean
        f_full.append(np.linalg.norm(dF)); p_full.append(np.linalg.norm(dP)); sig_full.append(np.linalg.norm(dS)); f_spec.append(np.linalg.norm(Q.T@dF@Q) if Q.size else 0.); p_spec.append(np.linalg.norm(Q.T@dP@Q) if Q.size else 0.)
        proj=Uport@(Uport.T@dS@Uport)@Uport.T; sig_res.append(np.linalg.norm(dS-proj))
    row=dict(element=el,npartners=len(items),valence_dim=FJ.shape[0],joint_fock_rank_1e2=fm['rank_1e2'],joint_fock_rank_1e3=fm['rank_1e3'],joint_fock_first_fraction=fm['first'],joint_fock_stable_rank=fm['stable'],joint_density_rank_1e2=pm['rank_1e2'],joint_density_rank_1e3=pm['rank_1e3'],joint_density_first_fraction=pm['first'],joint_density_stable_rank=pm['stable'],active_vector_coherence=float(aw[0]),fock_port_coherence=float(fw[0]),density_port_coherence=float(pw[0]),common_port_rank=rport,krylov_rank=rv.krylov_dim(Fmean,Uport),fock_traceless_rms=float(np.sqrt(np.mean(np.square(f_full)))),fock_spectator_rms=float(np.sqrt(np.mean(np.square(f_spec)))),density_rms=float(np.sqrt(np.mean(np.square(p_full)))),density_spectator_rms=float(np.sqrt(np.mean(np.square(p_spec)))),sigma_variation_rms=float(np.sqrt(np.mean(np.square(sig_full)))),sigma_common_port_residual_rms=float(np.sqrt(np.mean(np.square(sig_res)))))
    for j in range(occ.shape[1]): row[f'occ{j+1}_mean']=float(occ[:,j].mean()); row[f'occ{j+1}_std']=float(occ[:,j].std()); row[f'occ{j+1}_range']=float(np.ptp(occ[:,j]))
    for j,x in enumerate(fm['s']): row[f'joint_fsv{j+1}']=float(x)
    for j,x in enumerate(pm['s']): row[f'joint_psv{j+1}']=float(x)
    partner_rows.append(row)
pairdf=pd.DataFrame(pairs); epdf=pd.DataFrame(endpoints); pdf=pd.DataFrame(partner_rows)
pairdf.to_csv(args.output/'molecules.csv',index=False); epdf.to_csv(args.output/'endpoints.csv',index=False); pdf.to_csv(args.output/'partners.csv',index=False)
(args.output/'raw_summary.json').write_text(json.dumps(rv.js(raw),indent=2))
report=['# Parallel RHF/MINAO H/F/Cl/Br/I validation','',f'All 15 pair jobs completed. Virtual counts: {sorted(pairdf.nvirt.unique().tolist())}.','',f'Max idempotency error: {pairdf.density_idempotency_error.max():.3e}.',f'Max block identity error: {epdf.projector_identity_error.max():.3e}.','','## Partner summary','',pdf.round(7).to_markdown(index=False),'','## Endpoint summary','',epdf.round(7).to_markdown(index=False),'','## Interpretation','','Raw valence-Fock rank tests the literal H_AB hypothesis. The occupied-projector rank and local natural-occupation spectrum diagnose the lower-dimensional bond-active channel. MINAO co-rank must be treated as an exact finite-basis constraint, not universal evidence.']
(args.output/'report.md').write_text('\n'.join(report)); print(pdf.to_csv(index=False)); print(epdf.to_csv(index=False))
