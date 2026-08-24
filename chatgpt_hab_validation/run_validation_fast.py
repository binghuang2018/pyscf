#!/usr/bin/env python3
"""RHF/MINAO low-rank and dressed-atom validation for H/F/Cl/Br/I diatomics."""
from __future__ import annotations
import argparse, itertools, json, math, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from pyscf import gto, scf, lo, __version__ as pyscf_version

EV = 27.211386245988
ELS = ("H", "F", "Cl", "Br", "I")
VN = {"H":1, "F":2, "Cl":3, "Br":4, "I":5}
RREF = {
("H","H"):0.7414, ("H","F"):0.9168, ("H","Cl"):1.2746,
("H","Br"):1.4144, ("H","I"):1.6090, ("F","F"):1.4119,
("F","Cl"):1.6280, ("F","Br"):1.7580, ("F","I"):1.9100,
("Cl","Cl"):1.9879, ("Cl","Br"):2.1360, ("Cl","I"):2.3210,
("Br","Br"):2.2810, ("Br","I"):2.4690, ("I","I"):2.6660,
}
ETA = 0.05

def js(x):
    if isinstance(x,np.ndarray):
        if np.iscomplexobj(x): return {"real":x.real.tolist(),"imag":x.imag.tolist()}
        return x.tolist()
    if isinstance(x,(np.floating,np.integer)): return x.item()
    if isinstance(x,complex): return {"real":float(x.real),"imag":float(x.imag)}
    if isinstance(x,dict): return {str(k):js(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [js(v) for v in x]
    return x

def invsqrt(a,tol=1e-12):
    w,u=np.linalg.eigh((a+a.T.conj())/2)
    if w.min()<tol: raise np.linalg.LinAlgError(f"min eigenvalue {w.min():.3e}")
    return (u*(w**-0.5))@u.T.conj()

def metrics(x):
    s=np.asarray(np.linalg.svd(x,compute_uv=False),float)
    q=s*s; tot=float(q.sum())
    if tot<1e-28:
        return dict(s=s,rank_1e2=0,rank_1e3=0,first=0.,stable=0.,entropy=0.)
    p=q/tot; ent=-float(np.sum(p[p>0]*np.log(p[p>0])))
    return dict(s=s,rank_1e2=int(np.sum(s>1e-2*s[0])),
                rank_1e3=int(np.sum(s>1e-3*s[0])),first=float(q[0]/tot),
                stable=float(tot/s[0]**2),entropy=float(np.exp(ent)))

def make_mol(a,b,r,basis):
    return gto.M(atom=[(a,(0,0,-r/2)),(b,(0,0,r/2))],unit="Angstrom",
                 basis=basis,charge=0,spin=0,symmetry=False,cart=False,verbose=0)

def run_scf(m):
    mf=scf.RHF(m); mf.conv_tol=1e-11; mf.conv_tol_grad=1e-8
    mf.max_cycle=180; mf.diis_space=12; mf.verbose=0; mf.kernel()
    if not mf.converged:
        mf=mf.newton(); mf.conv_tol=1e-11; mf.max_cycle=100; mf.verbose=0; mf.kernel()
    if not mf.converged: raise RuntimeError("RHF did not converge")
    return mf

def orth_ao(m,S):
    try:
        X=lo.orth.orth_ao(m,method="meta_lowdin",pre_orth_ao="ANO",s=S)
        name="meta_lowdin/ANO"
    except Exception:
        X=invsqrt(S); name="symmetric_lowdin"
    err=np.linalg.norm(X.T@S@X-np.eye(S.shape[0]))
    if err>1e-7: raise RuntimeError(f"orthogonalization residual {err}")
    return X,name,err

def val_idx(m,atom,el):
    p0,p1=map(int,m.aoslice_by_atom()[atom][2:4]); n=VN[el]; labels=m.ao_labels(); out=[]
    for i in range(p0,p1):
        parts=labels[i].split(); tok=parts[2] if len(parts)>2 else ""
        if tok.startswith(f"{n}s") or tok.startswith(f"{n}p"): out.append(i)
    if not out: raise RuntimeError(f"No valence AOs for {el}: {labels[p0:p1]}")
    return out

def endpoint(m,F,H,P,atom,el,partner,homo,lumo):
    sl=m.aoslice_by_atom(); a0,a1=map(int,sl[atom][2:4]); b0,b1=map(int,sl[1-atom][2:4])
    ia=list(range(a0,a1)); ib=list(range(b0,b1)); va=val_idx(m,atom,el); vb=val_idx(m,1-atom,partner)
    Fab=F[np.ix_(va,vb)]; Hab=H[np.ix_(va,vb)]; Pab=P[np.ix_(va,vb)]
    Faa=F[np.ix_(va,va)]; Paa=P[np.ix_(va,va)]
    Pall=P[np.ix_(ia,ib)]; PA=P[np.ix_(ia,ia)]
    fm,hm,pm=metrics(Fab),metrics(Hab),metrics(Pab)
    occ,U=np.linalg.eigh((Paa+Paa.T)/2); order=np.argsort(occ)[::-1]; occ=occ[order]; U=U[:,order]
    activity=np.maximum(0,occ*(2-occ)); k=int(np.argmax(activity)); active=U[:,k]
    Uf,Sf,Vf=np.linalg.svd(Fab,full_matrices=False); Up,Sp,Vp=np.linalg.svd(Pab,full_matrices=False)
    iderr=np.linalg.norm(Pall@Pall.T-(2*PA-PA@PA))/max(np.linalg.norm(2*PA-PA@PA),1e-14)
    sigmas={}; Fbb=F[np.ix_(ib,ib)]; FaB=F[np.ix_(va,ib)]
    for key,z in {"vacuum":0+1j*ETA,"midgap":0.5*(homo+lumo)+1j*ETA}.items():
        sigmas[key]=FaB@np.linalg.solve(z*np.eye(len(ib))-Fbb,FaB.T)
    return dict(atom=atom,element=el,partner=partner,va=va,vb=vb,Fab=Fab,Hab=Hab,Pab=Pab,
                Faa=Faa,Paa=Paa,occ=occ,activity=activity,active=active,
                fport=Uf[:,0],pport=Up[:,0],fm=fm,hm=hm,pm=pm,
                active_f_overlap=float(abs(active@Uf[:,0])),
                active_p_overlap=float(abs(active@Up[:,0])),
                projector_identity_error=float(iderr),sigmas=sigmas)

def krylov_dim(H,U,tol=1e-8):
    cols=[]; X=np.array(U,copy=True)
    for _ in range(H.shape[0]):
        cols.append(X); M=np.concatenate(cols,axis=1)
        s=np.linalg.svd(M,compute_uv=False); rank=int(np.sum(s>tol*s[0])) if s[0]>0 else 0
        if rank==H.shape[0]: return rank
        X=H@X
    s=np.linalg.svd(np.concatenate(cols,axis=1),compute_uv=False)
    return int(np.sum(s>tol*s[0])) if s[0]>0 else 0

def projector_coherence(vecs):
    C=sum(np.outer(v/np.linalg.norm(v),v/np.linalg.norm(v)) for v in vecs)/len(vecs)
    w,u=np.linalg.eigh((C+C.T)/2); o=np.argsort(w)[::-1]
    return w[o],u[:,o]

def null_complement(U,tol=1e-10):
    d=U.shape[0]; M=np.eye(d)-U@U.T; w,V=np.linalg.eigh((M+M.T)/2)
    return V[:,w>1-tol]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--basis",default="minao"); ap.add_argument("--output",type=Path,default=Path("results")); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    pairs=[]; endpoints=[]; raw=[]; failures=[]
    for n,(a,b) in enumerate(itertools.combinations_with_replacement(ELS,2),1):
        tag=f"{a}-{b}"; r=RREF[(a,b)]; print(f"[{n:02d}/15] {tag} R={r:.4f} A",flush=True)
        try:
            m=make_mol(a,b,r,args.basis); mf=run_scf(m); S=m.intor_symmetric("int1e_ovlp")
            X,oname,oerr=orth_ao(m,S); F=X.T@mf.get_fock()@X; H=X.T@mf.get_hcore()@X
            C=np.linalg.solve(X,mf.mo_coeff); nocc=int(np.sum(mf.mo_occ>0)); P=2*C[:,:nocc]@C[:,:nocc].T
            idem=np.linalg.norm(P@P-2*P)/max(np.linalg.norm(2*P),1e-14)
            homo=float(mf.mo_energy[nocc-1]); lumo=float(mf.mo_energy[nocc])
            e0=endpoint(m,F,H,P,0,a,b,homo,lumo); e1=endpoint(m,F,H,P,1,b,a,homo,lumo)
            pairs.append(dict(pair=tag,a=a,b=b,r_ang=r,e_tot=float(mf.e_tot),nao=m.nao_nr(),nelectron=m.nelectron,
                              nocc=nocc,nvirt=m.nao_nr()-nocc,homo_ev=homo*EV,lumo_ev=lumo*EV,gap_ev=(lumo-homo)*EV,
                              orthogonalization=oname,orth_error=oerr,density_idempotency_error=idem))
            raw.append(dict(pair=tag,a=a,b=b,r_ang=r,endpoints=[e0,e1]))
            for ep in (e0,e1):
                row=dict(pair=tag,element=ep["element"],partner=ep["partner"],valence_dim=len(ep["va"]),partner_valence_dim=len(ep["vb"]),
                         fock_rank_1e2=ep["fm"]["rank_1e2"],fock_rank_1e3=ep["fm"]["rank_1e3"],fock_first_fraction=ep["fm"]["first"],fock_stable_rank=ep["fm"]["stable"],
                         hcore_rank_1e2=ep["hm"]["rank_1e2"],hcore_first_fraction=ep["hm"]["first"],
                         density_rank_1e2=ep["pm"]["rank_1e2"],density_rank_1e3=ep["pm"]["rank_1e3"],density_first_fraction=ep["pm"]["first"],density_stable_rank=ep["pm"]["stable"],
                         active_occ=float(ep["occ"][np.argmax(ep["activity"])]),active_f_overlap=ep["active_f_overlap"],active_p_overlap=ep["active_p_overlap"],
                         projector_identity_error=ep["projector_identity_error"],sigma_vacuum_norm=float(np.linalg.norm(ep["sigmas"]["vacuum"])),sigma_midgap_norm=float(np.linalg.norm(ep["sigmas"]["midgap"])))
                for j,x in enumerate(ep["occ"]): row[f"occ{j+1}"]=float(x)
                for j,x in enumerate(ep["fm"]["s"]): row[f"fsv{j+1}"]=float(x)
                for j,x in enumerate(ep["pm"]["s"]): row[f"psv{j+1}"]=float(x)
                endpoints.append(row)
        except Exception as ex:
            traceback.print_exc(); failures.append(dict(pair=tag,error=repr(ex)))
    if failures:
        (out/"failures.json").write_text(json.dumps(failures,indent=2)); raise RuntimeError(failures)

    partner_rows=[]
    for el in ELS:
        items=[]
        for rr in raw:
            for ep in rr["endpoints"]:
                if ep["element"]==el:
                    if rr["a"]==rr["b"] and ep["atom"]==1: continue
                    items.append(ep)
        FJ=np.concatenate([x["Fab"] for x in items],axis=1); PJ=np.concatenate([x["Pab"] for x in items],axis=1)
        fm,pm=metrics(FJ),metrics(PJ); occ=np.array([x["occ"] for x in items])
        aw,AU=projector_coherence([x["active"] for x in items]); fw,FU=projector_coherence([x["fport"] for x in items]); pw,PU=projector_coherence([x["pport"] for x in items])
        Uj,sj,Vj=np.linalg.svd(PJ,full_matrices=False); rport=max(1,pm["rank_1e2"]); Uport=Uj[:,:rport]
        Fmean=sum(x["Faa"] for x in items)/len(items); Pmean=sum(x["Paa"] for x in items)/len(items)
        Q=null_complement(Uport)
        f_full=[]; f_spec=[]; p_full=[]; p_spec=[]; sig_full=[]; sig_res=[]
        sigmean=sum(x["sigmas"]["vacuum"] for x in items)/len(items)
        for x in items:
            d=len(x["va"]); dF=x["Faa"]-Fmean; dF=dF-np.trace(dF)/d*np.eye(d); dP=x["Paa"]-Pmean; dS=x["sigmas"]["vacuum"]-sigmean
            f_full.append(np.linalg.norm(dF)); p_full.append(np.linalg.norm(dP)); sig_full.append(np.linalg.norm(dS))
            f_spec.append(np.linalg.norm(Q.T@dF@Q) if Q.size else 0.); p_spec.append(np.linalg.norm(Q.T@dP@Q) if Q.size else 0.)
            proj=Uport@(Uport.T@dS@Uport)@Uport.T; sig_res.append(np.linalg.norm(dS-proj))
        row=dict(element=el,npartners=len(items),valence_dim=FJ.shape[0],
                 joint_fock_rank_1e2=fm["rank_1e2"],joint_fock_rank_1e3=fm["rank_1e3"],joint_fock_first_fraction=fm["first"],joint_fock_stable_rank=fm["stable"],
                 joint_density_rank_1e2=pm["rank_1e2"],joint_density_rank_1e3=pm["rank_1e3"],joint_density_first_fraction=pm["first"],joint_density_stable_rank=pm["stable"],
                 active_vector_coherence=float(aw[0]),fock_port_coherence=float(fw[0]),density_port_coherence=float(pw[0]),
                 common_port_rank=rport,krylov_rank=krylov_dim(Fmean,Uport),
                 fock_traceless_rms=float(np.sqrt(np.mean(np.square(f_full)))),fock_spectator_rms=float(np.sqrt(np.mean(np.square(f_spec)))),
                 density_rms=float(np.sqrt(np.mean(np.square(p_full)))),density_spectator_rms=float(np.sqrt(np.mean(np.square(p_spec)))),
                 sigma_variation_rms=float(np.sqrt(np.mean(np.square(sig_full)))),sigma_common_port_residual_rms=float(np.sqrt(np.mean(np.square(sig_res)))))
        for j in range(occ.shape[1]):
            row[f"occ{j+1}_mean"]=float(occ[:,j].mean()); row[f"occ{j+1}_std"]=float(occ[:,j].std()); row[f"occ{j+1}_range"]=float(np.ptp(occ[:,j]))
        for j,x in enumerate(fm["s"]): row[f"joint_fsv{j+1}"]=float(x)
        for j,x in enumerate(pm["s"]): row[f"joint_psv{j+1}"]=float(x)
        partner_rows.append(row)

    pairdf=pd.DataFrame(pairs); epdf=pd.DataFrame(endpoints); pdf=pd.DataFrame(partner_rows)
    pairdf.to_csv(out/"molecules.csv",index=False); epdf.to_csv(out/"endpoints.csv",index=False); pdf.to_csv(out/"partners.csv",index=False)
    (out/"raw.json").write_text(json.dumps(js(raw),indent=2))
    nvirt=sorted(pairdf.nvirt.unique().tolist())
    report=["# RHF/MINAO H/F/Cl/Br/I low-rank validation","",
            f"PySCF version: {pyscf_version}. All 15 unordered diatomics were evaluated at fixed equilibrium-region reference geometries.","",
            "## Algebraic controls","",f"Virtual spatial-orbital counts: {nvirt}.",
            f"Maximum density idempotency residual: {pairdf.density_idempotency_error.max():.3e}.",
            f"Maximum block identity residual for $P_{{AB}}P_{{BA}}=2P_A-P_A^2$: {epdf.projector_identity_error.max():.3e}.","",
            "If the MINAO space has exactly one virtual orbital for every molecule, rank-one occupied-projector coupling is forced by co-rank. It is therefore a useful exact control but not, by itself, evidence that the raw Hamiltonian block is universally rank one.","",
            "## Cross-partner diagnostics","",pdf.round(7).to_markdown(index=False),"",
            "## Endpoint diagnostics","",epdf.round(7).to_markdown(index=False),"",
            "## Reading the result","",
            "The literal low-rank hypothesis is supported only if the valence Fock block has low numerical rank. If the Fock block is near full rank but the density block and local occupation spectrum contain one dominant active channel, the stronger conclusion is instead: the bond-active occupied subspace is low rank, while the raw Hamiltonian retains additional sigma, pi, Pauli and spectator couplings.",
            "A transferable dressed-A module requires more than pairwise rank: a coherent common A-side port across partners, a Krylov reach smaller than the full A valence dimension, and small variation in the port-complement local density/Fock blocks."]
    (out/"report.md").write_text("\n".join(report))
    print("\n=== MOLECULES ===\n"+pairdf.to_csv(index=False)); print("\n=== PARTNERS ===\n"+pdf.to_csv(index=False)); print("\n=== ENDPOINTS ===\n"+epdf.to_csv(index=False)); print("\n=== REPORT ===\n"+(out/"report.md").read_text())

if __name__=="__main__": main()
