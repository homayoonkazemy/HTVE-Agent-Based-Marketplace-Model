# Reproduction code for the HTVE 30-sector agent-based experiments
# Python 3; dependencies: numpy, pandas, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SECTOR_NAMES = [
    "Food retail","Agriculture","Household goods","Apparel",
    "Transport","Logistics","Housing maintenance","Construction",
    "Repair services","Home services","Healthcare","Dentistry",
    "Nursing and caregiving","Education","Childcare","IT services",
    "Digital services","Legal services","Accounting","Professional consulting",
    "Hospitality","Personal care","Fitness and wellness","Creative and media",
    "Manufacturing and crafts","Cleaning services","Administrative services",
    "Local delivery","Equipment services","Other local services"
]

def gini(x):
    x=np.asarray(x,float); x=np.clip(x,0,None)
    if x.sum()<=1e-12:return 0.0
    xs=np.sort(x); n=len(xs)
    return float(2*np.sum(np.arange(1,n+1)*xs)/(n*xs.sum())-(n+1)/n)

def controlled_balances(rng, sectors, mean_balance, top10_share=0.10, seed_floor_frac=0.15):
    N=len(sectors); K=int(sectors.max()+1); total=N*mean_balance
    n_top=max(1,int(round(.10*N)))
    floor=seed_floor_frac*mean_balance
    bal=np.full(N,floor,dtype=float)
    top=[]; per=max(1,n_top//K)
    for s in range(K):
        ids=np.where(sectors==s)[0]
        take=min(per,len(ids))
        top.extend(rng.choice(ids,size=take,replace=False).tolist())
    if len(top)<n_top:
        rest=np.setdiff1d(np.arange(N),np.array(top,int))
        top.extend(rng.choice(rest,size=n_top-len(top),replace=False).tolist())
    top=np.array(top[:n_top],int); mask=np.zeros(N,bool); mask[top]=True
    # 0.865 is the highest feasible top-decile share with the 15% floor.
    desired_top=min(top10_share,0.865)*total
    desired_bottom=total-desired_top
    add_top=max(0,desired_top-bal[mask].sum())
    add_bottom=max(0,desired_bottom-bal[~mask].sum())
    rem=total-bal.sum(); scale=rem/max(1e-12,add_top+add_bottom)
    bal[mask]+=scale*add_top/mask.sum()
    bal[~mask]+=scale*add_bottom/(~mask).sum()
    return bal

def run_htve_abm(seed=0,N=450,K=30,T=100,coverage=1.0,mean_balance=2.0,
                 top10_share=0.10,tau=0.01,social_share=0.50,demand_prob=0.60,
                 recycle_every=6,price_kappa=0.04,price_smooth=0.80,
                 acceptance_gamma=0.09,recycle=True,tight_capacity=False):
    rng=np.random.default_rng(seed)
    sectors=np.arange(N)%K; rng.shuffle(sectors)
    W=np.full((K,K),0.0015)
    essentials=[0,1,6]
    for s in range(K):
        pref=set(rng.choice(K,size=7,replace=False).tolist()); pref.update(essentials)
        pref=list(pref)
        vals=rng.gamma(shape=1.4,scale=1.0,size=len(pref))
        W[s,pref]+=vals; W[s,s]*=.20; W[s]/=W[s].sum()
    m=max(1,int(round(K*coverage))); active=np.zeros(K,bool)
    if m>=len(essentials):
        active[essentials]=True
        rem=[x for x in range(K) if x not in essentials]
        if m>len(essentials):
            active[rng.choice(rem,size=m-len(essentials),replace=False)]=True
    else:
        active[rng.choice(K,size=m,replace=False)]=True
    bal=controlled_balances(rng,sectors,mean_balance,top10_share)
    M0=float(bal.sum())
    base_avail=.72 if tight_capacity else 1.0
    willingness=np.where(active[sectors],base_avail,0).astype(float)
    prices=np.ones(K,float)
    window=12; purchases=np.zeros((window,N),np.uint8); earnings=np.zeros((window,N),np.uint8)
    social_pool=0.; operating_pool=0.
    txmat=np.zeros((K,K),float)
    rows=[]; conservation_max_error=0.0
    for t in range(T):
        avail=(rng.random(N)<willingness)&active[sectors]
        cap=np.bincount(sectors[avail],minlength=K)
        demanders=np.where(rng.random(N)<demand_prob)[0]
        u=rng.random(len(demanders)); cdf=np.cumsum(W[sectors[demanders]],axis=1)
        cats=(cdf<u[:,None]).sum(1)
        q20=np.quantile(bal,.2); bottom=(bal<=q20)
        bottom_demand=int(bottom[demanders].sum())
        bought=np.zeros(N,np.uint8); earned=np.zeros(N,np.uint8)
        values=rng.lognormal(np.log(1.6),.30,size=len(demanders))
        trades=0; value=0.; btrades=0; active_demand=0; active_capacity=int(cap[active].sum())
        for j in np.where(active)[0]:
            msk=(cats==j); idx=demanders[msk]; active_demand += len(idx)
            if cap[j]<=0 or len(idx)==0: continue
            vals=values[msk]
            eligible=idx[(bal[idx]>=prices[j])&(vals>=prices[j])]
            if len(eligible)==0: continue
            sellers=np.where(avail&(sectors==j))[0]
            n=min(len(eligible),len(sellers))
            if n<=0: continue
            buyers=rng.choice(eligible,size=n,replace=False)
            sellers=rng.choice(sellers,size=n,replace=False)
            # Prevent self-trades when a provider also demands their own sector.
            # Try several random rematchings; if unavoidable, drop remaining fixed points.
            if n > 0 and np.any(buyers == sellers):
                for _ in range(12):
                    rng.shuffle(sellers)
                    if not np.any(buyers == sellers):
                        break
                keep = buyers != sellers
                buyers = buyers[keep]; sellers = sellers[keep]; n = len(buyers)
            if n<=0: continue
            p=prices[j]; levy=tau*p
            bal[buyers]-=p; bal[sellers]+=p-levy
            social_pool+=social_share*levy*n; operating_pool+=(1-social_share)*levy*n
            bought[buyers]=1; earned[sellers]=1
            trades+=n; value+=p*n; btrades+=int(bottom[buyers].sum())
            uc,cnt=np.unique(sectors[buyers],return_counts=True); txmat[uc,j]+=cnt
        if recycle and (t+1)%recycle_every==0:
            if social_pool>0:
                rec=np.where((bal<=np.quantile(bal,.2))&active[sectors])[0]
                if len(rec)>0:
                    bal[rec]+=social_pool/len(rec); social_pool=0.
            if operating_pool>0:
                rec=np.where(active[sectors])[0]
                if len(rec)>0:
                    nrec=max(1,int(.04*len(rec)))
                    chosen=rng.choice(rec,size=nrec,replace=False)
                    bal[chosen]+=operating_pool/nrec; operating_pool=0.
        slot=t%window; purchases[slot]=bought; earnings[slot]=earned
        spend=purchases.sum(0); earn=earnings.sum(0)
        excess=np.maximum(0,earn-spend); hoard=np.maximum(0,bal/(mean_balance+1e-12)-1.25)
        target=base_avail*np.exp(-acceptance_gamma*excess*hoard)
        target=np.clip(target,.08,base_avail); target=np.where(active[sectors],target,0)
        willingness=.84*willingness+.16*target
        for j in np.where(active)[0]:
            idx=demanders[cats==j]
            affordable=(bal[idx]>=prices[j]).sum() if len(idx) else 0
            pressure=affordable/max(1,cap[j])
            proposed=prices[j]*np.exp(np.clip(price_kappa*(pressure-.88),-.12,.12))
            prices[j]=price_smooth*prices[j]+(1-price_smooth)*proposed
            prices[j]=float(np.clip(prices[j],.20,8.0))
        err=abs((bal.sum()+social_pool+operating_pool)-M0)
        conservation_max_error=max(conservation_max_error,err)
        rows.append({'t':t,'demand':len(demanders),'active_demand':active_demand,'trades':trades,
                     'completion':trades/max(1,len(demanders)),
                     'active_completion':trades/max(1,active_demand),
                     'active_demand_share':active_demand/max(1,len(demanders)),'value':value,
                     'price_index':float(prices[active].mean()),
                     'acceptance':float(willingness[active[sectors]].mean()),
                     'gini':gini(bal),'bottom_access':btrades/max(1,bottom_demand),
                     'capacity_pressure':active_demand/max(1,active_capacity)})
    pdf=pd.DataFrame(rows)
    reciprocal_vol=sum(txmat[i,j] for i in range(K) for j in range(K) if txmat[i,j]>0 and txmat[j,i]>0)
    def avg(col,a,b): return float(pdf.loc[(pdf.t>=a)&(pdf.t<b),col].mean())
    return {
        'completion_rate':float(pdf.trades.sum()/pdf.demand.sum()),
        'active_completion_rate':float(pdf.trades.sum()/max(1,pdf.active_demand.sum())),
        'active_demand_share':float(pdf.active_demand.sum()/max(1,pdf.demand.sum())),
        'velocity':float(pdf.value.sum()/(M0*T)),
        'balance_gini':float(pdf.iloc[-1].gini),
        'price_index':avg('price_index',T-30,T),
        'acceptance':avg('acceptance',T-30,T),
        'reciprocal_volume_share':float(reciprocal_vol/max(1,txmat.sum())),
        'launch5_completion':avg('completion',0,5),
        'early30_completion':avg('completion',0,30),
        'late30_completion':avg('completion',T-30,T),
        'late30_active_completion':avg('active_completion',T-30,T),
        'launch5_bottom_access':avg('bottom_access',0,5),
        'late30_bottom_access':avg('bottom_access',T-30,T),
        'launch5_price':avg('price_index',0,5),
        'late30_price':avg('price_index',T-30,T),
        'late30_capacity_pressure':avg('capacity_pressure',T-30,T),
        'conservation_error':float(conservation_max_error),
        'coverage':coverage,'mean_balance':mean_balance,'top10_share':top10_share,
        'tau':tau,'recycle':recycle,'tight_capacity':tight_capacity,'seed':seed
    }

def ci_summary(df, group_cols, metrics):
    out=[]
    for key,g in df.groupby(group_cols, dropna=False):
        if not isinstance(key,tuple): key=(key,)
        row={c:v for c,v in zip(group_cols,key)}
        for m in metrics:
            vals=g[m].to_numpy(float); n=len(vals); mean=vals.mean(); se=vals.std(ddof=1)/np.sqrt(n)
            row[m+'_mean']=mean; row[m+'_lo']=mean-1.96*se; row[m+'_hi']=mean+1.96*se
        out.append(row)
    return pd.DataFrame(out)

import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent / 'replication_results'; OUT.mkdir(parents=True,exist_ok=True)
R=20

def worker(t):
    exp, kw = t
    d=run_htve_abm(**kw); d['experiment']=exp; return d

tasks=[]
for cov in [0.20,0.50,0.80,1.00]:
    for r in range(R): tasks.append(('breadth',dict(seed=1000+r,coverage=cov,mean_balance=2.0,top10_share=.10,recycle=True)))
for mb in [0.75,1.5,2.0,4.0,8.0]:
    for r in range(R): tasks.append(('stock',dict(seed=2000+r,coverage=1.0,mean_balance=mb,top10_share=.10,recycle=True)))
for share in [0.10,0.30,0.50,0.70,0.865]:
    for rec in [False,True]:
        for r in range(R): tasks.append(('concentration',dict(seed=3000+r,coverage=1.0,mean_balance=1.5,top10_share=share,recycle=rec)))
for tight in [False,True]:
    for rec in [False,True]:
        for r in range(R): tasks.append(('capacity_recycling',dict(seed=4000+r,coverage=1.0,mean_balance=1.5,top10_share=.50,recycle=rec,tau=.03,social_share=.70,tight_capacity=tight,demand_prob=.65)))

if __name__=='__main__':
    with Pool(processes=5) as pool:
        recs=list(pool.imap_unordered(worker,tasks,chunksize=4))
    raw=pd.DataFrame(recs)
    raw.to_csv(OUT/'simulation_all_runs.csv',index=False)
    A=raw[raw.experiment=='breadth'].copy(); B=raw[raw.experiment=='stock'].copy(); C=raw[raw.experiment=='concentration'].copy(); D=raw[raw.experiment=='capacity_recycling'].copy()
    A.to_csv(OUT/'experiment_A_market_breadth_runs.csv',index=False)
    B.to_csv(OUT/'experiment_B_unit_stock_runs.csv',index=False)
    C.to_csv(OUT/'experiment_C_genesis_concentration_runs.csv',index=False)
    D.to_csv(OUT/'experiment_D_capacity_recycling_runs.csv',index=False)
    metrics=['completion_rate','active_completion_rate','active_demand_share','velocity','balance_gini','price_index','acceptance','reciprocal_volume_share','launch5_completion','late30_completion','late30_active_completion','launch5_bottom_access','late30_bottom_access','late30_price']
    As=ci_summary(A,['coverage'],metrics).sort_values('coverage'); Bs=ci_summary(B,['mean_balance'],metrics).sort_values('mean_balance')
    Cs=ci_summary(C,['top10_share','recycle'],metrics).sort_values(['top10_share','recycle']); Ds=ci_summary(D,['tight_capacity','recycle'],metrics).sort_values(['tight_capacity','recycle'])
    As.to_csv(OUT/'summary_A_market_breadth.csv',index=False); Bs.to_csv(OUT/'summary_B_unit_stock.csv',index=False)
    Cs.to_csv(OUT/'summary_C_genesis_concentration.csv',index=False); Ds.to_csv(OUT/'summary_D_capacity_recycling.csv',index=False)
    # Figures
    plt.figure(figsize=(7.2,4.8)); x=As.coverage*100; y=As.completion_rate_mean*100
    plt.plot(x,y,marker='o'); plt.fill_between(x,As.completion_rate_lo*100,As.completion_rate_hi*100,alpha=.18)
    plt.xlabel('Active sector coverage (%)'); plt.ylabel('Completed demand (%)'); plt.title('Market breadth strongly increases feasible closed-loop exchange'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT/'Fig1_market_breadth_completion.png',dpi=220); plt.close()
    plt.figure(figsize=(7.2,4.8)); plt.plot(x,As.active_completion_rate_mean*100,marker='o'); plt.fill_between(x,As.active_completion_rate_lo*100,As.active_completion_rate_hi*100,alpha=.18)
    plt.xlabel('Active sector coverage (%)'); plt.ylabel('Completion within active categories (%)'); plt.title('Breadth improves exchange even within categories already present'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT/'Fig1b_market_breadth_conditional_completion.png',dpi=220); plt.close()
    plt.figure(figsize=(7.2,4.8)); plt.plot(Bs.mean_balance,Bs.late30_completion_mean*100,marker='o'); plt.fill_between(Bs.mean_balance,Bs.late30_completion_lo*100,Bs.late30_completion_hi*100,alpha=.18)
    plt.xlabel('Mean genesis balance (units per participant)'); plt.ylabel('Late-period completed demand (%)'); plt.title('Additional units eventually yield diminishing transaction gains'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT/'Fig2_unit_stock_completion.png',dpi=220); plt.close()
    plt.figure(figsize=(7.2,4.8)); plt.plot(Bs.mean_balance,Bs.late30_price_mean,marker='o'); plt.fill_between(Bs.mean_balance,Bs.late30_price_lo,Bs.late30_price_hi,alpha=.18)
    plt.xlabel('Mean genesis balance (units per participant)'); plt.ylabel('Late-period internal price index'); plt.title('Excess nominal liquidity is absorbed partly through higher internal prices'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT/'Fig3_unit_stock_prices.png',dpi=220); plt.close()
    plt.figure(figsize=(7.2,4.8))
    for rec,label in [(False,'No levy recycling'),(True,'Levy recycling')]:
        s=Cs[Cs.recycle==rec].sort_values('top10_share'); plt.plot(s.top10_share*100,s.launch5_completion_mean*100,marker='o',label=label); plt.fill_between(s.top10_share*100,s.launch5_completion_lo*100,s.launch5_completion_hi*100,alpha=.12)
    plt.xlabel('Share of genesis units held by top 10% (%)'); plt.ylabel('Completed demand in first five periods (%)'); plt.title('Concentrated genesis can create a launch-stage liquidity cliff'); plt.legend(); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT/'Fig4_genesis_concentration.png',dpi=220); plt.close()
    labels=[]; vals=[]; errs=[]
    for tight,name in [(False,'Spare capacity'),(True,'Tight capacity')]:
        for rec,rname in [(False,'No recycling'),(True,'Recycling')]:
            s=Ds[(Ds.tight_capacity==tight)&(Ds.recycle==rec)].iloc[0]; labels.append(name+'\n'+rname); vals.append(s.late30_completion_mean*100); errs.append((s.late30_completion_hi-s.late30_completion_mean)*100)
    plt.figure(figsize=(8.0,4.8)); xx=np.arange(len(labels)); plt.bar(xx,vals,yerr=errs,capsize=4); plt.xticks(xx,labels); plt.ylabel('Late-period completed demand (%)'); plt.title('Recycling helps exchange more when capacity is available'); plt.grid(axis='y',alpha=.2); plt.tight_layout(); plt.savefig(OUT/'Fig5_recycling_capacity.png',dpi=220); plt.close()
    pd.DataFrame({'sector_id':range(1,len(SECTOR_NAMES)+1),'sector_archetype':SECTOR_NAMES}).to_csv(OUT/'sector_archetypes.csv',index=False)
    print('runs',len(raw),'max conservation',raw.conservation_error.max())
    print('A\n',As[['coverage','completion_rate_mean','active_completion_rate_mean','velocity_mean','acceptance_mean','balance_gini_mean']].to_string(index=False))
    print('B\n',Bs[['mean_balance','late30_completion_mean','late30_price_mean','velocity_mean']].to_string(index=False))
    print('C\n',Cs[['top10_share','recycle','launch5_completion_mean','late30_completion_mean','launch5_bottom_access_mean']].to_string(index=False))
    print('D\n',Ds[['tight_capacity','recycle','late30_completion_mean','late30_price_mean']].to_string(index=False))
