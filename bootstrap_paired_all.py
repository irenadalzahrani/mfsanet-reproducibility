import numpy as np, pandas as pd, json
P='/home/claude/work/final_outputs/predictions/'
def auc(y,s):
    # y=1 positive; rank-based AUC with tie handling via average ranks
    from scipy.stats import rankdata
    r=rankdata(s); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def aucs_boot(y,sa,sb,idxs):
    from scipy.stats import rankdata
    da=np.empty(len(idxs))
    for k,ix in enumerate(idxs):
        yy=y[ix]
        n1=yy.sum(); n0=len(yy)-n1
        if n1==0 or n0==0: da[k]=np.nan; continue
        ra=rankdata(sa[ix]); rb=rankdata(sb[ix])
        a=(ra[yy==1].sum()-n1*(n1+1)/2)/(n1*n0); b=(rb[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
        da[k]=a-b
    return da
pairs=[('mfsanet_A4','specxnet'),('mfsanet_A4','defax'),('mfsanet_A3','specxnet'),('mfsanet_A3','defax'),
       ('mfsanet_A3','mfsanet_A1'),('mfsanet_A4','mfsanet_A3'),('mfsanet_v3','specxnet_da'),('mfsanet_v3','defax_da'),('mfsanet_v3','mfsanet_A4')]
res=[]
for ds in ['ffpp','celebdf','dfdc']:
    d=pd.read_csv(P+f'{ds}_seed_mean_video_scores.csv')
    piv=d.pivot(index='video_id',columns='model',values='p_real_mean'); y=d.drop_duplicates('video_id').set_index('video_id').loc[piv.index,'y_true'].values.astype(int)
    rng=np.random.default_rng(42); N=len(y)
    idxs=[rng.integers(0,N,N) for _ in range(100000)]
    for a,b in pairs:
        if a not in piv or b not in piv: continue
        obs=auc(y,piv[a].values)-auc(y,piv[b].values)
        bd=aucs_boot(y,piv[a].values,piv[b].values,idxs); bd=bd[~np.isnan(bd)]
        lo,hi=np.percentile(bd,[2.5,97.5])
        res.append(dict(dataset=ds,a=a,b=b,auc_a=round(auc(y,piv[a].values),4),auc_b=round(auc(y,piv[b].values),4),delta=round(obs,4),lo=round(lo,4),hi=round(hi,4),frac_pos=round((bd>0).mean(),3),n=N))
        print(res[-1],flush=True)
pd.DataFrame(res).to_csv('/home/claude/work/tables/T_bootstrap_all.csv',index=False)
