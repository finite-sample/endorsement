#!/usr/bin/env python3
"""
Generate all tables and figures from the actual replication data.
Data: endorse R package (Bullock, Imai, Shapiro 2011) / Blair et al. (2013).
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment

df = pd.read_csv("data/pakistan.csv")
POLICIES = ["Polio", "FCR", "Durand", "Curriculum"]
GROUPS = {"b":"Afghan Taliban","c":"Pakistani Taliban (TTP)",
          "d":"Kashmiri Militants","e":"Sectarian Groups"}
for p in POLICIES:
    for s in list(GROUPS)+["a"]:
        df[f"{p}.{s}"] = (df[f"{p}.{s}"] - 1) / 4

ctrl_all = np.concatenate([df[f"{p}.a"].dropna().values for p in POLICIES])
ALPHA = float(ctrl_all.mean())
CTRL_SD = float(ctrl_all.std(ddof=1))
CATS = np.array([0, 0.25, 0.5, 0.75, 1.0])
CTRL_DIST = np.array([np.mean(np.abs(ctrl_all - c) < 0.01) for c in CATS])

GS = {}
for gs, gn in GROUPS.items():
    t = np.concatenate([df[f"{p}.{gs}"].dropna().values for p in POLICIES])
    GS[gn] = {"ate":float(t.mean()-ALPHA),"tmean":float(t.mean()),
              "tsd":float(t.std(ddof=1)),"nt":len(t)}

D_SYM = 0.05; SIG_N = 0.05
os.makedirs("tabs",exist_ok=True); os.makedirs("figs",exist_ok=True)

def disc(beta, s=1):
    shift = s*0.25
    dp = np.dot(CTRL_DIST, np.minimum(CATS+shift,1)-CATS)
    dm = np.dot(CTRL_DIST, np.maximum(CATS-shift,0)-CATS)
    den = dp-dm
    p = (beta-dm)/den if abs(den)>1e-12 else 0.5
    return {"p":np.clip(p,0,1),"dp":dp,"dm":dm,"ceil":shift-dp}

def wtex(path, tex):
    with open(path,"w") as f: f.write(tex.replace("%","\\%"))
    print(f"  wrote {path}")

# Table 1
def t1():
    tex = "\\begin{tabular}{lcccc}\n\\toprule\nGroup & ATE & Treat.\\ Mean & Blair et al. & Binary Switching \\\\\n\\midrule\n"
    for gn,gs in GS.items():
        tex += f"{gn} & ${gs['ate']:.3f}$ & ${gs['tmean']:.3f}$ & ``Low regard'' & {gs['tmean']:.1%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    wtex("tabs/blair_reinterpretation.tex", tex)

# Table 2
def t2():
    b = np.mean([gs["ate"] for gs in GS.values()])  # pooled average ATE
    d1=disc(b,1); d2=disc(b,2)
    rows = [
        ("Binary switching (with baseline)", f"$\\alpha + \\beta$; $\\alpha={ALPHA:.3f}$", ALPHA+b),
        ("5-point scale, 2-step shift", "Shift $\\pm 2$ categories, ceiling/floor", d2["p"]),
        ("5-point scale, 1-step shift", "Shift $\\pm 1$ category, ceiling/floor", d1["p"]),
        ("Binary switching (no baseline)", "$(\\beta+1)/2$, no constant", (b+1)/2),
        (f"Normal heterogeneity ($\\sigma={SIG_N}$)", f"$\\delta_i \\sim \\mathcal{{N}}({b:.3f},{SIG_N}^2)$", 1-stats.norm.cdf(-b/SIG_N)),
        (f"Symmetric effects ($d={D_SYM}$)", f"$\\delta^+=+{D_SYM}$, $\\delta^-=-{D_SYM}$", b/(2*D_SYM)+0.5),
        ("Extreme opponent reaction", "$\\delta^-=-1$, $\\delta^+=0$", 1+b),
    ]
    tex = "\\begin{tabular}{lll}\n\\toprule\nModel & Key Assumption & Implied Support \\\\\n\\midrule\n"
    for m,a,s in rows: tex += f"{m} & {a} & {s:.1%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    wtex("tabs/method_comparison.tex", tex)

# Table 3: discrete scale detail
def t3():
    b = np.mean([gs["ate"] for gs in GS.values()])  # pooled average ATE
    tex = "\\begin{tabular}{lrrrr}\n\\toprule\nShift & Eff.\\ $\\delta^+$ & Eff.\\ $\\delta^-$ & Ceiling Loss & Implied $p$ \\\\\n\\midrule\n"
    for s in [1,2,3]:
        r = disc(b,s)
        tex += f"{s} category & {r['dp']:.3f} & {r['dm']:.3f} & {r['ceil']:.3f} & {r['p']:.1%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    wtex("tabs/discrete_scale.tex", tex)

# Table 4: variance test
def t4():
    tex = "\\begin{tabular}{lrrrr}\n\\toprule\nGroup & Treat.\\ Mean & Predicted SD & Actual SD & Var.\\ Ratio \\\\\n\\midrule\n"
    for gn,gs in GS.items():
        p=gs["tmean"]; psd=np.sqrt(p*(1-p)); r=gs["tsd"]**2/(p*(1-p))
        tex += f"{gn} & {p:.3f} & {psd:.3f} & {gs['tsd']:.3f} & {r:.3f} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    wtex("tabs/variance_test.tex", tex)

# Table 5: simulation
def t5():
    rng = np.random.default_rng(42)
    tex = "\\begin{tabular}{lrrr}\n\\toprule\nGroup & ATE & $P(T>C)$ & Hungarian Max \\\\\n\\midrule\n"
    for gn,gs in GS.items():
        t = np.concatenate([df[f"{p}.{s}"].dropna().values for p in POLICIES for s,g in GROUPS.items() if g==gn])
        n_s=min(len(t),len(ctrl_all),50000)
        ptc = np.mean(rng.choice(t,n_s)>rng.choice(ctrl_all,n_s))
        n_h=500; ch=rng.choice(ctrl_all,n_h); th=rng.choice(t,n_h)
        cost=np.zeros((n_h,n_h))
        for i in range(n_h):
            for j in range(n_h):
                if th[j]>ch[i]: cost[i,j]=-1
        ri,ci=linear_sum_assignment(cost); pos=int(np.sum(th[ci]>ch[ri]))
        tex += f"{gn} & ${gs['ate']:.3f}$ & {ptc:.0%} & {pos/n_h:.0%} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    wtex("tabs/simulation_results.tex", tex)

# Figure 1
def f1():
    b=GS["Afghan Taliban"]["ate"]; d1=disc(b,1); d2=disc(b,2)
    methods=["Binary\nswitching","5-pt scale\n2-step","5-pt scale\n1-step",
             "Binary\n(no baseline)","Normal\nheterog.","Symmetric\neffects","Extreme\nopponent"]
    est=[ALPHA+b,d2["p"],d1["p"],(b+1)/2,1-stats.norm.cdf(-b/SIG_N),b/(2*D_SYM)+0.5,1+b]
    cols=["#f4a460","#e8b87a","#dcc69a","#a8d5e5","#c4b7e0","#fffacd","#90ee90"]
    fig,ax=plt.subplots(figsize=(10,5.5))
    bars=ax.bar(methods,est,color=cols,edgecolor="black",lw=0.8)
    ax.axhline(0.5,color="red",ls="--",lw=1.5,label="50% majority threshold")
    ax.set_ylabel("Implied proportion supporting group",fontsize=12)
    ax.set_title(f"Implied support: Blair et al. (2013), Afghan Taliban\nATE = {b:.3f}, baseline = {ALPHA:.3f}",fontsize=12)
    ax.set_ylim(0,1.08)
    for bar,e in zip(bars,est):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,f"{e:.1%}",ha="center",va="bottom",fontsize=10,fontweight="bold")
    ax.legend(loc="upper left",fontsize=10); plt.tight_layout()
    fig.savefig("figs/method_comparison.pdf",bbox_inches="tight"); plt.close(fig)
    print("  wrote figs/method_comparison.pdf")

# Figure 2
def f2():
    ate_range=np.linspace(-0.10,0.10,200)
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(ate_range,ALPHA+ate_range,"-",lw=2.2,color="#d45500",label=f"Binary switching (baseline={ALPHA:.2f})")
    ax.plot(ate_range,(ate_range+1)/2,"--",lw=1.8,color="#3377bb",label="Binary switching (no baseline)")
    ax.plot(ate_range,np.clip(1+ate_range,0,1),":",lw=1.8,color="#228833",label="Extreme opponent reaction")
    ax.plot(ate_range,np.clip(ate_range/(2*D_SYM)+0.5,0,1),"-.",lw=1.8,color="#aa8822",label=f"Symmetric effects (d={D_SYM})")
    ob=GS["Afghan Taliban"]["ate"]
    ax.axvline(ob,color="grey",ls="--",lw=1.2,alpha=0.7)
    ax.text(ob+0.003,0.05,f"Afghan Taliban\nATE = {ob:.3f}",fontsize=9,color="grey")
    ax.axhline(0.5,color="red",ls="--",lw=1,alpha=0.5)
    ax.set_xlabel("Average treatment effect",fontsize=12); ax.set_ylabel("Implied proportion supporting group",fontsize=12)
    ax.set_title("Sensitivity of support estimates to the ATE",fontsize=13)
    ax.set_ylim(0,1.05); ax.set_xlim(-0.10,0.10); ax.legend(fontsize=9,loc="lower right"); ax.grid(True,alpha=0.25)
    plt.tight_layout(); fig.savefig("figs/sensitivity.pdf",bbox_inches="tight"); plt.close(fig)
    print("  wrote figs/sensitivity.pdf")

def summary():
    print(f"\n{'='*60}\nNUMBERS FOR MANUSCRIPT\n{'='*60}")
    print(f"  Control mean (alpha):   {ALPHA:.4f}")
    print(f"  Control SD:             {CTRL_SD:.4f}")
    print(f"  Control distribution:   {dict(zip(CATS, CTRL_DIST.round(3)))}")
    for gn,gs in GS.items():
        p=gs["tmean"]; r=gs["tsd"]**2/(p*(1-p)); d1=disc(gs["ate"],1)
        print(f"\n  {gn}: ATE={gs['ate']:.4f}, treat_mean={gs['tmean']:.4f}")
        print(f"    Binary={gs['tmean']:.1%}, Disc1={d1['p']:.1%}, VarRatio={r:.3f}")

if __name__=="__main__":
    print("Generating from microdata ...\n\nTables:")
    t1(); t2(); t3(); t4(); t5()
    print("\nFigures:"); f1(); f2(); summary(); print("\nDone.")
