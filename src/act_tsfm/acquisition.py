import numpy as np
from .evaluation import Action

def choose(kind: str,mu: np.ndarray,cov: np.ndarray,noise: np.ndarray,cost: np.ndarray,
           available: np.ndarray,remaining: float,w: np.ndarray,rng: np.random.Generator,
           step: int,fantasies: int=12,batches=(1,5)) -> Action | None:
    actions=[Action(p,b) for p in range(len(mu)) for b in batches if available[p]>=b and cost[p]*b<=remaining]
    if not actions: return None
    m=len(mu)//12
    score_mean=mu.reshape(m,12)@w
    if kind=='random': return actions[int(rng.integers(len(actions)))]
    if kind=='fixed':
        return min(actions,key=lambda a:((a.pair-step)%len(mu),a.size))
    variance=np.maximum(np.diag(cov),0)
    if kind=='risk': scores=np.array([w[a.pair%12]/(1+step//len(mu)) for a in actions])
    elif kind in ('uncertainty','ucb','no_decision'):
        scores=np.array([variance[a.pair]/(noise[a.pair]/a.size+variance[a.pair]) for a in actions])
        if kind=='ucb':
            cell_best=mu.reshape(m,12).min(0)
            scores*=np.array([max(0,cell_best[a.pair%12]-mu[a.pair]+2*np.sqrt(variance[a.pair]))*w[a.pair%12] for a in actions])
        if kind!='uncertainty': scores/=np.array([cost[a.pair]*a.size for a in actions])
    elif kind=='racing':
        keep=np.argsort(score_mean)[:max(1,m//(1+step//max(1,len(mu))))]
        scores=np.array([variance[a.pair]/(cost[a.pair]*a.size) if a.pair//12 in keep else -1 for a in actions])
    elif kind=='no_uncertainty':
        return min(actions,key=lambda a:((a.pair-step)%len(mu),a.size))
    else:
        # Gaussian fantasy update: weighted model means shift by Cov(score,Y)/sd(Y).
        # min(E loss) - E[min(E loss | Y)] equals one-step Bayes regret reduction.
        cross=np.stack([w@cov[j*12:(j+1)*12] for j in range(m)])
        half=rng.normal(size=max(1,fantasies//2))
        innovations=np.r_[half,-half]  # Antithetic draws preserve zero-mean update.
        pairs=np.array([a.pair for a in actions]);sizes=np.array([a.size for a in actions])
        delta=(cross[:,pairs]/np.sqrt(variance[pairs]+noise[pairs]/sizes)).T
        future=score_mean[None,None,:]+innovations[None,:,None]*delta[:,None,:]
        scores=np.maximum(0,score_mean.min()-future.min(axis=2).mean(axis=1))
        if kind!='no_cost':scores/=cost[pairs]*sizes
    # Frozen fallback: informative affordable acquisition when MC KG is zero.
    if scores.max()<=0:
        scores=np.array([variance[a.pair]/(cost[a.pair]*a.size) for a in actions])
    return actions[int(np.argmax(scores))]
