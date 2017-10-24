#from collections import Counter

class Eval:
    def __init__(self, gold, pred):
        assert len(gold)==len(pred)
        self.gold = gold
        self.pred = pred
        self.matrices={}
        self.f1={}
        self.precision={}
        self.recall={}

    def accuracy(self):
        numer = sum(1 for p,g in zip(self.pred,self.gold) if p==g)
        return numer / len(self.gold)

    def confusionMatrices(self):
        self.matrices={l:[0,0,0,0] for l in set(self.gold)} #tp, fp, fn, tn
        for i in range(len(self.gold)):
            if self.pred[i]==self.gold[i]:
                self.matrices[self.pred[i]][0]+=1
                for l in set(self.gold):
                    self.matrices[l][3]+=1
                self.matrices[self.pred[i]][3]-=1
            else:
                self.matrices[self.gold[i]][2]+=1
                self.matrices[self.pred[i]][1]+=1
                
    def measures(self):
        self.precision={l:self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][1]) for l in set(self.gold)}
        self.recall={l:self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][2]) for l in set(self.gold)}
        self.f1={l:2*self.precision[l]*self.recall[l]/(self.precision[l]+self.recall[l]) for l in set(self.gold)}
        