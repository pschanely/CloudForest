package CloudForest

import (
	"math"
)

/*
AdaBoostTarget wraps a numerical feature as a target for us in Adaptive Boosting (AdaBoost)
*/
type AdaBoostTarget struct {
	CatFeature
	Weights []float64
}

/*
NewAdaBoostTarget creates a categorical adaptive boosting target and initializes its weights.
*/
func NewAdaBoostTarget(f CatFeature) (abt *AdaBoostTarget) {
	nCases := f.Length()
	abt = &AdaBoostTarget{f, make([]float64, nCases)}
	for i := range abt.Weights {
		abt.Weights[i] = 1 / float64(nCases)
	}
	return
}

/*
SplitImpurity is an AdaBoosting version of SplitImpurity.
*/
func (target *AdaBoostTarget) SplitImpurity(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs) (impurityDecrease float64) {
	nl := float64(len(*l))
	nr := float64(len(*r))
	nm := 0.0

	impurityDecrease = nl * target.Impurity(l, allocs.LCounter)
	impurityDecrease += nr * target.Impurity(r, allocs.RCounter)
	if m != nil && len(*m) > 0 {
		nm = float64(len(*m))
		impurityDecrease += nm * target.Impurity(m, allocs.Counter)
	}

	impurityDecrease /= nl + nr + nm
	return
}

//UpdateSImpFromAllocs willl be called when splits are being built by moving cases from r to l as in learning from numerical variables.
//Here it just wraps SplitImpurity but it can be implemented to provide further optimization.
func (target *AdaBoostTarget) UpdateSImpFromAllocs(l *[]int, r *[]int, m *[]int, allocs *BestSplitAllocs, movedRtoL *[]int) (impurityDecrease float64) {
	return target.SplitImpurity(l, r, m, allocs)
}

//Impurity is an AdaBoosting that uses the weights specified in weights.
func (target *AdaBoostTarget) Impurity(cases *[]int, counter *[]int) (e float64) {
	e = 0.0
	m := target.Modei(cases)
	for _, c := range *cases {
		if target.IsMissing(c) == false {
			cat := target.Geti(c)
			if cat != m {
				e += target.Weights[c]
			}
		}

	}
	return
}

//Boost performs categorical adaptive boosting using the specified partition and
//returns the weight that tree that generated the partition should be given.
func (t *AdaBoostTarget) Boost(leaves *[][]int) (weight float64) {
	weight = 0.0
	for _, cases := range *leaves {
		weight += t.Impurity(&cases, nil)
	}
	if weight >= .5 {
		return 0.0
	}
	weight = .5 * math.Log((1-weight)/weight)

	for _, cases := range *leaves {
		m := t.Modei(&cases)
		for _, c := range cases {
			if t.IsMissing(c) == false {
				cat := t.Geti(c)
				if cat != m {
					t.Weights[c] = t.Weights[c] * math.Exp(weight)
				} else {
					t.Weights[c] = t.Weights[c] * math.Exp(-weight)
				}
			}

		}
	}
	normfactor := 0.0
	for _, v := range t.Weights {
		normfactor += v
	}
	for i, v := range t.Weights {
		t.Weights[i] = v / normfactor
	}
	return
}
