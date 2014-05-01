package CloudForest

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

type GrowOpts struct {
	imp *string
	costs *string
	rfweights *string
	blacklist *string
	nCores int
	StringnSamples string
	StringmTry string
	StringleafSize string
	shuffleRE string
	blockRE string
	includeRE string
	nTrees int
	nContrasts int
	cpuprofile *string
	contrastAll bool
	impute bool
	splitmissing bool
	l1 bool
	density bool
	vet bool
	evaloob bool
	force bool
	entropy bool
	oob bool
	caseoob string
	progress bool
	adaboost bool
	gradboost float64
	multiboost bool
	nobag bool
	balance bool
	balanceby string
	ordinal bool
	permutate bool
	dotest bool
	testfm string
}

func newEmpty() *string {
	var s = ""
	return &s
}

func (me *GrowOpts) SetDefaults() {
	me.imp = newEmpty()
	me.costs = newEmpty()
	me.rfweights = newEmpty()
	me.blacklist = newEmpty()
	me.cpuprofile = newEmpty()
	me.nCores = 1
	me.StringnSamples = "0"
	me.StringmTry = "0"
	me.StringleafSize = "0"
	me.nTrees = 100
}


func Grow(data *FeatureMatrix, forestwriter *ForestWriter, targetname *string, o GrowOpts) {

	if *o.cpuprofile != "" {
		f, err := os.Create(*o.cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	rand.Seed(time.Now().UTC().UnixNano())

	if o.testfm != "" {
		o.dotest = true
	}

	if o.multiboost {
		fmt.Println("MULTIBOOST!!!!1!!!!1!!11 (things may break).")
	}
	var boostMutex sync.Mutex
	boost := (o.adaboost || o.gradboost != 0.0)
	if boost && !o.multiboost {
		o.nCores = 1
	}

	if o.nCores > 1 {

		runtime.GOMAXPROCS(o.nCores)
	}
	fmt.Printf("Threads : %v\n", o.nCores)
	fmt.Printf("nTrees : %v\n", o.nTrees)

	if o.nContrasts > 0 {
		fmt.Printf("Adding %v Random Contrasts\n", o.nContrasts)
		data.AddContrasts(o.nContrasts)
	}
	if o.contrastAll {
		fmt.Printf("Adding Random Contrasts for All Features.\n")
		data.ContrastAll()
	}

	blacklisted := 0
	blacklistis := make([]bool, len(data.Data))
	if *o.blacklist != "" {
		fmt.Printf("Loading blacklist from: %v\n", *o.blacklist)
		blackfile, err := os.Open(*o.blacklist)
		if err != nil {
			log.Fatal(err)
		}
		tsv := csv.NewReader(blackfile)
		tsv.Comma = '\t'
		for {
			id, err := tsv.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatal(err)
			}
			i, ok := data.Map[id[0]]
			if !ok {
				fmt.Printf("Ignoring blacklist feature not found in data: %v\n", id[0])
				continue
			}
			if !blacklistis[i] {
				blacklisted += 1
				blacklistis[i] = true
			}

		}
		blackfile.Close()

	}

	//find the target feature
	fmt.Printf("Target : %v\n", *targetname)
	targeti, ok := data.Map[*targetname]
	if !ok {
		log.Fatal("Target not found in data.")
	}

	if o.blockRE != "" {
		re := regexp.MustCompile(o.blockRE)
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}

			}

		}

	}

	if o.includeRE != "" {
		re := regexp.MustCompile(o.includeRE)
		for i, feature := range data.Data {
			if targeti != i && !re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}

			}

		}
	}

	nFeatures := len(data.Data) - blacklisted - 1
	fmt.Printf("Non Target Features : %v\n", nFeatures)

	mTry := ParseAsIntOrFractionOfTotal(o.StringmTry, nFeatures)
	if mTry <= 0 {

		mTry = int(math.Ceil(math.Sqrt(float64(nFeatures))))
	}
	fmt.Printf("mTry : %v\n", mTry)

	if o.impute {
		fmt.Println("Imputing missing values to feature mean/mode.")
		data.ImputeMissing()
	}

	if o.permutate {
		fmt.Println("Permutating target feature.")
		data.Data[targeti].Shuffle()
	}

	if o.shuffleRE != "" {
		re := regexp.MustCompile(o.shuffleRE)
		shuffled := 0
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				data.Data[i].Shuffle()
				shuffled += 1

			}

		}
		fmt.Printf("Shuffled %v features matching %v\n", shuffled, o.shuffleRE)
	}

	targetf := data.Data[targeti]
	unboostedTarget := targetf.Copy()

	var bSampler Bagger
	if o.balance {
		bSampler = NewBalancedSampler(targetf.(*DenseCatFeature))
	}

	if o.balanceby != "" {
		bSampler = NewSecondaryBalancedSampler(targetf.(*DenseCatFeature), data.Data[data.Map[o.balanceby]].(*DenseCatFeature))
		o.balance = true

	}

	nNonMissing := 0

	for i := 0; i < targetf.Length(); i++ {
		if !targetf.IsMissing(i) {
			nNonMissing += 1
		}

	}
	fmt.Printf("non-missing cases: %v\n", nNonMissing)

	leafSize := ParseAsIntOrFractionOfTotal(o.StringleafSize, nNonMissing)

	if leafSize <= 0 {
		if boost {
			leafSize = nNonMissing / 3
		} else if targetf.NCats() == 0 {
			//regression
			leafSize = 4
		} else {
			//classification
			leafSize = 1
		}
	}
	fmt.Printf("leafSize : %v\n", leafSize)

	//infer nSamples and mTry from data if they are 0
	nSamples := ParseAsIntOrFractionOfTotal(o.StringnSamples, nNonMissing)
	if nSamples <= 0 {
		nSamples = nNonMissing
	}
	fmt.Printf("nSamples : %v\n", nSamples)

	if o.progress {
		o.oob = true
	}
	if o.caseoob != "" {
		o.oob = true
	}
	var oobVotes VoteTallyer
	if o.oob {
		fmt.Println("Recording oob error.")
		if targetf.NCats() == 0 {
			//regression
			oobVotes = NewNumBallotBox(data.Data[0].Length())
		} else {
			//classification
			oobVotes = NewCatBallotBox(data.Data[0].Length())
		}
	}

	//****** Set up Target for Alternative Impurity  if needed *******//
	var target Target
	if o.density {
		fmt.Println("Estimating Density.")
		target = &DensityTarget{&data.Data, nSamples}
	} else {

		switch targetf.(type) {

		case NumFeature:
			fmt.Println("Performing regression.")
			if o.l1 {
				fmt.Println("Using l1/absolute deviance error.")
				targetf = &L1Target{targetf.(NumFeature)}
			}
			if o.ordinal {
				fmt.Println("Using Ordinal (mode) prediction.")
				targetf = NewOrdinalTarget(targetf.(NumFeature))
			}
			switch {
			case o.gradboost != 0.0:
				fmt.Println("Using Gradiant Boosting.")
				targetf = &GradBoostTarget{targetf.(NumFeature), o.gradboost}

			case o.adaboost:
				fmt.Println("Using Numeric Adaptive Boosting.")
				//BUG(ryan): gradiant boostign should expose learning rate.
				targetf = NewNumAdaBoostTarget(targetf.(NumFeature))
			}
			target = targetf

		case CatFeature:
			fmt.Println("Performing classification.")
			switch {
			case *o.costs != "":
				fmt.Println("Using missclasification costs: ", *o.costs)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(*o.costs), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				regTarg := NewRegretTarget(targetf.(CatFeature))
				regTarg.SetCosts(costmap)
				targetf = regTarg
			case *o.rfweights != "":
				fmt.Println("Using rf weights: ", *o.rfweights)
				weightmap := make(map[string]float64)
				err := json.Unmarshal([]byte(*o.rfweights), &weightmap)
				if err != nil {
					log.Fatal(err)
				}

				wrfTarget := NewWRFTarget(targetf.(CatFeature), weightmap)
				targetf = wrfTarget

			case o.entropy:
				fmt.Println("Using entropy minimization.")
				targetf = &EntropyTarget{targetf.(CatFeature)}

			case boost:

				fmt.Println("Using Adaptive Boosting.")
				targetf = NewAdaBoostTarget(targetf.(CatFeature))

			}
			target = targetf

		}
	}

	//****************** Needed Collections and vars ******************//
	var trees []*Tree
	trees = make([]*Tree, 0, o.nTrees)

	var imppnt *[]*RunningMean
	var mmdpnt *[]*RunningMean
	if *o.imp != "" {
		fmt.Println("Recording Importance Scores.")

		imppnt = NewRunningMeans(len(data.Data))
		mmdpnt = NewRunningMeans(len(data.Data))
	}

	treechan := make(chan *Tree, 0)

	//****************** Good Stuff Stars Here ;) ******************//
	trainingStart := time.Now()
	for core := 0; core < o.nCores; core++ {
		go func() {
			weight := -1.0
			canidates := make([]int, 0, len(data.Data))
			for i := 0; i < len(data.Data); i++ {
				if i != targeti && !blacklistis[i] {
					canidates = append(canidates, i)
				}
			}
			tree := NewTree()
			tree.Target = *targetname
			cases := make([]int, 0, nSamples)
			oobcases := make([]int, 0, nSamples)

			if o.nobag {
				for i := 0; i < nSamples; i++ {
					if !targetf.IsMissing(i) {
						cases = append(cases, i)
					}
				}
			}

			var depthUsed *[]int
			if mmdpnt != nil {
				du := make([]int, len(data.Data))
				depthUsed = &du
			}

			allocs := NewBestSplitAllocs(nSamples, targetf)
			for {
				nCases := data.Data[0].Length()
				//sample nCases case with replacement
				if !o.nobag {
					cases = cases[0:0]

					if o.balance {
						bSampler.Sample(&cases, nSamples)

					} else {
						for j := 0; len(cases) < nSamples; j++ {
							r := rand.Intn(nCases)
							if !targetf.IsMissing(r) {
								cases = append(cases, r)
							}
						}
					}

				}

				if o.nobag && nSamples != nCases {
					cases = cases[0:0]
					for i := 0; i < nSamples; i++ {
						if !targetf.IsMissing(i) {
							cases = append(cases, i)
						}
					}
					SampleFirstN(&cases, nil, nCases, 0)
				}

				if o.oob || o.evaloob {
					ibcases := make([]bool, nCases)
					for _, v := range cases {
						ibcases[v] = true
					}
					oobcases = oobcases[0:0]
					for i, v := range ibcases {
						if !v {
							oobcases = append(oobcases, i)
						}
					}
				}

				tree.Grow(data, target, cases, canidates, oobcases, mTry, leafSize, o.splitmissing, o.force, o.vet, o.evaloob, imppnt, depthUsed, allocs)

				if mmdpnt != nil {
					for i, v := range *depthUsed {
						if v != 0 {
							(*mmdpnt)[i].Add(float64(v))
							(*depthUsed)[i] = 0
						}

					}
				}

				if boost {
					boostMutex.Lock()
					weight = targetf.(BoostingTarget).Boost(tree.Partition(data))
					boostMutex.Unlock()
					if weight == math.Inf(1) {
						fmt.Printf("Boosting Reached Weight of %v\n", weight)
						close(treechan)
						break
					}

					tree.Weight = weight
				}

				if o.oob {
					tree.VoteCases(data, oobVotes, oobcases)
				}

				treechan <- tree
				tree = <-treechan
			}
		}()

	}

	for i := 0; i < o.nTrees; i++ {
		tree := <-treechan
		if tree == nil {
			break
		}
		if forestwriter != nil {
			forestwriter.WriteTree(tree, i)
		}

		if o.dotest {
			trees = append(trees, tree)

			if i < o.nTrees-1 {
				//newtree := new(Tree)
				treechan <- NewTree()
			}
		} else {
			if i < o.nTrees-1 {
				treechan <- tree
			}
		}
		if o.progress {
			fmt.Printf("Model oob error after tree %v : %v\n", i, oobVotes.TallyError(unboostedTarget))
		}

	}

	trainingEnd := time.Now()
	fmt.Printf("Training model took %v.\n", trainingEnd.Sub(trainingStart))

	if o.oob {
		fmt.Printf("Out of Bag Error : %v\n", oobVotes.TallyError(unboostedTarget))
	}
	if o.caseoob != "" {
		caseoobfile, err := os.Create(o.caseoob)
		if err != nil {
			log.Fatal(err)
		}
		defer caseoobfile.Close()
		for i := 0; i < unboostedTarget.Length(); i++ {
			fmt.Fprintf(caseoobfile, "%v\t%v\t%v\n", data.CaseLabels[i], oobVotes.Tally(i), unboostedTarget.GetStr(i))
		}
	}

	if *o.imp != "" {
		impfile, err := os.Create(*o.imp)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		for i, v := range *imppnt {
			mean, count := v.Read()
			meanMinDepth, treeCount := (*mmdpnt)[i].Read()
			fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n", data.Data[i].GetName(), mean, count, mean*float64(count)/float64(o.nTrees), mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)

		}
	}

	if o.dotest {
		var bb VoteTallyer

		testdata := data
		testtarget := unboostedTarget
		if o.testfm != "" {
			var err error
			testdata, err = LoadAFM(o.testfm)
			if err != nil {
				log.Fatal(err)
			}
			targeti, ok = testdata.Map[*targetname]
			if !ok {
				log.Fatal("Target not found in test data.")
			}
			testtarget = testdata.Data[targeti]

			for _, tree := range trees {
				// tree.Root.Climb(func(n *Node) {
				// 	if n.Splitter == nil && n.CodedSplit != nil {
				// 		fmt.Printf("node %v \n %v nil should be %v \n", *n, n.Splitter, n.CodedSplit)
				// 	}
				// 	if n.Splitter != nil && n.CodedSplit == nil {
				// 		fmt.Printf("node %v \n %v nil should be %v \n", *n, n.Splitter, n.CodedSplit)
				// 	}

				// 	switch n.CodedSplit.(type) {
				// 	case float64:
				// 		v := n.Splitter.Value
				// 		if n.CodedSplit.(float64) != v {
				// 			fmt.Printf("%v splits not equal.\n", *n)
				// 		}
				// 		if n.Featurei != testdata.Map[n.Splitter.Feature] {
				// 			fmt.Printf("Feature %v at %v not at %v \n", n.Splitter.Feature, testdata.Map[n.Splitter.Feature], n.Featurei)
				// 		}
				// 	}
				// })
				tree.StripCodes()

			}
		}

		if unboostedTarget.NCats() == 0 {
			//regression
			bb = NewNumBallotBox(testdata.Data[0].Length())
		} else {
			//classification
			bb = NewCatBallotBox(testdata.Data[0].Length())
		}

		for _, tree := range trees {
			tree.Vote(testdata, bb)
		}

		fmt.Printf("Error: %v\n", bb.TallyError(testtarget))

		if testtarget.NCats() != 0 {
			correct := 0
			length := testtarget.Length()
			for i := 0; i < length; i++ {
				if bb.Tally(i) == testtarget.GetStr(i) {
					correct++
				}

			}
			fmt.Printf("Classified: %v / %v = %v\n", correct, length, float64(correct)/float64(length))
		}

	}

}
