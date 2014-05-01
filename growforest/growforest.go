package main

import (
	"flag"
	"fmt"
	".."
	"log"
	"os"
)

func main() {
	fm := flag.String("train",
		"featurematrix.afm", "AFM formated feature matrix containing training data.")
	rf := flag.String("rfpred",
		"", "File name to output predictor forest in sf format.")
	targetname := flag.String("target",
		"", "The row header of the target in the feature matrix.")
	
	//Parse Data
	fmt.Printf("Loading data from: %v\n", *fm)
	data, err := CloudForest.LoadAFM(*fm)
	if err != nil {
		log.Fatal(err)
	}

	var forestwriter *CloudForest.ForestWriter
	if *rf != "" {
		forestfile, err := os.Create(*rf)
		if err != nil {
			log.Fatal(err)
		}
		defer forestfile.Close()
		forestwriter = CloudForest.NewForestWriter(forestfile)
	}
/*

	imp := flag.String("importance",
		"", "File name to output importance.")
	costs := flag.String("cost",
		"", "For categorical targets, a json string to float map of the cost of falsely identifying each category.")

	rfweights := flag.String("rfweights",
		"", "For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.")

	blacklist := flag.String("blacklist",
		"", "A list of feature id's to exclude from the set of predictors.")

	var nCores int
	flag.IntVar(&nCores, "nCores", 1, "The number of cores to use.")

	var StringnSamples string
	flag.StringVar(&StringnSamples, "nSamples", "0", "The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.")

	var StringmTry string
	flag.StringVar(&StringmTry, "mTry", "0", "Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.")

	var StringleafSize string
	flag.StringVar(&StringleafSize, "leafSize", "0", "The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.")

	var shuffleRE string
	flag.StringVar(&shuffleRE, "shuffleRE", "", "A regular expression to identify features that should be shuffled.")

	var blockRE string
	flag.StringVar(&blockRE, "blockRE", "", "A regular expression to identify features that should be filtered out.")

	var includeRE string
	flag.StringVar(&includeRE, "includeRE", "", "Filter features that DON'T match this RE.")

	var nTrees int
	flag.IntVar(&nTrees, "nTrees", 100, "Number of trees to grow in the predictor.")

	var nContrasts int
	flag.IntVar(&nContrasts, "nContrasts", 0, "The number of randomized artificial contrast features to include in the feature matrix.")

	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

	var contrastAll bool
	flag.BoolVar(&contrastAll, "contrastall", false, "Include a shuffled artificial contrast copy of every feature.")

	var impute bool
	flag.BoolVar(&impute, "impute", false, "Impute missing values to feature mean/mode before growth.")

	var splitmissing bool
	flag.BoolVar(&splitmissing, "splitmissing", false, "Split missing values onto a third branch at each node (experimental).")

	var l1 bool
	flag.BoolVar(&l1, "l1", false, "Use l1 norm regression (target must be numeric).")

	var density bool
	flag.BoolVar(&density, "density", false, "Build density estimating trees instead of classifcation/regression trees.")

	var vet bool
	flag.BoolVar(&vet, "vet", false, "Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.")

	var evaloob bool
	flag.BoolVar(&evaloob, "evaloob", false, "Evaluate potential splitting features on OOB cases after finding split value in bag.")

	var force bool
	flag.BoolVar(&force, "force", false, "Force at least one non constant feature to be tested for each split.")

	var entropy bool
	flag.BoolVar(&entropy, "entropy", false, "Use entropy minimizing classification (target must be categorical).")

	var oob bool
	flag.BoolVar(&oob, "oob", false, "Calculate and report oob error.")

	var caseoob string
	flag.StringVar(&caseoob, "oobpreds", "", "Calculate and report oob predictions in the file specified.")

	var progress bool
	flag.BoolVar(&progress, "progress", false, "Report tree number and running oob error.")

	var adaboost bool
	flag.BoolVar(&adaboost, "adaboost", false, "Use Adaptive boosting for regression/classification.")

	var gradboost float64
	flag.Float64Var(&gradboost, "gbt", 0.0, "Use gradiant boosting with the specified learning rate.")

	var multiboost bool
	flag.BoolVar(&multiboost, "multiboost", false, "Allow multithreaded boosting which may have unexpected results. (highly experimental)")

	var nobag bool
	flag.BoolVar(&nobag, "nobag", false, "Don't bag samples for each tree.")

	var balance bool
	flag.BoolVar(&balance, "balance", false, "Balance bagging of samples by target class for unbalanced classification.")

	var balanceby string
	flag.StringVar(&balanceby, "balanceby", "", "Roughly balanced bag the target within each class of this feature.")

	var ordinal bool
	flag.BoolVar(&ordinal, "ordinal", false, "Use ordinal regression (target must be numeric).")

	var permutate bool
	flag.BoolVar(&permutate, "permute", false, "Permute the target feature (to establish random predictive power).")

	var dotest bool
	flag.BoolVar(&dotest, "selftest", false, "Test the forest on the data and report accuracy.")

	var testfm string
	flag.StringVar(&testfm, "test", "", "Data to test the model on.")

	flag.Parse()
*/	
	CloudForest.Grow(data, forestwriter, targetname)
}
