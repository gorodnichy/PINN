// BEGIN_COPYRIGHT
/*------------------------------------------------------------------------------
*  Code:      Library that allows to create, train and evaluate
*             various kinds of Associative Neural Networks.
*  Author:    Oleksiy K. Dekhtyarenko, 2003-2005
*             name@domain, name=olexii, domain=mail.ru
*  Copyright: The author grants the right to use and modify the code
*             provided suitable acknowledgements and citations are made.
*-----------------------------------------------------------------------------*/
// END_COPYRIGHT

# include "modular.h"


bool Module::train(Vector *in, Properties *prop)
{
    if(net == 0)
        net = getNetwork();

    assertion(in->size()==net->dim(), "[Module::train] wrong input size");

    bool res = net->train(in, prop);
    net->doEndTraining();

    return res;
}


void Module::test(Vector *in, Vector *out)
{
    assertion(net!=0, "[Module::test] net == 0 (no previous call of train)");
    assertion(in->size()==out->size() && out->size()==net->dim(), "[Module::test] in/out/net dim mismatch");

    net->converge(in, out);
}


string Module::getDescription()
{
    bool hasBeenEmpty = isEmpty();
    if(hasBeenEmpty)
        net = getNetwork();

    string res = "# Module with capacity = " + toString(capacity, 3)
                        + "\n# Based on:\n" + net->getDescription();

    if(hasBeenEmpty)
        clear();

    return res;
}


int Module::dim()
{
    bool hasBeenEmpty = isEmpty();
    if(hasBeenEmpty)
        net = getNetwork();

    int res = net->dim();

    if(hasBeenEmpty)
        clear();

    return res;
}


//------------------------------------------------------------------------------

string ModularNet::getDescription()
{
    return "# ----- Modular Associative Network -----\n# moduleType = " + getModuleName()
        + ", levelCount = " + toString(levelCount)
        + "\n# moduleCapacity = " + toString(moduleCapacity, 3)
        + ", threshold = " + toString(threshold, 3)
        + ", train\\testEpsilon = " + toString(trainEpsilon, 3) + "\\" + toString(testEpsilon, 3)
        + "\n# Based on:\n" + modules[0]->getDescription();
}

// normalized difference of stored patternd in right and left subtrees
double ModularNet::getInbalanceDegree()
{
    if(trainedCount == 0)
        return NOT_A_NUMBER;

    int rightLeftDiff = 0, m = 1;
    for(int lev=1; lev<levelCount; lev++)
    {
        int halfCount = round( pow(2, lev-1) );
        for(int i=0; i<halfCount; i++)
            rightLeftDiff -= modules[m++]->numStored();
        for(int i=0; i<halfCount; i++)
            rightLeftDiff += modules[m++]->numStored();
    }

    return ((double)rightLeftDiff)/trainedCount;
}


// average number of actually filled levels
double ModularNet::getActualLevelCount()
{
    return log( (trainedCount/moduleCapacity)/modules[0]->dim() + 1) / log(2);
}

// bulds a path tree using current threshold and provided epsilon
// path - sorted list of modules included in search tree
// modulesToTrain - sorted list of AVAILABLE modules to train
// numModulesSelected - number of modules that must have been trained
//                      (but possibly not all of them are included into modulesToTrain due to tree size limitation)
void ModularNet::buildTree(Vector *in, double epsilon, vector<int> *path,
    vector<double> *pathDifference, vector<int> *modulesToTrain, int *numModulesSelected)
{
    path->clear();
    pathDifference->clear();
    modulesToTrain->clear();

    vector<int> currentLevel, nextLevel, tmp;
    currentLevel.push_back(0); // start from module #0
    for(int level=0; level<levelCount; level++) // go through each level of a tree
    {
        if(currentLevel.size() == 0) // no modules left to consider
            break;

        nextLevel.clear();
        for(int m=0; m<currentLevel.size(); m++)
        {
            int currMod = currentLevel[m];
            Module *module = modules[currMod];
            double difference = module->getDifference(in);
            if(!module->isEmpty())  // add to path (for further testing)
            {
                path->push_back(currMod);
                pathDifference->push_back(difference);
            }

            if(!module->isFull()) // add to train
                modulesToTrain->push_back(currMod);
            else // fully filled module; continue building tree
            {
                if(difference <  threshold + epsilon)
                    nextLevel.push_back(2*currMod + 1);
                if(difference >= threshold - epsilon)
                    nextLevel.push_back(2*currMod + 2);
            }
        }
        currentLevel = nextLevel;
    }

    // next level has modules that have to be trained, but this level is out of existing tree
    *numModulesSelected = modulesToTrain->size() + nextLevel.size();

    assertion(*numModulesSelected>0, "[ModularNet::buildTree] alogrithm error");
}


// discrete train
void ModularNet::train(Vector *in, Properties *prop)
{
    assertion(in->size() == modules[0]->dim(), "[ModularNet::train] in-size() != modules[0]->dim()");

    vector<int> path, modulesToTrain;
    vector<double> pathDifference;
    int numModulesSelected;
    buildTree(in, trainEpsilon, &path, &pathDifference, &modulesToTrain, &numModulesSelected);

    vector<int> dest;
    for(int m=0; m<modulesToTrain.size(); m++)
    {
        bool trained = modules[ modulesToTrain[m] ]->train(in, prop);

        assertion(trained, "[ModularNet::train] module #" + toString(modulesToTrain[m])
            + ": training failed");

        if(trained)
        {
            trainedCount++;
            dest.push_back(modulesToTrain[m]);
        }
    }
    destinations.push_back(dest);
    if(dest.size()>0)
        numStored++;

    (*prop)["tr##Vec"] = destinations.size();
    (*prop)["tr#Stored"] = numStored;
    (*prop)["trEfficiency"] = modulesToTrain.size()/numModulesSelected;
    (*prop)["trRedundancy"] = ((double)trainedCount)/numStored;
    (*prop)["trInbalance"] = getInbalanceDegree();
    (*prop)["trActLevels"] = getActualLevelCount();
}

// discrete test
void ModularNet::test(Vector *in, int inIndex, Vector *out, bool *pathError, bool *belongingError, double *searchComplexity)
{
    assertion(in->size() == modules[0]->dim(), "[ModularNet::test] in-size() != modules[0]->dim()");
    assertion(out->size() == modules[0]->dim(), "[ModularNet::test] out-size() != modules[0]->dim()");
    assertion(inIndex>=0 && inIndex<destinations.size(), "[ModularNet::test] inIndex out of range");

    vector<int> path, modulesToTrain;
    vector<double> pathDifference;
    int numModulesSelected;
    buildTree(in, testEpsilon, &path, &pathDifference, &modulesToTrain, &numModulesSelected);

    int minDiffIndex;
    double minDiff = minElement(&pathDifference, &minDiffIndex);
    int resModule = path[minDiffIndex];
    modules[resModule]->test(in, out);

    *pathError = true;
    vector<int> &dest = destinations[inIndex];
    for(int i=0; i<dest.size(); i++)
        if(indexOfSorted(&path, dest[i]) >= 0) // if one of destinations is within path
        {
            *pathError = false;
            break;
        }

    *belongingError = indexOfSorted(&dest, resModule) < 0; // selected module is not among destinations
    *searchComplexity = path.size()/getActualLevelCount();
}

// cumulative train
void ModularNet::train(IOData *data, Properties *prop)
{
    data->setToBeginning();
    Vector in(data->inDim()), out(data->outDim());
    Properties singleTrainProp;
    PropertyStatCounter propSC;
    for(int d=0; d<data->count(); d++)
    {
        data->getNext(&in, &out);
        train(&in, &singleTrainProp);
        propSC.addValue(&singleTrainProp);
        saveProperties(singleTrainProp, d==0, 13, "_ModularNet_train.rep");
    }

    propSC.getStatistics(prop, SC_AVERAGE | SC_SD);
}

// cumulative test
// returns fraction of datapoints with correctly recalled modules
double ModularNet::test(IOData *data, int count, int hamming, int runs, Properties *prop)
{
    assertion(count>0, "[ModularNet::test] count <= 0");
    assertion(count<=data->count(), "[ModularNet::test] count > data->count()");
    data->setToBeginning();
    Vector in(data->inDim()), inH(data->inDim()), out(data->outDim()), outH(data->outDim());
    bool pathError, belongingError;
    double searchComplexity;
    StatCounter pathErrorSC("_ePath"), belongingErrorSC("_eBel"), totalErrorSC("_eTot"),
        hammingSC("_resH"), searchComplexitySC("_readCX");

    for(int d=0; d<count; d++)
    {
        data->getNext(&in, &out);
        for(int r=0; r<runs; r++)
        {
            inH.setBipolarNoise(&in, hamming);
            test(&inH, d, &outH, &pathError, &belongingError, &searchComplexity);
            int resH = out.hammingDistance(&outH);
            hammingSC.addValue(resH);
            pathErrorSC.addValue(pathError? 1 : 0);
            totalErrorSC.addValue(belongingError? 1 : 0);
            if(!pathError)
                belongingErrorSC.addValue(belongingError? 1 : 0);
            searchComplexitySC.addValue(searchComplexity);
        }
    }

    pathErrorSC.getData(prop, SC_AVERAGE | SC_SD);
    belongingErrorSC.getData(prop, SC_AVERAGE | SC_SD);
    totalErrorSC.getData(prop, SC_AVERAGE | SC_SD);
    hammingSC.getData(prop, SC_AVERAGE | SC_SD);
    searchComplexitySC.getData(prop, SC_AVERAGE | SC_SD);

    return 1. - totalErrorSC.getAverage();
}



