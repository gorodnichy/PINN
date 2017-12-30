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

# include "ClusterTest.h"


// [0, x2], f = x
class LinearDistrRand : public RandomGenerator
{
public:
    LinearDistrRand(double x2) : RandomGenerator(0, x2, x2) { }

protected:
    double distribution(double arg) { return arg; }
};

// finds attractors using data as initial points
// found attractors sorted by (1) their frequencies and (2) difference coefficients
//
// NOTE:
// if numTests <=0 then all data points are used
void findAttractors(IOData *data, int numTests, AssociativeNet *net, TopHitAttr *topHitAttr, TopDiffAttr *topDiffAttr)
{
    assertion(data->inDim()==net->dim(), "[findAttractors] data/net dim mismatch");

    int dim = net->dim();
    BipolarVector in(dim), out(dim), dataOut(0);
    map<BipolarVector, int> attractors;
    bool isDynamicAttr;

    topHitAttr->clear();
    topDiffAttr->clear();

    for(int i=0; (numTests<=0||i<numTests) && data->hasNext(); i++)
    {

        data->getNext(&in, &dataOut);

        net->converge(&in, &out, &isDynamicAttr, false);
        //out.assign(&in); // to test without convergence

        out.mult(out[0], &out); // !! to avoid sign ambiguity

        //---------------------------------------------------------
        //attractors[out]++; // Calls copy constructor even if "attractors" already contains "out" !!!!!
        //                                                                (during the creation of value_type)
        //---------------------------------------------------------

        map<BipolarVector, int>::iterator iter;
        if((iter = attractors.find(out)) != attractors.end())
            iter->second++;
        else
            attractors.insert(map<BipolarVector, int>::value_type(out, 1));

        //---------------------------------------------------------

        printProgress("[findAttractors] vectors passed/different", i, 100, attractors.size());
    }
    cout << endl;

    int iterNum = 0;
    for(map<BipolarVector, int>::iterator iter = attractors.begin(); iter!=attractors.end(); iter++)
    {
        Attractor attr;
        attr.attractor = iter->first;
        attr.diff = net->getInputContentDiff(&attr.attractor);
        attr.freq = iter->second;

        topHitAttr->insert(multimap<int, Attractor, greater<int> >::value_type(attr.freq, attr));
        topDiffAttr->insert(multimap<double, Attractor>::value_type(attr.diff, attr));
        printProgress("[findAttractors] reordering attractors", ++iterNum, 100, attractors.size());
    }
    cout << endl;
}

// reads data file (MNIST or other format), runs convergence starting from stored images,
// stores found attractors sorted by (1) their frequencies and (2) difference coefficients
void testDataFile(Parameters params)
{
    FullProjectiveNet net(0);
    net.loadFromFile(params.getString("weightsFile"));

    //MnistData data(params.getString("inputsFile"), 8*1024*1024);
    TextRAMData data(params.getString("inputsFile"));
    data.open();

    int numTests = params.getInt("numTests");

    TopHitAttr topHitAttractors;
    TopDiffAttr topDiffAttractors;
    findAttractors(&data, numTests, &net, &topHitAttractors, &topDiffAttractors);

    string attrFile = params.getString("attrFile");
    int numSaved = 0, numToSave = params.getInt("numToSave");

    // save attractors ordered by frequency
    for(TopHitAttr::iterator iterHit = topHitAttractors.begin(); iterHit!=topHitAttractors.end(); iterHit++)
    {
        Attractor attractor = iterHit->second;
        Vector attr = attractor.attractor;
        attr.transpose();

        saveString(setWidth(toString(attractor.freq), 10) + "  " + toString(attractor.diff, 5), attrFile+"_freq", numSaved==0);
        saveMatrix(&attr, attrFile+"_freq_attr", false, numSaved==0);

        printProgress("[testDataFile] attractors saved", ++numSaved, 1, numToSave);

        if(numSaved >= numToSave)
            break;
    }
    cout << endl;

    // save attractors ordered by difference coefficient
    numSaved = 0;
    for(TopDiffAttr::iterator iterDiff = topDiffAttractors.begin(); iterDiff!=topDiffAttractors.end(); iterDiff++)
    {
        Attractor attractor = iterDiff->second;
        Vector attr = attractor.attractor;
        attr.transpose();

        saveString(setWidth(toString(attractor.diff, 5), 10) + "  " + toString(attractor.freq), attrFile+"_diff", numSaved==0);
        saveMatrix(&attr, attrFile+"_diff_attr", false, numSaved==0);

        printProgress("[testDataFile] attractors saved", ++numSaved, 1, numToSave);

        if(numSaved >= numToSave)
            break;
    }
    cout << endl;
}


// generate clusters
// save clusters
// generate projection matrices
// average them
// save averaged matrix
void testRandomClustersPartI(Parameters params)
{
    cout << "[testRandomClustersPart I]\n";

    // generate clusters
    int dim = params.getInt("dim");
    int numClusters = params.getInt("numClusters");

    //-------------------------------------------------------

    srand(6852167956);
    vector<BipolarVector> centers;
    Matrix centersM(0, dim);
    for(int i=0; i<numClusters; i++)
    {
        BipolarVector center(dim);
        center.fillBipolarRand();
        centers.push_back(center);
        centersM.append(&(center.transpose()));
    }

    // save clusters
    string clusterFile = params.getString("clusterFile");
    saveMatrix(&centersM, clusterFile, true, true);

    //---------------------------------------------------------------
    /*
    // load centers
    string clusterFile = params.getString("clusterFile");
    Matrix centersM;
    readMatrix(&centersM, clusterFile);
    assertion(centersM.sizeX()==dim, "[testRandomClustersPartI] centersM.sizeX() != net.dim()");
    assertion(centersM.sizeY() == numClusters, "[testRandomClustersPartI] centersM.sizeX() != net.dim()");
    centersM.transpose();
    vector<BipolarVector> centers;
    for(int i=0; i<numClusters; i++)
    {
        BipolarVector center(dim);
        center.setFromColumn(&centersM, i);
        //center.mult(center[0], &center); // !! to avoid sign ambiguity
        centers.push_back(center);
    }
    */
    //-------------------------------------------------------


    // generate projection matrices
    int clusterRadius = params.getInt("clusterRadius");
    int numProjMatrices = params.getInt("numProjMatrices");
    int projRank = params.getInt("projRank");

    Vector dataPoint(dim);
    Matrix sumProj(dim, dim), proj(dim, dim);
    sumProj.init(0);

    srand(497326378979);
    //LinearDistrRand linearDistr(clusterRadius);
    for(int m=0; m<numProjMatrices; m++)
    {
        proj.init(0);
        for(int r=0; r<projRank; r++) // find projective matrix
        {
            //----------------------------------------------
            //genaration of non-uniform data (concentrated around cluster centers)
            int hamming = rand(clusterRadius);
            //int hamming = linearDistr.rand();
            int clusterIndex = rand(numClusters);
            dataPoint.setBipolarNoise(&centers[clusterIndex], hamming);

            //----------------------------------------------
            // generation uniform data (might be slow for small clusterRadius)
            // ... it appeard to be very slow - for 8 clusters and clusterRadius = 96 about 1 datum/sec
            //bool flag = true;
            //while(flag)
            //{
            //    dataPoint.fillBipolarRand();
            //    for(int c=0; c<numClusters; c++)
            //        if(dataPoint.hammingDistance(&centers[c]) <= clusterRadius)
            //        {
            //            flag = false;
            //            cout << "yes! ";
            //            break;
            //        }
            //}
            //-------------------------------------------------------------------

            proj.expandProjectiveMatrix(&dataPoint);
        }

        sumProj.plus(&proj, &sumProj);
        printProgress("[testRandomClustersPartI] proj matrices constructed", m, 1, numProjMatrices);
    }
    cout << endl;

    // save averaged projective matrix
    string aveProjMatrixFile = params.getString("aveProjMatrixFile");
    saveMatrix(&sumProj, aveProjMatrixFile, true, true);
}


// load weight matrix
// find attractors
// load centers
// find correspondence between attractors/centers
// save correspondence
void testRandomClustersPartII(Parameters params)
{
    cout << "[testRandomClustersPart II]\n";

    srand(34624727);

    // load weight matrix
    string weightsFile = params.getString("weightsFile");
    FullProjectiveNet net(0);
    net.loadFromFile(weightsFile);

    // find attractors
    int dim = net.dim();
    int numRuns = params.getInt("numRuns");
    RandomBipolarRAMData data(dim, 0, numRuns);
    TopHitAttr topHitAttr;
    TopDiffAttr topDiffAttr;
    findAttractors(&data, -1, &net, &topHitAttr, &topDiffAttr);

    // load centers
    string clusterFile = params.getString("clusterFile");
    Matrix centersM;
    readMatrix(&centersM, clusterFile);
    assertion(centersM.sizeX()==dim, "[testRandomClustersPartII] centersM.sizeX() != net.dim()");
    int numClusters = centersM.sizeY();
    assertion(numClusters == params.getInt("numClusters"), "[testRandomClustersPartII] inner error");
    centersM.transpose();
    vector<BipolarVector> centers;
    for(int i=0; i<numClusters; i++)
    {
        BipolarVector center(dim);
        center.setFromColumn(&centersM, i);
        center.mult(center[0], &center); // !! to avoid sign ambiguity
        centers.push_back(center);
    }

    // find correspondence between attractors/centers
    // save correspondence
    // save attractors ordered by frequency
    string resFile = params.getString("resFile");
    saveString(params.toString(), resFile, true);
    int numAttractorsToSave = params.getInt("numAttractorsToSave");
    int numSaved = 0;
    for(TopHitAttr::iterator iterHit = topHitAttr.begin(); iterHit!=topHitAttr.end(); iterHit++)
    {
        Attractor attractor = iterHit->second;
        Vector attr = attractor.attractor;

        int closestIndex, smallestHamming;
        for(int i=0; i<centers.size(); i++)
        {
            int hamming = attr.hammingDistance(&centers[i]);
            if(i==0 || hamming<smallestHamming)
            {
                closestIndex = i;
                smallestHamming = hamming;
            }
        }

        int fieldWidth = 17;
        if(numSaved==0)
            saveString(string("\n\n") + setWidth("Num", fieldWidth) + setWidth("Freq", fieldWidth)
            + setWidth("Diff", fieldWidth) + setWidth("Min H to Center", fieldWidth)
            + setWidth("Closest Center #", fieldWidth), resFile);

        saveString(setWidth(toString(numSaved+1), fieldWidth)+ setWidth(toString(attractor.freq), fieldWidth)
            + setWidth(toString(attractor.diff, 6), fieldWidth)
            + setWidth(toString(smallestHamming), fieldWidth)
            + setWidth(toString(closestIndex), fieldWidth), resFile);
        printProgress("[testRandomClustersPartII] attractors saved", numSaved, 10, numAttractorsToSave);

        if(++numSaved >= numAttractorsToSave)
            break;
    }
    cout << endl;

    numSaved = 0;
    for(TopDiffAttr::iterator iterDiff = topDiffAttr.begin(); iterDiff!=topDiffAttr.end(); iterDiff++)
    {
        Attractor attractor = iterDiff->second;
        Vector attr = attractor.attractor;

        int closestIndex, smallestHamming;
        for(int i=0; i<centers.size(); i++)
        {
            int hamming = attr.hammingDistance(&centers[i]);
            if(i==0 || hamming<smallestHamming)
            {
                closestIndex = i;
                smallestHamming = hamming;
            }
        }

        int fieldWidth = 17;
        if(numSaved==0)
            saveString(setWidth("\n\nNum", fieldWidth) + setWidth("Freq", fieldWidth)
            + setWidth("Diff", fieldWidth) + setWidth("Min H to Center", fieldWidth)
            + setWidth("Closest Center #", fieldWidth), resFile);

        saveString(setWidth(toString(numSaved+1), fieldWidth)+ setWidth(toString(attractor.freq), fieldWidth)
            + setWidth(toString(attractor.diff, 6), fieldWidth)
            + setWidth(toString(smallestHamming), fieldWidth)
            + setWidth(toString(closestIndex), fieldWidth), resFile);
        printProgress("[testRandomClustersPartII] attractors saved", numSaved, 10, numAttractorsToSave);

        if(++numSaved >= numAttractorsToSave)
            break;
    }
    cout << endl;
}

