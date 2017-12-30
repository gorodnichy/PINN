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

# include "SmallWorld.h"


// net's description
string SmallWorldNet::getDescription()
{
    return PseudoInverseNet::getDescription() + "\n# Small-World cellular net, connR = "
        + toString(connR)
        + ", rewiringType = "+ (rewiringType==rtRandom? "rtRandom" : "rtSystematic")
        + ", rewiringProbability " + toString(rewiringProbability, 3);
}


double SmallWorldNet::getParameter(string parameterName)
{
    if(!parameterName.compare(PARAM_REWIRING_DEG))
        return rewiringProbability;

    return PseudoInverseNet::getParameter(parameterName);
}


void SmallWorldNet::setParameter(string parameterName, double value)
{
    if(!parameterName.compare(PARAM_REWIRING_DEG))
        rewiringProbability = value;
    else
        PseudoInverseNet::setParameter(parameterName, value);
}


void SmallWorldNet::subSetArchitecture()
{
    if(!keepDataInSSA)
    {
        projMatrix.setSizeYX(dim(), dim());
        projMatrix.init(0);
    }
    PseudoInverseNet::subSetArchitecture();
    setFreeArchitecture();
}


void SmallWorldNet::clear()
{
    projMatrix.init(0);
    PseudoInverseNet::clear();
}


bool SmallWorldNet::subTrain(Vector *in, Properties *prop)
{
    assertion(projMatrix.expandProjectiveMatrix(in), "[SmallWorldNet::subTrain] linearly dependent input");

    return PseudoInverseNet::subTrain(in, prop);
}


// form mask
// init(mask)
// fill weights using PseudoInverseNet train procedure
void SmallWorldNet::doEndTraining(Properties *prop)
{
    // set architecture (do rewiring)
    int numRewired;
    keepDataInSSA = true;
    switch (rewiringType)
    {
        case rtRandom:
            numRewired = doRandomRewiring();
        break;

        case rtSystematic:
            numRewired = doSystematicRewiring();
        break;

        default:
            assertion(false, "[SmallWorldNet::doEndTraining] unknown rewiring type");
    }
    keepDataInSSA = false;

    if(prop)
        (*prop)["numRewired"] = numRewired;

    for(int i=0; i<inputList.size(); i++)
        assertion(PseudoInverseNet::subTrain(inputList[i], prop),
        "[AdaptiveCellularNet::doEndTraining] PseudoInverseNet::subTrain failed");
    PseudoInverseNet::doEndTraining(prop);
}


// Sets regular 1D architecture and then does random rewiring
// returns the number of rewired connections
int SmallWorldNet::doRandomRewiring()
{
    setLocal1DArchitecture(dim(), connR);

    int dimension = mask.size();
    vector<int> connections;
    for(int i=0; i<mask.size(); i++) // connections are sorted
    {
        vector<int> &maski = mask[i];
        for(int j=0; j<maski.size(); j++)
            connections.push_back(i*dimension + maski[j]);
    }

    int numConnectioons = connections.size();

    int res = 0; // number of actually rewired connections
    set<int> newConnections;
    for(int i=0; i<numConnectioons; i++)
    {
        int connection = connections.back();
        connections.pop_back();
        if(rand(0, 1) < rewiringProbability)
        {
            //int rewiredConnection = rand(dimension*dimension); // ERROR - new connection may be not connected with initial neuron
            int rewiredConnection = (connection/dimension)*dimension + rand(dimension); // FIXED
            if(rewiredConnection != connection    // if new connection is different from any existing
               && indexOfSorted(&connections, rewiredConnection) == -1
               && newConnections.find(rewiredConnection) == newConnections.end()
               && (!noDiagonalWeights || rewiredConnection/dimension!=rewiredConnection%dimension) )
            {
                newConnections.insert(rewiredConnection);
                res++;
            }
            else // no rewiring because the connection already exists
            {
                assertion(newConnections.find(connection) == newConnections.end(),
                    "[SmallWorldNet::doRandomRewiring] algorithmic error");
                newConnections.insert(connection);
            }
        }
        else // no rewiring, copy as is
        {
                assertion(newConnections.find(connection) == newConnections.end(),
                    "[SmallWorldNet::doRandomRewiring] algorithmic error");
                newConnections.insert(connection);
        }
    }

    assertion(numConnectioons == newConnections.size(), "[SmallWorldNet::doRandomRewiring] algorithmic error");

    // create mask using newConnections
    mask.clear();
    mask.resize(dimension);

    for(set<int>::iterator iter = newConnections.begin(); iter!=newConnections.end(); iter++)
    {
        int connection = *iter;
        mask[connection/dimension].push_back(connection%dimension);
    }

    for(int i=0; i<mask.size(); i++)
        weights[i].assign(mask[i].size(), 0);
    onArchitectureChange();

    return res;
}


// Sets regular 1D architecture and then does systematic rewiring
// takes the set of the weakest connections from the grid (the ones with the list values of corresponding
// projMatrix elements) and rewires them to the set of the strongest connections out of the grid
// returns the number of rewired connections
int SmallWorldNet::doSystematicRewiring()
{
    setLocal1DArchitecture(dim(), connR);
    int dimension = dim();
    int res = 0; // the total number of actually rewired connections

    for(int neuron=0; neuron<dimension; neuron++) // cycle over all neurons
    {
        set<int> connections;
        vector<int> &maskN = mask[neuron];
        for(int j=0; j<maskN.size(); j++)
            connections.insert(maskN[j]);

        multimap<double, int, greater<double> > nonGridConnections; // descending order
        multimap<double, int, less<double> > gridConnections;       // ascending order

        for(set<int>::iterator iter=connections.begin(); iter!=connections.end(); iter++) // form gridConnections
        {
            int connection = *iter;
            double absel = fabs(projMatrix.el(neuron, connection));
            gridConnections.insert(multimap<double, int, less<double> >::value_type(absel, connection));
        }

        for(int i=0; i<dimension; i++) // form nanGridConnections
        {
            int connection = i;
            if(connections.find(i) == connections.end()) // not in the grid
            {
                double absel = fabs(projMatrix.el(neuron, connection));
                nonGridConnections.insert(multimap<double, int, greater<double> >::value_type(absel, connection));
            }
        }

        assertion(gridConnections.size()+nonGridConnections.size() == dimension,
            "[SmallWorldNet::doRandomRewiring] algorithmic error");

        multimap<double, int, less<double> >::iterator gridIter = gridConnections.begin();
        multimap<double, int, greater<double> >::iterator nonGridIter = nonGridConnections.begin();

        int integerPart = rewiringProbability*connections.size();
        double fractionalPart = rewiringProbability*connections.size() - integerPart;
        int numToRewire = integerPart + (rand(0, 1)<fractionalPart? 1 : 0);
        int rewired = 0;
        while(gridIter != gridConnections.end() && nonGridIter != nonGridConnections.end())
        {
            if(gridIter->first >= nonGridIter->first || rewired >= numToRewire)
                break; // if required number of rewires achieved or there no sence to do further rewiring

            set<int>::iterator grid = connections.find(gridIter->second);
            set<int>::iterator nonGrid = connections.find(nonGridIter->second);
            assertion(grid != connections.end(), "[SmallWorldNet::doRandomRewiring] algorithmic error");
            assertion(nonGrid == connections.end(), "[SmallWorldNet::doRandomRewiring] algorithmic error");

            if(!noDiagonalWeights || nonGridIter->second!=neuron)
            {
                connections.erase(grid);
                connections.insert(nonGridIter->second);
                gridIter++;
                rewired++;
                res++;
            }
            nonGridIter++;
        }

        // create mask using newConnections
        maskN.clear();

        for(set<int>::iterator iter=connections.begin(); iter!=connections.end(); iter++)
            maskN.push_back(*iter);
    }

    for(int i=0; i<mask.size(); i++)
        weights[i].assign(mask[i].size(), 0);
    onArchitectureChange();

    return res;
}

/* // Old version - with an ERROR - new connection may be not connected with initial neuron
// Sets regular 1D architecture and then does systematic rewiring
// takes the set of the weakest connections from the grid (the ones with the list values of corresponding
// projMatrix elements) and rewires them to the set of the strongest connections out of the grid
// returns the number of rewired connections
int SmallWorldNet::doSystematicRewiring()
{
    setLocal1DArchitecture(dim(), connR);
    int dimension = dim();

    set<int> connections;
    for(int i=0; i<mask.size(); i++)
    {
        vector<int> &maski = mask[i];
        for(int j=0; j<maski.size(); j++)
            connections.insert(i*dimension + maski[j]);
    }

    //sort(connections.begin(), connections.end()); // not necessarily

    multimap<double, int, greater<double> > nonGridConnections; // descending order
    multimap<double, int, less<double> > gridConnections;       // ascending order

    for(set<int>::iterator iter=connections.begin(); iter!=connections.end(); iter++) // form gridConnections
    {
        int connection = *iter;
        double absel = fabs(projMatrix.el(connection/dimension, connection%dimension));
        gridConnections.insert(multimap<double, int, less<double> >::value_type(absel, connection));
    }

    for(int i=0; i<dimension*dimension; i++) // form nanGridConnections
    {
        int connection = i;
        if(connections.find(i) == connections.end()) // not in the grid
        {
            double absel = fabs(projMatrix.el(connection/dimension, connection%dimension));
            nonGridConnections.insert(multimap<double, int, greater<double> >::value_type(absel, connection));
        }
    }

    assertion(gridConnections.size()+nonGridConnections.size() == dimension*dimension,
        "[SmallWorldNet::doRandomRewiring] algorithmic error");

    multimap<double, int, less<double> >::iterator gridIter = gridConnections.begin();
    multimap<double, int, greater<double> >::iterator nonGridIter = nonGridConnections.begin();

    int numToRewire = rewiringProbability*connections.size();
    int res = 0; // the number of actually rewired connections
    while(gridIter != gridConnections.end() && nonGridIter != nonGridConnections.end())
    {
        if(gridIter->first > nonGridIter->first || res >= numToRewire)
            // if required number of rewires achieved or there no sence to do further rewiring
            break;

        set<int>::iterator grid = connections.find(gridIter->second);
        set<int>::iterator nonGrid = connections.find(nonGridIter->second);
        assertion(grid != connections.end(), "[SmallWorldNet::doRandomRewiring] algorithmic error");
        assertion(nonGrid == connections.end(), "[SmallWorldNet::doRandomRewiring] algorithmic error");

        if(!noDiagonalWeights || nonGridIter->second/dimension!=nonGridIter->second%dimension)
        {
            connections.erase(grid);
            connections.insert(nonGridIter->second);
            gridIter++;
            res++;
        }
        nonGridIter++;
    }

    // create mask using newConnections
    mask.clear();
    mask.resize(dimension);

    for(set<int>::iterator iter=connections.begin(); iter!=connections.end(); iter++)
    {
        int connection = *iter;
        mask[connection/dimension].push_back(connection%dimension);
    }

    for(int i=0; i<mask.size(); i++)
        weights[i].assign(mask[i].size(), 0);
    onArchitectureChange();

    return res;
}*/

