/// BEGIN_COPYRIGHT
/*------------------------------------------------------------------------------
*  Code:      Library that allows to create, train and evaluate
*             various kinds of Associative Neural Networks.
*  Author:    Oleksiy K. Dekhtyarenko, 2003-2005
*             name@domain, name=olexii, domain=mail.ru
*  Copyright: The author grants the right to use and modify the code
*             provided suitable acknowledgements and citations are made.
*-----------------------------------------------------------------------------*/
// END_COPYRIGHT

# include "nets.h"


# define CONV_COEFF 1.0  // dim*CONV_COEFF is a maximum number of iterations during convergence
# define PROCESS_BINARY  // convergence optimization (uses incrementsl updates to calculate the next state of the network)


//-------------------------- AssociativeUnit -----------------------------------

// basic description
string AssociativeUnit::getDescription()
{
    return string("# AssociativeUnit, id ") + getID() +
        ", input dim = " + toString(dim()) +
        ", num stored = " + toString(numStored());
}


//-------------------------- AssociativeNet ------------------------------------

string AssociativeNet::getDescription()
{
    return AssociativeUnit::getDescription() + "\n# AssociativeNet, desat coeff =  " + toString(desatCoeff, 5);
}

double AssociativeNet::getParameter(string parameterName)
{
    if(!parameterName.compare(PARAM_NET_DIM))
        return dim();
    else if(!parameterName.compare(PARAM_DESAT_COEFF))
        return desatCoeff;
    else
        assertion(false, string("[AssociativeNet::getParameter] uknown parameter: ")+parameterName);
}


void AssociativeNet::setParameter(string parameterName, double value)
{
    if(!parameterName.compare(PARAM_DESAT_COEFF))
    {
        desatCoeff = value;
        onWeightsChange();
    }
    else
        assertion(false, string("[AssociativeNet::setParameter] uknown parameter: ")+parameterName);
}

bool AssociativeNet::train(Vector *in, Properties *prop)
{
    assertion(in->size() == dim(), "[AssociativeNet::train] diff sizes");
    AssociativeUnit::train(in);

    bool res;
    if(res = subTrain(in, prop))
        inputList.push_back(new Vector(in));

    log(string("[train] numStored = ") + toString(numStored()), getID());

    return res;
}


void AssociativeNet::doEndTraining(Properties *prop)
{
    AssociativeUnit::doEndTraining();
    onWeightsChange();
    # ifdef PROCESS_BINARY
    processBinaryCheck(); //  check that increment convergence process works properly
    # endif

    if(prop != 0)
        getProperties(prop);
}


void AssociativeNet::clear()
{
    while(inputList.size())
    {
        delete inputList.back();
        inputList.pop_back();
    }

    onWeightsChange();
    AssociativeUnit::clear();
}


// net properties
void AssociativeNet::getProperties(Properties *prop)
{
    AssociativeUnit::getProperties(prop);

    (*prop)["trace(W)"]	    = getTrace();
    (*prop)["|W-I|^2"]	    = getIDiff();
    (*prop)["|W|^2"]	    = getWSqNorm();
    (*prop)["|W-W_tr|^2"]   = getSymDiff();
    (*prop)["WijWji/|W|^2"] = getNormalizedSymDiff();
    (*prop)["numStored"]    = numStored();
    (*prop)["desatCoeff"]	= desatCoeff;
    (*prop)["lastDiff"]     = getInputContentDiff(inputList.back());
    (*prop)["<diff>"]       = getAverageContentDiff();
    (*prop)["|WV-V|^2"]     = getDiscrepancy();
    (*prop)["k_measure"]    = getKMeasure();
    (*prop)["min_ALF"]      = getMinALF();
    (*prop)["gammaAve"]     = getAveGammaMeasure();
}


// update auxilary data & weights
void AssociativeNet::onWeightsChange()
{
    currState.assign(dim(), 0.5);
    oneHPS.setSize(dim());
    currHPS.setSize(dim());
    process(&currState, &oneHPS);
    currState.assign(dim(), 1);
}


// checks if processBinary works correct
void AssociativeNet::processBinaryCheck()
{
    Vector in(dim()), outBinary(dim()), outRegular(dim());
    in.fillBipolarRand();
    processBinary(&in, &outBinary, false);
    process(&in, &outRegular, true);
    double sqdist = outBinary.sqdist(&outRegular);
    assertion(sqdist < PSEUDO_INVERSE_EPS, "[AssociativeNet::processBinaryCheck] failed");
}


// returns num of iterations if settled in less then dim*CONV_COEFF or 0 otherwise
// isDynamicAttractor is set to true if settled in dymamic attractor, false otherwise
// if singleIter then no convergence is performed, "in" is processed by the net just once
int AssociativeNet::converge(Vector *in, Vector *out, bool *isDynamicAttractor, bool singleIter)
{
    AssociativeUnit::converge(in, out, isDynamicAttractor, singleIter);

	assertion(in->size()==dim() && out->size() == dim(), "[AssociativeNet::converge] diff sizes");

	Vector *v1, *v2, *v3, *tmp, vec1(*in), vec2(dim()), vec3(dim());
	v1 = &vec1;
	v2 = &vec2;
	v3 = &vec3;

# ifdef PROCESS_BINARY
    processBinary(v1, v2, false);
# else
    process(v1, v2, true);
# endif

	v2->setSign(v2);
	if(singleIter)
	{
		out->assign(v2);
        if(isDynamicAttractor != 0)
    		*isDynamicAttractor = false;
		return 1;
	}

# ifdef PROCESS_BINARY
    processBinary(v2, v3, true);
# else
	process(v2, v3, true);
# endif

	v3->setSign(v3);
	int maxNumIter = dim()*CONV_COEFF/2, numIter;
	bool isDynamic = false;
	for(numIter=1; numIter<=maxNumIter; numIter++)
	{
		if(v1->binaryEquals(v2))
		{
			out->assign(v1);
			isDynamic = false;
			break;
		}
		else if(v1->binaryEquals(v3))
		{
			out->assign(v1);
			isDynamic = true;
			break;
		}
		
		tmp = v1;
		v1 = v3;
		v3 = tmp;

# ifdef PROCESS_BINARY
		processBinary(v1, v2, true);
		v2->setSign(v2);
        processBinary(v2, v3, true);
# else
        process(v1, v2, true);
		v2->setSign(v2);
		process(v2, v3, true);
# endif

        v3->setSign(v3);
		if(numIter == maxNumIter)
			out->assign(v3);
	}

	if(isDynamicAttractor != 0)
		*isDynamicAttractor = isDynamic;

	return numIter<=maxNumIter? numIter*2 : 0;
}


// rather slow implementation of convergence as it tracks all the possible cycles,
// that is          network_state(t + delta) = network_state(t)
// once the cycle is reached, its length is stored in "cycleLength"
// cycleLength==1 <=> fixed Point
// rerutns 0 if no such state has been achieved within dim()*CONV_COEFF iterations
int AssociativeNet::extendedConverge(Vector *in, Vector *out, int *cycleLength)
{
    AssociativeUnit::extendedConverge(in, out, cycleLength);

	assertion(in->size()==dim() && out->size() == dim(), "[AssociativeNet::extendedConverge] diff sizes");

	BipolarVector *v1, *v2, *tmp, vec1(dim()), vec2(dim());
    vec1.assign(in);
	v1 = &vec1;
	v2 = &vec2;

    map<BipolarVector, int> visitedStates;
    map<BipolarVector, int>::iterator iter;

    visitedStates.insert(map<BipolarVector, int>::value_type(*v1, 0));
    int maxNumIter = dim()*CONV_COEFF;
    for(int i=1; i<maxNumIter; i++)
    {
        # ifdef PROCESS_BINARY
        processBinary(v1, v2, i>0);
        # else
        process(v1, v2, true);
        # endif

        v2->setSign(v2);

        iter = visitedStates.find(*v2);
        if(iter != visitedStates.end())
        {
            *cycleLength = i - iter->second;
            return i;
        }

        visitedStates.insert(map<BipolarVector, int>::value_type(*v2, i));
        tmp = v1;
		v1 = v2;
		v2 = tmp;
    }

    // did not converge
    *cycleLength = -1;
    return 0;
}


// |in - process(in)|^2/|in|^2
double AssociativeNet::getInputContentDiff(Vector *in)
{
	assertion(in->size() == dim(), "[AssociativeNet::getInputContentDiff] diff sizes");

	Vector vec(dim());
	process(in, &vec, false);
	vec.minus(in, &vec);

    double inSqNorm = in->sqnorm();
	double ortSqNorm = vec.sqnorm();

	return inSqNorm>0? ortSqNorm/inSqNorm : 0;
}


// averaged over all trained vectors
double AssociativeNet::getAverageContentDiff()
{
    assertion(numStored() > 0, "[AssociativeNet::getAverageContentDiff] numStored == 0");

    double sumDiff = 0;
	for(int d=0; d<inputList.size(); d++)
		sumDiff += getInputContentDiff(inputList[d]);

    return sumDiff/inputList.size();
}


// |WV-V|^2
double AssociativeNet::getDiscrepancy()
{
    assertion(numStored() > 0, "[AssociativeNet::getDiscrepancy] numStored == 0");

    double res = 0;
    Vector vec(dim());
	for(int d=0; d<inputList.size(); d++)
    {
        process(inputList[d], &vec, false);
	    vec.minus(inputList[d], &vec);
		res += vec.sqnorm();
    }

    return res;
}


// minimum/average (normalized) aligned local field
double AssociativeNet::getGammaMeasure(bool isNormalized, bool isMinimum)
{
    assertion(numStored()>0, "[AssociativeNet::getGammaMeasure] numStored == 0");

    Vector postSynapse(dim()), norms(dim());
    if(isNormalized)
        for(int i=0; i<dim(); i++)
        {
            norms[i] = sqrt(getWSqNorm(i));
            //assertion(norms[i]>0, "[AssociativeNet::getGammaMeasure] weight with zero norm");
            if(norms[i] == 0)
                return NOT_A_NUMBER;
        }

    StatCounter res;
    double currVal;
    for(int d=0; d<inputList.size(); d++)
    {
        process(inputList[d], &postSynapse, false);
        for(int i=0; i<dim(); i++)
        {
            currVal = (*inputList[d])[i]*postSynapse[i];
            if(isNormalized)
                currVal /= norms[i];
            res.addValue(currVal);
        }
    }

    return isMinimum? res.getMin() : res.getAverage();
}


//----------------------------- FullNet ----------------------------------------

void FullNet::setParameter(string parameterName, double value)
{
    if(!parameterName.compare("dim"))
        setDim(round(value));
    else
        AssociativeNet::setParameter(parameterName, value);
}


// clears net content without structural reconfiguration
void FullNet::clear()
{
    weights.init(0);
	log("[FullNet::clear]", getID());

    AssociativeNet::clear();
}


void FullNet::saveWeights(string fileName)
{
    saveString(getDescription() + "\n", fileName, true);
    saveMatrix(&weights, fileName);
}


// trace of the weight matrix
double FullNet::getTrace()
{
    double *data = weights.getData(), res = 0;
    int aDim = dim();
    for(int i=0; i<aDim; i++)
    {
        res += *data;
        data += aDim+1;
    }
    return res;
}


// |W-I|^2,
// where |M|^2 = sum el(i,j)^2
double FullNet::getIDiff()
{
    double *data = weights.getData(), res = 0;
    int aDim = dim();
    for(int i=0; i<aDim; i++)
        for(int j=0; j<aDim; j++)
        {
            double diff = i==j? *data++ -1 : *data++;
            res += diff*diff;
        }
    return res;
}


// |W|^2,
// where |M|^2 = sum el(i,j)^2
double FullNet::getWSqNorm()
{
    return weights.sqnorm();
}


// |W_neuron|^2,
// where |M|^2 = sum el(i,j)^2
double FullNet::getWSqNorm(int neuron)
{
    assertion(neuron>=0 && neuron<dim(), "[FullNet::getWSqNorm] neuron index out of bounds");

    Matrix neuronWeights;
    neuronWeights.setFromRow(&weights, neuron);
    return neuronWeights.sqnorm();
}


// |W - W_tr|^2,
// where |M|^2 = sum el(i,j)^2
double FullNet::getSymDiff()
{
    double res = 0;
    int aDim = dim();
    for(int i=0; i<aDim; i++)
    {
        double *datacol = weights.getData()+i, *datarow = weights.getData()+i*aDim;
        for(int j=0; j<aDim; j++)
        {
            double diff = *datacol - *datarow;
            res += diff*diff;
            datacol += aDim;
            datarow += 1;
        }
    }
    return res;
}


// Sum WijWji / Sum Wij^2
double FullNet::getNormalizedSymDiff()
{
    double crossSum = 0, sqSum = 0;
    int aDim = dim();
    for(int i=0; i<aDim; i++)
    {
        double *datacol = weights.getData()+i, *datarow = weights.getData()+i*aDim;
        for(int j=0; j<aDim; j++)
        {
            crossSum += *datacol * *datarow;
            sqSum += *datarow * *datarow;
            datacol += aDim;
            datarow += 1;
        }
    }

    assertion(sqSum>0, "[FullNet::getNormalizedSymDiff] zero norm weights");
	return crossSum/sqSum;
}


// out = desaturated_weights*in + bias
// desaturated_weights(i,j) = weights(i,j) * (1-delta(i,j)*(1-desatCoeff))
// direct multiplication
void FullNet::process(Vector *in, Vector *out, bool useDesaturation)
{
	assertion(in->size() == dim() && out->size() == dim(), "[FullNet::process] diff sizes");

    double *data = weights.getData(), ps, *inBuf = in->getData(), *outBuf = out->getData();
    int aDim = dim();
    for(int i=0; i<aDim; i++)
    {
        ps = 0;
        for(int j=0; j<i; j++)
            ps += inBuf[j] * *data++;
        ps += inBuf[i] * *data++ * (useDesaturation? desatCoeff : 1);
        for(int j=i+1; j<aDim; j++)
            ps += inBuf[j] * *data++;
        outBuf[i] = ps;
    }
}


// out = weights * in
// "cached-style" multiplication for binary (+1/-1) data
// set useCurrState = true if "in" in current call is close to "in" in previous
void FullNet::processBinary(Vector *in, Vector *out, bool useCurrState)
{
    checkIfTrainingEnded("FullNet::processBinary");

	assertion(in->size() == dim() && out->size() == dim(), "[FullNet::processBinary] diff sizes");

    int aDim = dim();
    double *inBuf = in->getData();

	if(!useCurrState)
    {
        int nPlusOne = 0;
        for(int i=0; i<aDim; i++)
	        if(inBuf[i] > 0)
                nPlusOne++;
        if(nPlusOne > aDim/2)
        {
            currState.assign(aDim, 1);
            currHPS.assign(&oneHPS);
        }
        else
        {
            currState.assign(aDim, -1);
            oneHPS.mult(-1, &currHPS);
        }
        processBinary(in, out, true);
    }
    else
    {
        double *currStateBuf = currState.getData(), *currHPSBuf = currHPS.getData();
        for(int i=0; i<aDim; i++)
            if(fabs(inBuf[i]-currStateBuf[i]) > 0.5)
            {
                double *datacol = weights.getData()+i; 
                currStateBuf[i] = inBuf[i];
                if(currStateBuf[i] > 0)
                {
                    for(int j=0; j<i; j++)
                    {
                        currHPSBuf[j] += *datacol;
                        datacol += aDim;
                    }
                    currHPSBuf[i] += *datacol*desatCoeff;
                    datacol += aDim;
                    for(int j=i+1; j<aDim; j++)
                    {
                        currHPSBuf[j] += *datacol;
                        datacol += aDim;
                    }
                }
                else
                {
                    for(int j=0; j<i; j++)
                    {
                        currHPSBuf[j] -= *datacol;
                        datacol += aDim;
                    }
                    currHPSBuf[i] -= *datacol*desatCoeff;
                    datacol += aDim;
                    for(int j=i+1; j<aDim; j++)
                    {
                        currHPSBuf[j] -= *datacol;
                        datacol += aDim;
                    }
                }
            }
        currHPS.mult(2, out);
    }
}


//--------------------------- CellularNet --------------------------------------


string CellularNet::getDescription()
{
    return AssociativeNet::getDescription() + "\n# CellularNet, topology " + getTopology()
        + "(" + toString(sizeY) + "x" + toString(sizeX) + "), connR " + toString(getConnR())
        + ", noDiagonalWeights " + toStringB(noDiagonalWeights)
        + ", num weights " + toString(numWeights())
        + ", connectivityPortion " + toString(connectivityPortion, 3)
        + " (" + toString(dim()*dim()*connectivityPortion) + " weights)";
}


void CellularNet::getProperties(Properties *prop)
{
    (*prop)["numWghs"] = numWeights();
    (*prop)["numWgtsDiag"] = numDiagonalWeights();
    (*prop)["numWgtsMin"] = minNumNeuronWeights();
    (*prop)["numWgtsMax"] = maxNumNeuronWeights();
    (*prop)["ttlConnL"] = totalConnectionLength();
    AssociativeNet::getProperties(prop);
}


string CellularNet::getTopology()
{
    if(topology == tpUnset)
        return "tpUnset";
    else if(topology == tpPredefined)
        return "tpPredefined";
    else if(topology == tp1D)
        return "tp1D";
    else if(topology == tp2D)
        return "tp2D";
    else if(topology == tpRandom)
        return "tpRandom";
    else if(topology == tpFree)
        return "tpFree";
    else
    {
        assertion(false, "[CellularNet::getTopology] undefined topology");
        return "";
    }
}


// number of mask elements
int CellularNet::numWeights()
{
    int res = 0;
    for(int i=0; i<dim(); i++)
        res += mask[i].size();
    return res;
}


// number of diagonal mask elements
int CellularNet::numDiagonalWeights()
{
    int res = 0;
    for(int i=0; i<dim(); i++)
        if(diagonalInd[i] < mask[i].size())
            res++;
    return res;
}


// minimum number of weights a neuron has
int CellularNet::minNumNeuronWeights()
{
    if(dim() == 0)
        return NOT_A_NUMBER;

    StatCounter sc;
    for(int i=0; i<dim(); i++)
        sc.addValue(mask[i].size());

    return round(sc.getMin());
}


// maximum number of weights a neuron has
int CellularNet::maxNumNeuronWeights()
{
    if(dim() == 0)
        return NOT_A_NUMBER;

    StatCounter sc;
    for(int i=0; i<dim(); i++)
        sc.addValue(mask[i].size());

    return round(sc.getMax());
}


// total connection length
int CellularNet::totalConnectionLength()
{
    if(dim() == 0)
        return NOT_A_NUMBER;

    int res = 0;
    for(int i=0; i<dim(); i++)
    {
        vector<int> &maski = mask[i];
        for(int j=0; j<maski.size(); j++)
            res += abs(i-maski[j]);
    }

    return res;
}


// inits mask and weights according to dim and connR values
// 1D (linear) neurons' topology
// neurons are wrapped into circle, two neurons (i,j) are connected
// if dist(i,j) <= connR
void CellularNet::setLocal1DArchitecture(int dimension, int connR)
{
    topology = tp1D;
	if(connR > dimension/2)
		connR = dimension/2;
	this->connR = connR;
    mask.clear();
	mask.resize(dimension);
    weights.clear();
	weights.resize(dimension);
	for(int i=0; i<dimension; i++)
	{
		vector<int> &maski = mask[i];

		for(int j=i-connR; j<=i+connR; j++)
            if(noDiagonalWeights && (j+dimension)%dimension==i)
                continue;
            else
                maski.push_back((j+dimension)%dimension);

		sort(maski.begin(), maski.end());
		vector<int>::iterator end = unique(maski.begin(), maski.end());
        maski.erase(end, maski.end());
		weights[i].assign(maski.size(), 0);
	}
    bias.assign(dimension, 0);
    indices.assign(dimension, 0);

    onArchitectureChange();

	log("[CellularNet::set1DArchitecture]", getID());
}


// inits mask and weights according to sizeX,Y and connR values
// 2D (planar) neurons' topology
// two neurons (i,j)&(k,l) are connected if max(|i-k|,|j-l|) <= connR
void CellularNet::setLocal2DArchitecture(int sizeY, int sizeX, int connR)
{
    topology = tp2D;
    this->sizeY = sizeY;
    this->sizeX = sizeX;
    this->connR = connR;
	mask.clear();
    int dimension = sizeX*sizeY;
    mask.clear();
    mask.resize(dimension);
    weights.clear();
	weights.resize(dimension);
    for(int y=0; y<sizeY; y++)
        for(int x=0; x<sizeX; x++)
        {
            int x0 = (x-connR<0)? 0 : x-connR;
            int x1 = (x+connR>=sizeX)? sizeX-1 : x+connR;
            int y0 = (y-connR<0)? 0 : y-connR;
            int y1 = (y+connR>=sizeY)? sizeY-1 : y+connR;
            vector<int> &maskxy = mask[y*sizeX+x];
            for(int j=y0; j<=y1; j++)
                for(int i=x0; i<=x1; i++)
                    if(noDiagonalWeights && y*sizeX+x==j*sizeX+i)
                        continue;
                    else
                        maskxy.push_back(j*sizeX+i);
            weights[y*sizeX+x].assign(maskxy.size(), 0);
        }
    bias.assign(dimension, 0);
    indices.assign(dimension, 0);

    onArchitectureChange();

	log("[CellularNet::set2DArchitecture]", getID());
}


// inits mask with random architecture
// with connectivityPortion * dimension^2 weights
void CellularNet::setRandomArchitecture(int dimension, float connectivityPortion)
{
    assertion(connectivityPortion>=0 && connectivityPortion<=1,
        "[CellularNet::setRandomArchitecture] connectivityPortion must be within [0, 1]");

    this->connectivityPortion = connectivityPortion;

    topology = tpRandom;
    mask.clear();
	mask.resize(dimension);
    weights.clear();
	weights.resize(dimension);

    vector<int> indices;
    indices.reserve(dimension*dimension);
    for(int i=0; i<dimension*dimension; i++)
        indices.push_back(i);
    random_shuffle(indices.begin(), indices.end());

    int numWeigths = connectivityPortion*dimension*dimension;
	for(int i=0; i<numWeigths; i++)
        if(noDiagonalWeights && indices[i]/dimension==indices[i]%dimension)
            continue;
        else
            mask[ indices[i]/dimension ].push_back( indices[i]%dimension );

    for(int i=0; i<dimension; i++)
    {
        sort(mask[i].begin(), mask[i].end());
        if(mask[i].size() == 0)
            mask[i].push_back(rand(dimension));
        weights[i].assign(mask[i].size(), 0);
    }

    onArchitectureChange();

	log("[CellularNet::setRandomArchitecture]", getID());
}


// inits mask according to the argument
// expands weights
void CellularNet::setPredefinedArchitecture(vector<vector<int> > *scheme)
{
    topology = tpPredefined ;
    int dimension = scheme->size();
    mask.clear();
	mask.resize(dimension);
    weights.clear();
	weights.resize(dimension);
	for(int i=0; i<dimension; i++)
    {
        mask[i].assign((*scheme)[i].begin(), (*scheme)[i].end());
        weights[i].assign(mask[i].size(), 0);
    }

    onArchitectureChange();

	log("[CellularNet::setPredefinedArchitecture]", getID());
}


// clears net content without structural reconfiguration
void CellularNet::clear()
{
    assertion(topology != tpUnset, "[CellularNet::clear] topology == tpUnset");

	for(int i=0; i<dim(); i++)
		weights[i].init(0);
    bias.init(0);
    AssociativeNet::clear();
	log("[CellularNet::clear]", getID());
}


// inits random weights (without structural reconfiguration)
void CellularNet::initRandomWeights(double range)
{
    assertion(topology != tpUnset, "[CellularNet::initRandomWeights] topology == tpUnset");

	for(int i=0; i<dim(); i++)
		weights[i].fillRand(-range, range);

    onWeightsChange();
}


// saves full (zeros instead non-existent connections) weight matrix to file
void CellularNet::saveWeights(string fileName)
{
    assertion(topology != tpUnset, "[CellularNet::saveWeights] topology == tpUnset");
    checkIfTrainingEnded("CellularNet::saveWeights");

    ofstream ofs(fileName.c_str());	// :: ???
	assertion(!ofs.fail(), string("[CellularNet::saveWeights] cant open file ") + fileName);

    ofs << getDescription() << endl << endl;
    ofs << dim() << " " << dim() << endl;
    for(int i=0; i<dim(); i++)
    {
        vector<int> &maski = mask[i];
        Vector &weightsi = weights[i];
        for(int j=0; j<dim(); j++)
        {
            int index = indexOfSorted(&maski,j);
            if(index<0)
                ofs << 0;
            else
                ofs << weightsi[index];
		    if(j < dim()-1)
			    ofs << " ";
        }
        ofs << endl;
    }
	ofs.close();
}


// saves full (zeros instead non-existent connections) mask matrix to file
void CellularNet::saveMask(string fileName)
{
    assertion(topology != tpUnset, "[CellularNet::saveMask] topology == tpUnset");
    checkIfTrainingEnded("CellularNet::saveMask");

    ofstream ofs(fileName.c_str());	// :: ???
	assertion(!ofs.fail(), string("[CellularNet::saveMask] cant open file ") + fileName);

    ofs << getDescription() << endl << endl;
    ofs << dim() << " " << dim() << endl;
    for(int i=0; i<dim(); i++)
    {
        vector<int> &maski = mask[i];
        for(int j=0; j<dim(); j++)
        {
            int index = indexOfSorted(&maski, j);
            if(index<0)
                ofs << 0;
            else
                ofs << 1;
		    if(j < dim()-1)
			    ofs << " ";
        }
        ofs << endl;
    }
	ofs.close();
}


// trace of the weight matrix
double CellularNet::getTrace()
{
    assertion(topology != tpUnset, "[CellularNet::getTrace] topology == tpUnset");

    double res = 0;
	for(int i=0; i<dim(); i++)
		res += diagonalInd[i]<mask[i].size()? weights[i][ diagonalInd[i] ] : 0;
	return res;
}


// |W-I|^2,
// where |M|^2 = sum el(i,j)^2
double CellularNet::getIDiff()
{
    assertion(topology != tpUnset, "[CellularNet::getIDiff] topology == tpUnset");

	int index;
	double res = 0;
	for(int i=0; i<dim(); i++)
	{
		index = indexOfSorted(&mask[i], i);
		for(int j=0; j<mask[i].size(); j++)
			res += j==index? (weights[i][j]-1)*(weights[i][j]-1) : weights[i][j]*weights[i][j];
	}

	return res;
}


// |W|^2,
// where |M|^2 = sum el(i,j)^2
double CellularNet::getWSqNorm()
{
    assertion(topology != tpUnset, "[CellularNet::getWSqNorm] topology == tpUnset");

	int index;
	double res = 0;
	for(int i=0; i<dim(); i++)
    {
        double *weightsi = weights[i].getData();
		for(int j=0; j<mask[i].size(); j++)
        {
            double weight = weightsi[j];
            res += weight*weight;
        }
    }

	return res;
}


// |W_neuron|^2,
// where |M|^2 = sum el(i,j)^2
double CellularNet::getWSqNorm(int neuron)
{
    assertion(topology != tpUnset, "[CellularNet::getWSqNorm] topology == tpUnset");
    assertion(neuron>=0 && neuron<dim(), "[CellularNet::getWSqNorm] neuron index out of bounds");

    double res = 0;
    double *weightsN = weights[neuron].getData();
    int size = mask[neuron].size();
    for(int i=0; i<size; i++)
        res += weightsN[i]*weightsN[i];

    return res;
}


// |W - W_tr|^2,
// where |M|^2 = sum el(i,j)^2
double CellularNet::getSymDiff()
{
    assertion(topology != tpUnset, "[CellularNet::getSymDiff] topology == tpUnset");

    double res = 0, diff;
    for(int i=0; i<dim(); i++)
	{
        vector<int> &maski = mask[i];
        double *weightsi = weights[i].getData();
        int maskisize = maski.size();
        vector<int> &maskTi = maskT[i];
        double *weightsTi = weightsT[i].getData();
        for(int j=0; j<maskisize; j++)
        {
            int maskij = maski[j];
            int index = indexOfSorted(&maskTi, maskij);
            diff = weightsi[j] - (index>=0? weightsTi[index] : 0);
            res += diff*diff;
        }
    }
	return res;
}


// Sum WijWji / Sum Wij^2
double CellularNet::getNormalizedSymDiff()
{
    assertion(topology != tpUnset, "[CellularNet::getNormalizedSymDiff] topology == tpUnset");

    double crossSum = 0, sqSum = 0;
    for(int i=0; i<dim(); i++)
	{
        vector<int> &maski = mask[i];
        double *weightsi = weights[i].getData();
        int maskisize = maski.size();
        vector<int> &maskTi = maskT[i];
        double *weightsTi = weightsT[i].getData();
        for(int j=0; j<maskisize; j++)
        {
            int maskij = maski[j];
            int index = indexOfSorted(&maskTi, maskij);
            crossSum += weightsi[j] * (index>=0? weightsTi[index] : 0);
            sqSum += weightsi[j]*weightsi[j];
        }
    }

    assertion(sqSum>0, "[CellularNet::getNormalizedSymDiff] zero norm weights");
	return crossSum/sqSum;
}


// fills weightsT
// thens oneHPS by calling AssociativeNet::onWeightsChange();
void CellularNet::onWeightsChange()
{
    indices.assign(dim(),0);

    for(int i=0; i<dim(); i++) // loop over weightsT[i]
    {
        int index = 0;
        double *weightsTi = weightsT[i].getData();
        for(int j=0; j<dim(); j++)
            // indices[j] is the index of weight in row j with matrix index >= i
            if(indices[j]>=0 && mask[j][indices[j]]==i)
            {
                weightsTi[index++] = weights[j][indices[j]];
                indices[j]++;
                if(indices[j] == mask[j].size())
                    indices[j] = -1; // end of j-th row is reached
            }
        // check that weightsT[i] is filled completely
        assertion(index == maskT[i].size(), "[CellularNet::setAuxilary] algorithmic error");
    }

    AssociativeNet::onWeightsChange();
}


// constructs diagonalInd(T), maskT and weightsT
// fills weightsT
// thens oneHPS by calling AssociativeNet::onWeightsChange();
void CellularNet::onArchitectureChange()
{
    maskRangeOrderingConnectivityCheck();
    maskWeightCorrespondenceCheck();

    indices.assign(dim(),0);

    diagonalInd.clear();
    diagonalIndT.clear();
    maskT.clear();
    maskT.resize(dim());
    weightsT.clear();
    weightsT.resize(dim());

    for(int i=0; i<dim(); i++) // loop over weightsT[i]
    {
        vector<int> &maskTi = maskT[i];
        Vector &weightsTi = weightsT[i];
        for(int j=0; j<dim(); j++)
            // indices[j] is the index of weight in row j with matrix index >= i
            if(indices[j]>=0 && mask[j][indices[j]]==i)
            {
                maskTi.push_back(j);
                weightsTi.push_back(weights[j][indices[j]]);
                indices[j]++;
                if(indices[j] == mask[j].size())
                    indices[j] = -1;
            }
        int index = indexOfSorted(&mask[i],i);
        diagonalInd.push_back(index>=0? index : mask[i].size());
        index = indexOfSorted(&maskTi,i);
        diagonalIndT.push_back(index>=0? index : maskTi.size());
    }

    subSetArchitecture();
    AssociativeNet::onWeightsChange();
}


// checks if mask elements are in range [0,dim()), ordered, and there is no empty rows in it
void CellularNet::maskRangeOrderingConnectivityCheck()
{
    assertion(topology != tpUnset, "[CellularNet::maskRangeOrderingConnectivityCheck] topology == tpUnset");

	for(int i=0; i<dim(); i++)
	{
		vector<int> &maski = mask[i];
		assertion(maski.size() > 0, "[CellularNet::maskRangeOrderingConnectivityCheck] connectivity failed");
		int prev = -1;
		for(int j=0; j<maski.size(); j++)
		{
			assertion(maski[j] > prev, "[CellularNet::maskRangeOrderingConnectivityCheck] ordering failed");
			assertion(maski[j]>=0 && maski[j]<dim(), "[CellularNet::maskRangeOrderingConnectivityCheck] range failed");
			prev = maski[j];
		}
	}
}


// checks if mask is symmetric
void CellularNet::maskSymmetryCheck()
{
    assertion(topology != tpUnset, "[CellularNet::maskSymmetryCheck] topology == tpUnset");

	for(int i=0; i<dim(); i++)
	{
		vector<int> &maski = mask[i];
		for(int j=0; j<maski.size(); j++)
			assertion(indexOfSorted(&mask[maski[j]],i) >= 0, "[CellularNet::maskSymmetryCheck] failed");
    }
}


// checks if weights correspond to mask
void CellularNet::maskWeightCorrespondenceCheck()
{
    assertion(topology != tpUnset, "[CellularNet::maskWeightCorrespondenceCheck] topology == tpUnset");

    assertion(mask.size()==weights.size(), "[CellularNet::maskWeightCorrespondenceCheck] failed");
    for(int i=0; i<dim(); i++)
        assertion(mask[i].size()==weights[i].size(), "[CellularNet::maskWeightCorrespondenceCheck] failed");
}


// out = desaturated_weights*in + bias
// desaturated_weights(i,j) = weights(i,j) * (1-delta(i,j)*(1-desatCoeff))
// direct multiplication
void CellularNet::process(Vector *in, Vector *out, bool useDesaturation)
{
    assertion(topology != tpUnset, "[CellularNet::process] topology == tpUnset");
	assertion(in->size() == dim() && out->size() == dim(), "[CellularNet::process] diff sizes");

    int aDim = dim();

	for(int i=0; i<aDim; i++)
	{
		double sp = 0;
		vector<int> &maski = mask[i];
		double *weightsi = weights[i].getData();
		int size = maski.size(), index = diagonalInd[i];
		for(int j=0; j<index; j++)
			sp += weightsi[j]*(*in)[maski[j]];
        if(index < size)
            sp += weightsi[index] * (*in)[maski[index]] * (useDesaturation? desatCoeff : 1);
        for(int j=index+1; j<size; j++)
			sp += weightsi[j]*(*in)[maski[j]];
		(*out)[i] = sp + (useBias? bias[i] : 0);
	}
}


// out = weights * in
// "cached-style" multiplication for binary (+1/-1) data
// set useCurrState = true if "in" in current call is similar to "in" in previous
void CellularNet::processBinary(Vector *in, Vector *out, bool useCurrState)
{
    checkIfTrainingEnded("CellularNet::processBinary");

    assertion(topology != tpUnset, "[CellularNet::processBinary] topology == tpUnset");
	assertion(in->size() == dim() && out->size() == dim(), "[CellularNet::processBinary] diff sizes");

	int aDim = dim();
    double *inBuf = in->getData();

	if(!useCurrState)
    {
        int nPlusOne = 0;
        for(int i=0; i<aDim; i++)
	        if(inBuf[i] > 0)
                nPlusOne++;
        if(nPlusOne > aDim/2)
        {
            currState.assign(aDim, 1);
            currHPS.assign(&oneHPS);
        }
        else
        {
            currState.assign(aDim, -1);
            oneHPS.mult(-1, &currHPS);
        }
        processBinary(in, out, true);
    }
    else
    {
        double *currStateBuf = currState.getData(), *currHPSBuf = currHPS.getData();
        for(int i=0; i<aDim; i++)
            if(fabs((*in)[i]-currStateBuf[i]) > 0.5)
            {
                currState[i] = (*in)[i];
                vector<int> &maskTi = maskT[i];
                double *weightsTi = weightsT[i].getData();
                int size = maskTi.size(), index = diagonalIndT[i];
                if(currState[i] > 0)
                {
                    for(int j=0; j<index; j++)
                        currHPSBuf[maskTi[j]] += weightsTi[j];
                    if(index < size)
                        currHPSBuf[maskTi[index]] += weightsTi[index]*desatCoeff;
                    for(int j=index+1; j<size; j++)
                        currHPSBuf[maskTi[j]] += weightsTi[j];
                }
                else
                {
                    for(int j=0; j<index; j++)
                        currHPSBuf[maskTi[j]] -= weightsTi[j];
                    if(index < size)
                        currHPSBuf[maskTi[index]] -= weightsTi[index]*desatCoeff;
                    for(int j=index+1; j<size; j++)
                        currHPSBuf[maskTi[j]] -= weightsTi[j];
                }
            }
        currHPS.mult(2, out);
    }
}










