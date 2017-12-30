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

# ifndef _NETS_H_
# define _NETS_H_

// Names of network parameters
# define PARAM_NET_TYPE         "netType"
# define PARAM_NT_CELLPROJ      "cellprojective"
# define PARAM_NT_PSEUDOINVERSE "pseudoinverse"
# define PARAM_NT_FULLPROJ      "fullprojective"
# define PARAM_NT_ADAPTIVE      "adaptive"
# define PARAM_NT_HEBBIAN       "hebbian"
# define PARAM_NT_DELTACELL     "deltacellular"
# define PARAM_NT_SMALLWORLD    "smallworld"

# define PARAM_TOPOLOGY_TYPE    "topology"
# define PARAM_TP_1D            "tp1D"
# define PARAM_TP_2D            "tp2D"
# define PARAM_TP_RANDOM        "tpRandom"
# define PARAM_TP_2D_SIZEX      "tp2DsizeX"
# define PARAM_TP_2D_SIZEY      "tp2DsizeY"

# define PARAM_NET_DIM          "netDim"
# define PARAM_CONN_R           "connectionRadius"
# define PARAM_NO_DIAG_WGTS     "noDiagonalWeights"
# define PARAM_DESAT_COEFF      "desatCoeff"
# define PARAM_USE_BIAS         "useBias"
# define PARAM_ALPHA            "alpha"
# define PARAM_IS_SYMMETRIC     "isSymmetric"
# define PARAM_IS_PROJECTIVE    "isProjective"
# define PARAM_HAS_QUOTA        "hasQuota"
# define PARAM_USE_CORRELATIOM  "useCorrelation"
# define PARAM_CONNECTIVITY_DEG "connectivityDegree"
# define PARAM_REWIRING_TYPE    "rewiringType"
# define PARAM_REWIRING_DEG     "rewiringDegree"
# define PARAM_INPUT_DIM        "inputDim"
# define PARAM_HIDDEN_DIM       "hiddenDim"
# define PARAM_INIT_RANGE       "initRange"
# define PARAM_LEARNING_RATE    "learningRate"
# define PARAM_TRAIN_EPOCHS     "trainEpochs"
# define PARAM_TRAIN_PRECISION  "trainPrecision"
# define PARAM_USE_T_VALUE      "useTValue"
# define PARAM_T_VALUE          "tValue"

# include "..\lib\math_aux.h"

enum Topology { tpUnset = 0, tpPredefined = 1, tp1D = 2, tp2D = 3, tpRandom = 4, tpFree = 5 };

class StatMaker;

// base associative memory class (interface mostly)
class AssociativeUnit
{
public:
    AssociativeUnit()       { id = toString(abs((int)getTime_ms())); trainingEnded = false; }

    virtual int dim() = 0;
    virtual int numStored() = 0;
    virtual double getInputContentDiff(Vector *in) = 0;

    virtual string getDescription();
    virtual void getProperties(Properties *prop) { checkIfTrainingEnded("getProperties"); }

    virtual bool train(Vector *in, Properties *prop = 0) { trainingEnded = false; return true; }
    // is called by test methods by the time they want to start testing
    // useful for networks with off-line training procedures
    virtual void doEndTraining(Properties *prop = 0) { trainingEnded = true; }
    virtual void adjustBeforeTesting() = 0;
    // sets network to initial state (like it was before training)
    virtual void clear() { trainingEnded = false; }

    // should be in AssociativeNet...
    virtual int converge(Vector *in, Vector *out, bool *isDynamicAttr = 0, bool singleIter = false) { checkIfTrainingEnded("converge"); return 0; }
    virtual int extendedConverge(Vector *in, Vector *out, int *cycleLength) { checkIfTrainingEnded("extendedConverge"); return 0; }

    virtual double getParameter(string parameterName) = 0;
    virtual void setParameter(string parameterName, double value) = 0;

    string getID() { return id; }

    virtual void saveDetails(string fileName)
    {
        saveString(getDescription(), fileName);
        Properties prop;
        getProperties(&prop);
        saveProperties(prop, true, 14, fileName);
        saveString("\n", fileName);
    }

protected:
    void checkIfTrainingEnded(string funcName) { assertion(trainingEnded, string("[AssociativeUnit::") + funcName + "] there was no call to doEndTraining()"); }

private:
    string id;                          // unit descriptor (used for logging)
    bool trainingEnded;
};

// base associative neural net class
// implements convergence process
class AssociativeNet : public AssociativeUnit
{
public:

    AssociativeNet()  { desatCoeff = 1; }
    // used to be virtual ~AssociativeNet() { clear(); } => caused "Pure Virtual Function Called"
    virtual ~AssociativeNet() { while(inputList.size()) { delete inputList.back(); inputList.pop_back(); } }
    virtual string getDescription();
    virtual void saveDetails(string fileName) { saveWeights(fileName+"_weights"); AssociativeUnit::saveDetails(fileName); }
    virtual void saveWeights(string fileName) = 0;
    virtual void adjustBeforeTesting() { adjustDesaturationCoefficient(); }
    void adjustDesaturationCoefficient()
    {
        assertion(getTrace()>0, "[AssociativeNet::adjustDesaturationCoefficient] getTrace() == 0");
        setParameter(PARAM_DESAT_COEFF, 0.15*numStored()/fabs(getTrace()));
    }

    virtual double getParameter(string parameterName);
    virtual void setParameter(string parameterName, double value);

    bool train(Vector *in, Properties *prop = 0);
	virtual void doEndTraining(Properties *prop = 0);
    virtual void clear();

    int converge(Vector *in, Vector *out, bool *isDynamicAttr = 0, bool singleIter = false);
    int extendedConverge(Vector *in, Vector *out, int *cycleLength);
    void processBinaryCheck();

    virtual void getProperties(Properties *prop);
    virtual double getTrace() = 0;
	virtual double getIDiff() = 0;
    virtual double getWSqNorm() = 0;
    virtual double getWSqNorm(int neuron) = 0;
	virtual double getSymDiff() = 0;
    virtual double getNormalizedSymDiff() = 0;
	double getInputContentDiff(Vector *in);
    double getAverageContentDiff();
    double getDiscrepancy();
    double getMinALF()           { return getGammaMeasure(false, true);  }
    double getKMeasure()         { return getGammaMeasure(true,  true);  }
    double getAveGammaMeasure()  { return getGammaMeasure(true,  false); }

    int numStored() { return inputList.size(); };

protected:
    double desatCoeff;                                         // desaturation coefficient
    Vector currState, oneHPS, currHPS;                         // current state of the net, oneHPS = W*{1,...,1}/2, currHPS = W*currState/2

    virtual bool subTrain(Vector *in, Properties *pr = 0) = 0; // must return true if "in" successfully stored
    virtual void onWeightsChange();
    virtual void process(Vector *in, Vector *out, bool useDesaturation = true) = 0;
    virtual void processBinary(Vector *in, Vector *out, bool useCurrState) = 0;

    vector<Vector*> inputList;                                  // input vectors feed durind train

    double getGammaMeasure(bool isNormalized, bool isMinimum);
};


// base class for fully connected nets
//
// derived cluss must implement
// subTrain()
//
// may implement
// getDescription()
// clear()
// getProperties(Properties *prop)
// get/setParameter()
class FullNet : public AssociativeNet
{
public:
    virtual string getDescription() { return AssociativeNet::getDescription(); }

    FullNet(int dimension) { setDim(dimension); }

    int dim() { return weights.sizeX(); }

    virtual void setDim(int value)
    {
        weights.setSizeYX(value, value);
        weights.init(0);
        onWeightsChange();
    }

    virtual void setParameter(string parameterName, double value);
    virtual void clear();
    void saveWeights(string fileName);
    virtual void getProperties(Properties *prop) { AssociativeNet::getProperties(prop); }
    double getTrace();
	double getIDiff();
    double getWSqNorm();
    double getWSqNorm(int neuron);
	double getSymDiff();
    double getNormalizedSymDiff();

protected:
    Matrix weights;

private:
    void process(Vector *in, Vector *out, bool useDesaturation = true);
    void processBinary(Vector *in, Vector *out, bool useCurrState);
};


// base cellular net class
// provides sparse weights operations,
// such as different initializations, save
//
// derived class must implement
// subSetArchitecture(), subTrain()
//
// may implement
// doEndTraining()
// clear()
// getDescription()
// getProperties(Properties *prop)
// get/setParameter()
class CellularNet : public AssociativeNet
{
friend StatMaker;
public:
    CellullarNet() { topology = tpUnset; }

    virtual string getDescription();
    virtual void getProperties(Properties *prop);
    virtual void saveDetails(string fileName) { saveMask(fileName+"_mask"); AssociativeNet::saveDetails(fileName); }
    void saveWeights(string fileName);
    void saveMask(string fileName);

    virtual double getParameter(string parameterName)
    {
        if(!parameterName.compare(PARAM_CONN_R))
            return getConnR();
        else if(!parameterName.compare(PARAM_CONNECTIVITY_DEG))
            return connectivityPortion;
        else
            return AssociativeNet::getParameter(parameterName);
    }

    virtual void setParameter(string parameterName, double value)
    {
        if(!parameterName.compare(PARAM_CONN_R))
            setConnR(round(value));
        else if(!parameterName.compare(PARAM_CONNECTIVITY_DEG))
        {
            connectivityPortion = value;
            setRandomArchitecture(dim(), connectivityPortion);
        }
        else
            AssociativeNet::setParameter(parameterName, value);
    }

    int dim()		{ return mask.size(); }
    int numWeights();
    int numDiagonalWeights();
    int minNumNeuronWeights();
    int maxNumNeuronWeights();
    int totalConnectionLength();
    int getConnR()	{ return connR; }
    void setConnR(int connR)
    {
        this->connR = connR;
        if(topology == tp1D)
            setLocal1DArchitecture(dim(), connR);
        else if(topology == tp2D)
            setLocal2DArchitecture(sizeY, sizeX, connR);
        else
            throw GenException("[CellularNet::setConnR] unsupported topology");
    }

    // no "clear" is called during "set*Architecture" !!!
	void setLocal1DArchitecture(int dimension, int connR);
    void setLocal2DArchitecture(int sizeY, int sizeX, int connR);
    void setRandomArchitecture(int dimension, float connectivityPortion);
    void setPredefinedArchitecture(vector<vector<int> > *scheme);
    void setFreeArchitecture() { topology = tpFree; }

    virtual void clear();
    void initRandomWeights(double range);

	double getTrace();
	double getIDiff();
    double getWSqNorm();
    double getWSqNorm(int neuron);
	double getSymDiff();
    double getNormalizedSymDiff();

    string getTopology();
	
protected:
	vector<vector<int> > mask;  // weights' mask
	vector<Vector> weights;     // weights
    Vector bias;                // bias vector
    bool useBias;               // if use bias flag
    double connectivityPortion; // used for mask random initialization
    bool noDiagonalWeights;     // if there is no direct feedback connections

    void process(Vector *in, Vector *out, bool useDesaturation);
    void processBinary(Vector *in, Vector *out, bool useCurrState); // always uses desatCoeff

	CellularNet()
	{
        useBias = false;
        connectivityPortion = 0;
        noDiagonalWeights = true;
        topology = tpUnset;
		connR = -1;
        sizeY = sizeX = 0;
    }

	virtual void subSetArchitecture() = 0;    // configure sub net content according to mask
    void onWeightsChange();                   // call after net weights have been changed
    void onArchitectureChange();              // call after architecture has been changed (extends onWeightsChange)

	void maskRangeOrderingConnectivityCheck();
	void maskSymmetryCheck();
    void maskWeightCorrespondenceCheck();

private:
    Topology topology;
	int connR;                          // connectivity radius
    int sizeY, sizeX;                   // size of planar net
	// auxilary data used for binary convergence optimization
    vector<vector<int> > maskT;         // transposed mask
	vector<Vector> weightsT;            // transposed weights
    vector<int> diagonalInd;            // indices of diagonal elements in mask
    vector<int> indices, diagonalIndT;  // auxilary vector, indices of diagonal elements in maskT
};


#endif /* _NETS_H_ */
