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

# ifndef _MODULAR_H_
# define _MODULAR_H_

# include "..\associative\nets.h"
# include "..\associative\FullProjective.h"
# include "..\lib\data.h"


enum ModuleType { mtFullProjective = 0, mtCellProjective = 1, mtCellPseudoInverse = 2 };
# define MT_FULL_PROJECTIVE "mtFullProjective"
# define MT_CELL_PROJECTIVE "mtCellProjective"
# define MT_CELL_PSEUDOINVERSE "mtCellPseudoInverse"

# define PARAM_MODULE_TYPE      "moduleType"
# define PARAM_LEVEL_COUNT      "levelCount"
# define PARAM_MODULE_CAPACITY  "moduleCapacity"
# define PARAM_MODULE_THRESHOLD "moduleThreshold"
# define PARAM_TRAIN_EPSILON    "trainEpsilon"
# define PARAM_TEST_EPSILON     "testEpsilon"


// Abstract base class - basic bulding block of the modular net
// Has one of associative networks used for difference calculation and data retrieval
// currently supports only autoassociative memory type, but can be made heteroassociative
class Module
{
public:
    Module(double capacity, Parameters netParams)
    {
        this->capacity = capacity;
        this->netParams = netParams;
        net = 0;
    }

    ~Module()                           { delete net; }

    void setCapacity(double capacity)   { this->capacity = capacity; }
    int numStored()                     { return net!=0 ? net->numStored() : 0; }
    bool isEmpty()                      { return net==0; }
    bool isFull()                       { return net!=0 && net->numStored()>=capacity*net->dim(); }

    double getDifference(Vector *in)    { return net==0 ? 1 : net->getInputContentDiff(in); }
    bool train(Vector *in, Properties *prop);
    void clear()                        { delete net; net = 0; }
    void test(Vector *in, Vector *out);

    string getDescription();
    int dim();

protected:
    Parameters netParams;
    virtual AssociativeUnit* getNetwork() = 0;

private:
    double capacity;
    AssociativeUnit *net;
};


// uses fully-connected network with projective learning rule
class FullProjectiveModule : public Module
{
public:
    FullProjectiveModule(double capacity, Parameters netParams) : Module(capacity, netParams) {} 

protected:
    AssociativeUnit* getNetwork() { return new FullProjectiveNet(netParams); }
};

//------------------------------ ModularNet ------------------------------------

// contains ser of hetero/auto-associative NN modules
// organized in a binary tree structure
// to train/test a data pair the path is constructed from a root module
// using the value of diffence coefficient calculated for each module in the path
class ModularNet
{
public:

    ModularNet(ModuleType moduleType, Parameters moduleParams, int levelCount, double moduleCapacity, double threshold, double trainEpsilon, double testEpsilon)
    {
        _ModularNet(moduleType, moduleParams, levelCount, moduleCapacity, threshold, trainEpsilon, testEpsilon);
    }

    ModularNet(Parameters params)
    {
        _ModularNet(getModuleType(params.getString(PARAM_MODULE_TYPE)), params, params.getInt(PARAM_LEVEL_COUNT),
            params.getFloat(PARAM_MODULE_CAPACITY), params.getFloat(PARAM_MODULE_THRESHOLD),
            params.getFloat(PARAM_TRAIN_EPSILON), params.getFloat(PARAM_TEST_EPSILON));
    }

    ~ModularNet()
    {
        for(int i=0; i<modules.size(); i++)
            delete modules[i];
    }

    string getDescription();
    void setTestEpsolon(double testEpsilon) { this->testEpsilon = testEpsilon; }
    double getInbalanceDegree();
    double getActualLevelCount();

    // discrete train/test
    void train(Vector *in, Properties *prop);
    void test(Vector *in, int inIndex, Vector *out, bool *pathError, bool *belongingError, double *searchComplexity);
    // cumulative train/test
    void train(IOData *data, Properties *prop);
    double test(IOData *data, int count, int hamming, int runs, Properties *prop);


    void clear() // clear net content
    {
        for(int i=0; i<modules.size(); i++)
            modules[i]->clear();

        destinations.clear();
        trainedCount = 0;
        numStored = 0;
    }

    double getTestEpsilon()          { return testEpsilon; }
    void setTestEpsilon(double eps)  { testEpsilon = eps; }

protected:
    Module* getModule(double moduleCapacity, Parameters moduleParams)
    {
        switch (moduleType)
        {
            case mtFullProjective:
                return new FullProjectiveModule(moduleCapacity, moduleParams);

            default:
                assertion(false, "[ModularNet::getModule] unsupported moduleType");
        }
    }

    ModuleType getModuleType(string strModuleType)
    {
        if(!strModuleType.compare(MT_FULL_PROJECTIVE))
            return mtFullProjective;
        else
            assertion(false, "[ModularNet::getModuleType] unsupported moduleType = " + strModuleType);
    }

    string getModuleName()
    {
        switch (moduleType)
        {
            case mtFullProjective:
                return MT_FULL_PROJECTIVE;

            default:
                assertion(false, "[ModularNet::getModuleName] unsupported moduleType");
        }
    }


private:
    _ModularNet(ModuleType moduleType, Parameters moduleParams, int levelCount, double moduleCapacity,
        double threshold, double trainEpsilon, double testEpsilon)
    {
        assertion(levelCount>0, "[ModularNet] levelCount must be > 0");
        assertion(moduleCapacity>0, "[ModularNet] moduleCapacity must be > 0");

        this->moduleType = moduleType;
        this->levelCount = levelCount;
        this->moduleCapacity = moduleCapacity;
        this->threshold = threshold;
        this->trainEpsilon = trainEpsilon;
        this->testEpsilon = testEpsilon;

        int moduleCount = round(pow(2, levelCount) - 1);
        modules.resize(moduleCount);
        for(int i=0; i<moduleCount; i++)
            modules[i] = getModule(moduleCapacity, moduleParams);

        trainedCount = 0;
        numStored = 0;
    }

    ModuleType moduleType;   // type of base network
    int levelCount;          // number of levels, total num of modules would be (2^levelCount-1)
    double moduleCapacity;   // each module stores moduleCapacity*inDim vectors
    double threshold;        // tree construction threshold
    double trainEpsilon;     // tree split inrerval during training (causes storage redundancy)
    double testEpsilon;      // tree split inrerval during testing (causes read redundancy)

    vector<Module*> modules; // set of modules (binary tree)

    // sets of sorted modules containing each data vector (used to trace module selection errors)
    vector< vector<int> > destinations;
    int trainedCount;   // total module load (sum of all 'destination' sizes, redundant value)
    int numStored;      // number of stored vectors (num of non-empty 'destination'-s, redundant value)

    void buildTree(Vector *in, double epsilon, vector<int> *path,
        vector<double> *pathDifference, vector<int> *modulesToTrain, int *numModulesSelected);
};


/*
class FullProjectiveModularNet : public ModularNet
{
public:
    FullProjectiveModularNet(Parameters params) : ModularNet(params) {}

protected:
    Module* getModule(double moduleCapacity, Parameters moduleParams)
    {
        return new FullProjectiveModule(moduleCapacity, moduleParams);
    }
};*/

# endif /* _MODULAR_H_ */




