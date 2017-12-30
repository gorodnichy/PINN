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

# ifndef _SMALL_WORLD_H_
# define _SMALL_WORLD_H_

# include "pseudoinverse.h"

enum RewiringType { rtRandom = 0, rtSystematic = 1 };

//--------------------------
//!!! TO DO: EnumReader erRewiringType("rtRandom", "rtSystematic"); // (RewiringType)erRewiringType.get(string enumVal)
//--------------------------

// Cellular associative net with Small-World architecture
// Small-World is obtaines from the regular architecture (1D topology) with neighbourhood criterion
// Then certain regular connections are rewired
// Two ways of rewiring are possible
// 1. Random rewiring ()
// 2. Systematic rewiring (according to the biggest absolute values in projective matrix)
// After the rewiring the cellular net is trained using pseudoinverse algorithm

// References:
// Watts, D. & Strogatz, S. Collective dynamics of 'small-world' networks. Nature, 1998 , 393 , 440-442
// Dekhtyarenko, O. Systematic Rewiring in Associative Neural Networks with Small-World Architecture 2005 , 1178-1181 International Joint Conference on Neural Networks (IJCNN'05)
class SmallWorldNet : public PseudoInverseNet
{
public:
    SmallWorldNet(int dim, int connR, bool noDiagonalWeights, RewiringType rt, double rewiringProbability)
        : PseudoInverseNet(noDiagonalWeights, false /*useBias*/, 1 /*alpha*/, false /*isSymmetric*/)
    {
        _SmallWorldNet(dim, connR, rt, rewiringProbability);
    }

    SmallWorldNet(Parameters params)
        : PseudoInverseNet(params.getBool(PARAM_NO_DIAG_WGTS), false /*useBias*/, 1 /*alpha*/, false /*isSymmetric*/)
    {
        _SmallWorldNet(params.getInt(PARAM_NET_DIM), params.getInt(PARAM_CONN_R), (RewiringType)params.getInt(PARAM_REWIRING_TYPE),
            params.getFloat(PARAM_REWIRING_DEG));
    }


    void _SmallWorldNet(int dim, int connR, RewiringType rt, double rewiringProbability)
    {
        assertion(rewiringProbability>=0 && rewiringProbability<=1, "[SmallWorldNet::SmallWorldNet] rewiringProbability must be in [0,1]");
        this->connR = connR;
        rewiringType = rt;
        this->rewiringProbability = rewiringProbability;
        keepDataInSSA = false;
        setLocal1DArchitecture(dim, connR);
    }

    string getDescription();

    virtual double getParameter(string parameterName);
    virtual void setParameter(string parameterName, double value);

protected:
	virtual void subSetArchitecture();
    virtual void clear();
    virtual bool subTrain(Vector *in, Properties *prop = 0);
    virtual void doEndTraining(Properties *prop = 0);

private:
    int connR;
    RewiringType rewiringType;
    double rewiringProbability;
    Matrix projMatrix;
    bool keepDataInSSA; // flag that shows that subSetArchitecture() was called during doEndTraining

    int doRandomRewiring();
    int doSystematicRewiring();
};

# endif /* _ADAPTIVE_CELLULAR_H_ */