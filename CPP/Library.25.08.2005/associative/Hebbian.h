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

# ifndef _HEBBIAN_H_
# define _HEBBIAN_H_

# include "nets.h"

class HebbianCellularNet : public CellularNet
{
public:
    HebbianCellularNet(int dim, int connR, bool noDiagonalWeights, int trainEpochs,
        double tValue, bool useTValue, bool useDeltaRule, double learningRate)
    {
        _HebbianCellularNet(dim, connR, noDiagonalWeights, trainEpochs,
            tValue, useTValue);
    }

    HebbianCellularNet(Parameters params)
    {
        _HebbianCellularNet(params.getInt(PARAM_NET_DIM), params.getInt(PARAM_CONN_R), params.getBool(PARAM_NO_DIAG_WGTS),
         params.getInt(PARAM_TRAIN_EPOCHS), params.getFloat(PARAM_T_VALUE), params.getBool(PARAM_USE_T_VALUE));
    }

    void _HebbianCellularNet(int dim, int connR, bool noDiagonalWeights, int trainEpochs, double tValue, bool useTValue)
    {
        this->noDiagonalWeights = noDiagonalWeights;
        this->trainEpochs = trainEpochs;
        this->tValue = tValue;
        this->useTValue = useTValue;

        setLocal1DArchitecture(dim, connR);
    }

    string getDescription()
    {
        return CellularNet::getDescription() + "\n# HebbianCellularNet\n"
            + "useTValue = " + toStringB(useTValue)
            + ", tValue = " + toString(tValue, 3)
            + ", trainEpochs = " + toString(trainEpochs);
    }

    void subSetArchitecture() { clear(); }
    bool subTrain(Vector *in, Properties *prop = 0) { return true; }
    void doEndTraining(Properties *prop = 0);
    //void normalizeWeights();

private:
    int trainEpochs;
    double tValue;
    bool useTValue;
    bool useDeltaRule;
    double learningRate;

};

# endif /* _HEBBIAN_H_ */
