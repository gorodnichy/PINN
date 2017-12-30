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

# ifndef _DELTA_H_
# define _DELTA_H_

# include "nets.h"

class DeltaCellularNet : public CellularNet
{
public:
    DeltaCellularNet(int dim, int connR, bool noDiagonalWeights, int trainEpochs,
        double learningRate, double trainPrecision)
    {
        _DeltaCellularNet(dim, connR, noDiagonalWeights, trainEpochs,
            learningRate, trainPrecision);
    }

    DeltaCellularNet(Parameters params)
    {
        _DeltaCellularNet(params.getInt(PARAM_NET_DIM), params.getInt(PARAM_CONN_R),
        params.getBool(PARAM_NO_DIAG_WGTS), params.getInt(PARAM_TRAIN_EPOCHS),
        params.getFloat(PARAM_LEARNING_RATE), params.getFloat(PARAM_TRAIN_PRECISION));
    }

    void _DeltaCellularNet(int dim, int connR, bool noDiagonalWeights, int trainEpochs,
        double learningRate, double trainPrecision)
    {
        this->noDiagonalWeights = noDiagonalWeights;
        this->trainEpochs = trainEpochs;
        this->learningRate = learningRate;
        this->trainPrecision = trainPrecision;

        setLocal1DArchitecture(dim, connR);
    }

    string getDescription()
    {
        return CellularNet::getDescription() + "\nDeltaCellularNet\n"
            + "learningRate = " + toString(learningRate, 4)
            + ", trainPrecision = " + toString(trainPrecision, 4)
            + ", trainEpochs = " + toString(trainEpochs);
    }

    void subSetArchitecture() { clear(); }
    bool subTrain(Vector *in, Properties *prop = 0) { return true; }
    void doEndTraining(Properties *prop = 0);

private:
    int trainEpochs;
    double trainPrecision;
    double learningRate;
};

# endif /* _DELTA_H_ */
