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

# ifndef _BAMCELLULAR_H_
# define _BAMCELLULAR_H_

# include "nets.h"

class BAMCellularNet : public AssociativeUnit
{
public:
    BAMCellularNet(int inputDim, int hiddenDim, int connR, double initRange, double learningRate, int trainEpochs)
    {
        _BAMCellularNet(inputDim, hiddenDim, connR, initRange, learningRate, trainEpochs);
    }

    BAMCellularNet(Parameters params)
    {
        _BAMCellularNet(params.getInt(PARAM_INPUT_DIM), params.getInt(PARAM_HIDDEN_DIM), params.getInt(PARAM_CONN_R), params.getFloat(PARAM_INIT_RANGE),
            params.getFloat(PARAM_LEARNING_RATE), params.getInt(PARAM_TRAIN_EPOCHS));
    }

    void _BAMCellularNet(int inputDim, int hiddenDim, int connR, double initRange, double learningRate, int trainEpochs)
    {
        weights.setSizeYX(hiddenDim, inputDim);
        weights.fillRand(-initRange, initRange);

        initMask(&mask, inputDim, hiddenDim, connR);

        input.setSize(inputDim);
        tmpInput.setSize(inputDim);
        output.setSize(hiddenDim);

        this->initRange = initRange;
        this->learningRate = learningRate;
        this->trainEpochs = trainEpochs;
    }

    int dim()            { return weights.sizeX(); }
    int numStored()      { return inputs.size(); }
    string getDescription();
    void getProperties(Properties *prop) {}
    void clear()         { weights.fillRand(-initRange, initRange); inputs.clear(); }
    bool train(Vector *in, Properties *prop = 0);
    void doEndTraining(Properties *prop = 0);
    int converge(Vector *in, Vector *out, bool *isDynamicAttr = 0, bool singleIter = false);
    double getInputContentDiff(Vector *in) { return getSqError(in); }
    double getSqError(Vector *in);
    double getParameter(string parameterName)             { assertion(false, "BAMCellularNet::getParameter"); return 0; }
    void setParameter(string parameterName, double value) { assertion(false, "BAMCellularNet::setParameter"); }

    int hiddenDim() { return weights.sizeY(); }
    double getTrainingError();

protected:
    void initMask(Matrix *mtr, int inDim, int outDim, int connR);
    void applyMask(Matrix *mask, Matrix *weights);

private:
    Matrix weights, mask;
    vector<Vector> inputs;
    Vector input, tmpInput, output;
    double initRange;
    double learningRate;
    int trainEpochs;
};

# endif /* _BAMCELLULAR_H_ */
 