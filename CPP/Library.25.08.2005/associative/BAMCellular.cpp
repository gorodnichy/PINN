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

# include "BAMCellular.h"


string BAMCellularNet::getDescription()
{
    return AssociativeUnit::getDescription() + "\n# BAMCellularNet, hidden dim = " + toString(hiddenDim());
}

bool BAMCellularNet::train(Vector *in, Properties *prop)
{
    assertion(in->size() == dim(), "[BAMCellularNet::train] in->size() != dim()");
    inputs.push_back(*in);

    return true;
}

// iterative training process
// out = W * in
// e = in - W.tr * out
// deltaW = LR * out * e.tr
void BAMCellularNet::doEndTraining(Properties *prop)
{
    int trainingLength = trainEpochs*inputs.size();

    for(int t=0; t<trainingLength; t++)
    {
        Vector &in = inputs[rand(inputs.size())];
        weights.mult(&in, &output);
        weights.multTN(&output, &input);
        in.minus(&input, &tmpInput); // tmpIn = error

        weights.plusVVT(&output, &tmpInput, learningRate);
        weights.truncate(&weights, 1e10);
        applyMask(&mask, &weights);

        if(t%100 == 0)
            printProgress("[BAMCellularNet::doEndTraining]", t, 1, trainingLength,
                string("error = ") + toString(getTrainingError(),5));
    }
    cout << endl;

    saveMatrix(&weights, "_weights", true, true);
    saveMatrix(&mask, "_mask", true, true);

    AssociativeUnit::doEndTraining();
}

int BAMCellularNet::converge(Vector *in, Vector *out, bool *isDynamicAttr, bool singleIter)
{
    return 0;
}

// |in - W.tr * W * in|^2
double BAMCellularNet::getSqError(Vector *in)
{
    weights.mult(in, &output);
    weights.multTN(&output, &input);
    in->minus(&input, &tmpInput); // tmpIn = error

    return tmpInput.sqnorm();
}

double BAMCellularNet::getTrainingError()
{
    assertion(numStored() > 0, "[BAMCellularNet::getTrainingError] numStored == 0");

    double sumSqErr = 0;
    for(int i=0; i<inputs.size(); i++)
    {
        Vector &in = inputs[i];
        sumSqErr += getSqError(&in);
    }

    return sqrt(sumSqErr/inputs.size()/dim());
}

// inits mask so that each input neuron is connected with [-connR, connR] neighbourhood
// of corresponding out neuron
void BAMCellularNet::initMask(Matrix *mtr, int inDim, int outDim, int connR)
{
    assertion(connR>0 && connR<=outDim, "[BAMCellularNet::initMask] musty be (connR>0 && connR<=outDim)");
    mtr->setSizeYX(outDim, inDim);
    mtr->init(0);

    for(int i=0; i<inDim; i++)
    {
        int outNeuron = i*outDim / inDim; // correspondind out neuron
        for(int j=outNeuron-connR; j<=outNeuron+connR; j++) // out neighbourhood
            mtr->el((j+outDim)%outDim, i) = 1;
    }
}

// is mask element is 0 than set corresponding weight element to 0 too
void BAMCellularNet::applyMask(Matrix *mask, Matrix *weights)
{                         
    assertEqualSize(mask, weights, "[BAMCellularNet::applyMask]");

    int size = mask->sizeX()*mask->sizeY();
    double *maskData = mask->getData(), *weightsData = weights->getData();
    for(int i=0; i<size; i++)
        if(!maskData[i])
            weightsData[i] = 0;
}
