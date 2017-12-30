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

# include "DeltaCellular.h"


// iterative training process based on Widrow-Hoff Delta rule
// 0. do trainEpochs times
// 1. select next "in" from inputs
// 2. calculate postsynapse
// 3. update weights of i-th neuron(s) as dW_ij = -learningRate(ps_i-input_i)*input_j
void DeltaCellularNet::doEndTraining(Properties *prop)
{
    initRandomWeights(0); // init with zero weights

    Vector postSynapse(dim());

    int epochIndex = 0, neuronUpdateCount = 0;
    double discrepancy = -1;
    while(true)
    {
        for(int d=0; d<numStored(); d++)
        {
            Vector *in = inputList[d]; // pick next input
            process(in, &postSynapse, false);
            postSynapse.minus(in, &postSynapse);

            double *inBuf = in->getData(), *psBuf = postSynapse.getData();
            for(int n=0; n<dim(); n++)
            {
                vector<int> &maskN = mask[n];
                Vector &weightsN = weights[n];
                int size = maskN.size();
                double *weightsBuf = weightsN.getData();
                double coeff = -learningRate*psBuf[n];

                for(int i=0; i<size; i++)
                    weightsBuf[i] += coeff*inBuf[maskN[i]];

                neuronUpdateCount++;
            }
        }

        if(++epochIndex % 10 == 0)
        {
            CellularNet::doEndTraining(0);
            printProgress("[DeltaCNN::train]", epochIndex, 1, trainEpochs,
                string("|WV-V|^2 = ") + setWidth(toString(discrepancy = getDiscrepancy(), 4), 8, saLeft) +
                string(", kappa = ") + setWidth(toString(getKMeasure(), 4), 8, saLeft));
        }

        // termination criterion
        if(epochIndex >= trainEpochs || (discrepancy>0 ? discrepancy<trainPrecision : false))
            break;

        /* // used to record train dynamics 
        if(t%100 == 0)
        {
            string reportFile = "_trainReport";
            Properties trainProp;
            trainProp["trainIter"] = t;
            CellularNet::doEndTraining(&trainProp);
            if(t==0)
            {
                saveString(getDescription(), reportFile);
                saveProperties(trainProp, true, 14, reportFile);
            }
            else
                saveProperties(trainProp, false, 14, reportFile);

        }*/
    }
    cout << endl;

    if(prop!=0)
    {
        (*prop)["epochCount"] = epochIndex;
        (*prop)["neuronUpdates"] = neuronUpdateCount;
    }

    CellularNet::doEndTraining(prop);
}
