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

# include "hebbian.h"


/*
// iterative training process based on Hebbian Learning Rule
//
// 1. select next neuron - i
// 2. select the input vector with smallest "alligned local field" input_i*ps_i
// 4. update weight of i-th neuron as dW_i = input_i*in / n
//
// Reference: Krauth, W., and Mezard, M. (1987).
// Learning algorithms with optimal stability in neural networks.
// Journal of Physics A: Mathematical and General 20, L745-L752.
void HebbianCellularNet::doEndTraining(Properties *prop)
{
    initRandomWeights(0);

    int neuronIndex = 0;

    for(int t=0; t<trainEpochs; t++)
    {
        vector<int> &maskInd = mask[neuronIndex];
        double *weightsIndBuf = weights[neuronIndex].getData();
        int size = maskInd.size();

        // search for dtum providing min aligbed local field at neuronIndex neuron
        double minALF;
        int inputIndex;
        for(int d=0; d<inputList.size(); d++)
        {
            double *inBuf = inputList[d]->getData(), alf = 0;
            for(int i=0; i<size; i++)
                alf += weightsIndBuf[i]*inBuf[maskInd[i]];
            alf *= inBuf[neuronIndex];
            if(d==0 || alf<minALF)
            {
                minALF = alf;
                inputIndex = d;
            }
        }

        // update weights of neuronIndex neuron
        double *inBuf = inputList[inputIndex]->getData();
        double coeff = inBuf[neuronIndex]/dim();
        for(int i=0; i<size; i++)
            weightsIndBuf[i] += coeff*inBuf[maskInd[i]];

        // increment neuronIndex
        neuronIndex = ++neuronIndex%dim();

        if(t%100 == 0)
            printProgress("[HebbianCellNet::doEndTraining]", t, 1, trainEpochs,
                string("k = ") + toString(getKMeasure(), 8) + "        ");
    }
    cout << endl;

    CellularNet::doEndTraining(prop);
}
*/

// iterative training process based on Hebbian Learning Rule
// 0. do until (useTValue? all ALF > tValue : trainEpochs times)
// 1. select next "in" from inputs
// 2. calculate postsynapse
// 3. useTValue? find neurons with ALF < tValue :
//                  find neuron "i" with the largest output error (the least Aligned Local Field)
// 4. update weight of i-th neuron(s) as dW_i = input_i*input / num_connections_i
void HebbianCellularNet::doEndTraining(Properties *prop)
{
    initRandomWeights(0); // init with zero weights

    Vector postSynapse(dim()), product(dim());

    int epochIndex = 0, neuronUpdateCount = 0;
    while(true)
    {
        bool noLessThanT = true;
        for(int d=0; d<numStored(); d++)
        {
            Vector *in = inputList[d]; // pick next input
            process(in, &postSynapse, false);
            in->multCW(&postSynapse, &product);

            vector<int> neuronList;

            if(useTValue)
            for(int i=0; i<product.size(); i++)
            {
                if(product[i] < tValue)
                {
                    neuronList.push_back(i); // update neurons with ALF < tValue
                    noLessThanT = false;
                }
            }
            else
                neuronList.push_back(product.minElementIndex()); // update only worst neuron

            for(int n=0; n<neuronList.size(); n++) // update weights of selected neurons
            {
                int worstNeuron = neuronList[n];
                vector<int> &maskWN = mask[worstNeuron];
                Vector &weightsWN = weights[worstNeuron];
                int size = maskWN.size();
                double *inBuf = in->getData(), *weightsWNBuf = weightsWN.getData();
                double coeff = inBuf[worstNeuron]/size;

                for(int i=0; i<size; i++)
                    weightsWNBuf[i] += coeff*inBuf[maskWN[i]];

                neuronUpdateCount++;
            }
        }

        if(++epochIndex % 10 == 0)
        {
            CellularNet::doEndTraining(0);
            printProgress("[HebbCNN::train]", epochIndex, 1, useTValue? 0 : trainEpochs,
                string("minALF = ") + setWidth(toString(getMinALF(), 4), 8, saLeft) +
                string(", kappa = ") + setWidth(toString(getKMeasure(), 4), 8, saLeft));
        }

        // termination criterion
        if((useTValue && noLessThanT)
                 || (!useTValue && epochIndex >= trainEpochs))
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

    if(prop!=0 && useTValue)
    {
        (*prop)["epochCount"] = epochIndex;
        (*prop)["neuronUpdates"] = neuronUpdateCount;
    }

    CellularNet::doEndTraining(prop);
}

/*
void HebbianCellularNet::normalizeWeights()
{
    double sumSqNorm = 0;
    for(int i=0; i<dim(); i++)
        sumSqNorm += weights[i].sqnorm();
    double coeff = 1/sqrt(sumSqNorm);

    for(int i=0; i<dim(); i++)
        weights[i].mult(coeff, &weights[i]);
}*/
