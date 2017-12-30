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

# include "fullprojective.h"

// vector<int> deletedNeurons;

// loads from text file of format
//
// dim
// w[0,0] w[0,1] .... w[0,dim-1]
// w[1,0] w[1,1] .... w[1,dim-1]
// ...
void FullProjectiveNet::loadFromFile(string fileName)
{
    Matrix wgt;
    readMatrix(&wgt, fileName);
    assertion(wgt.sizeX()==wgt.sizeY(), "[FullProjectiveNet::loadFromFile] weights must be square matrix");

    setDim(wgt.sizeX());
    weights.assign(&wgt);
    onWeightsChange();
    doEndTraining();
}


string FullProjectiveNet::getDescription()
{
    return FullNet::getDescription() + "\n# Full projective net, retrain type = " + toString(retrainType);
}

// retrains image according to retrainType
// returns true is weights have been changed
// 0 - classic retrain - add (I-C)v*[(I-C)v].Tr / v.Tr*(I-C)v
// 1 -                 - add (I-C)v*v.Tr / |v|^2
// 2 -                 - add v.Tr*(I-C)v / |v|^2
// 3 -                 - add (I-C)v*v.Tr / v.Tr(I-C)*v
// 4 -                 - add (I-C)v*[(I-C)v].Tr / [(I-C)v].Tr*(I-C)v
bool FullProjectiveNet::retrain(Vector *in)
{
    if(getInputContentDiff(in) < PSEUDO_INVERSE_EPS)
        return false;

    Vector v1(dim()), v2(dim()), invec(*in);
    double coeff;

    // v1 = (I-C)v
    weights.mult(&invec, &v1);
    double *v1data = v1.getData(), *indata = invec.getData();
    int aDim = dim();
    for(int i=0; i<aDim; i++)
        v1data[i] = -v1data[i] + indata[i];

    if(retrainType == 0)
    {
        v2.assign(&v1);
        coeff = invec.sproduct(&v1);
    }
    else if(retrainType == 1)
    {
        v2.assign(&invec);
        coeff = invec.sproduct(&invec);
    }
    else if(retrainType == 2)
    {
        v2.assign(&v1);
        v1.assign(&invec);
        coeff = invec.sproduct(&invec);
    }
    else if(retrainType == 3)
    {
        v2.assign(&invec);
        coeff = invec.sproduct(&v1);
    }
    else if(retrainType == 4)
    {
        v2.assign(&v1);
        coeff = v1.sproduct(&v1);
    }
    else
        throw GenException("[FullProjectiveNet::retrain] unknown retrain type");

    if(coeff == 0)
        return false;

    double *v2data = v2.getData(), *wdata = weights.getData();
    for(int i=0; i<aDim; i++)
        for(int j=0; j<aDim; j++)
            *wdata++ += v1data[i]*v2data[j]/coeff;

    onWeightsChange();
    return true;
}

// deletes (sets to zero) weights if random "count" neurons
void FullProjectiveNet::deleteNeurons(int count)
{
    assertion(count>=0 && count<=dim(), "[FullProjectiveNet::deleteNeurons] wrong count");

    //deletedNeurons.clear();

    vector<char> deleted(dim(),0);
    int nDeleted = 0;
    while(nDeleted < count)
    {
        int index = rand(0,dim());
        if(index == dim())
            index--;
        if(deleted[index])
            continue;
        for(int i=0; i<dim(); i++)
            weights.el(index,i) = weights.el(i,index) = 0;
        //deletedNeurons.push_back(index);
        nDeleted++;
        deleted[index] = 1;
    }
    //sort(deletedNeurons.begin(),deletedNeurons.end());
    onWeightsChange();
}

// deletes (sets to zero) random portion*dim()*dim() weights
void FullProjectiveNet::deleteWeights(double portion)
{
    assertion(portion>=0 && portion<=1, "[FullProjectiveNet::deleteWeights] wrong portion");

    int count = portion*dim()*dim();
    double *wdata = weights.getData();

    vector<char> deleted(dim()*dim(),0);
    int nDeleted = 0;
    while(nDeleted < count)
    {
        int index = rand(0,dim()*dim());
        if(index == dim()*dim())
            index--;
        if(deleted[index])
            continue;
        wdata[index] = 0;
        nDeleted++;
        deleted[index] = 1;
    }
    onWeightsChange();
}
