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

# include "AdaptiveCellular.h"

typedef multimap<double, int, greater<double> > MapD2I; // descending order


void AdaptiveCellularNet::subSetArchitecture()
{
    if(!keepDataInSSA)
    {
        projMatrix.setSizeYX(dim(), dim());
        projMatrix.init(0);
        corrMatrix.init_corr(); // no need to set size
    }
    PseudoInverseNet::subSetArchitecture();
    setFreeArchitecture();
}

void AdaptiveCellularNet::clear()
{
    projMatrix.init(0);
    corrMatrix.init_corr();
    PseudoInverseNet::clear();
}

// net's description
string AdaptiveCellularNet::getDescription()
{
    return PseudoInverseNet::getDescription() + "\n# Adaptive cellular net, isProjective = " + toStringB(isProjective)
        + ", hasQuota " + toStringB(hasQuota) + ", useCorrelation " + toStringB(useCorrelation);
}


bool AdaptiveCellularNet::subTrain(Vector *in, Properties *prop)
{
    assertion(projMatrix.expandProjectiveMatrix(in), "[AdaptiveCellularNet::subTrain] linearly dependent input");
    if(useCorrelation)
        corrMatrix.addItem(in);

    return PseudoInverseNet::subTrain(in, prop);
}


// select connectivityPortion greatest all (!hasQuota) or per-row elements (hasQuota)
// form mask
// init(mask)
// fill weights using either projectiveMatrix (isProjective) or PseudoInverseNet (!isProjective) train procedure
// calculate WeightRatio and put it into prop (ratio of sum abs selected weights to sum abs all weights)
void AdaptiveCellularNet::doEndTraining(Properties *prop)
{
    // set architecture
    keepDataInSSA = true;
    double weightRatio = setArchitecture(useCorrelation? &corrMatrix : &projMatrix);
    keepDataInSSA = false;
    if(prop)
        (*prop)["weight_ratio"] = weightRatio;

    // set weights
    if(isProjective)
        setWeights(&projMatrix);
    else
    {
        for(int i=0; i<inputList.size(); i++)
            assertion(PseudoInverseNet::subTrain(inputList[i], prop),
            "[AdaptiveCellularNet::doEndTraining] PseudoInverseNet::subTrain failed");
    }

    //if(prop != 0)
    //    (*prop)["trainTime"] = getTime_ms() - timeMark;

    PseudoInverseNet::doEndTraining(prop);
}


// sets net's architecture according to the connectivityPortion and hasQuota
// returns ratio sum_abs_selected / sum_abs_all
// changeDegree is set to the number of mismatching positions in the new and current architectures
double AdaptiveCellularNet::setArchitecture(Matrix *mtr, int *changeDegree)
{
    assertion(mtr->sizeX()==mtr->sizeY() && mtr->sizeX()==dim(), "[AdaptiveCellular::setArchitecture(Matrix *mtr)] improper matrix size");

    vector< vector<int> > maskCopy(mask); // copy of the current mask

    vector<MapD2I> sortedPerRowWeights(dim());
    MapD2I sortedAllWeights;
    double sumAbsAll = 0, sumAbsSelected = 0;

    // ordering of elements ----------------------------------------------------
    vector<int> indices; // to prevent selection of the whole weight blocks
    for(int i=0; i<dim()*dim(); i++)
        indices.push_back(i);
    random_shuffle(indices.begin(),indices.end());

    for(int i=0; i<dim()*dim(); i++)
    {
        MapD2I &mmp = hasQuota? sortedPerRowWeights[indices[i]/dim()] : sortedAllWeights;
        double absel = fabs(mtr->getData()[indices[i]]);
        mmp.insert(MapD2I::value_type(absel, indices[i]));
        sumAbsAll += absel;
    }

    // scheme construction -----------------------------------------------------
    int count = connectivityPortion*dim()*(hasQuota? 1 : dim()); // num of elements to select
    if(hasQuota)
    {
        for(int i=0; i<dim(); i++)
        {
            vector<int> &maski = mask[i];
            maski.clear();
            MapD2I::iterator iter = sortedPerRowWeights[i].begin();
            for(int j=0; j<count; j++)
            {
                assertion(iter != sortedPerRowWeights[i].end(), "[AdaptiveCellularNet::setArchitecture] algorithmic error");
                int iind = iter->second/dim();
                assertion(iind==i, "[AdaptiveCellular::setArchitecture] algorithmic error");
                int jind = iter->second%dim();
                maski.push_back(jind);
                sumAbsSelected += iter->first;
                iter++;
            }
        }
    }
    else
    {
        for(int i=0; i<dim(); i++)
            mask[i].clear();
        MapD2I::iterator iter = sortedAllWeights.begin();
        for(int i=0; i<count && iter!=sortedAllWeights.end(); i++)
        {
            int iind = iter->second/dim();
            int jind = iter->second%dim();
            iter++;
            if(noDiagonalWeights && iind==jind)
            {
                i--;
                continue;
            }
            mask[iind].push_back(jind);
            sumAbsSelected += iter->first;
        }
    }

    weights.clear();
    weights.resize(dim());
    for(int i=0; i<dim(); i++)
    {
        vector<int> &maski = mask[i];
        sort(maski.begin(), maski.end());
        if(maski.size() == 0)
            maski.push_back(i);
        weights[i].assign(maski.size(), 0);
    }

    // changeDegree calculation
    if(changeDegree)
    {
        *changeDegree = 0;
        for(int i=0; i<dim(); i++)
        {
            vector<int> &maski = mask[i];
            vector<int> &maskCopyi = maskCopy[i];
            vector<int> diff;
            back_insert_iterator< vector<int> > iter(diff);
            set_symmetric_difference(maski.begin(), maski.end(), maskCopyi.begin(), maskCopyi.end(), iter);
            *changeDegree += diff.size();
        }                                
    }

    onArchitectureChange();

    return sumAbsAll>0 ? sumAbsSelected/sumAbsAll : -1;
}


// fills weights from mtr according to mask
void AdaptiveCellularNet::setWeights(Matrix *mtr)
{
    assertion(mtr->sizeX()==mtr->sizeY() && mtr->sizeX()==dim(), "[AdaptiveCellular::setWeights(Matrix *mtr)] improper matrix size");

    for(int i=0; i<dim(); i++)
    {
        vector<int> &maski = mask[i];
        Vector &weightsi = weights[i];
        for(int j=0; j<maski.size(); j++)
            weightsi[j] = mtr->el(i, maski[j]);
    }

    onWeightsChange();
}


// used for projective type only
// retrains net with already stored data then selects weights
// alpha used as the adaptive filtration coefficient
void AdaptiveCellularNet::retrain(double alpha, Properties *prop)
{
    assertion(isProjective, "[AdaptiveCellularNet::retrain] called for non-projective net");
    int adim = dim();

    // copy weights to wgt
    Matrix wgt(adim, adim);
    wgt.init(0);
    for(int i=0; i<adim; i++)
    {
        vector<int> &maski = mask[i];
        Vector &weightsi = weights[i];
        for(int j=0; j<maski.size(); j++)
            wgt.el(i, maski[j]) = weightsi[j];
    }

    // retrain wgt
    for(int i=0; i<inputList.size(); i++)
        assertion(wgt.expandProjectiveMatrix(inputList[i], 0, alpha), "[AdaptiveCellularNet::retrain()] expand failed");

    int changeDegree;
    (*prop)["weight_ratio"] = setArchitecture(&wgt, &changeDegree);
    (*prop)["changeDegree"] = changeDegree;
    
    setWeights(&wgt);
}


/*
// sets all "scheme" elements to 0s
// then fills "scheme" positions that correspond to "connectivity" greatest elements of "weights" with 1s
// returns ratio sum_abs_selected / sum_abs_all
double AdaptiveCellularNet::createScheme(Matrix *weights, Matrix *scheme, double connectivity)
{
    assertion(weights->sizeX()==scheme->sizeX() && weights->sizeY()==scheme->sizeY(),
        "[AdaptiveCellularNet::createScheme] size mismatch");
    assertion(connectivity>0 && connectivity<=1, "[AdaptiveCellularNet::createScheme] connectivity must be in (0,1]");


    int size = weights->sizeX() * weights->sizeY();
    double *weightsdata = weights->getData();
    MapD2I sortedAllWeights;
    double sumAbsAll = 0, sumAbsSelected = 0;

    vector<int> indices; // to prevent selection of the whole weight blocks
    for(int i=0; i<size; i++)
        indices.push_back(i);
    random_shuffle(indices.begin(), indices.end());

    for(int i=0; i<size; i++)
    {
        double absel = fabs(weightsdata[indices[i]]);
        sortedAllWeights.insert(MapD2I::value_type(absel, indices[i]));
        sumAbsAll += absel;
    }

    scheme->init(0);
    int count = connectivity*size; // num of elements to select
    double *schemedata = scheme->getData();
    MapD2I::iterator iter = sortedAllWeights.begin();
    for(int i=0; i<count; i++)
    {
        assertion(iter != sortedAllWeights.end(), "[AdaptiveCellular::createScheme] algorithmic error");
        schemedata[iter->second] = 1;
        sumAbsSelected += iter->first;
        iter++;
    }

    return sumAbsAll>0 ? sumAbsSelected/sumAbsAll : -1;
}*/
