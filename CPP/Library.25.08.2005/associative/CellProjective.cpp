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

# include "cellprojective.h"


// net's description
string CellProjectiveNet::getDescription()
{
	return CellularNet::getDescription() + "\n# Cellular Projective net";
}


// intitializes net 
void CellProjectiveNet::subSetArchitecture()
{
	projectiveMatrix.setSizeYX(dim(), dim());
    projectiveMatrix.init(0);
}


// expands subspace of projective matrix 
bool CellProjectiveNet::subTrain(Vector *in, Properties *prop)
{
    double diff;
    bool res = projectiveMatrix.expandProjectiveMatrix(in, &diff);
    if(prop)
        (*prop)["expand_diff"] = diff;
    return res;
}


void CellProjectiveNet::doEndTraining(Properties *prop)
{
    double *pmdata = projectiveMatrix.getData();
    int size = dim();
    for(int i=0; i<size; i++)
    {
        vector<int> &maski = mask[i];
        Vector &weightsi = weights[i];
        int maskisize = maski.size();
        for (int j=0; j<maskisize; j++)
            weightsi[j] = pmdata[maski[j]];
        pmdata += size;
    }
}



