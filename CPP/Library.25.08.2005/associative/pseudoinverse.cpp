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

# include "pseudoinverse.h"


// net's description
string PseudoInverseNet::getDescription()
{
	return CellularNet::getDescription() + "\n# Pseudoinverse net, isSymmetric " + toStringB(isSymmetric) +
        ", alpha " + toString(alpha,4) + ", useBias " + toStringB(useBias);
}


// intitializes net
void PseudoInverseNet::subSetArchitecture()
{
	if(isSymmetric)
        maskSymmetryCheck();

    int i;

    systemRanks.assign(dim(),0);
	inputs.setSizeYX(0,dim());

	if(isSymmetric)													// isSymmetric	
	{
		boundaryIndices.resize(dim());
		for(i=0; i<dim(); i++)
		{
			int index = indexOfSorted(&mask[i],i);
			assertion(index >= 0, "[PseudoInverseNet::subInit] algorithmic error");
			boundaryIndices[i] = index;
		}
	}

    subInputs.resize(dim());
    subInputsPI.resize(dim());

	for(i=0; i<dim(); i++)											// isSymmetric	
	{
        int sizeX = mask[i].size()-(isSymmetric?boundaryIndices[i]:0)+(useBias?1:0);
		subInputs[i].setSizeYX(0,sizeX);
		subInputsPI[i].setSizeYX(0,sizeX);
    }
}


// clears net contents
void PseudoInverseNet::clear()
{
    systemRanks.assign(dim(),0);
    inputs.setSizeY(0);
    for(int i=0; i<dim(); i++)
    {
        subInputs[i].setSizeY(0);
        subInputsPI[i].setSizeY(0);
    }
    CellularNet::clear();
}


// see M. Bruccoli, L. Carnimeo, G. Grassi. Heteroassociative memories via
// cellular neural networks. International journal of circuit theory and
// applications, 26, 231-241 (1998)
bool PseudoInverseNet::subTrain(Vector *in, Properties *prop)
{
    int i;

    // append "in" to "inputs"
    Matrix inm(*in);
    inm.transpose();
    inputs.append(&inm);

    // cycle over neurons
    int nIndep = 0;
    for(int n=0; n<dim(); n++)
    {
        vector<int> &maskn = mask[n];
        Vector &weightsn = weights[n];

        // append "in" to "subInputs"
        Matrix &subn = subInputs[n];
        int boundary = isSymmetric? boundaryIndices[n] : 0;			// isSymmetric
        Vector row(subn.sizeX());
        for(i=boundary; i<maskn.size(); i++)
            row[i-boundary] = (*in)[maskn[i]];
        if(useBias)
            row[i-boundary] = 1;

        // recalculate subInputsPI
        Matrix &subPIn = subInputsPI[n];
        bool isIndep;
        Matrix::pseudoInverseRecalc(false,&subn,&row,&subPIn,&isIndep);
        if(isIndep)
        {
            nIndep++;
            systemRanks[n]++;
        }

        // calculate postsynapse ps
        Matrix ps(subPIn.sizeY(),1);
        ps.init(0);
        for(i=0; i<ps.sizeY(); i++)
        {
            double sp = 0;
			if(isSymmetric)											// isSymmetric	
				for(int j=0; j<boundary; j++)
					sp += weightsn[j]*inputs.el(i,maskn[j]);
            ps.el(i,0) = inputs.el(i,n)*alpha - sp;
        }

        // find n-th neuron weigths and fill weight matrix symmetrically
        subPIn.multTN(&ps,&row);
        for(i=boundary; i<maskn.size(); i++)
        {
            weightsn[i] = row.el(i-boundary,0);
			if(isSymmetric)											// isSymmetric	
			{
				int index = indexOfSorted(&mask[maskn[i]],n);
				if(index >= 0)
					weights[maskn[i]][index] = row.el(i-boundary,0);
				else
					throw GenException("[PseudoInverseNet::subTrain] algorithmic error");
			}
        }
        if(useBias)
            bias[n] = row.el(i-boundary,0);
    }

    if(prop != 0)
    {
        (*prop)["numIndep"] = nIndep;
    }

    return true;
}

double PseudoInverseNet::getAveRank()
{
    double sum = 0;
    for(int i=0; i<systemRanks.size(); i++)
        sum += systemRanks[i];
    return systemRanks.size()? sum/systemRanks.size() : -1;
}

/*
// recalcs mtp accordingly to row that will be added to mt
// mtp has to be equal to mt+ before function call
// isIndep is set to proper value accordingly to the algorithm 
double PseudoInverseNet::pseudoInverseRecalc(Matr *mt, Vec *row, Matr * mtp, bool *isIndep)
{
    if(mt->sizeX()!=mtp->sizeX() || mt->sizeX()!=row->size() || mt->sizeY()!=mtp->sizeY())
        throw GenException("[PseudoInverseNet::pseudoInverseRecalc] diff sizes");
    if(mt->sizeX() == 0)
         throw GenException("[PseudoInverseNet::pseudoInverseRecalc] sizeX == 0");

    int i;
    int sX = mt->sizeX();
    int sY = mt->sizeY();

    // mtp = row / |row|^2
    if(sY == 0)
    {
		if(isIndep != 0)
			*isIndep = true;
        double sqnorm = sproduct(row,row);
        if(sqnorm == 0)
            throw GenException("[PseudoInverseNet::pseudoInverseRecalc] |row| = 0");
        mt->setSizeY(1);
        mtp->setSizeY(1);
        for(i=0; i<sX; i++)
        {
            mt->el(0,i) = (*row)[i];
            mtp->el(0,i) = (*row)[i]/sqnorm;
        }
        return sqnorm;
    }

    // vecX = (I - mtp.T*mt) * row
    Vec vecX(sX), vecY(sY);
    mt->mult(row,&vecY);
    mtp->multTr(&vecY,&vecX);
    for(i=0; i<sX; i++)
        vecX[i] = (*row)[i]-vecX[i];
    double sqnorm = sproduct(&vecX,&vecX);

    // vecX = (I - mtp.T*mt) * row / |...|^2
    bool hasVecY = false;
    if(sqrt(sqnorm)/sX > PSEUDO_INVERSE_EPS)
    {
		if(isIndep != 0)
			*isIndep = true;
        for(i=0; i<sX; i++)
            vecX[i] /= sqnorm;
    }
    // vecX = mtp.T*mtp*row / (1 + |mtp*row|^2)
    else
    {
		if(isIndep != 0)
			*isIndep = false;
        hasVecY = true;
        mtp->mult(row,&vecY);
        double denom = sproduct(&vecY,&vecY);
        denom += 1;
        mtp->multTr(&vecY,&vecX);
        for(i=0; i<sX; i++)
            vecX[i] /= denom;
    }

    // vecY = mtp*row
    if(!hasVecY)
        mtp->mult(row,&vecY);
    mt->setSizeY(sY+1);
    mtp->setSizeY(sY+1);
    for(i=0; i<sY; i++)
        for(int j=0; j<sX; j++)
            mtp->el(i,j) -= vecY[i]*vecX[j];
    for(i=0; i<sX; i++)
    {
        mt->el(sY,i) = (*row)[i];
        mtp->el(sY,i) = vecX[i];
    }

	return sqnorm;
}
*/





