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

# ifndef _PSEUDOINVERSE_H_
# define _PSEUDOINVERSE_H_

# include "nets.h"

// class for sparse network with Pseudoinverse Learning Rule
// the network can have arbitrary architecture and is trained by finding
// the optimal solutions for each neuron, that is W {p} = {p}, where {p} is
// a set of patterns to be stored
//
// for the original description of the algorithm see
// Brucoli, M.; Carnimeo, L. & Grassi, G. Discrete-time cellular neural networks for associative memories with learning and forgetting capabilities IEEE Transactions on Circuits and Systems, 1995 , 42 , 396–399
class PseudoInverseNet : public CellularNet
{
public:
	PseudoInverseNet(bool noDiagonalWeights, bool useBias, double alpha, bool isSymmetric)
	{
        _PseudoInverseNet(noDiagonalWeights, useBias, alpha, isSymmetric);
	}

    PseudoInverseNet(Parameters params)
    {
        _PseudoInverseNet(params.getBool(PARAM_NO_DIAG_WGTS),
            params.getBool(PARAM_USE_BIAS), params.getFloat(PARAM_ALPHA), params.getBool(PARAM_IS_SYMMETRIC));

        setLocal1DArchitecture(params.getInt(PARAM_NET_DIM), params.getInt(PARAM_CONN_R));
    }

    string getDescription();

    virtual void getProperties(Properties *prop)
    {
        (*prop)["aveRank"] = getAveRank();
        CellularNet::getProperties(prop);
    }

    double getAveRank();

protected:
	virtual void subSetArchitecture();
	virtual void clear();
	virtual bool subTrain(Vector *in, Properties *prop = 0);

private:
	bool isSymmetric;
    double alpha;
	Matrix inputs;
    vector<int> boundaryIndices, systemRanks;
    vector<Matrix> subInputs, subInputsPI;

    void _PseudoInverseNet(bool noDiagonalWeights, bool useBias, double alpha, bool isSymmetric)
    {
        this->noDiagonalWeights = noDiagonalWeights;
		this->useBias = useBias;
        this->alpha = alpha;
		this->isSymmetric = isSymmetric;
    }
};


# endif /* _PSEUDOINVERSE_H_ */
