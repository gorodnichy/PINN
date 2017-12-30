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

# ifndef _FULL_PROJECTIVE_H_
# define _FULL_PROJECTIVE_H_

# include "nets.h"

// fully connected projective net
// weight matrix is formed as a projective onto training vectors set
class FullProjectiveNet : public FullNet
{
public:
    int retrainType;

    FullProjectiveNet(int dim) : FullNet(dim) { setDim(dim); }
    FullProjectiveNet(Parameters params) : FullNet(params.getInt(PARAM_NET_DIM))
    {
        setDim(params.getInt(PARAM_NET_DIM));
        setParameter(PARAM_DESAT_COEFF, params.getFloat(PARAM_DESAT_COEFF));
    }


    virtual void setDim(int dim)
    {
        FullNet::setDim(dim);
        retrainType = 0;
        projectiveWeights.setSizeYX(dim, dim);
        projectiveWeights.init(0);
        clear();
    }

    void loadFromFile(string fileName);
    string getDescription();
    virtual void clear() { projectiveWeights.init(0); FullNet::clear(); }
    bool retrain(Vector *in);
    void multiplyWeights() { Matrix wgt; wgt.assign(&weights); wgt.mult(&wgt,&weights); }
    double projDiff() { return weights.sqdist(&projectiveWeights); }
    double firstSecondDegreeDiff() { Matrix w2(dim(),dim()); weights.mult(&weights,&w2); return weights.sqdist(&w2); }
    void deleteNeurons(int count);
    void deleteWeights(double portion);
    
    virtual void getProperties(Properties *prop)
    {
        (*prop)["|W-Proj|^2"] = projDiff();
        (*prop)["|W-W^2|^2"] = firstSecondDegreeDiff();
        FullNet::getProperties(prop);
    }

protected:
    Matrix projectiveWeights;
    bool subTrain(Vector *in, Properties *pr)
    {
        projectiveWeights.expandProjectiveMatrix(in);
        return weights.expandProjectiveMatrix(in);
    }
};

# endif /* _FULL_PROJECTIVE_H_ */
 