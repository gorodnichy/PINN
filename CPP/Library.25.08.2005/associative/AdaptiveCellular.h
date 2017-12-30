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

# ifndef _ADAPTIVE_CELLULAR_H_
# define _ADAPTIVE_CELLULAR_H_

# include "pseudoinverse.h"

// Cellular associative net with adaptive architecture
// Mask is selected dynamically according to the biggest absolute
// values in projective matrix
// after that the cellular net is trained using projective or pseudoinverse algorithm
//
// References:
// Dekhtyarenko, O.; Reznik, A. & Sitchov, A. Associative Cellular Neural Networks with Adaptive Architecture 2004 , 219-224 The 8th IEEE International Biannual Workshop on Cellular Neural Networks and their Application (CNNA'04)
// Dekhtyarenko, O.; Tereshko, V. & Fyfe, C. Phase transition in sparse associative neural networks 2005 , ??? European Symposium on Artificial Neural Networks (ESANN'05)
class AdaptiveCellularNet : public PseudoInverseNet
{
public:
    AdaptiveCellularNet(int dim, bool noDiagonalWeights, bool isProjective, bool hasQuota, bool useCorrelation,
                        double connectivityPortion)
                        : PseudoInverseNet(noDiagonalWeights, false /*useBias*/, 1 /*alpha*/, false /*isSymmetric*/)
    {
        _AdaptiveCellularNet(dim, isProjective, hasQuota, useCorrelation, connectivityPortion);
    }

    AdaptiveCellularNet(Parameters params)
                        : PseudoInverseNet(params.getBool(PARAM_NO_DIAG_WGTS), false /*useBias*/, 1 /*alpha*/, false /*isSymmetric*/)
    {
        _AdaptiveCellularNet(params.getInt(PARAM_NET_DIM), params.getBool(PARAM_IS_PROJECTIVE), params.getBool(PARAM_HAS_QUOTA),
            params.getBool(PARAM_USE_CORRELATIOM), params.getFloat(PARAM_CONNECTIVITY_DEG));
    }


    void _AdaptiveCellularNet(int dim, bool isProjective, bool hasQuota, bool useCorrelation,
                        double connectivityPortion)
    {
        assertion(connectivityPortion>0 && connectivityPortion<=1, "[AdaptiveCellularNet::AdaptiveCellularNet] connectivityPortion must be in (0,1]");
        this->isProjective = isProjective;
        this->hasQuota = hasQuota;
        this->useCorrelation = useCorrelation;
        this->connectivityPortion = connectivityPortion;
        setDim(dim);
    }

    void setDim(int dim) { keepDataInSSA = false; setLocal1DArchitecture(dim, 1); }
    string getDescription();
    void retrain(double alpha, Properties *prop);

    //static double createScheme(Matrix *weights, Matrix *scheme, double connectivity);

protected:
	virtual void subSetArchitecture();
    virtual void clear();
    virtual bool subTrain(Vector *in, Properties *prop = 0);
    virtual void doEndTraining(Properties *prop = 0);

private:
    bool isProjective;                                                                       
    bool hasQuota;
    bool useCorrelation;
    Matrix projMatrix;
    CorrelationMatrix corrMatrix;
    bool keepDataInSSA; // flag that shows that subSetArchitecture() was called during doEndTraining

    double setArchitecture(Matrix *mtr, int *changeDegree = 0);
    void setWeights(Matrix *mtr);
};

# endif /* _ADAPTIVE_CELLULAR_H_ */
 