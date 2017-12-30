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

# ifndef _CELL_PROJECTIVE_H_
# define _CELL_PROJECTIVE_H_

# include "nets.h"

// cellular net with projective learning rule
// weight matrix is a projective one with only mask elements left
class CellProjectiveNet : public CellularNet
{
private:
	Matrix projectiveMatrix;

protected:
	virtual void subSetArchitecture();
	virtual void clear() { projectiveMatrix.init(0); CellularNet::clear(); }
	virtual bool subTrain(Vector *in, Properties *prop = 0);
    virtual void doEndTraining(Properties *prop = 0);

public:
	string getDescription();
};


# endif /* _CELL_PROJECTIVE_H_ */
