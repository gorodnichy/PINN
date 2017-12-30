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

# ifndef _CLUSTERTEST_H_
# define _CLUSTERTEST_H_

# include "..\associative\FullProjective.h"
# include "..\lib\Data.h"

struct Attractor
{
    Vector attractor;
    int freq;
    double diff;
};

typedef multimap<int, Attractor, greater<int> > TopHitAttr;
typedef multimap<double, Attractor> TopDiffAttr;

void findAttractors(IOData *data, int numTests, AssociativeNet *net, TopHitAttr *topHitAttr, TopDiffAttr *topDiffAttr);
void testDataFile(Parameters params);

void testRandomClustersPartI(Parameters params);
void testRandomClustersPartII(Parameters params);

# endif /* _CLUSTERTEST_H_ */
