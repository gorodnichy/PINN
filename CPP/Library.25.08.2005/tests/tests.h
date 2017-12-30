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

# ifndef _TESTS_H_
# define _TESTS_H_

# include "..\associative\cellprojective.h"
# include "..\associative\pseudoinverse.h"
# include "..\associative\fullprojective.h"
# include "..\associative\AdaptiveCellular.h"
# include "..\associative\BAMCellular.h"
# include "..\associative\Hebbian.h"
# include "..\associative\DeltaCellular.h"
# include "..\associative\SmallWorld.h"
# include "..\modular\modular.h"

# include "..\lib\Data.h"
# include "..\lib\crawler.h"

# define H_THRESHOLD 0              // test 1
# define R_ATTRACTION 50            // test 2,3
# define CHECK_POINT_COUNT 3        // test 1,2,3
# define START_PORTION 1            // test 2,3
# define START_SHIFT 0              // test 2,3
# define REPORT_WIDTH_VAL 14
# define EXIT_AVEH_IF_ERROR         // exit in getAveHamming if at least one error takes place && H_THRESHOLD == 0


class StatMaker
{
public:
    void calculateDistribution(CellularNet *net, double halfR, int halfC,
                                Vector *arg, Vector *func, double *mean, double *disp, int type);
    double getCorrCoeff(CellularNet *net, IOData *data);
};

class AdaptiveCellularSubj : protected AdaptiveCellularNet, public CrawlerSubject
{
public:
    AdaptiveCellularSubj(Parameters params) : AdaptiveCellularNet(params) {}
    void doTask(Parameters params, Properties *prop);
};

AssociativeNet *createNetwork(Parameters params);
void generalTestingScript(Parameters params);

double getAveHamming(int numTests, AssociativeUnit *unit, IOData *data, int initialH, bool singleIter, double *errorPortion = 0, Properties *prop = 0, Vector *errorDistribution = 0);
int getRAttraction(int numTests, bool singleIter, AssociativeUnit *unit, IOData *data, Properties *prop, bool probCriterion = false);
void test1(int numTestEpochs, AssociativeUnit *unit, RAMData *data, Parameter numStoredScale, string reportFile, bool adjustBeforeTesting, bool needSave = false, bool singleIter = false);
void test4(int numTestEpochs, AssociativeUnit *unit, RAMData *data, int numStored, string variableName, Parameter variableValue, string reportFile, bool adjustBeforeTesting, bool needSave = false, bool singleIter = false);
void testStability(int numInitialStates, AssociativeUnit *unit, RAMData *data, int numStored, string reportFile);

//--------------------------- New Tests (in accordance with Neil Davey et al) ---------------------------------
void getNormalizedRAttraction(int numTestEpochs, int numStored, AssociativeUnit *unit, bool contiguousNoise, Properties *prop);
double getAttrSimilarity(AssociativeUnit *unit, Vector *pattern, Vector *startPoint, bool contiguousNoise, double *aveIter, double *convergedRatio);
double getCrossSimilarity(RAMData *data, int patternIndex, Vector *pattern);
void test5(AssociativeUnit *unit, Parameters params, bool contiguousNoise);
//-------------------------------------------------------------------------------------------------------------


/*void test2(int numTests, AssociativeUnit *unit, IOData *data, string parameterName, Parameter parameterValue, string reportFile, bool adjustBeforeTesting, bool needSave = false, bool singleIter = false);
void test3(int numTests, CellularNet *net, vector<IOData> *dataSet, string reportFile, bool singleIter = false);
*/


void trainAndSave(AssociativeUnit *unit, IOData *data, int netFill, string netFile);

/*void testPIAlgorithm(int from, int to, int step, int nTests, string reportFile);
void testStatistics(CellularNet *net, IOData *data, int from, int to, int step, string reportFile);
void testErrorDistribution(int numTests, AssociativeNet *net, IOData *data, int from, int to, int step, string reportFile);
void testRetrain(int numTests, FullProjectiveNet *net, IOData *data, int numStored, double eraseValue, int eraseType, int retrainCount, int multCount, string reportFile);
void retrainScript();
void modularScript();
void adaptiveScript();
void weightsSelectScript();*/

void adaptiveTransitionScript(Parameters params);
void modularStatScript(Parameters params);
void modularTestScript(Parameters params);
void convergenceStabilityScript(Parameters params);


# endif /* _TESTS_H_ */