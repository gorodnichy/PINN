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

# include "associative\nets.h"
# include "tests\tests.h"
# include "tests\ClusterTest.h"

# include "lib\Crawler.h"

void main(int argvc, char **argv)
{
    initRNG();

    try
	{
        if(argvc != 2)
        {
            cout << "[nets] args: ini_file_name";
            return;
        }

        Parameters params(argv[1]);

        generalTestingScript(params);
        //convergenceStabilityScript(params);
        //adaptiveTransitionScript(params);

    }
	catch(GenException &exc)
	{
		cout << "[nets] exception: " << exc.getMessage();
	}
}