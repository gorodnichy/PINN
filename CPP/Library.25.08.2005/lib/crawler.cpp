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

# include "crawler.h"

# define REPORT_WIDTH 14
# define REPORT_OFFSET 20
# define REPORT_PRECISION 4


// runs through 1D or 2D grid of parameters (there MUST BE 1 or 2 variable parameters in "params")
// stores the results and some basic statistics
void runCrawler(CrawlerSubject *subject, Parameters params, string reportFile)
{
    int varCount = params.getVarCount();
    assertion(varCount==1||varCount==2,
        "[runCrawler] there must be 1 or 2 variable parameters (found "+toString(varCount)+")");

    string var0 = params.getVarName(0);
    string var1 = varCount==2? params.getVarName(1) : string("#");

    map<string, StatCounter> counters;

    bool firstRecord = true;
    Properties taskResults;
    string header = setWidth(var0 + "\\" + var1, REPORT_OFFSET, saLeft);
    if(varCount==2)
    {
         params.init(var1);
         while(params.setNext(var1))
            header += setWidth(toString(params.getFloat(var1), REPORT_PRECISION), REPORT_WIDTH);
    }                 

    params.init(var0);
    while(params.setNext(var0)) // cycle over var0 (always exists)
    {
        bool newLine = true;
        if(varCount==2)
            params.init(var1);

        while(varCount==2? params.setNext(var1) : newLine) // cycle over var1 (might not exist)
        {
            string varString = var0 + " = " + toString(params.getFloat(var0), REPORT_PRECISION);
            if(varCount==2)
                varString += ", " + var1 + " = " + toString(params.getFloat(var1), REPORT_PRECISION);

            try
            {
                subject->doTask(params, &taskResults);
            }
            catch(GenException &exc)
            {
                string excReport = "\n!!! [runCrowler] exception occured at " + getTimeStr() + "\n"
                    + varString + "\n" + exc.getMessage() + "\n";
                saveString(excReport, reportFile);
            }

            // save parameters
            if(firstRecord)
            {
                saveString(params.toString(), reportFile, true, true);
                saveString("\nExternal loop parameter: " + var0, reportFile);
            }

            // save task results and update counters
            for(Properties::iterator iter=taskResults.begin(); iter!=taskResults.end(); iter++)
            {
                string name = iter->first;
                double value = iter->second;
                string subReportFile = reportFile + "_" + name;
                if(firstRecord)
                    saveString(header, subReportFile, true, false);
                if(newLine)
                    saveString("\n" + setWidth(toString(params.getFloat(var0), REPORT_PRECISION), REPORT_OFFSET, saLeft), subReportFile, false, false);
                // save value
                saveString(setWidth(toString(value, REPORT_PRECISION), REPORT_WIDTH, saRight), subReportFile, false, false);

                counters[iter->first].addValue(value, varString);
            }
            firstRecord = false;
            newLine = false;
        }
        // save time info and var0 value
        saveString(setWidth(getTimeStr(), REPORT_OFFSET, saLeft)
            + toString(params.getFloat(var0), REPORT_PRECISION), reportFile);
    }

    // save statistics
    saveString("\nStatistics:\n", reportFile);
    for(map<string, StatCounter>::iterator iter=counters.begin(); iter!=counters.end(); iter++)
    {
        string name = iter->first;
        StatCounter counter = iter->second;
        saveString(name + ":\n" + counter.toString()+"\n", reportFile);
    }
}
