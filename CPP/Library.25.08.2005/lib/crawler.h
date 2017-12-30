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

# ifndef _CRAWLER_H_
# define _CRAWLER_H_

# include "math_aux.h"

class CrawlerSubject
{
public:
    // results of task execution must be returned via properties
    // can throw GenException
    virtual void doTask(Parameters params, Properties *prop) = 0;
};

void runCrawler(CrawlerSubject *subject, Parameters params, string reportFile);

# endif /* _CRAWLER_H_ */
