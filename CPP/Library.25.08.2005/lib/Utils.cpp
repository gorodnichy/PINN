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

# include "utils.h"
# include "math_aux.h"

# define PROPERTIES_SAVE_PRECISION 6
# define MATRIX_SAVE_PRECISION     6

//------------------------------ EXCEPTIONS ------------------------------------

void GenException::save()
{
    saveString(setWidth(getTimeStr(), 20, saLeft) + "GenException generated: " + message, "_!_exceptions.log");
}

void assertion(bool statement, string message)
{
    if(!statement)
    {
        cout << "Assertion failed: " << message.c_str() << endl;
        throw GenException(string("Assertion failed: ") + message);
    }
}

//-------------------------------- TIME ----------------------------------------


// yyyy/mm/dd-hh:mm:ss
string getTimeStr()
{
    SYSTEMTIME time;
	GetSystemTime(&time);
	static char buf[128];
	sprintf(buf, "%d/%d/%d-%d:%d:%d", time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond);
    return string(buf);
}

__int64 getTime_ms()
{
	SYSTEMTIME time;
	GetSystemTime(&time);
    return (((((__int64)time.wDay)*24 + time.wHour)*60 + time.wMinute)*60 + time.wSecond)*1000 + time.wMilliseconds;
}


//-------------------------------- SAVE ----------------------------------------


void saveString(string str, string fileName, bool overwrite, bool endWithNewLine)
{
	ofstream ofs(fileName.c_str(), overwrite? ios::trunc : ios::app);
	assertion(!ofs.fail(), string("[saveString] cant open file ")+=fileName);
	ofs << str.c_str();
    if(endWithNewLine)
        ofs << endl;
	ofs.close();
}

ostream& operator<<(ostream &os, Properties prop)
{
    /*os << setw(20) << "Time(GMT)";
    Properties::iterator iter;
    for(iter=prop.begin(); iter!=prop.end(); iter++)
        os << setw(12) << iter->first.c_str();
    os << endl;
    os << setw(20) << getTimeStr().c_str();
    for(iter=prop.begin(); iter!=prop.end(); iter++)
        os << setw(12) << setprecision(fmod(iter->second,1)||fabs(iter->second)>1e15? SAVE_PRECISION : 15) << iter->second;
	os << endl;
    return os;*/

    os << "Time(GMT)=" << getTimeStr().c_str();
    Properties::iterator iter;
    for(iter=prop.begin(); iter!=prop.end(); iter++)
        os << "; " << iter->first.c_str() << "="
           << setprecision(fmod(iter->second,1)||fabs(iter->second)>1e15? PROPERTIES_SAVE_PRECISION : 15) << iter->second;
	return os << endl;
}

//           Time(GMT)    { Names }
// yyyy/mm/dd-hh:mm:ss    { Values }
void saveProperties(Properties prop, bool withHeader, int width, string fileName, bool overwrite)
{
    Properties::iterator iter;
	ostrstream ostr;
    //ostr.setf(ios_base::fixed);
    if(withHeader)
    {
        ostr << setw(20) << "Time(GMT)";
        for(iter=prop.begin(); iter!=prop.end(); iter++)
		    ostr << setw(width) << iter->first.c_str();
	    ostr << endl;
    }
    ostr << setw(20) << getTimeStr().c_str();
    for(iter=prop.begin(); iter!=prop.end(); iter++)
        ostr << setw(width) << setprecision(fmod(iter->second,1)||fabs(iter->second)>1e15? PROPERTIES_SAVE_PRECISION : 15) << iter->second;
	ostr << ends;
    saveString(string(ostr.str()), fileName, overwrite);
    ostr.freeze(false); // there is a memory leak without this call ()
}

//           Time(GMT)    { Names1  Names2 }
// yyyy/mm/dd-hh:mm:ss    { Values1 Values2 }
void saveProperties(Properties prop1, Properties prop2, bool withHeader, int width, string fileName, bool overwrite)
{
    Properties::iterator iter;
	ostrstream ostr;
    //ostr.setf(ios_base::fixed);
    if(withHeader)
    {
        ostr << setw(20) << "Time(GMT)";
        for(iter=prop1.begin(); iter!=prop1.end(); iter++)
		    ostr << setw(width) << iter->first.c_str();
        for(iter=prop2.begin(); iter!=prop2.end(); iter++)
		    ostr << setw(width) << iter->first.c_str();
	    ostr << endl;
    }
    ostr << setw(20) << getTimeStr().c_str();
    for(iter=prop1.begin(); iter!=prop1.end(); iter++)
        ostr << setw(width) << setprecision(fmod(iter->second,1)? PROPERTIES_SAVE_PRECISION : 20) << iter->second;
    for(iter=prop2.begin(); iter!=prop2.end(); iter++)
        ostr << setw(width) << setprecision(fmod(iter->second,1)? PROPERTIES_SAVE_PRECISION : 20) << iter->second;
	ostr << ends;
    saveString(string(ostr.str()), fileName, overwrite);
    ostr.freeze(false); // there is a memory leak without this call ()
}

// add "time str" to fileName.log
void log(string str, string fileName, bool overwrite)
{
# ifdef LOGGING
	saveString(setWidth(getTimeStr(),20,saLeft)+str, fileName+string(".log"), overwrite);
# endif
}


// saves vector as a grayscale image to ppm format
// saves image*scaleFactor
void saveToPPM(int height, int width, double scaleFactor, Vector *image, string fileName)
{
    assertion(height*width == image->size(), "[saveToPPM] height*width must be equal to image->size()");

    saveString("P3", fileName, true); // format identifier
    saveString(toString(width) + " " + toString(height) + " 255", fileName);

    for(int i=0; i<width*height; i++)
    {
        int val = (*image)[i]*scaleFactor;
        if(val == 256)
            val = 255;
        assertion(val>0 && val<256, "[saveToPPM] must be  0 < image)[i]*scaleFactor < 256");
        string strVal = toString(val);
        saveString(strVal + " " + strVal + " " + strVal, fileName);
    }
}

// saves to file in text form
//
// { sizeY sizeX }
// w[0,0] w[0,1] .... w[0,sizeX-1]
// w[1,0] w[1,1] .... w[1,sizeX-1]
// ...
// w[sizeY-1,0]  ....
void saveMatrix(Matrix *mtr, string fileName, bool writeDim, bool overwrite, bool formatted, int precision)
{
    ofstream ofs(fileName.c_str(), overwrite? ios::trunc : ios::app);
	assertion(!ofs.fail(), string("[saveMatrix] cant open file ")+=fileName);
    ofs.precision(precision>0 ? precision : MATRIX_SAVE_PRECISION);

    bool isVerbose = mtr->sizeY()>=100 && mtr->sizeX()>1;

    int sizeY = mtr->sizeY(), sizeX = mtr->sizeX();
    if(writeDim)
        ofs << sizeY << ' ' << sizeX << endl;

    double *data = mtr->getData();
	for(int i=0; i<sizeY; i++)
    {
        if(isVerbose)
            printProgress("[saveMatrix]", i+1, 10, sizeY);
        for(int j=0; j<sizeX; j++)
        {
            if(formatted)
                ofs << setw(8);
            ofs << *data++ << " ";
        }
        ofs << endl;
    }
    if(isVerbose)
        cout << endl;

	ofs.close();
}


// reads from text file in format
//
// sizeY sizeX
// w[0,0] w[0,1] .... w[0,sizeX-1]
// w[1,0] w[1,1] .... w[1,sizeX-1]
// ...
// w[sizeY-1,0]  ....
//
// delimeters can be ' ', '\t', ';', ','
void readMatrix(Matrix *mtr, string fileName)
{
    FileLiner fl(fileName, "[readMatrix]", '#');
    assertion(fl.hasMoreLines(), "[readMatrix] empty file");

    readMatrix(&fl, mtr);
}


// the same as previous, but reads two matrices
void readMatrix(Matrix *mtr1, Matrix *mtr2, string fileName)
{
    FileLiner fl(fileName, "[readMatrix]", '#');
    assertion(fl.hasMoreLines(), "[readMatrix] empty file");

    readMatrix(&fl, mtr1);
    assertion(fl.hasMoreLines(), "[readMatrix(Matrix *mtr1, Matrix *mtr2)] no 2nd matrix");
    readMatrix(&fl, mtr2);
}


// the same as previous, but reads all matrices in a file
// terurned matrices are allocated by NEW (!), so must be DELETEd after use
vector<Matrix*> readMatrix(string fileName)
{
    FileLiner fl(fileName, "[readMatrix]", '#');
    assertion(fl.hasMoreLines(), "[readMatrix] empty file");

    vector<Matrix*> res;
    bool hasNext;

    while(true)
    {
        Matrix *mtr = new Matrix();
        bool hasNext = readMatrix(&fl, mtr);
        res.push_back(mtr);
        if(!hasNext)
            break;
    }

    return res;
}


// service function
// reads matrix from FileLiner, returns TRUE if more data (matrices) is left
bool readMatrix(FileLiner *fl, Matrix* mtr)
{
    assertion(fl->hasMoreLines(), "[readMatrix] no data to read from the very beginning");

    char *delims = " \t;,";
    int lineLength, tokenLength;
    char *line = fl->nextLineBuf(&lineLength);
    StringTokenizer st(line, lineLength, false, delims);
    assertion(st.tokenCount()==2, "[readMatrix] wrong file format");
    char *value = st.nextToken(&tokenLength);
    int sizeY = atoi(value);
    value = st.nextToken(&tokenLength);
    int sizeX = atoi(value);

    mtr->setSizeYX(sizeY, sizeX);
    double *data = mtr->getData();

    for(int i=0; i<sizeY; i++)
    {
        printProgress("[readMatrix]", i+1, 10, sizeY);
        assertion(fl->hasMoreLines(), "[readMatrix] wrong file format");
        line = fl->nextLineBuf(&lineLength);
        StringTokenizer st(line, lineLength, false, delims);
        assertion(st.tokenCount()==sizeX, "[readMatrix] wrong file format");
        for(int j=0; j<sizeX; j++)
        {
            value = st.nextToken(&tokenLength);
            *data++ = atof(value);
        }
    }
    cout << endl;

    return fl->hasMoreLines();
}


//------------------------------- STRINGS --------------------------------------


ostream& operator<<(ostream &os, const Matrix &mtr)
{
    double *data = mtr.getData();
    int sX = mtr.sizeX(), sY = mtr.sizeY();
    for(int i=0; i<sY; i++)
    {
        for(int j=0; j<sX; j++)
            os << *data++ << " ";
        os << endl;
    }
    return os;
}


string toString(Matrix *mtr, bool printTransposed, bool printSize, int precision)
{
    string res = "";
    double *data = mtr->getData();
    int sX = mtr->sizeX(), sY = mtr->sizeY();

    int resPrecision = precision>0? precision : MATRIX_SAVE_PRECISION;

    if(printSize)
    {
        res.append("Matrix (" + toString(sY) + "x" + toString(sX) + ")");
        if(printTransposed)
            res.append(", printed transposed");
        res += "\n";
    }

    if(!printTransposed)
        for(int i=0; i<sY; i++)
        {
            for(int j=0; j<sX; j++)
            {
                res.append(toString(mtr->el(i, j), resPrecision));
                res.append(" ");
            }
            res.append("\n");
        }
    else
        for(int j=0; j<sX; j++)
        {
            for(int i=0; i<sY; i++)
            {
                res.append(toString(mtr->el(i, j), resPrecision));
                res.append(" ");
            }
            res.append("\n");
        }

    return res;
}


// trim whitespaces and tab characters
string trim(string str)
{
    if(str.length() == 0)
        return str;
    unsigned int from = 0;
    int to = str.length()-1;
    while(from<str.length() && (str.at(from)==' ' || str.at(from)=='\t'))
        from++;
    while(to>=0 && (str.at(to)==' ' || str.at(from)=='\t'))
        to--;
    if((int)from > to)
        return "";
    else
        return str.substr(from, to-from+1);
}

// set width and align
// do nothing if required width less than trimmed string length
string setWidth(string str, unsigned int width, StringAlign align)
{
    string strTr = trim(str);
    if(width <= strTr.length())
        return strTr;
    string spaces(width-strTr.length(),' ');
    if(align == saLeft)
        return strTr+spaces;
    else
        return spaces+strTr;
}

string toString(int value)
{
    static char buf[32];
    sprintf(buf, "%d", value);
    return string(buf);
}

string toString(double value, int precision)
{
	/*ostrstream ostr;
    ostr << setprecision(precision) << value << ends;
    string res(ostr.str());
    ostr.freeze(false);
    return res;
    */
    static char buf[32];
    static char formatLine[32];
    sprintf(formatLine, "%%.%dg", precision);
    sprintf(buf, formatLine, value);
    return string(buf);
}


void deleteLine()
{ printf("\r%s\r",
"                                                                               ");
}


void printProgress(char *caption, int value, int step, int value2, string afterString)
{
    static int counter = 0;

    if(value%step == 0)
    {
        char spinner;
        if(counter == 0)
            spinner = '|';
        else if(counter == 1)
            spinner = '/';
        else if(counter == 2)
            spinner = '-';
        else
            spinner = '\\';

        if(value2 == 0)
            printf("\r%s %8.d  %c  %s", caption, value, spinner, afterString.c_str());
        else
            printf("\r%s %8.d  %c %8.d  %s", caption, value, spinner, value2, afterString.c_str());

            
        counter = (counter == 3)? 0 : counter+1;
    }
}


// delims must ne null-terminated string
void StringTokenizer::init(char *source, int size, bool doCopy, char *delims)
{
    length = size;
    ownBuffer = doCopy;
    if(doCopy)
    {
        str = new char[length+1];
        memcpy(str, source, length);
        str[length] = 0;
    }
    else
        str = source;
    this->delims = string(delims);
    pos = 0;
    while(pos<length && this->delims.find(str[pos])!=string::npos)
        pos++;
}

// returns pointer to the start of next token in internal buffer, "size" is set to token length
char* StringTokenizer::nextToken(int *size)
{
    assertion(pos < length, "[StringTokenizer::nextToken] no more tokens");

    while(pos<length && delims.find(str[pos])!=string::npos)
        pos++;
    char *res = &str[pos];
    *size = 0;
    while(pos<length && delims.find(str[pos])==string::npos)
    {
        pos++;
        (*size)++;
    }
    while(pos<length && delims.find(str[pos])!=string::npos)
        pos++;
    return res;
}

string StringTokenizer::theRest()
{
    assertion(pos < length, "[StringTokenizer::theRest] no more tokens");
    int from = pos;
    pos = length;
    return string(&str[from], length-from);
}

unsigned int StringTokenizer::tokenCount()
{
    unsigned int res = 0, i = 0;
    while(i < length)
    {
        while(i<length && delims.find(str[i])!=string::npos)
            i++;
        if(i < length)
            res++;
        while(i<length && delims.find(str[i])==string::npos)
            i++;
    }
    return res;
}


FileLiner::FileLiner(string fileName, string owner, char commentChar)
{
    this->commentChar = commentChar;
    lineSize = 1024;
    line = (char*)malloc(lineSize);
    ifs.open(fileName.c_str());
    assertion(ifs.is_open(), string("[FileLiner::FileLiner] cant open file: \"") + fileName + "\"" + (owner.compare("")? string(" for ")+owner : string("")));

    // skip new lines and lines starting from comments
    while((ifs.peek()=='\n'||ifs.peek()==commentChar) && ifs.peek()!=char_traits<char>::eof())
        if(ifs.peek()==commentChar) // goto new line
            while(ifs.peek()!='\n' && ifs.peek()!=char_traits<char>::eof())
                ifs.get();
        else
            ifs.get();

    hasMore = ifs.peek()!=char_traits<char>::eof();
}


FileLiner::~FileLiner()
{
    free(line);
    if(ifs.is_open())
        ifs.close();
}


// returns null terminated string
// NOTE: returns own buffer, do not delete
// all text after comment char is discarded (till the end of the line)
char* FileLiner::nextLineBuf(int *size)
{
    assertion(hasMore, "[FileLiner::nextLine] no more lines");
    int pos = 0;
    do
    {
        line[pos++] = ifs.get();
        if(pos+1 == lineSize) // extend buffer
        {
            lineSize *= 2;
            line = (char*)realloc(line, lineSize);
        }
    }
    while(ifs.peek()!='\n' && ifs.peek()!=commentChar && ifs.peek()!=char_traits<char>::eof());

    if(size)
        *size = pos;
    line[pos++] = 0;

    // skip new lines and lines starting from comments
    while((ifs.peek()=='\n'||ifs.peek()==commentChar) && ifs.peek()!=char_traits<char>::eof())
        if(ifs.peek()==commentChar) // goto new line
            while(ifs.peek()!='\n' && ifs.peek()!=char_traits<char>::eof())
                ifs.get();
        else
            ifs.get();

    hasMore = ifs.peek()!=char_traits<char>::eof();

    return line;
}


//----------------------------- PARAMETERS -------------------------------------


/* use variable parameter as
while(par.setNext())
{
    .... par.GetXXX()...
}

Value of a list parameter is specified as "item_1; item_2; ... item_N"
NOTE: items are trimmed when returned
*/
Parameter::Parameter(string value)
{
    this->value = value;
    StringTokenizer st(value, ":+*");
    if(st.tokenCount() > 1)
    {
        isVar = isInited = true;
        assertion(st.tokenCount() == 3, string("[Parameter::Parameter] improper format of variable parameter \"")
                                                         +value+"\", \nuse \"begin:end:step+\" or \"begin:end:coeff*\"");
        begin = atof(st.nextToken().c_str());
        end = atof(st.nextToken().c_str());
        increment = atof(st.nextToken().c_str());
        assertion(increment > 0, "[Parameter::Parameter] increment must be > 0");
        if(value.at(value.length()-1) == '+')
        {
            isAdditive = true;
            current = begin-increment;
        }
        else if(value.at(value.length()-1) == '*')
        {
            isAdditive = false;
            current = begin/increment;
        }
        else
            assertion(false, string("[Parameter::Parameter] improper format of variable parameter \"")
                                +value+"\", \nuse \"begin:end:step+\" or \"begin:end:coeff*\"");

    }
    else
        isVar = false;

    StringTokenizer st2(value, ";");
    isList = st2.tokenCount()>1? true : false;
    assertion(!(isVar&&isList), "[Parameter::Parameter] isVar&&isList");
}


// goto next value of variable-type parameter
// return true if new value is in specified range, false otherwise
bool Parameter::setNext()
{
    assertion(isVar, "[Parameter::setNext] is not a variable");

    isInited = false;

    if(current >= end)
        return false;
    if(isAdditive)
        current += increment;
    else
        current *= increment;
    return current<end? true : false;
}


// goto first value of variable-type parameter
void Parameter::init()
{
    assertion(isVar, "[Parameter::init] is not a variable");

    isInited = true;

    if(isAdditive)
        current = begin-increment;
    else
        current = begin/increment;
}


bool Parameter::getBool()
{
    assertion(!isList, "[Parameter::getBool] isList");

    if(!value.compare("true"))
        return true;
    if(!value.compare("false"))
        return false;
    throw GenException("[Parameter::getBool] specify proper boolean value");
}


int Parameter::getInt()
{
    assertion(!isList, "[Parameter::getInt] isList");

    if(isVar)
    {
        assertion(current<end, "[Parameter::getInt] end of range");
        return isInited? begin : current;
    }
    else
        return atoi(value.c_str());
}

float Parameter::getFloat()
{
    assertion(!isList, "[Parameter::getFloat] isList");

    if(isVar)
    {
        assertion(current<end, "[Parameter::getFloat] end of range");
        return isInited? begin : current;
    }
    else
        return atof(value.c_str());
}


// returns vector of int list-type parameter values
vector<int> Parameter::getIntList()
{
    assertion(isList, "[Parameter::getIntList] is not a list");

    StringTokenizer st(value, ";");
    vector<int> res;
    while(st.hasMoreTokens())
        res.push_back(atoi(trim(st.nextToken()).c_str()));

    return res;
}

// returns vector of float list-type parameter values
vector<float> Parameter::getFloatList()
{
    assertion(isList, "[Parameter::getFloatList] is not a list");

    StringTokenizer st(value, ";");
    vector<float> res;
    while(st.hasMoreTokens())
        res.push_back(atof(trim(st.nextToken()).c_str()));

    return res;
}


// returns vector of string list-type parameter values
vector<string> Parameter::getStringList()
{
    assertion(isList, "[Parameter::getStringList] is not a list");

    StringTokenizer st(value, ";");
    vector<string> res;
    while(st.hasMoreTokens())
        res.push_back(trim(st.nextToken()));

    return res;
}


//------------------------------ Parameters ------------------------------------


// read from file of a form "name value"
// value can be of a form "value" or "begin:end:step+" or "begin:end:coeff*"
// skip empty lines
// everything after '#' till the end of the line is discarded (as comments)
Parameters::Parameters(string fileName)
{
    FileLiner fl(fileName, "[Parameters::Parameters]", '#');
    while(fl.hasMoreLines())
    {
        string str = trim(fl.nextLine());
        if(str.length()==0)  // skip empty lines
            continue;

        StringTokenizer st(str," \t");
        assertion(st.tokenCount() >= 2, string("[Parameters::Parameters] wrong string format: ")+str);
        //map<string,Parameter>::value_type entry(st.nextToken(),Parameter(st.nextToken())); // Parameter(st.nextToken()) is calculated first?!!!
        string name = st.nextToken();
        assertion(find(name) == end(), string("[Parameters::Parameters] parameter \'")+name+"\' redefinition");
        Parameter par = Parameter(st.theRest());
        insert(map<string,Parameter>::value_type(name,par));
        if(par.isVariable())
            variables.push_back(name);
    }
    insert(map<string,Parameter>::value_type("#",Parameter("1:-1:1+")));
}

bool Parameters::getBool(string name)                { try { return getParameter(name).getBool(); } catch(GenException &exc) { throw GenException(string("[Parameters::getBool] name = ")+name,exc); } }
int Parameters::getInt(string name)                  { try { return getParameter(name).getInt(); } catch(GenException &exc) { throw GenException(string("[Parameters::getInt] name = ")+name,exc); } }
float Parameters::getFloat(string name)              { try { return getParameter(name).getFloat(); } catch(GenException &exc) { throw GenException(string("[Parameters::getFloat] name = ")+name,exc); } }
vector<int> Parameters::getIntList(string name)      { try { return getParameter(name).getIntList(); } catch(GenException &exc) { throw GenException(string("[Parameters::getIntList] name = ")+name,exc); } }
vector<float> Parameters::getFloatList(string name)  { try { return getParameter(name).getFloatList(); } catch(GenException &exc) { throw GenException(string("[Parameters::getFloatList] name = ")+name,exc); } }
vector<string> Parameters::getStringList(string name){ try { return getParameter(name).getStringList(); } catch(GenException &exc) { throw GenException(string("[Parameters::getStringList] name = ")+name,exc); } }
string Parameters::getString(string name)            { try { return getParameter(name).getString(); } catch(GenException &exc) { throw GenException(string("[Parameters::getString] name = ")+name,exc); } }
bool Parameters::setNext(string name)                { try { return getParameter(name).setNext(); } catch(GenException &exc) { throw GenException(string("[Parameters::setNext] name = ")+name,exc); } }
void Parameters::init(string name)                   { try { getParameter(name).init(); } catch(GenException &exc) { throw GenException(string("[Parameters::init] name = ")+name,exc); } }

string Parameters::toString()
{
    string res;
    for(map<string, Parameter>::iterator iter = begin(); iter!=end(); iter++)
        res += iter->first.at(0)=='#'? string("") : setWidth(iter->first, 25, saLeft)+iter->second.getString()+'\n';
    return res;
}

Parameter& Parameters::getParameter(string name)
{
    map<string, Parameter>::iterator iter = find(name);
    assertion(iter != end(), string("[Parameters] specify value for ")+name);
    return iter->second;
}


// data generation with random or functional dependance ------------------------


vector<double> dataGeneratorFunc(vector<double> arg)
{
    //assertion(arg.size() == 2, "[dataGeneratorFun] wrong argument dim");

    vector<double> res;

    //res.push_back(rand(0.2, 0.8));
    res.push_back(rand(-1, 1)>0? 1 : 0);

    return res;
}

void generateDataScript(string fileName, int count)
{
    int inDim = 10;

    double lowRange = -1;
    double highRange = 1;

    ofstream dataFile(fileName.c_str(), ios::trunc);

    for(int i=0; i<count; i++)
    {
        vector<double> in;
        for(int j=0; j<inDim; j++)
            in.push_back(rand(lowRange, highRange));
        vector<double> out = dataGeneratorFunc(in);

        if(i == 0)
            dataFile << inDim << " " << out.size() << endl;

        for(int j=0; j<in.size(); j++)
            dataFile << in[j] << ";";
        for(int j=0; j<out.size(); j++)
            dataFile << out[j] << (j==out.size()-1? "" : ";");
        dataFile << endl;

    }

    dataFile.close();
}


//------------------------------------------------------------------------------






