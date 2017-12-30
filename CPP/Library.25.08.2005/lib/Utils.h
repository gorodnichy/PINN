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

# ifndef _UTILS_H_
# define _UTILS_H_

# include <math.h>
# include <stdlib.h>

# include <fstream>
# include <iostream>
# include <iomanip>
# include <strstream>

# include <algorithm>
# include <vector>
# include <map>
# include <set>
# include <string>

# include <windows.h>

using namespace std;

typedef map<string, double> Properties;

//------------------------------ Compilation -----------------------------------

# ifdef BORLAND_COMPILER
# include <mem.h>
# endif

# ifdef MS_COMPILER
# include <memory.h>
# define for if (0) {} else for // scope of definition in “for” statements
# endif

//---------------------------- STL Vector Debugging ----------------------------
//# define VECTOR_DEBUG
# ifdef VECTOR_DEBUG
template <class T, class Allocator = allocator<T> > class _vector : public vector<T,Allocator>
{
public:
    _vector() : vector<T,Allocator>()                       { _size = 0; }
    _vector(int n) : vector<T,Allocator>(n)                 { _size = n; }
    _vector(int n, T el) : vector<T,Allocator>(n,el)        { _size = n; }
    void assign(vector<T,Allocator>::iterator first, vector<T,Allocator>::iterator last)
                                                            { vector<T,Allocator>::assign(first,last); _size = size(); }
    void assign(int n, T x = T())                           { vector<T,Allocator>::assign(n,x); _size = n; }
    void resize(int n, T x = T())                           { vector<T,Allocator>::resize(n,x); _size = n; }
    void push_back(const T& x)                              { vector<T,Allocator>::push_back(x); _size = size(); }
    T &operator[](int i)
    {
        _size = size();
        assertion(i>=0 && i<_size, "[_vector::operator[]] index out of range");
        //return ::operator[](i); // ???
        return at(i);
    }

    int _size;
};
# define vector _vector
# endif /* VECTOR_DEBUG */
//------------------------------------------------------------------------------

class GenException
{
    public:
        GenException()                                      { message = ""; }
	    GenException(string message)	                    { this->message = message; save(); }
	    GenException(string message, GenException exc)      { this->message = message + string("\nNested exception is:\n") + exc.getMessage(); save(); }
        string getMessage()                                 { return message; }
    private:
	    string message;
        void save();
};

void assertion(bool statement,  string message);


// time ------------------------------------------------------------------------

string getTimeStr();
__int64 getTime_ms();


// save & load -----------------------------------------------------------------

// wrapper class, provides correct << operator usage
class outBinStream : private ofstream
{
public:
    outBinStream(string fileName, bool trunc) : ofstream(fileName.c_str(), ios::binary | (trunc? ios::trunc : 0)) { assertion(is_open(), string("[outBinStream::outBinStream] cant open file ") + fileName); }
    bool is_open() { return ofstream::is_open(); }
    outBinStream& write(char* pch, int nCount) { ofstream::write(pch, nCount); return *this; }
    int tellp() { return ofstream::tellp(); }
    void close() { ofstream::close(); }

    outBinStream& operator<<(int ival)      { ofstream::write((char*)&ival, sizeof(int)); return *this; }
    outBinStream& operator<<(float fval)    { ofstream::write((char*)&fval, sizeof(float)); return *this; }
    outBinStream& operator<<(double dval)   { ofstream::write((char*)&dval, sizeof(double)); return *this; }
    outBinStream& operator<<(bool bval)     { ofstream::write((char*)&bval, sizeof(bool)); return *this; }
};

// wrapper class, provides correct >> operator usage
class inBinStream : protected ifstream
{
public:
    inBinStream(string fileName) : ifstream(fileName.c_str(), ios::binary) { assertion(is_open(), string("[inBinStream::inBinStream] cant open file ") + fileName); }
    bool is_open() { return ifstream::is_open(); }
    inBinStream& read(char* pch, int nCount) { ifstream::read(pch, nCount); return *this; }
    int tellg() { return ifstream::tellg(); }
    void close() { ifstream::close(); }

    inBinStream& operator>>(int &ival)     { ifstream::read((char*)&ival, sizeof(int)); return *this; }
    inBinStream& operator>>(float &fval)   { ifstream::read((char*)&fval, sizeof(float)); return *this; }
    inBinStream& operator>>(double &dval)  { ifstream::read((char*)&dval, sizeof(double)); return *this; }
    inBinStream& operator>>(bool &bval)    { ifstream::read((char*)&bval, sizeof(bool)); return *this; }
};
void saveString(string str, string fileName, bool overwrite = false, bool endWithNewLine = true);
ostream& operator<<(ostream &os, Properties prop);
void saveProperties(Properties prop, bool withHeader, int width, string fileName, bool overwrite = false);
void saveProperties(Properties prop1, Properties prop2, bool withHeader, int width, string fileName, bool overwrite = false);
void log(string str, string fileName, bool overwrite = false);
void saveStr(string &line, outBinStream &stream) { stream << (int)line.size(); stream.write((char*)line.c_str(), line.size()); }
void loadStr(string &line, inBinStream &stream) { int size; stream >> size; char *buf = new char[size]; stream.read(buf, size); line.assign(buf, size); delete[] buf; }

class Vector;
void saveToPPM(int height, int width, double scaleFactor, Vector *image, string fileName);

class Matrix;
class FileLiner;
void saveMatrix(Matrix *mtr, string fileName, bool writeDim = true, bool overwrite = false, bool formatted = false, int precision = -1);
bool readMatrix(FileLiner *fl, Matrix* matr);
void readMatrix(Matrix *mtr, string fileName);
void readMatrix(Matrix *mtr1, Matrix *mtr2, string fileName);
vector<Matrix*> readMatrix(string fileName);

// strings ---------------------------------------------------------------------

ostream& operator<<(ostream &os, const Matrix &mtr);
string toString(Matrix *mtr, bool printTransposed = false, bool printSize = false, int precision = -1);

string trim(string str);
enum StringAlign { saLeft = 0, saRight = 1 };
string setWidth(string str, unsigned int width, StringAlign align = saRight);
string toStringB(bool value) { return value? "true" : "false"; }
string toString(int value);
string toString(double value, int precision);
int stringLength(char *str) { int res = 0; while(str[res]) res++; return res; }

void deleteLine();
void printProgress(char *caption, int value, int step, int value2 = 0, string afterString = "");

class StringTokenizer
{
public:
    // delims must be null-terminated string
    StringTokenizer(char *source, int size, bool doCopy, char *delims) { init(source, size, doCopy, delims); }
    // delims must be null-terminated string
    StringTokenizer(string source, char *delims) { init((char*)source.c_str(), source.length(), true, delims); }
    ~StringTokenizer()              { if(ownBuffer) delete[] str; }
	bool hasMoreTokens()            { return pos < length; }
    char* nextToken(int *size);
	string nextToken()              { int size; char *token = nextToken(&size); return string(token, size); }
    string theRest();
	unsigned int tokenCount();

private:
	char *str;
    bool ownBuffer;
    unsigned int length, pos;
    string delims;
    void init(char *source, int size, bool doCopy, char *delims);
};

class FileLiner
{
public:
    FileLiner(string fileName, string owner = "", char commentChar = -1);
    ~FileLiner();
    bool hasMoreLines()         { return hasMore; }
    char *nextLineBuf(int *size = 0);
    string nextLine()           { return string(nextLineBuf()); }

private:
    FileLiner(const FileLiner &);
    void operator = (const FileLiner &);
    ifstream ifs;
    char *line;
    int lineSize;
    bool hasMore;
    char commentChar;
};


// parameters ------------------------------------------------------------------

class Parameter
{
public:
    Parameter(string value);
    bool isVariable()       { return isVar; }
    bool setNext();
    void init();
    bool getBool();
    int getInt();
    float getFloat();
    string getString()      { return value; }
    string toString()       { return value; }
    vector<int> getIntList();
    vector<float> getFloatList();
    vector<string> getStringList();

private:
    string value;
    bool isVar, isAdditive, isInited, isList; // isInited - to return "begin" at getInt/Float if no "setNext" called
    double begin, end, increment, current;
};

class Parameters : protected map<string, Parameter>
{
public:
    Parameters() {}
    Parameters(string fileName);

    bool getBool(string name);
    int getInt(string name);
    float getFloat(string name);
    vector<int> getIntList(string name);
    vector<float> getFloatList(string name);
    vector<string> getStringList(string name);
    string getString(string name);
    bool setNext(string name);
    void init(string name);
    int getVarCount()               { return variables.size(); }
    string getVarName(int index)    { return variables.at(index); }
    string toString();
    Parameter& getParameter(string name);

protected:
    vector<string> variables;
};


// pointers & values -----------------------------------------------------------

void inline swapPointers(void **p1, void **p2) { void *vp; vp = *p1; *p1 = *p2; *p2 = vp; }
template<class T> void inline swapValues(T *p1, T *p2) { T val; val = *p1; *p1 = *p2; *p2 = val; }


// Provides keeping multiple references to the same object
// Never delete such an object, call to delRef() indstead
// If hierarchy with more than one descending level is used
// then all descendents (except of the last one) must have virtual destructor
class RefObject
{
public:
    RefObject()          { refCount = 1; }
    virtual ~RefObject() { assertion(refCount <= 0, "[RefObject::~RefObject] attempt to delete \"this\" directly. Likely caused by non-dynamic instance creation or by multi-thread access."); }
    RefObject *addRef()  { refCount++; return this; }
    void delRef()        { if(--refCount <= 0) ::delete this; /* not an operator */ }

private:
    int refCount;
# ifdef _BORLANDC_
	// to prevent direct deletion at a compile-time
    void operator delete(void *arg) { }
# endif
};


// sound :) --------------------------------------------------------------------

void inline beep() { Beep(1000,20); }


// misc ------------------------------------------------------------------------

void generateDataScript(string fileName, int count);

# endif /* _UTILS_H_ */