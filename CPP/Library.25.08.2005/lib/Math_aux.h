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

# ifndef _MATH_AUX_H_
# define _MATH_AUX_H_

# include "utils.h"
# include "templates.h"


# define min(X, Y)  ((X) < (Y) ? (X) : (Y))  // Beware of "Duplication of Side Effects"!!! ( akin min(X, foo(Z)) )
# define max(X, Y)  ((X) > (Y) ? (X) : (Y))  //                 ----------------"----------------

# define PSEUDO_INVERSE_EPS 1e-10
# define NOT_A_NUMBER -11111111

// StatCounter fields
# define SC_COUNT   (((char)1)<<0)
# define SC_AVERAGE (((char)1)<<1)
# define SC_SD      (((char)1)<<2)
# define SC_DISP    (((char)1)<<3)
# define SC_MIN     (((char)1)<<5)
# define SC_MAX     (((char)1)<<4)
# define SC_ALL     ((char)0xFF)

int indexOfSorted(vector<int> *vec, int el);

void initRNG();
int inline round(double arg) { int intPart = (int)arg; double res = arg-intPart; return fabs(res)<0.5? intPart : (res>0 ? intPart+1 : intPart-1); }
double inline rand(double from, double to) { if(from == to) return 0; return ((double)rand())*(to-from)/RAND_MAX + from; }
int inline rand(int to) { if(to<=0) return 0; int res = ((double)rand())/RAND_MAX*to; return res==to ? res-1: res; }
double inline trunc(double value, double threshold) { return fabs(value)>threshold? (value>0? threshold : -threshold) : value; }
double inline trunc(double value, double thresholdLow, double thresholdHigh) { return value<thresholdLow? thresholdLow : (value>thresholdHigh? thresholdHigh : value); }
int changeEndianType(unsigned int value) { return value>>24 | (value&0x00FF0000)>>8 | (value&0x0000FF00)<<8 | value<<24; }

// generates random value with specified distribution
class RandomGenerator
{
public:
    double rand();

protected:
    RandomGenerator(double x1, double x2, double distrMax) { this->x1 = x1; this->x2 = x2; this->distrMax = distrMax; }
    double virtual distribution(double arg) = 0;

private:
    double x1, x2, distrMax;
};

class Vector;

class Matrix
{
public:
    Matrix()                { data = 0; sX = 0; sY = 0; }
    Matrix(int sY, int sX)  { data = 0; this->sX = sX; this->sY = sY; resize(sX*sY); }
    template <class T> Matrix(vector<T> *vec)
                            { data = 0; setSizeYX(vec->size(), 1); for(int i=0; i<sY; i++) data[i] = (*vec)[i]; }
    Matrix(Matrix *mt)      { data = 0; setSizeYX(mt->sY, mt->sX); memcpy(data, mt->data, sizeof(double)*sX*sY); }
    Matrix(const Matrix &mt){ data = 0; setSizeYX(mt.sY, mt.sX);   memcpy(data, mt.data,  sizeof(double)*sX*sY); }
    Matrix &operator=(const Matrix &mt) { if(this != &mt) { sX = mt.sX; sY = mt.sY; resize(sX*sY); memcpy(data, mt.data, sizeof(double)*sX*sY); } return *this; }
    ~Matrix()               { free(data); }

    int sizeX() const  { return sX; }
    int sizeY() const  { return sY; }
    virtual void setSizeX(int sX)          { if(data!=0 && this->sX==sX) return; this->sX = sX; resize(sX*sY); }
    void setSizeY(int sY)                  { if(data!=0 && this->sY==sY) return; this->sY = sY; resize(sX*sY); }
    virtual void setSizeYX(int sY, int sX) { if(data!=0 && this->sY==sY && this->sX==sX) return; this->sY = sY;  this->sX = sX; resize(sX*sY); }

    void init(double val)   { int size = sX*sY; for(int i=0; i<size; i++) data[i] = val; }
    virtual void assign(Matrix *mt) { setSizeYX(mt->sY, mt->sX); memcpy(data, mt->data, sizeof(double)*sX*sY); }
    
    void fillRand(double from, double to) { for(int i=0; i<sX*sY; i++) data[i] = rand(from, to); }
    void fillBipolarRand()	{ for(int i=0; i<sX*sY; i++) data[i] = rand(-1,1)>0? 1 : -1; }
    void setSign(Matrix *in);
    void setAbs(Matrix *in);
    void setBipolarNoise(Matrix *in, int hamming);
    void truncate(Matrix *out, double threshold);

    double& el(int i, int j);
    double *getData() const	{ return data; }

    double maxElement(int *yPos = 0, int *xPos = 0);
    double minElement(int *yPos = 0, int *xPos = 0);
    double sqdist(Matrix *mt);
    double sproduct(Matrix *mtr);
    double sqnorm();
    void append(Matrix *mt);
    void appendTransposed(Vector *vec);
    void copyToRow(Vector *vec, int rowIndex);
    void setFromRow(Matrix *in, int rowIndex);
    void setFromColumn(Matrix *in, int columnIndex);
    Matrix& transpose();
    void plus(Matrix *in, Matrix *out);
    void minus(Matrix *in, Matrix *out);
    void plusVVT(Vector *vec1, Vector *vec2, double coeff = 1);
    void mult(double value, Matrix *out);
    void multCW(Matrix *in, Matrix *out);
    void mult(Matrix *in, Matrix *out);
    void multTN(Matrix *in, Matrix *out, bool byItself = false);
    void multNT(Matrix *in, Matrix *out);
    void multTT(Matrix *in, Matrix *out);
    void multVec(Vector *in, Vector *out);
    double static pseudoInverseRecalc(bool forceDep, Matrix *mt, Vector *row, Matrix * mtp, bool *isIndep = 0);
    int pseudoInverse(Matrix *arg);
    bool solveAxb(Vector *xvec, Vector *bvec, double *singDegree = 0, bool keepThis = true);
    bool solveOverdeterminedAxb(Vector *xvec, Vector *bvec, double *singDegree = 0);
    bool expandProjectiveMatrix(Vector *vec, double *diff = 0, double alpha = 1);
    void expandMMT(Vector *vec);

    double getDifference(Vector *vec);
    bool binaryEquals(Matrix *mtr);
    int hammingDistance(Matrix *mtr);
    double getSimilarity(Matrix *mtr);

    void save(outBinStream &str) { str << sY << sX; str.write((char *)data, sY*sX*sizeof(double)); }
    void load(inBinStream &str) { str >> sY >> sX; setSizeYX(sY, sX); str.read((char *)data, sY*sX*sizeof(double)); }

protected:
    void resize(int size);
    double *data;

private:
    int sX, sY;
};

void assertEqualSize(Matrix *mtr1, Matrix *mtr2, string msg = "")
{
    assertion(mtr1->sizeX()==mtr2->sizeX() && mtr1->sizeY()==mtr2->sizeY(), msg + " matrices must be of the same size");
}

void assertNonzeroSize(Matrix *mtr, string msg = "")
{
    assertion(mtr->sizeX()*mtr->sizeX(), msg + " the matrix must be of nonzero size");
}

// column-type vector stored in (sizeY x 1) matrix
class Vector : public Matrix
{
public:
    Vector() : Matrix()                  {}
    Vector(int size) : Matrix(size, 1)   {}
    template <class T> Vector(vector<T> *vec) : Matrix(vec) {}
    Vector(Vector *vec) : Matrix(vec)    {}
    int size() const                     { return sizeY(); }
    void setSize(int _size)              { Matrix::setSizeYX(_size, 1); }
    void setSizeX(int sX)                { assertion(sX==1, "[Vector::setSizeX] an attempt to set sizeX!=1"); }
    void setSizeYX(int sY, int sX)       { assertion(sX==1, "[Vector::setSizeYX] an attempt to set sizeX!=1"); Matrix::setSizeYX(sY, 1); }
    void assign(Matrix *mt)              { assertion(mt->sizeX()==1, "[Vector::assign] an attempt to assign non vector"); Matrix::assign(mt); }
    void assign(int _size, double val)   { setSize(_size); init(val); }
    void push_back(double val)           { int _sY = sizeY(); setSize(_sY+1); getData()[_sY] = val; }
    double &operator[](int i);
    void setFromRow(Matrix *in, int rowIndex);
    int maxElementIndex()                { int res; maxElement(&res, 0); return res; }
    int minElementIndex()                { int res; minElement(&res, 0); return res; }
};


// stores binary vector of {-1, 1} components
// imposes total ordering
class BipolarVector : public Vector
{
public:

    BipolarVector(int size) : Vector(size) {}

    friend bool operator<(const BipolarVector& x, const BipolarVector& y) // return ( x < y )
    {
        assertion(x.size()==y.size(), "[BipolarVector] operator\"< \"size mismatch");
        int sz = x.size();
        double *xBuf = x.getData(), *yBuf = y.getData();
        for(int i=0; i<sz; i++)
        {
            if(xBuf[i] < yBuf[i]-0.5)
                return true;
            if(yBuf[i] < xBuf[i]-0.5)
                return false;
        }
        return false;
    }
};


// element(i, j) = cov(Vi, Vj)/SD(Vi)SD(Vj)
// cov(Vi, Vj) = M( (Vi - M(Vi))(Vj - M(Vj)) )
class StatCounter;
class CorrelationMatrix : public Matrix
{
public:
    CorrelationMatrix() { numItems = 0; }
    void addItem(Vector *vec);
    void init_corr()         { init(0); numItems = 0; }

private:
    int numItems;
    vector<StatCounter> counters;
    Matrix mmt;
};

class StatCounter
{
public:
    StatCounter()               { name = ""; init(); }
    StatCounter(string name)    { this->name = name; init(); }
    void init()                 { minLabel = maxLabel = ""; count = 0; sum = sumsq = min = max = 0; }
    void addValue(double value);
    void addValue(double value, string label);
    int getCount()              { return count; }
    double getAverage()         { checkCount(); return sum/count; }
    double getSD()              { checkCount(); double val = sumsq/count - sum*sum/count/count; return val>0? sqrt(val) : 0; }
    double getDisp()            { checkCount(); double val = sumsq/count - sum*sum/count/count; return val; }
    double getMin()             { checkCount(); return min; }
    double getMax()             { checkCount(); return max; }
    string getMinLabel()        { checkCount(); return minLabel; }
    string getMaxLabel()        { checkCount(); return maxLabel; }
    void getData(Properties *prop, char fields = SC_ALL);
    string toString();

protected:
    void checkCount()           { assertion(count, "Address to [StatCounter] in initial state"); }
    string name;

private:
    string minLabel, maxLabel;
    int count;
    double sum;
    double sumsq;
    double min, max;
};

// Provides probability and probability's dencity distributions
class AdvancedStatCounter : public StatCounter
{
public:
    AdvancedStatCounter(double from, double to, int granularity, string name = ""); // real-valued
    AdvancedStatCounter(int from, int to, string name = "");                        // discrete
    void getProbabilityDensity(Vector *func);
    void getProbabilityDistribution(Vector *func);
    void getScale(Vector *scale);
    void init()                 { counter.assign(counter.size(), 0); StatCounter::init(); }
    void addValue(double value);
    string toString();

private:
    double from, to;
    vector<int> counter;
    bool isDiscrete;
};

// statistics for each field (entry) of properties
class PropertyStatCounter
{
public:
    //PropertyStatCounter() {}
    void addValue(Properties *prop);
    void addValue(string name, double value) { counters[name].addValue(value); }
    void init() { counters.clear(); }
    void getStatistics(Properties *prop, char fields = SC_ALL);

protected:
    map<string, StatCounter> counters;
};

# endif /* _MATH_AUX_H_ */
