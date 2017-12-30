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

# ifndef _DATA_H_
# define _DATA_H_

# include "math_aux.h"

enum DataType { dtUChar = 0, dtShortInt = 1, dtInt = 2, dtFloat = 3 };

void convertNNAtoVVC(string srcFileName, string destFileName);


// interface class
class Preprocessor
{
public:
    virtual void preprocess(Vector *vec) = 0;
    virtual string getDescription() = 0;
};


// does nothing
class IdenticalPreprocessor : public Preprocessor
{
public:
    void preprocess(Vector *vec) {}
    string getDescription() { return "IdenticalPreprocessor"; }
};


// threshold normalization: out = in<threshold? minVal : maxVal
class ThresholdPreprocessor : public Preprocessor
{
public:
    ThresholdPreprocessor(double th, double min, double max) { threshold = th; minVal = min; maxVal = max; }
    void preprocess(Vector *vec) { int size = vec->size(); double *data = vec->getData(); for(int i=0; i<size; i++) data[i] = data[i]<threshold? minVal : maxVal; }
    string getDescription()
    {
        return "ThresholdPreprocessor: th = " + toString(threshold, 4) +
               ", minVal = " + toString(minVal, 4) + ", maxVal = " + toString(maxVal, 4);
    }

private:
    double threshold, minVal, maxVal;
};


// basic interface class for data storage that contains pairs of input/output vectors
// derived class must set _inDim, _outDim, check in/out dims and call "preprocess"
class IOData
{
public:
    virtual ~IOData()                   { delete inPreproc; delete outPreproc; }
    int inDim()                         { assertion(isOpen(), "[IOData::inDim] not opened");  return _inDim; }
    int outDim()                        { assertion(isOpen(), "[IOData::outDim] not opened"); return _outDim; }
    virtual int count() = 0;
    virtual bool hasNext() = 0;
    virtual void getNext(Vector *in, Vector *out) = 0;
    virtual void setToBeginning() = 0;
    virtual void setToRandomPos() = 0;
    virtual bool isOpen() { return true; }
    virtual void open()   {}
    virtual void close()  {}

    void setInPreproc(Preprocessor *preproc) { delete inPreproc; inPreproc = preproc; }
    void setOutPreproc(Preprocessor *preproc) { delete outPreproc; outPreproc = preproc; }
    Preprocessor* getInPreproc() { return inPreproc; }
    Preprocessor* getOutPreproc() { return outPreproc; }

    void getInputStatistics(Vector *ave, Vector *sd, int *dataCount = 0);
    void saveToTextFile(string fileName);
    virtual string getDescription()
    {
        return "# Input: Dim = " + toString(_inDim) + " Preprocessor: " + inPreproc->getDescription() +
            "\n# Output: Dim = " + toString(_outDim) + " Preprocessor: " + outPreproc->getDescription() +
            "\n# DataCount = " + toString(count());
    };

protected:
    int _inDim, _outDim;
    IOData()                            { _inDim = _outDim = 0; inPreproc = new IdenticalPreprocessor(); outPreproc = new IdenticalPreprocessor(); }
    void preprocess(Vector *in, Vector *out) { inPreproc->preprocess(in); outPreproc->preprocess(out); }

private:
    Preprocessor *inPreproc, *outPreproc;
};



// data container that is supposed to keep all its content in RAM
// allows random access and shuffling of data order
struct IOPair
{
    IOPair(Vector *in, Vector *out) { this->in = in, this->out = out; }
    Vector *in, *out;
};

// child class must set:
//
// _inDim, _outDim and "data" buffer
//
// must implement:
//
// string getDescription()
class RAMData : public IOData
{
public:
    ~RAMData()                          { while(data.size()>0) { delete data.back().in; delete data.back().out; data.pop_back(); } }
    // IOData interface
    int count()                         { return data.size(); }
    bool hasNext()                      { return getIndex<(int)data.size(); }
    void getNext(Vector *in, Vector *out);
    void setToBeginning()               { getIndex = 0; }
    void setToRandomPos()               { getIndex = rand(data.size()); }
    // extras
    void shuffle()                      { random_shuffle(data.begin(), data.end()); }
    void get(Vector *in, Vector *out, int index);
    void add(Vector *in, Vector *out);

protected:
    RAMData() : IOData() { getIndex = 0; }
    vector<IOPair> data;
    int getIndex;
};


// Bipolar vectors (with -1/+1 components)
// components are random, independent and equiprobable
class RandomBipolarRAMData : public RAMData
{
public:
    RandomBipolarRAMData(int inDim, int outDim, int count, bool isAutoassociative = false);
    string getDescription() { return "# RandomBipolarRAMData, isAutoassociative = " + toStringB(isAutoassociative)
        + "\n" + IOData::getDescription(); }
private:
    bool isAutoassociative;
};


// RAMData implementation which can be read from text file
class TextRAMData : public RAMData
{
public:
    TextRAMData(string fileName);
    virtual string getDescription();

private:
    string fileName;
};

// RAMData implementation which can be read from VVC file
class VvcRAMData : public RAMData
{
public:
    VvcRAMData(string fileName, int count = -1);
    virtual string getDescription();

private:
    string fileName;
};


// base class for HD binary data storage, provides buffer for reading
class BufferedData : public IOData
{
public:
    BufferedData(string fileName, int bufferSize) : IOData() // buffer size in bytes
                                        { this->fileName = fileName; this->bufferSize = bufferSize; buffer = 0; }
    virtual ~BufferedData()             { stream.close(); /*delete[] buffer;*/ }
    // IOData interface
    virtual bool isOpen()               { return buffer!=0; }
    virtual void open();
    virtual void close()                { stream.close(); /*delete[] buffer;*/ buffer = 0; }
    // extras
    virtual void setToRandomPos() = 0;
    virtual string getDescription();

protected:
    ifstream stream;
    string fileName;

private:
    char *buffer;
    int bufferSize;
};


// base class for HD binary inline data storage, provides reading and positioning
class InlineData : public BufferedData
{
public:
    InlineData(string fileName, int bufferSize) : BufferedData(fileName, bufferSize) {}
    virtual ~InlineData()               { close(); }
    // IOData interface
    int count()                         { assertion(isOpen(), "[InlineData::count] is not opened"); return recordCount; }
    bool hasNext()                      { assertion(isOpen(), "[InlineData::hasNext] is not opened"); return recordIndex < recordCount; }
    void getNext(Vector *in, Vector *out);
    void setToBeginning()               { assertion(isOpen(), "[InlineData::setToBeginning] is not opened"); recordIndex = 0; stream.seekg(headerSize, ios::beg); }
    void setToRandomPos()               { assertion(isOpen(), "[InlineData::setToRandomPos] is not opened"); recordIndex = rand(recordCount); stream.seekg(headerSize + recordIndex*recordLength, ios::beg); }
    virtual void open();
    void close()                        { delete[] record; BufferedData::close(); }
    // extras
    int getRecordIndex()                { assertion(isOpen(), "[InlineData::getRecordIndex] is not opened"); return recordIndex; }

protected:
    DataType dataType;
    int headerSize, ioRecordLength, recordLength, recordIndex, recordCount;
    virtual void setFields() = 0;       // child class must set _inDim, _outDim, dataType, headerSize, ioRecordLength, recordLength, recordCount

private:
    void *record;
};


// VVC format data -------------------------------------------------------------

// header of VVC file format
struct VVCHeader
{
    char Pass_Word[4];                  // Keyword
    int len_file;                       // File length in bytes
    int type_data;                      // Data type  0 - short int
                                        //            1 - int
                                        //            2 - float
                                        //            3 - bit mask for RSC and PSC
                                        //            4 - char
    int quantity_inputs;                // Number of inputs
    int quantity_outputs;               // Number of outputs
    int quantity_outputs_class;         // Number of output classes
                                        // 0 - interpolator; 1 - Classifier
    int len_record;                     // Record length, including additional fields
    char  description_outputs[25][20];  // Description of network outputs
    char  description_data[496];        // Description of data
};

class VvcData : public InlineData
{
public:
    VvcData(string fileName, int bufferSize) : InlineData(fileName, bufferSize) {}

protected:
    void setFields();
};


// MNIST format data -----------------------------------------------------------

struct MnistInHeader
{
    int magicNumber;
    int numImages;
    int rows;
    int columns;
};

struct MnistOutHeader
{
    int magicNumber;
    int numItems;
};

class MnistData : public InlineData
{
public:
    MnistData(string fileName, int bufferSize) : InlineData(fileName, bufferSize)
    {
        setInPreproc(new ThresholdPreprocessor(25, -1, +1));
    }

protected:
    void setFields();
};


// Random Bipolar Data ---------------------------------------------------------

// Threading data, generated 'on fly', no replay possible
class RandomBipolarThreadingData : public IOData
{
public:
    RandomBipolarThreadingData(int inDim, int outDim, int count)
    {
        initRNG();
        _inDim = inDim;
        _outDim = outDim;
        _count = count;
        index = 0;
    }

    int count() { return _count; }
    bool hasNext() { return index < _count; }
    void getNext(Vector *in, Vector *out)
    {
        assertion(index>=0 && index<_count, "[RandomBipolarThreadingData::getNext] no next");
        assertion(in->size()==_inDim && out->size()==_outDim, "[RandomBipolarThreadingData::getNext] wronf in/out sizes");
        in->fillBipolarRand();
        out->fillBipolarRand();
        preprocess(in, out);
        index++;
    }
    void setToBeginning() { /*index = 0;*/ assertion(false, "[RandomBipolarThreadingData::setToBeginning] prohibited call"); }
    void setToRandomPos() { /*index = randon(_count);*/ assertion(false, "[RandomBipolarThreadingData::setToRandomPos] prohibited call"); }

    string getDescription() { return "# RandomBipolarThreadingData\n" + IOData::getDescription(); }

private:
    int _count, index;
};

# endif /* _DATA_H_ */

