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

# include "Data.h"

# define lineLength 32*1024

void convertNNAtoVVC(string srcFileName, string destFileName)
{
    assertion(srcFileName.size()>=5, "[convertNNAtoVVC] must be *.nna file");
    assertion(srcFileName.substr(srcFileName.size()-4, 4).compare(".nna")==0, "[convertNNAtoVVC] must be *.nna file");

    char line[lineLength]; // suppose that's enough

    ifstream nnaFile(srcFileName.c_str());
    assertion(nnaFile.is_open(), string("[convertNNAtoVVC] cant open file: ") + srcFileName);
    int bufferSize = 1024*1024;
    char *buffer = new char[bufferSize];
    nnaFile.rdbuf()->pubsetbuf(buffer, bufferSize);

    ofstream vvcFile((destFileName.substr(0, destFileName.size()-4)+".vvc").c_str(), ios::binary | ios::trunc);
    VVCHeader header;
    vvcFile.write((char*)&header, sizeof(VVCHeader));

    float fval;
    int index = 0, recordsProcessed = 0, recordDim = -1;
    while(nnaFile.getline(line, lineLength))
    {
        if(line[0] == '!' || line[0] == '\n') // skip comment and new lines
            continue;

        if(line[0] != '&') // new record
        {
            if(recordsProcessed == 0) ;     // do nothing
            else if(recordsProcessed == 1)  // first record, set dim
                recordDim = index;
            else                            // check that dim is the same
                assertion(recordDim == index, "[convertNNAtoVVC] inconsistent record dim");
            printProgress("[convertNNAtoVVC] processing records", ++recordsProcessed, 100);
            index = 0;
        }

        // parse line
        int length = stringLength(line);
        int pos = line[0]=='&'? 1 : 0;
        while(pos < length)
        {
            while(line[pos] == ' ')
                pos++;
            fval = atof(&line[pos]);
            vvcFile.write((char*)&fval, sizeof(float));
            index++;
            while(line[pos]!=' ' && line[pos]!=0)
                pos++;
        }
    }
    nnaFile.close();

    assertion(recordsProcessed>0, "[convertNNAtoVVC] file must have at least two records");

    header.len_file = sizeof(VVCHeader) + recordsProcessed*recordDim*sizeof(float);
    header.type_data = 2; // float
    header.quantity_inputs = recordDim-1;
    header.quantity_outputs = 1;
    header.quantity_outputs_class = 2;
    header.len_record = recordDim*sizeof(float);

    assertion(header.len_file == (int)vvcFile.tellp(), "[convertNNAtoVVC] internal error");
    vvcFile.seekp(0, ios::beg);
    vvcFile.write((char*)&header, sizeof(VVCHeader));
    vvcFile.close();
    cout << endl << "[convertNNAtoVVC] conversion done, " << recordsProcessed << " records processed";
}


//------------------------------- IOData ---------------------------------------


// calculate average and standard deviation
void IOData::getInputStatistics(Vector *ave, Vector *sd, int *dataCount)
{
    bool closeData = false;
    if(!isOpen())
    {
        open();
        closeData = true;
    }

    vector<StatCounter> counters(inDim());
    Vector in(inDim()), out(outDim());
    double *inBuf = in.getData();

    int processed = 0;
    setToBeginning();
    while(hasNext())
    {
        getNext(&in, &out);
        for(int i=0; i<_inDim; i++)
            counters[i].addValue(inBuf[i]);
        printProgress("[IOData::getStatistics]", ++processed, 100, count());
    }
    cout << endl;

    ave->setSize(inDim());
    sd->setSize(inDim());
    for(int i=0; i<inDim(); i++)
    {
        (*ave)[i] = counters[i].getAverage();
        (*sd)[i]  = counters[i].getSD();
    }

    if(dataCount)
        *dataCount = counters[0].getCount();

    if(closeData)
        close();
}


// saves as two consecutive matrices (inputs and outputs), whith data vectors as rows
void IOData::saveToTextFile(string fileName)
{
    bool closeData = false;
    if(!isOpen())
    {
        open();
        closeData = true;
    }

    int dataCount = count(), datumIndex = 0;
    Vector in(inDim()), out(outDim());
    Matrix inM(dataCount, inDim()), outM(dataCount, outDim());
    setToBeginning();
    while(hasNext())
    {
        getNext(&in, &out);
        inM.copyToRow(&in,   datumIndex);
        outM.copyToRow(&out, datumIndex);
        datumIndex++;
        //printProgress("[IOData::saveToTextFile] cashing", datumIndex, 100, dataCount);
    }
    //cout << endl;

    saveString(getDescription() + "\n", fileName, true); // overwrite
    saveMatrix(&inM,  fileName);
    saveMatrix(&outM, fileName);

    if(closeData)
        close();
}


//---------------------------- RAMData -----------------------------------------


// reads the next input/output pair
void RAMData::getNext(Vector *in, Vector *out)
{
    assertion(getIndex<count(), "[RAMData::getNext] end of data reached");
    get(in, out, getIndex++);
}


// reads input/output pair at "index" position
// normalizes output if needed
void RAMData::get(Vector *in, Vector *out, int index)
{
    assertion(index>=0 && index<count(), "[RAMData::get] index out of range");
    assertion(in->size()==_inDim && out->size()==_outDim, "[RAMData::get] wrong in/out sizes");

    in->assign(data[index].in);
    out->assign(data[index].out);

    preprocess(in, out);
}


// adds in-out pair to the end of data set
void RAMData::add(Vector *in, Vector *out)
{
    assertion(in->size()==_inDim && out->size()==_outDim, "[RAMData::add] wrong in/out sizes");

    data.push_back(IOPair(new Vector(*in), new Vector(*out)));
}


//--------------------------- RandomBipolarRAMData -----------------------------

RandomBipolarRAMData::RandomBipolarRAMData(int inDim, int outDim, int count, bool isAutoassociative)
{
    initRNG();

    assertion(!isAutoassociative || inDim==outDim, "[RandomBipolarRAMData::RandomBipolarRAMData] isAutoassociative && inDim!=outDim");

    this->isAutoassociative = isAutoassociative;
    _inDim = inDim;
    _outDim = outDim;

    for(int d=0; d<count; d++)
    {
        Vector *in = new Vector(_inDim);
        Vector *out = new Vector(_outDim);
        in->fillBipolarRand();
        isAutoassociative? out->assign(in) : out->fillBipolarRand();
        data.push_back(IOPair(in, out));
    }
}


//---------------------------- TextRAMData -------------------------------------

// reads Input data from text file that stores two consecutive matrices (inputs and outputs),
// whith data vectors as columns
TextRAMData::TextRAMData(string fileName)
{
    this->fileName = fileName;

    Matrix inMatrix, outMatrix;
    readMatrix(&inMatrix, &outMatrix, fileName);
    assertion(inMatrix.sizeY()==outMatrix.sizeY(),
        "[TextRAMData::TextRAMData(string fileName)] different number of rows in in/out matrices");

    _inDim = inMatrix.sizeX();
    _outDim = outMatrix.sizeX();

    int dataCount = inMatrix.sizeY();
    for(int r=0; r<dataCount; r++)
    {
        Vector *in = new Vector(inMatrix.sizeX());
        Vector *out = new Vector(outMatrix.sizeX());
        in->setFromRow( &inMatrix,  r);
        out->setFromRow(&outMatrix, r);
        data.push_back(IOPair(in, out));
        //printProgress("[TextRAMData::TextRAMData] inserting", r, 100, dataCount);
    }
    //cout << endl;
}

string TextRAMData::getDescription()
{
    return "# TextRAMData, file: " + fileName + "\n" + IOData::getDescription();
}


//---------------------------- VvcRAMData --------------------------------------


// reads from VVC file
// reads "count" records, if(count <= 0) => reads all file
VvcRAMData::VvcRAMData(string fileName, int count)
{
    this->fileName = fileName;

    VvcData vvcData(fileName, 1024*1024);
    vvcData.open();

    _inDim = vvcData.inDim();
    _outDim = vvcData.outDim();

    while(vvcData.hasNext())
    {
        IOPair pair(new Vector(_inDim), new Vector(_outDim));
        vvcData.getNext(pair.in, pair.out);
        data.push_back(pair);

        if(count > 0 && data.size() >= count)
            break;
    }

    assertion(getIndex==0, "[VvcRAMData::VvcRAMData] default parent constructor not working ?!!");
}

string VvcRAMData::getDescription()
{
    return "# VvcRAMData, file: " + fileName + "\n" + IOData::getDescription();
}


//------------------------------ BufferedData ----------------------------------


// opens file stream and appends buffer to it
void BufferedData::open()
{
    assertion(!isOpen(), "[BufferedData::open] already opened for reading");

    stream.open(fileName.c_str(), ios::binary);
    assertion(stream.is_open(), "[BufferedData::open] cant open file: " + fileName);

    buffer = new char[bufferSize];
    stream.rdbuf()->pubsetbuf(buffer, bufferSize);
}


string BufferedData::getDescription()
{
    assertion(isOpen(), "[BufferedData::getDescription] not opened");
    return "# BufferedData, file: " + fileName + "\n" + IOData::getDescription();
}


//------------------------------- InlineData -----------------------------------


// opens stream and sets data parameters
void InlineData::open()
{
    BufferedData::open();

    setFields();

    if(dataType == dtShortInt)        // short int
        record = new short int[_inDim + _outDim];
    else if(dataType == dtInt)        // int
        record = new int[_inDim + _outDim];
    else if(dataType == dtFloat)      // float
        record = new float[_inDim + _outDim];
    else if(dataType == dtUChar)      // unsigned char
        record = new unsigned char[_inDim + _outDim];

    recordIndex = 0;
}


// reads current record
void InlineData::getNext(Vector *in, Vector *out)
{
    assertion(in->size()==_inDim && out->size()==_outDim, "[InlineData::getNext] wrong in/out sizes");
    assertion(recordIndex<recordCount, getDescription()+ " [InlineData::getNext] doesnt have next");

    stream.read((char*)record, ioRecordLength);
    if(!stream.good())
        assertion(false, getDescription() + " [InlineData::getNext] read problem (unexpected end of file)");
    stream.seekg(recordLength-ioRecordLength, ios::cur);

    double *inBuf = in->getData(), *outBuf = out->getData();

    if(dataType == dtShortInt)
    {
        short int *recordBuf = (short int*)record;
        for(int i=0; i<_inDim; i++)
            *inBuf++ = *recordBuf++;
        for(int i=0; i<_outDim; i++)
            *outBuf++ = *recordBuf++;
    }
    else if(dataType == dtInt)
    {
        int *recordBuf = (int*)record;
        for(int i=0; i<_inDim; i++)
            *inBuf++ = *recordBuf++;
        for(int i=0; i<_outDim; i++)
            *outBuf++ = *recordBuf++;
    }
    else if(dataType == dtFloat)
    {
        float *recordBuf = (float*)record;
        for(int i=0; i<_inDim; i++)
            *inBuf++ = *recordBuf++;
        for(int i=0; i<_outDim; i++)
            *outBuf++ = *recordBuf++;
    }
    else if(dataType == dtUChar)
    {
        unsigned char *recordBuf = (unsigned char*)record;
        for(int i=0; i<_inDim; i++)
            *inBuf++ = *recordBuf++;
        for(int i=0; i<_outDim; i++)
            *outBuf++ = *recordBuf++;
    }

    recordIndex++;
    preprocess(in, out);
}


//------------------------------- VvcData --------------------------------------


// InlineData: child class must set _inDim, _outDim, dataType, headerSize, ioRecordLength, recordLength, recordCount
void VvcData::setFields()
{
    assertion(fileName.size()>=5, getDescription() + " [VvcData::setFields] must be *.vvc file");
    assertion(fileName.substr(fileName.size()-4, 4).compare(".vvc")==0, "[VvcData::setFields] must be *.vvc file");


    VVCHeader header;
    stream.read((char*)&header, sizeof(VVCHeader));
    assertion(stream.good(), getDescription() + " [VvcData::setFields] cant read VVC header");

    _inDim = header.quantity_inputs;
    _outDim = header.quantity_outputs;
    headerSize = sizeof(VVCHeader);
    recordLength = header.len_record;
    recordCount = (header.len_file-sizeof(VVCHeader))/header.len_record;

    if(header.type_data == 0)       // short int
    {
        dataType = dtShortInt;
        ioRecordLength = (_inDim+_outDim)*sizeof(short int);
    }
    else if(header.type_data == 1)  // int
    {
        dataType = dtInt;
        ioRecordLength = (_inDim+_outDim)*sizeof(int);
    }
    else if(header.type_data == 2)  // float
    {
        dataType = dtFloat;
        ioRecordLength = (_inDim+_outDim)*sizeof(float);
    }
    else if(header.type_data == 4)  // unsigned char
    {
        dataType = dtUChar;
        ioRecordLength = (_inDim+_outDim)*sizeof(char);
    }
    else
        throw GenException(getDescription() + " [VvcBufferedData::open] unsupported data type. Use short int/int/float/char (0/1/2/4)");
}


//---------------------------------- MnistData ---------------------------------


// InlineData: child class must set _inDim, _outDim, dataType, headerSize, ioRacordLength, recordLength, recordCount
void MnistData::setFields()
{
    MnistInHeader header;
    stream.read((char*)&header, sizeof(header));
    assertion(stream.good(), getDescription() + " [MnistData::setFields] cant read MnistInHeader");

    header.magicNumber = changeEndianType(header.magicNumber);
    header.numImages = changeEndianType(header.numImages);
    header.rows = changeEndianType(header.rows);
    header.columns = changeEndianType(header.columns);

    assertion(header.magicNumber==0x00000803, "[MnistData::setFields] wrong signature");

    _inDim = header.rows*header.columns;
    _outDim = 0;
    dataType = dtUChar;
    headerSize = sizeof(header);
    ioRecordLength = _inDim;
    recordLength = _inDim;
    recordCount = header.numImages;
}









