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

# include "math_aux.h"


// Debugging -------------------------------------------------------------------
//# define DISABLE_RANDOMIZE
//# define MATH_DEBUG
//# define MATRIX_MAX_SIZE 1000000
//------------------------------------------------------------------------------


void initRNG()
{
    # ifndef DISABLE_RANDOMIZE
    srand((unsigned int)getTime_ms());
    # endif
}


// index of el if such el is in vec, -1 otherwise;
// vector must be sorted in ascending order
int indexOfSorted(vector<int> *vec, int el)
{
	vector<int>::iterator iter = lower_bound(vec->begin(),vec->end(),el);
	if(iter!=vec->end() && *iter==el)
		return iter-vec->begin();
	else
		return -1;
}


double RandomGenerator::rand()
{
    double (RandomGenerator::*distr)(double) = &RandomGenerator::distribution;
    double x, y, f;
    while(true)
    {
        x = ::rand(x1, x2);
        y = ::rand(0, distrMax);
        
        f = (this->*distr)(x);
        assertion(f >= 0, "[RandomGenerator::rand] distr(x)<0");

        if(y < f)
            return x;
    }
}


//------------------------------- Matrix ---------------------------------------

double & Matrix::el(int i, int j)
{
    # ifdef MATH_DEBUG
    assertion(i>=0 && i<sY && j>=0 && j<sX, "[Matrix::el] index out of range");
    # endif
    return data[i*sX+j];
}

void Matrix::resize(int size)
{
    //if(data==0 && size==0) // to avoid CodeGuard stopping here
    //    return;
    # ifdef MATH_DEBUG
    assertion(size>=0 && size<MATRIX_MAX_SIZE, "[Matrix::resize] size out of range");
    # endif
    data = (double*)realloc(data, sizeof(double)*size);
}


// this = sign(in) (with {-1, 1} codomain)
void Matrix::setSign(Matrix *in)
{
    assertion(in->sY==sY && in->sX==sX, "[Matrix::setSign] diff sizes");
    vecsign(in->data, data, sX*sY);
}

// this = abs(in)
void Matrix::setAbs(Matrix *in)
{
    assertion(in->sY==sY && in->sX==sX, "[Matrix::setAbs] diff sizes");
    vecabs(in->data, data, sX*sY);
}


// copies in to this with exactly hamming inverted (*-1) values at random positions
// designed for use WITH BINARY VECTORS ONLY
// NOTE: "in" and "this" must be different objects !!!
void Matrix::setBipolarNoise(Matrix *in, int hamming)
{
    assertion(in->sizeY()==sY && in->sizeX()==sX, "[Matrix::setNoise] diff sizes");
    assertion(hamming>=0 && hamming<=sX*sY, "[Matrix::setNoise] wrong hamming value");

    assign(in);
    double *inbuf = in->getData(), *thisbuf = data;
    int size = sX*sY;
	for(int h=0; h<hamming; h++)
	{
		while(true)
		{
			int index = rand()*size/RAND_MAX;
			if(index == size)
				index--;
			if(fabs(inbuf[index]-thisbuf[index]) < 0.5)
			{
				thisbuf[index] *= -1;
				break;
			}
		}
	}
}


// out[i] = fabs(this[i])>threshold? (this[i]>0? threshold : -threshold) : this[i]
void Matrix::truncate(Matrix *out, double threshold)
{
    assertion(sX==out->sX && sY==out->sY, "[Matrix::truncate] diff sizes");
    assertion(threshold >= 0, "[Matrix::truncate] threshold < 0");
    ::truncate(data, out->data, sX*sY, threshold); 
}


// max element and its position
double Matrix::maxElement(int *yPos, int *xPos)
{
    assertion(sX>0 && sY>0, "[Matrix::maxElement] zero size matrix");

    int maxElIndex;
    double res = ::maxElement(data, sX*sY, &maxElIndex);

    if(yPos)
        *yPos = maxElIndex/sX;
    if(xPos)
        *xPos = maxElIndex%sX;

    return res;
}


// min element and its position
double Matrix::minElement(int *yPos, int *xPos)
{
    assertion(sX>0 && sY>0, "[Matrix::minElement] zero size matrix");

    int minElIndex;
    double res = ::minElement(data, sX*sY, &minElIndex);

    if(yPos)
        *yPos = minElIndex/sX;
    if(xPos)
        *xPos = minElIndex%sX;

    return res;
}


// sum (this(ij) - mt(ij))^2
double Matrix::sqdist(Matrix *mt)
{
    assertion(sX==mt->sX && sY==mt->sY, "[Matrix::sqdist] diff sizes");
    return ::sqdist(data, mt->data, sX*sY);
}

// sum ( this(ij) * mt(ij) )
double Matrix::sproduct(Matrix *mtr)
{
    assertion(sX==mtr->sX && sY==mtr->sY, "[Matrix::sproduct] size mismatch");
    return ::sproduct(data, mtr->data, sX*sY);
}

// sum (el^2)
double Matrix::sqnorm()
{
    return ::sqnorm(data, sX*sY);
}

// appends mt to the bottom of this
void Matrix::append(Matrix *mt)
{
     assertion(sX == mt->sX, "[Matrix::append] diff sizes");

     setSizeY(sY+mt->sY);
     double *dest = &data[sX*(sY-mt->sY)];
     memcpy(dest, mt->data, mt->sX*mt->sY*sizeof(double));
}

// appends vec.Tr to the bottom of this
void Matrix::appendTransposed(Vector *vec)
{
     assertion(sX == vec->size(), "[Matrix::appendTransposed] diff sizes");

     setSizeY(sY + 1);
     double *dest = &data[sX * (sY-1)];
     memcpy(dest, vec->data, sX*sizeof(double));
}

// copies vec.Tr to the rowIndex-th row of this
void Matrix::copyToRow(Vector *vec, int rowIndex)
{
     assertion(sX == vec->size(), "[Matrix::copyToRow] diff sizes");
     assertion(rowIndex>=0 && rowIndex<sY, "[Matrix::copyToRow] rowIndex out of range");

     double *dest = &data[sX * rowIndex];
     memcpy(dest, vec->data, sX*sizeof(double));
}

// this = in(rowIndex, 0:in->sX-1)
void Matrix::setFromRow(Matrix *in, int rowIndex)
{
    assertion(rowIndex>=0 && rowIndex<in->sY, "[Matrix::setFromRow] wrong index");

    setSizeY(1);
    setSizeX(in->sX);
    double *src = &in->data[rowIndex*sX];
    memcpy(data, src, sX*sizeof(double));
}

// this = in(0:in->sY-1, columnIndex)
void Matrix::setFromColumn(Matrix *in, int columnIndex)
{
    assertion(columnIndex>=0 && columnIndex<in->sX, "[Matrix::setFromColumn] wrong index");

    setSizeY(in->sY);
    setSizeX(1);
    double *indata = &in->data[columnIndex];
    int insizeX = in->sX;
    for(int j=0; j<sY; j++)
    {
        data[j] = *indata;
        indata += insizeX;
    }
}

// this = this.Tr
Matrix& Matrix::transpose()
{
    swapValues<int>(&sX,&sY);
    if(sX==0 || sX==1 || sY==0 || sY==1)
        return *this;

    double *dataTr = (double*)malloc(sX*sY*sizeof(double)), *dtr = dataTr;
    for(int i=0; i<sY; i++)
        for(int j=0; j<sX; j++)
            *dtr++ = data[j*sY+i];
    free(data);
    data = dataTr;

    return *this;
}

// out = this + in
void Matrix::plus(Matrix *in, Matrix *out)
{
    assertion(in->sY==sY && in->sX==sX && out->sY==sY && out->sX==sX, "[Matrix::plus] diff sizes");
    vecplus(data, in->data, sX*sY, out->data);
}

// out = this - in
void Matrix::minus(Matrix *in, Matrix *out)
{
    assertion(in->sY==sY && in->sX==sX && out->sY==sY && out->sX==sX, "[Matrix::minus] diff sizes");
    vecminus(data, in->data, sX*sY, out->data);
}

// this += coeff * vec1 * vec2'
void Matrix::plusVVT(Vector *vec1, Vector *vec2, double coeff)
{
    assertion(vec1->size()==sY && vec2->size()==sX, "[Matrix::plusVVT] size mismatch");
    vecplus(vec1->getData(), vec2->getData(), vec1->size(), vec2->size(), data, coeff);
}

// out = this * value
void Matrix::mult(double value, Matrix *out)
{
    assertion(out->sY==sY && out->sX==sX, "[Matrix::mult] diff sizes");
    vecmult(data, value, sX*sY, out->data);
}

// out = this $ in
// where '$' stands for componentwise multiplication
void Matrix::multCW(Matrix *in, Matrix *out)
{
    assertion(in->sY==sY && in->sX==sX && out->sY==sY && out->sX==sX, "[Matrix::multCW] diff sizes");
    vecmultComponentwise(data, in->data, sX*sY, out->data);
}

// out = this*in
void Matrix::mult(Matrix *in, Matrix *out)
{
    assertion(in->sY==sX && in->sX==out->sX && sY==out->sY, "[Matrix::mult] diff sizes");
    multiplyNN(out->data, data, in->data, out->sX, out->sY, sX);
}

// out = this.Tr*in
void Matrix::multTN(Matrix *in, Matrix *out, bool byItself)
{
    assertion(in->sY==sY && in->sX==out->sX && sX==out->sY, "[Matrix::multTN] diff sizes");
    assertion(!byItself || in->sX==sX, "[Matrix::multTN] not itself");
    multiplyTN(out->data, data, in->data, out->sX, out->sY, sY, byItself);
}

// out = this*in.Tr
void Matrix::multNT(Matrix *in, Matrix *out)
{
    assertion(in->sX==sX && in->sY==out->sX && sY==out->sY, "[Matrix::multNT] diff sizes");
    multiplyNT(out->data, data, in->data, out->sX, out->sY, sX);
}

// out = this.Tr*in.Tr
void Matrix::multTT(Matrix *in, Matrix *out)
{
    assertion(in->sX==sY && in->sY==out->sX && sX==out->sY, "[Matrix::multTT] diff sizes");

    double sp, *datai, *dataj;
    for(int i=0; i<sX; i++)
    {
        for(int j=0; j<in->sY; j++)
        {
            datai = &data[i];
            dataj = &in->data[in->sX*j];
            sp = 0;
            for(int k=0; k<sY; k++)
            {
                sp += (*datai)*dataj[k];
                datai += sX;
            }
            out->el(i,j) = sp;
        }
    }
}

// out = [this,in] - vector product
void Matrix::multVec(Vector *in, Vector *out)
{
     assertion(sX==1 && sY==3 && in->size()==3 && out->size()==3, "[Matrix::multVec] wrong sizes");

     double *indata = in->getData(), *outdata = out->getData();
     outdata[0] = indata[2]*data[1] - data[2]*indata[1];
     outdata[1] = indata[0]*data[2] - data[0]*indata[2];
     outdata[2] = indata[1]*data[0] - data[1]*indata[0];
}

// recalcs mtp accordingly to row that will be added to mt
// mtp has to be equal to mt+ before function call
// if forceDep = true then row is added as dependent vector
// isIndep is set to proper value accordingly to the algorithm
// returns |(I - mtp.T*mt) * row|^2 / |row|^2 i.e. |ort(row)|^2 / |row|^2
// NOTE: row is a vector
double Matrix::pseudoInverseRecalc(bool forceDep, Matrix *mt, Vector *row, Matrix *mtp, bool *isIndep)
{
    assertion(mt->sX==mtp->sX && mt->sX==row->size() && mt->sY==mtp->sY, "[Matrix::pseudoInverseRecalc] wrong sizes");
    assertion(mt->sX != 0, "[Matrix::pseudoInverseRecalc] sizeX == 0");

    int i;
    int sX = mt->sX;
    int sY = mt->sY;
    double *rowdata = row->getData();

    // mtp = row / |row|^2
    if(sY == 0)
    {
		if(isIndep != 0)
			*isIndep = true;
        double sqnorm = row->sproduct(row);
        assertion(sqnorm > 0, "[Matrix::pseudoInverseRecalc] |row| = 0");
        mt->setSizeY(1);
        mtp->setSizeY(1);
        double *mtdata = mt->data, *mtpdata = mtp->data;
        for(i=0; i<sX; i++)
        {
            mtdata[i] = rowdata[i];
            mtpdata[i] = rowdata[i]/sqnorm;
        }
        return 1;
    }

    double sqrow = row->sproduct(row);
    assertion(sqrow > 0, "[Matrix::pseudoInverseRecalc] |row| = 0");

    // vecX = (I - mtp.T*mt) * row
    Matrix  vecX(sX,1), vecY(sY,1);
    mt->mult(row,&vecY);
    mtp->multTN(&vecY,&vecX);
    double *vecXdata = vecX.data, *vecYdata = vecY.data;
    for(i=0; i<sX; i++)
        vecXdata[i] = rowdata[i]-vecXdata[i];
    double sqnorm = vecX.sproduct(&vecX), diff = sqnorm/sqrow;

    // vecX = (I - mtp.T*mt) * row / |...|^2
    bool hasVecY = false;
    if(diff>PSEUDO_INVERSE_EPS && !forceDep)
    {
		if(isIndep != 0)
			*isIndep = true;
        for(i=0; i<sX; i++)
            vecXdata[i] /= sqnorm;
    }
    // vecX = mtp.T*mtp*row / (1 + |mtp*row|^2)
    else
    {
		if(isIndep != 0)
			*isIndep = false;
        hasVecY = true;
        mtp->mult(row,&vecY);
        double denom = vecY.sproduct(&vecY);
        denom += 1;
        mtp->multTN(&vecY,&vecX);
        for(i=0; i<sX; i++)
            vecXdata[i] /= denom;
    }

    // vecY = mtp*row
    if(!hasVecY)
        mtp->mult(row,&vecY);
    mt->setSizeY(sY+1);
    mtp->setSizeY(sY+1);
    double *mtpdata = mtp->data, *mtdata = mt->data+sY*sX;
    for(i=0; i<sY; i++)
        for(int j=0; j<sX; j++)
        {
            *mtpdata -= vecYdata[i]*vecXdata[j];
            mtpdata++;
        }
    for(i=0; i<sX; i++)
    {
        *mtdata = rowdata[i];
        *mtpdata = vecXdata[i];
        mtdata++;
        mtpdata++;
    }

	return diff;
}

// this = (arg+).Tr
// returns row-rank of arg
int Matrix::pseudoInverse(Matrix *arg)
{
    assertion(arg->sX!=0 && arg->sY!=0, "[Matrix::pseudoInverse] sizeX||Y == 0");

    setSizeX(arg->sizeX());
    setSizeY(0);
    Vector row(arg->sizeX());
    Matrix mt(0, arg->sizeX());
    double *argdata = arg->data, *rowdata = row.getData();
    int argSizeX = arg->sizeX(), res = 0;
    bool isIndep;
    for(int i=0; i<arg->sizeY(); i++)
    {
        for(int j=0; j<argSizeX; j++)
        {
            rowdata[j] = *argdata;
            argdata++;
        }
        pseudoInverseRecalc(/*res>=sizeX()*/false,&mt,&row,this,&isIndep);
        if(isIndep)
            res++;
    }
    return res;
}

// solves square matrix system Ax = b
// using matrix reduction to upper-triangular form
// returns true if A is not singular, false otherwise
bool Matrix::solveAxb(Vector *xvec, Vector *bvec, double *singDegree, bool keepThis)
{
    assertion(sX==sY && sY==xvec->size() && xvec->size()==bvec->size(), "[Matrix::solveAxb] wrong sizes");

    int dim = sX;
    double *buffer = new double[dim + (keepThis? dim*dim : 0)];
    double *bv = buffer;
    double *am = keepThis? buffer+dim : data;
    double *xv = xvec->getData();

    memcpy(bv,bvec->getData(),dim*sizeof(double));
    if(keepThis)
        memcpy(am,data,dim*dim*sizeof(double));

    double minmax = 0;
    for(int i=0; i<dim-1; i++)
    {
        double *aii = am + i*dim+i;
        int subdim = dim-i;

        // pivoting (max element selection)
        double maxel = 0, *aji = aii;
        int maxind = -1;
        for(int j=i; j<dim; j++)
        {
            if(fabs(*aji) > maxel)
            {
                maxel = fabs(*aji);
                maxind = j;
            }
            aji += dim;
        }
        if(maxel<minmax || i==0)
            minmax = maxel;
        if(maxel == 0)
        {
            if(singDegree != 0)
                *singDegree = 0;
            delete []buffer;
            return false;
        }

        // i-th and maxind-th rows' exchange (incl. bv elements)
        if(maxind != i)
        {
            double *amaxi = am + maxind*dim+i, dval;
            for(int j=0; j<subdim; j++)
            {
                dval = aii[j];
                aii[j] = amaxi[j];
                amaxi[j] = dval;
            }
            dval = bv[i];
            bv[i] = bv[maxind];
            bv[maxind] = dval;
        }

        // rows' substraction (incl. bv elements)
        for(int j=i+1; j<dim; j++)
        {
            aji = am + j*dim+i;
            double factor = aji[0]/aii[0];
            for(int k=1; k<subdim; k++)
                aji[k] -= aii[k]*factor;
            bv[j] -= bv[i]*factor;
        }
    }

    // back substitution
    for(int i=dim-1; i>=0; i--)
    {
        double sum = 0, *ai0 = am + i*dim;
        for (int j=i+1; j<dim; j++)
            sum += xv[j]*ai0[j];
        xv[i] = (bv[i]-sum)/ai0[i];
    }

    delete []buffer;

    if(singDegree != 0)
        *singDegree = minmax;
    return true;
}

// solves overdetermined square matrix system Ax = b where dim(x) <= dim(b)
// x = (A.tr*A)^-1 * A.tr * b
bool Matrix::solveOverdeterminedAxb(Vector *xvec, Vector *bvec, double *singDegree)
{
    assertion(sX==xvec->size() && sY==bvec->size(), "[Matrix::solveOverdeterminedAxb] wrong sizes");
    assertion(xvec->size() <= bvec->size(), "[Matrix::solveOverdeterminedAxb] is not overdetermined");

    Matrix ata(sX,sX);
    Vector atv(sX);
    multTN(this,&ata,true);
    multTN(bvec,&atv);
    return ata.solveAxb(xvec,&atv,singDegree,false);
}

// expands projective matrix in "this" by "vec"
// returns true if "vec" is linearly independent from "this" subspace, and, thus, "this" was altered
// false otherwise
// "diff" contains |ort(vec)|^2 / |vec|^2
// ort = vec - this*vec
// NOTE: uses PSEUDO_INVERSE_EPS as linear dependence criterion
// To expand as a projective matrix alpha must not be specified (i.e. default value 1 is used)
// Specifying alpha other then 1 may be used in adaptive associative training procedure
bool Matrix::expandProjectiveMatrix(Vector *vec, double *diff, double alpha)
{
    assertion(sX==sY && vec->size()==sX, "[Matrix::expandProjectiveMatrix] wrong sizes");

    Matrix ort(sX,1);
    mult(vec,&ort);
    vec->minus(&ort,&ort);

    double vecsq = vec->sproduct(vec);
    if(vecsq == 0)
    {
        if(diff)
            *diff = 0;
        return false;
    }

    double adiff = ort.sproduct(&ort) / vecsq;
    if(diff)
        *diff = adiff;
    double *ortdata = ort.getData();
    if(adiff > PSEUDO_INVERSE_EPS)
    {
        double factor = ort.sproduct(vec)/alpha, *adata = data;
        for(int i=0; i<sX; i++)
            for(int j=0; j<sX; j++)
                *adata++ += ortdata[i]*ortdata[j]/factor;
        return true;
    }
    else
        return false;
}

// expands projective matrix as this = this + vec*vec.Tr
// so this = M*M.Tr where M = { vec0, cvec1, ... }
void Matrix::expandMMT(Vector *vec)
{
    assertion(sX==sY && vec->size()==sX, "[Matrix::expandMMT] wrong sizes");
    double *vecdata = vec->getData();
    double *mtrdata = data;
    for(int i=0; i<sX; i++)
    {
        double veci = vecdata[i];
        for(int j=0; j<sX; j++)
            *mtrdata++ += veci*vecdata[j];
    }
}

// |vec - this*vec|^2/|vec|^2
double Matrix::getDifference(Vector *vec)
{
    assertion(sX==sY && vec->size()==sX, "[Matrix::getDifference] wrong sizes");

    Matrix ort(sX,1);
    mult(vec,&ort);
    ort.minus(vec,&ort);
    double sqnorm = vec->sproduct(vec);
    return sqnorm>0? ort.sproduct(&ort)/sqnorm : -1;
}


// returns false if (i,j) exists that |this[i,j] - mtr[i,j]| > 0.5, true otherwise
bool Matrix::binaryEquals(Matrix *mtr)
{
    assertEqualSize(this, mtr, "[Matrix::binaryEquals]");
    return binary_equals(data, mtr->data, sX*sY);
}


// returns number of elemenets such that |this[i,j] - mtr[i,j]| > 0.5
int Matrix::hammingDistance(Matrix *mtr)
{
    assertEqualSize(this, mtr, "[Matrix::hammingDistance]");
    return hamming_distance(data, mtr->data, sX*sY);
}

double Matrix::getSimilarity(Matrix *mtr)
{
    assertEqualSize(this, mtr, "[Matrix::getSimilarity]");
    assertNonzeroSize(this, "[Matrix::getSimilarity]");
    return fabs(sproduct(mtr)) / (sX*sY);
}


//---------------------------- Vector ------------------------------------------

double & Vector::operator[](int i)
{
    # ifdef MATH_DEBUG
    assertion(i>=0 && i<sizeY(), "[Vector::[]] index out of range");
    # endif
    return data[i];
}


// this = in(rowIndex, 0:in->sX-1)
void Vector::setFromRow(Matrix *in, int rowIndex)
{
    assertion(rowIndex>=0 && rowIndex<in->sizeY(), "[Matrix::setFromRow] wrong index");

    setSize(in->sizeX());
    double *src = &in->getData()[rowIndex*size()];
    memcpy(data, src, size()*sizeof(double));
}


//----------------------- CorrelationMatrix ------------------------------------

void CorrelationMatrix::addItem(Vector *vec)
{
    if(numItems == 0) // init stuff
    {
        counters.resize(0);
        counters.resize(vec->size());
        setSizeYX(vec->size(), vec->size());
        mmt.setSizeYX(vec->size(), vec->size());
        mmt.init(0);
    }
    else
        assertion(vec->size() == (int)counters.size(), "[CorrelationMatrix::addItem] wrong argument->sizeY()");

    numItems++;
    mmt.expandMMT(vec);
    int dim = vec->size();
    double *vecdata = vec->getData();
    for(int i=0; i<dim; i++)
        counters[i].addValue(vecdata[i]);

    double *thisdata = getData();
    for(int i=0; i<dim; i++)
        for(int j=0; j<dim; j++)
        {
            double sdi = counters[i].getSD();
            double sdj = counters[j].getSD();
            if(sdi*sdj != 0)
                *thisdata++ = (mmt.el(i, j)/numItems - counters[i].getAverage()*counters[j].getAverage()) / sdi / sdj;
            else
                *thisdata++ = 0;
        }
}

//-------------------------- StatCounter ---------------------------------------

void StatCounter::addValue(double value)
{
    sum += value;
    sumsq += value*value;
    min = value<min||!count? value : min;
    max = value>max||!count? value : max;
    count++;
}

void StatCounter::addValue(double value, string label)
{
    minLabel = value<min||!count? label : minLabel;
    maxLabel = value>max||!count? label : maxLabel;
    
    addValue(value);
}

void StatCounter::getData(Properties *prop, char fields)
{
    string prefix = name.compare("")? name+"_" : string("");

    if(fields & SC_COUNT)
        (*prop)[prefix+"n"] = getCount();
    if(fields & SC_AVERAGE)
        (*prop)[prefix+"ave"] = getAverage();
    if(fields & SC_SD)
        (*prop)[prefix+"sd"] = getSD();
    if(fields & SC_DISP)
        (*prop)[prefix+"Disp"] = getDisp();
    if(fields & SC_MIN)
        (*prop)[prefix+"min"] = getMin();
    if(fields & SC_MAX)
        (*prop)[prefix+"max"] = getMax();
}

string StatCounter::toString()
{
    int precision = 4, width = 6;

    string res = "[StatCounter]" + (name.compare("")? (" - " + name) : string("")) + "\n";
    res += setWidth("Count", width, saLeft) + ::toString(getCount()) + "\n";
    res += setWidth("Ave", width, saLeft) + ::toString(getAverage(), precision) + "\n";
    res += setWidth("SD", width, saLeft) + ::toString(getSD(), precision) + "\n";
    res += setWidth("Disp", width, saLeft) + ::toString(getDisp(), precision) + "\n";
    res += setWidth("min", width, saLeft)
        + setWidth(::toString(getMin(), precision), 10, saLeft)
        + (minLabel.compare("")? "label: " + minLabel : string("")) +"\n";
    res += setWidth("max", width, saLeft)
        + setWidth(::toString(getMax(), precision), 10, saLeft)
        + (maxLabel.compare("")? "label: " + maxLabel : string(""));

    return res;
}


//------------------------ AdvancedStatCounter ---------------------------------

AdvancedStatCounter::AdvancedStatCounter(double from, double to, int granularity, string name) : StatCounter(name)
{
    assertion(from<to && granularity>0, "[AdvancedStatCounter::AdvancedStatCounter] wrong params");

    isDiscrete = false;
    this->from = from;
    this->to = to;
    counter.assign(granularity, 0);
}

AdvancedStatCounter::AdvancedStatCounter(int from, int to, string name)
{
    assertion(from<to, "[AdvancedStatCounter::AdvancedStatCounter] wrong params");

    isDiscrete = true;
    this->from = from;
    this->to = to;
    counter.assign(to-from, 0);
}

void AdvancedStatCounter::getProbabilityDensity(Vector *func)
{
    checkCount();
    func->setSize(0);
    double factor = ((double)counter.size())/getCount()/(to-from);
    for(unsigned int i=0; i<counter.size(); i++)
        func->push_back(counter[i]*factor);
}

void AdvancedStatCounter::getProbabilityDistribution(Vector *func)
{
    checkCount();
    func->setSize(0);
    for(unsigned int i=0; i<counter.size(); i++)
        func->push_back(counter[i]+(i>0? (*func)[i-1] : 0));
    for(int i=0; i<counter.size(); i++)
        (*func)[i] /= getCount();
}

void AdvancedStatCounter::getScale(Vector *scale)
{
    checkCount();
    scale->setSize(0);
    double step = (to-from)/counter.size();
    for(unsigned int i=0; i<counter.size(); i++)
        scale->push_back(isDiscrete? round(from+i*step) : from+i*step);
}

void AdvancedStatCounter::addValue(double value)
{
    StatCounter::addValue(value);
    int index = isDiscrete? round( (value-from)*counter.size()/(to-from) )
                                        : (value-from)*counter.size()/(to-from);
    if(index < 0)
        index = 0;
    if(index >= (int)counter.size())
        index = counter.size()-1;
    counter[index]++;
}

string AdvancedStatCounter::toString()
{
    Vector scale, probDensity, probDistrib;
    getScale(&scale);
    getProbabilityDensity(&probDensity);
    getProbabilityDistribution(&probDistrib);

    string res = "[AdvancedStatCounter], isDiscrete = " + ::toStringB(isDiscrete) + ", "
                                                    + StatCounter::toString() + "\n";
    res += "Scale:            " + ::toString(&scale, true);
    res += "ProbDensity:      " + ::toString(&probDensity, true);
    res += "ProbDistribution: " + ::toString(&probDistrib, true);
    if(isDiscrete)
    res += "Events:           " + ::toString(&Vector(&counter), true);

    return res;
}


//--------------------------- PropStatCounter ----------------------------------

void PropertyStatCounter::addValue(Properties *prop)
{
    for(map<string, double>::iterator iter=prop->begin(); iter!=prop->end(); iter++)
        counters[iter->first].addValue(iter->second);
}

void PropertyStatCounter::getStatistics(Properties *prop, char fields)
{
    prop->clear();
    int count = -1;
    for(map<string, StatCounter>::iterator iter=counters.begin(); iter!=counters.end(); iter++)
    {
        string fieldName = iter->first;
        StatCounter &sc = iter->second;
        if(count == -1)
            count = sc.getCount();
        //else
        //    assertion(count==sc.getCount(), "[PropertyStatCounter] different number of field instances");

        if(fields & SC_COUNT)
            (*prop)[fieldName+"_n"]   = sc.getCount();
        if(fields & SC_AVERAGE)
            (*prop)[fieldName+"_ave"] = sc.getAverage();
        if(fields & SC_SD)
            (*prop)[fieldName+"_sd"]  = sc.getSD();
        if(fields & SC_DISP)
            (*prop)[fieldName+"_disp"]  = sc.getDisp();
        if(fields & SC_MIN)
            (*prop)[fieldName+"_min"] = sc.getMin();
        if(fields & SC_MAX)
            (*prop)[fieldName+"_max"] = sc.getMax();
    }
}




