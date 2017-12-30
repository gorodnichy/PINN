# ifndef _TEMPLATES_H_
# define _TEMPLATES_H_

# include "utils.h"

// res(vector) = mtr*vec
// mx = mtr->SizeX, my = mtr->SizeY
template <class T1, class T2, class T3> void multiplyN(T1 *res, T2 *mtr, T3 *vec, int mx, int my)
{
    T1 *resbuf = res;
    T2 *mtrbuf = mtr;
    for(int i=0; i<my; i++)
    {
        T3 *vecbuf = vec;
        double sum = 0;
        for(int j=0; j<mx; j++)
            sum += *mtrbuf++ * *vecbuf++;
        *resbuf++ = (T1)sum;
    }
}

// res(vector) = mtr.transposed*vec
// mx = mtr->SizeX, my = mtr->SizeY
template <class T1, class T2, class T3> void multiplyT(T1 *res, T2 *mtr, T3 *vec, int mx, int my)
{
    T1 *resbuf = res;
    for(int i=0; i<mx; i++)
    {
        T2 *mtrbuf = mtr+i;
        T3 *vecbuf = vec;
        double sum = 0;
        for(int j=0; j<my; j++)
        {
            sum += *mtrbuf * *vecbuf++;
            mtrbuf += mx;
        }
        *resbuf++ = (T1)sum;
    }
}

// res(matrix) = mtrA*mtrB
// rx = res->SizeX, ry = res->SizeY, mAx = mtrA->SizeX
template <class T1, class T2, class T3> void multiplyNN(T1 *res, T2 *mtrA, T3 *mtrB, int rx, int ry, int mAx)
{
    T1 *resbuf = res;
    for(int i=0; i<ry; i++)
    {
        for(int j=0; j<rx; j++)
        {
            double sum = 0;
            T2 *mtrAbuf = mtrA + i*mAx;
            T3 *mtrBbuf = mtrB + j;
            for(int k=0; k<mAx; k++)
            {
                sum += *mtrAbuf * *mtrBbuf;
                mtrAbuf++;
                mtrBbuf += rx;
            }
            *resbuf++ = (T1)sum;
        }
    }
}

// res(matrix) = mtrA.transposed*mtrB
// rx = res->SizeX, ry = res->SizeY, mAy = mtrA->SizeY
// if mtrA == mtrB then set byItself = true (2x times quicker)
template <class T1, class T2, class T3> void multiplyTN(T1 *res, T2 *mtrA, T3 *mtrB, int rx, int ry, int mAy, bool byItself = false)
{
    if(byItself && rx!=ry)
        throw GenException("[template <...> MultiplyTN] byItself && rx!=ry");
    T1 *resbuf = res;
    for(int i=0; i<ry; i++)
    {
        for(int j=0; j<rx; j++)
        {
            if(byItself && j<i) // set symmetric element
            {
                res[i*rx + j] = res[j*rx + i];
                continue;
            }
            double sum = 0;
            T2 *mtrAbuf = mtrA + i;
            T3 *mtrBbuf = mtrB + j;
            for(int k=0; k<mAy; k++)
            {
                sum += *mtrAbuf * *mtrBbuf;
                mtrAbuf += ry;
                mtrBbuf += rx;
            }
            *resbuf++ = (T1)sum;
        }
    }
}

// res(matrix) = mtrA*mtrB.transposed
// rx = res->SizeX, ry = res->SizeY, mAx = mtrA->SizeX
template <class T1, class T2, class T3> void multiplyNT(T1 *res, T2 *mtrA, T3 *mtrB, int rx, int ry, int mAx)
{
    T1 *resbuf = res;
    for(int i=0; i<ry; i++)
    {
        for(int j=0; j<rx; j++)
        {
            double sum = 0;
            T2 *mtrAbuf = mtrA + i*mAx;
            T3 *mtrBbuf = mtrB + j*mAx;
            for(int k=0; k<mAx; k++)
            {
                sum += *mtrAbuf * *mtrBbuf;
                mtrAbuf++;
                mtrBbuf++;
            }
            *resbuf++ = (T1)sum;
        }
    }
}

// |buf1 - buf2|^2
template <class T1, class T2> double sqdist(T1 *buf1, T2 *buf2, int size)
{
    double diff, res = 0;
    for(int i=0; i<size; i++)
    {
        diff = buf1[i] - buf2[i];
        res += diff*diff;
    }

    return res;
}

// scalar product
// buf1*buf2 = sum(i) buf1[i]*buf2[i]
template <class T1, class T2> double sproduct(T1 *buf1, T2 *buf2, int size)
{
    double res = 0;
    for(int i=0; i<size; i++)
        res += buf1[i] * buf2[i];
    return res;
}

// sum of squared elements
// sum(i) buf[i]*buf[i]
template <class T> double sqnorm(T *buf, int size)
{
    double res = 0;
    for(int i=0; i<size; i++)
        res += buf[i] * buf[i];
    return res;
}

// res = buf1 + buf2
template <class T1, class T2, class T3> void vecplus(T1 *buf1, T2 *buf2, int size, T3 *res)
{
    for(int i=0; i<size; i++)
        res[i] = buf1[i] + buf2[i];
}

// res = buf1 - buf2
template <class T1, class T2, class T3> void vecminus(T1 *buf1, T2 *buf2, int size, T3 *res)
{
    for(int i=0; i<size; i++)
        res[i] = buf1[i] - buf2[i];
}

// res = buf * value
template <class T1, class T2, class T3> void vecmult(T1 *buf, T2 value, int size, T3 *res)
{
    for(int i=0; i<size; i++)
        res[i] = buf[i] * value;
}

// res[i] = buf1[i] * buf2[i]
template <class T1, class T2, class T3> void vecmultComponentwise(T1 *buf1, T2 *buf2, int size, T3 *res)
{
    for(int i=0; i<size; i++)
        res[i] = buf1[i] * buf2[i];
}

// res[i][j] += coeff * buf1[i] * buf2[j]
template <class T1, class T2, class T3> void vecplus(T1 *buf1, T2 *buf2, int size1, int size2, T3 *res, double coeff)
{
    T3 *resbuf = res;
    for(int i=0; i<size1; i++)
    {
        double factor = coeff * buf1[i];
        for(int j=0; j<size2; j++)
            *resbuf++ += factor * buf2[j];
    }
}

// save to a binary stream
template <class T> void saveVec(vector<T> &vec, outBinStream &str)
{
    str << (int)vec.size();
    for(int i=0; i<vec.size(); i++)
        str << vec[i];
}

// load from a binary stream
template <class T> void loadVec(vector<T> &vec, inBinStream &str)
{
    int size;
    str >> size;
    vec.resize(size);
    for(int i=0; i<size; i++)
        str >> vec[i];
}

// returns false if i exists that |buf1[i] - buf2[i]| > 0.5, true otherwise
template <class T1, class T2> bool binary_equals(T1 *buf1, T2 *buf2, int size)
{
    for(int i=0; i<size; i++)
        if(fabs(buf1[i]-buf2[i]) > 0.5)
            return false;
    return true;
}

template <class T1, class T2> void vecsign(T1 *in, T2 *out, int size)
{
	for(int i=0; i<size; i++)
		out[i] = in[i]>0? 1 : -1;
}

template <class T1, class T2> void vecabs(T1 *in, T2 *out, int size)
{
	for(int i=0; i<size; i++)
		out[i] = fabs(in[i]);
}

// returns number of elemenets such that |buf1[i] - buf2[i]| > 0.5
template <class T1, class T2> int hamming_distance(T1 *buf1, T2 *buf2, int size)
{
    int res = 0;
    for(int i=0; i<size; i++)
        if(fabs(buf1[i]-buf2[i]) > 0.5)
            res++;
    return res;
}

// truncates the array
template <class T> void truncate(T *in, T *out, int size, double threshold)
{
    assertion(threshold >= 0, "[templates::truncate] threshold<0");
    T value;
    for(int i=0; i<size; i++)
    {
        value = in[i];
        out[i] = fabs(value)>threshold? (value>0? threshold : -threshold) : value;
    }
}

// returns max element in unsorted array
// if (maxElIndex!=0) then sets maxElIndex to its max element position
template <class T> T maxElement(T *buf, int size, int *maxElIndex = 0)
{
    assertion(size>0, "[templates::maxElement] must be size>0");

    T maxEl = buf[0];
    int index = 0;

    for(int i=1; i<size; i++)
        if(buf[i] > maxEl)
        {
            maxEl = buf[i];
            index = i;
        }

    if(maxElIndex)
        *maxElIndex = index;

    return maxEl;
}

// returns min element in unsorted array
// if (minElIndex!=0) then sets minElIndex to its max element position
template <class T> T minElement(T *buf, int size, int *minElIndex = 0)
{
    assertion(size>0, "[templates::minElement] must be size>0");

    T minEl = buf[0];
    int index = 0;

    for(int i=1; i<size; i++)
        if(buf[i] < minEl)
        {
            minEl = buf[i];
            index = i;
        }

    if(minElIndex)
        *minElIndex = index;

    return minEl;
}

// returns max element in unsorted vector
// if (maxElIndex!=0) then sets maxElIndex to its max element position
template <class T> T maxElement(vector<T> *vec, int *maxElIndex = 0)
{
    T maxEl = (*vec)[0];
    int index = 0, size = vec->size();

    for(int i=1; i<size; i++)
        if((*vec)[i] > maxEl)
        {
            maxEl = (*vec)[i];
            index = i;
        }

    if(maxElIndex)
        *maxElIndex = index;

    return maxEl;
}

// returns min element in unsorted array
// if (minElIndex!=0) then sets minElIndex to its max element position
template <class T> T minElement(vector<T> *vec, int *minElIndex = 0)
{
    T minEl = (*vec)[0];
    int index = 0, size = vec->size();

    for(int i=1; i<size; i++)
        if((*vec)[i] < minEl)
        {
            minEl = (*vec)[i];
            index = i;
        }

    if(minElIndex)
        *minElIndex = index;

    return minEl;
}



# endif /* _TEMPLATES_H_ */





