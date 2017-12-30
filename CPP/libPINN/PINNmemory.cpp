
// PINNmemory.cpp: implementation of the CPINNmemory class.
//
//////////////////////////////////////////////////////////////////////

#include "PINNmemory.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CPINNmemory::CPINNmemory()
{
	bCreated = false;
//	m_N=0;
}
CPINNmemory::~CPINNmemory()
{
	deleteMemory();
}

void CPINNmemory::deleteMemory()
{
	if (bCreated) 
	{
		delete m_Y;
		delete m_S;
		delete m_S0;
		delete m_aExcited;
		for (int i=0; i<m_N; i++)
			delete m_C[i];
		delete  m_C;
		bCreated = false;
//		m_N=0;
	}
}


void CPINNmemory::createMemory(int N)
{
	m_M = 0; // No pattern stored
	m_N = N;
    m_C = new float*[N];
	for (int i=0; i<m_N; i++)
	{
		m_C[i] = new float[m_N];
		for (int j=0; j<m_N; j++)
			m_C[i][j]=0;
	}
	m_Y = new signed char[N];
	m_S = new double[N];
	m_S0 = new double[N];
	m_aExcited = new int[N];
	bCreated = true;
	m_D = 1.0; //0.15; // The most optimal desaturation value for most cases
//	m_D = 0; implies no desaturation (i.e. the same as m_D=1.0)
	m_nNoise = 0;
	m_dFade = 1.0;

	m_nExcitedNow=0; 
	m_nExcitedLast=0;
	m_bCiiChanged = false;
}

void CPINNmemory::reset()
{
	m_M = 0; // No pattern stored
	m_nExcitedNow=0; 
	m_nExcitedLast=0;
	m_bCiiChanged = false;
	for (int i=0; i<m_N; i++)
	{
		for (int j=0; j<m_N; j++)
			m_C[i][j]=0;
	}
}

bool CPINNmemory::saveC(char *filename)
{
	FILE *f=NULL;

	f = fopen(filename, "wb");
	if (f==NULL) 
		return false;

//	fprintf(f, "%i %i %s.\n", m_N, m_M, filename);
	for (int i=0; i<m_N; i++)
		fwrite(m_C[i], m_N, sizeof(float), f);
	return true;
}

bool CPINNmemory::loadC(char *filename) 
{
	FILE *f=NULL;

	f = fopen(filename, "rb");
	if (f==NULL) 
		return false;

//	int N, M;
//	char str[100];
//	fscanf(f, "%i %i %s.\n", &N, &M, str);
//	if (N!=m_N)
//		m_M = M;

	for (int i=0, nRead=0; i<m_N; i++)
		nRead+=fread(m_C[i], m_N, sizeof(float), f);
	if (nRead != m_N*m_N) 
		return false;
	else
		return true;
}

bool CPINNmemory::putInMemory(signed char *V)
{
// if needed restore values (from [dynamic] desaturation)
//	restoreSaturationLevel();

	int i,j;
//	m_dAveCii = 0;
//	m_dAveCij2 = 0;
	m_E = m_N; 
	for(i=0; i<m_N; i++)
	{
		for(m_S[i]=0,j=0; j<m_N; j++)
			m_S[i] += m_C[j][i]*V[j];
//		m_dAveCii += (V[i]-m_S[i])*(V[i]-m_S[i]);
//		m_dAveCij2 += V[i]*(V[i]-m_S[i]);
		m_E -= m_S[i]*V[i];
	}
//	m_E = m_dAveCii; // Should be the same, but it is NOT!!!
	if (m_E <= 1) // <0.1
		return false; // Either the memory is full OR the same(similar) pattern has been already stored

#if 1 // <--------
	for(i=0;i<m_N;i++)
		for(j=0;j<m_N;j++)
			m_C[i][j]+=(float)((V[i]-m_S[i])*(V[j]-m_S[j])/m_E); // Notice that m_C is symetric.

/*	Blah Use this?...
	for(i=0;i<m_N;i++)
		for(j=0;j<=i;j++)
		{
			m_C[i][j]+=(float)((V[i]-m_S[i])*(V[j]-m_S[j])/m_E); 
			m_C[i][j] = m_C[j][i];
		}
*/
#else
// For dynamic desaturation:
    for(i=0;i<m_N;i++)
		for(j=0;j<m_N;j++)
			m_C[i][j] = (float) (m_dFade*m_C[i][j] + (V[i]-m_S[i])*(V[j]-m_S[j])/m_E); // Notice that m_C is symetric.
//			m_C[i][j] = m_dFade*m_C[i][j] + (1-m_dFade)*(V[i]-m_S[i])*(V[j]-m_S[j])/m_E; // Notice that m_C is symetric.
#endif
	m_M ++; // One pattern has been added to memory

// if needed desaturate
//	analyzeMemory();
/*	if (m_dAveCii >  m_dAveCij2*4)
		setDesaturationLevel(0.15);
*/
//	desaturateMemory();

	return true;
}

void CPINNmemory::analyzeMemory() // examine how filled the memory is 
{
	double X,Y,Z; int i, j;

// Compute the actual values
	for(X=0,Y=0,Z=0, i=0;i<m_N;i++)
    {
		for(j=0;j<m_N;j++)
		{
			Y+=m_C[i][j]*m_C[i][j];  Z+=fabs(m_C[i][j]);
		}
		X+=m_C[i][i];
    }
  
  /* Average values of weights Cii and Cij */
  X=X/m_N; 
  Y=Y/m_N/m_N; 
  Z=(Z/m_N/m_N)*(Z/m_N/m_N); 
//  printf("\nC:  <Cii> = %.5f,\t   <Cij^> = %.5f  vs  <|Cij|>^=%.5f\n", X,Y,Z);
  m_dAveCii = X; 
  m_dAveCii2 = X*X; 
  m_dAveCij2 = Y; 
  m_dAveAbsCij2 = Z;

 // Compute approximations of Average values of weights Cii and Cij by formula from Ref [2] 
  X=(double)m_M/m_N; 
  Y=(double)m_M*(m_N-m_M)/m_N/m_N/m_N;  
//  printf("vs   M/N  = %.5f,\tM(N-M)/N3 = %.5f \n", X,Y);

}     

void CPINNmemory::setDesaturationLevel(double dDesaturationLevel)
{
	m_D = (float)dDesaturationLevel;
}
void CPINNmemory::desaturateMemory()
{
	if (m_D==0)
		return;
	if (m_bCiiChanged)
		return;

	if (m_D > 0 ) // Gorodnichy's
		for(int i=0;i<m_N;i++)
			m_C[i][i]=m_D*m_C[i][i];
	else // Ueda's
		for(int i=0;i<m_N;i++)
			m_C[i][i]=m_C[i][i]-(1+m_D);

	m_bCiiChanged = true;


/* NB: To do: Compare vs. Ohta approach
	for(int i=0;i<m_N;i++)
		m_C[i][i]=m_C[i][i]-m_D; // m_D=0.98
*/
}

void CPINNmemory::desaturateMemory(double dD)
{
	setDesaturationLevel(dD);
	desaturateMemory();
}
void CPINNmemory::restoreSaturationLevel()
{
	if (m_D==0)
		return;
	if (! m_bCiiChanged)
		return;

	for(int i=0;i<m_N;i++)
		m_C[i][i]=m_C[i][i]/m_D; // Blah: errors will accumulate: C*D/D*D/D...
	m_bCiiChanged = false;
}

// By default, all neurons are not excited, i.e. are "-1"
void CPINNmemory::setMemoryToDormantState() // in which all neurons are in unexcited (-1) state
{
	int i,j;
	for(i=0;i<m_N;i++)
		for(j=0, m_S0[i]=0;j<m_N;j++)
		  m_S0[i]-=m_C[j][i];
}
 

// Recognition stage. Retrieval from memory. Testing/examining the network.
//
// Using the flood-fill neuro-processing technique described in Ref [3], 
// which is based on the fact that in brain only a small number of neurons 
// is actually excited (i.e. are "+1"), and therefore should be processed.

int CPINNmemory::retrieveFromMemory(signed char **Y0, signed char **Yattractor)
{	
	int i;
	bool bAttractorAchieved;
	int nOscilatingNeurons;
	// Initialize state of the network to Y0, by setting the buffer of '+1' neurons
	for(m_nExcitedNow=0, i=0; i<m_N; i++)
	{
		m_S[i]=m_S0[i];
		m_Y[i] = (*Y0)[i];
		if (m_Y[i]>0) 
		    m_aExcited[m_nExcitedNow++]=+i+1; 
	}		

#if 1 //SYNCHONOUS 
 // Gorodnichy's method: Parallel dynamics

//	printf("PARALLEL DYNAMICS");
	// Iterate until the network converges to an attractor, either static or dynamic(cycle).
	for(m_nIter=0, bAttractorAchieved=false; !bAttractorAchieved ;m_nIter++)
    {
		for(int e=0; e<m_nExcitedNow; e++) // Trace synapces of all excited neurons and update PSP of the connected neurons
			if (m_aExcited[e]<0)
				for(int j=0;j<m_N;j++)
					m_S[j] -= 2*m_C[-m_aExcited[e]-1][j]; // -1, because it starts from '+/-1' not '0'
			else
				for(int j=0;j<m_N;j++)
					m_S[j] += 2*m_C[m_aExcited[e]-1][j];
      
		m_nExcitedLast=m_nExcitedNow;
		m_nExcitedNow=0;
		nOscilatingNeurons=0;
		for(int i=0; i<m_N; i++) // Update the state of the network
		{
			if ( (m_S[i]<0) && (m_Y[i]>0) )
			{ 
				m_Y[i]=-1; 
				if (m_aExcited[m_nExcitedNow]==i+1)  // cycle check: see whether this neuron was excited in previous iteration
						nOscilatingNeurons++;	// if yes, update nOscilatingNeurons
					m_aExcited[m_nExcitedNow++]=-i-1; // update buffer
				}
			else if  ( (m_S[i]>=0) && (m_Y[i]<0) ) // another option: (m_S[i]>=0)
			{
				m_Y[i]=+1; 
				if (m_aExcited[m_nExcitedNow]==-i-1)  // for check of a cycle 
					nOscilatingNeurons++;
				m_aExcited[m_nExcitedNow++]=+i+1;
			}
			else
				i=i;
		}
		if (m_nExcitedNow==0) 
		{
			m_bCycleOccured = false; // it is a static attractor
			bAttractorAchieved = true;
		}
		if (m_nExcitedNow == m_nExcitedLast && nOscilatingNeurons==m_nExcitedNow) 
		{
			m_bCycleOccured = true;// it is a cycle  
			bAttractorAchieved = true;
		}	
	} // end of iteration loop
	
// 2005. Technque from UEDA: Asynchronous dynamics - 
//		in one iteration change only one m_Y[i]: m_S[i]>=0 * m_Y[i] -> max
	// I'll do some time later...
	
#else  // This doesnot work
printf("ASYNCHRONOUS DYNAMICS");


	int j;
	for(i=0;i<m_N;i++)
		for(m_S[i]=0,j=0;j<m_N;j++)
			m_S[i]+=m_C[i][j]*m_Y[j];


	m_nExcitedLast=-1;
	m_nExcitedNow=-1;
	for(m_nIter=0; ;m_nIter++)
    {
		m_nExcitedLast=m_nExcitedNow;
		
		int nNeuronsToUpdate=0; 
		double minE = 0;
		for(int i=0; i<m_N; i++) // Update the state of the network. Find best neuron
		{

			// Because m_S[i]  is not the same with reduction with m_D!!!!
			if (m_S[i] - m_D < 0 && m_Y[i]>0 )
//			if (m_S[i] < 0 && m_Y[i]>0 )
			{
				nNeuronsToUpdate++;
				if (m_S[i]<minE)				
				{ 
						m_nExcitedNow = i; 
						minE = m_S[i]; // m_S[i]*m_Y[i]; 
				}
			}
			else if (m_S[i]-m_D > 0 && m_Y[i]<0 )
			{
				nNeuronsToUpdate++;
				if (m_S[i]>-minE)				
				{ 
						m_nExcitedNow = i; 
						minE = -m_S[i]; // m_S[i]*m_Y[i]; 
				}
			}
		}
		if (nNeuronsToUpdate==0)
			break; // Attractor reached!

		int nExcitedRandom;

		printf( "  %i", m_nExcitedNow);
		if (m_nExcitedNow == m_nExcitedLast)
		{
			// select randomly any other neuron
			// the one changed last time should not change now
			do 
				nExcitedRandom=rand() % m_N;
			while (nExcitedRandom==m_nExcitedLast);
			m_nExcitedNow = nExcitedRandom;
			printf( "/%i", m_nExcitedNow);
			// neponqtno: chto wnosit' Noise gde ne nado?!
			// ili hotq by tam gde naimen'shij risk (E smallest)?
		}


//		// dlq debuga
//		nOscilatingNeurons = nNeuronsToUpdate;

		// Best neuron selected

		// Reverse m_Y[m_nExcitedNow] and the consequences of this flip
		m_Y[m_nExcitedNow] = - m_Y[m_nExcitedNow];
		for(int j=0;j<m_N;j++)
		{
			m_S[j] += 2*m_C[m_nExcitedNow][j]*m_Y[m_nExcitedNow]; // -1, because it starts from '+/-1' not '0'
// ??			m_S[j] -= 2*m_C[m_nExcitedNow][j]*m_Y[m_nExcitedNow]; // -1, because it starts from '+/-1' not '0'
			if (j==m_nExcitedNow)
				continue;
			m_Y[j] = (m_S[j] > 0) ? +1 : -1; 
		}
	}

#endif


// Check how close the retrieved pattern is to those which were stored: m_E = ||CY-Y||^2

// we can use extra condition; ??? Doesnot work. Jan 2005
/*
Y should be stable in main network;
E = ||CY - Y|| = should be 0
*/
	double E;
	for(m_E=m_N, i=0, E=0; i<m_N; i++)
	{
		(*Yattractor)[i] = m_Y[i];
		m_E -= m_S[i]*m_Y[i]; //  equivalent to E += (m_S[i]-m_V[i])*(m_S[i]-m_V[i]);
	}

	return 1;
}


void CPINNmemory::addNoise(signed char **V)
{
	add_noise(*V,V,m_N,m_nNoise);
}


/*** Utility functions ***
********************************************************************************************/

void CPINNmemory::add_noise(signed char *Y, signed char **V, int N, int nNoise)
{
	bool *bY = new bool [N];
	for(int i=0;i<N;i++)  
		bY[i]=false;
	for(i=0;i<nNoise; )  
    {  
		int kk=rand() % N;
		if (bY[kk] == false)
		{
			bY[kk]=true;  i++;
		}
    }
	for(i=0;i<N;i++)  
		if (bY[i] == true) 
			(*V)[i] = - Y[i];
		else
			(*V)[i] = Y[i];

	delete bY;
}

int CPINNmemory::computeHemming(signed char *Y, signed char *V, int N)
{
	for(int nHemming=0,i=0;i<N;i++) 
		if (Y[i]!=V[i]) 
			nHemming++;
	return nHemming;
}

bool CPINNmemory::writeRandomBinaryVectors(int N, int M, char *sFileName, double dPercentageOfOnes, int nRndSeed)
{
    FILE *f;
	double dRnd;
    f=fopen(sFileName, "w");
	if (f==NULL) return false; 

	char * sSample = new char[N+1];

    fprintf(f, "Prototype set: N=%i, M=%i, L=%.2f. Seed=%i\n\n",N, M, dPercentageOfOnes, nRndSeed);
    srand( (unsigned)nRndSeed ); 
    for (int m=0;m<M;m++)
    {
		for (int i=0;i<N;i++)
		{
			dRnd = (double)rand()/RAND_MAX;
			if (dRnd < dPercentageOfOnes)
				sSample[i] = '1';
			else 
				sSample[i] = '0';
		}
		sSample[i]='\0'; 
      	fprintf(f, "%s\n", sSample);
      }
    fclose(f); 

	delete sSample;
	return true;
}

void CPINNmemory::string2BinaryVector(signed char **V, int N, char *string)
{
    for(int i=0;i<N;i++)
		(*V)[i]=('1'-string[i])*2 - 1;
}


/*********************************************************************************************
*** End of Utility functions ***/


