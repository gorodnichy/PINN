// PINNtest2.cpp : Defines the entry point for the console application.
// For IJCNN tests.
//
// PINNtest.cpp : Defines the entry point for the console application.
//
//////////////////////////////////////////////////////////////////////


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

class CPINNmemory  
{
public:
	CPINNmemory();
	virtual ~CPINNmemory();
	
	void createMemory(int N);
	void deleteMemory();
	bool putInMemory(signed char *V);
	void analyzeMemory(); // examine how filled the memory is 
	void setDesaturationLevel(double dDesaturationLevel);
	void desaturateMemory();
	void desaturateMemory(double dD);
	void restoreSaturationLevel();
	void setMemoryToDormantState(); // in which all neurons are in unexcited (-1) state
	int  retrieveFromMemory(signed char *Y0, signed char **Yattractor);
	bool CPINNmemory::saveC(char *filename);
	bool CPINNmemory::loadC(char *filename);
	
	void addNoise(signed char **V);
	// Utility functions
	void add_noise(signed char *Y, signed char **V, int N, int nNoise);
	int computeHemming(signed char *Y, signed char *V, int N);
	bool writeRandomBinaryVectors(int N, int M, char *sFileName, double dPercentageOfOnes, int nRndSeed);
	void string2BinaryVector(signed char **V, int N, char *string);
	
	bool bCreated;
	int m_N; 
	int m_M; // number of stored prototypes in network's memory 
	float 	**m_C;	// matrix C of weightes (aka synaptic weights) of the network
	double	*m_S,	// S = CY, postsynaptic potentials (PSP) of the network
		*m_S0;
	double	m_E;	// E = || CY - Y ||
	double m_D;
	double m_dFade;
	signed char *m_Y;	// state (aka potentials, or neurons, or neuron states) of the network
	
	int *m_aExcited; // buffer which stored indices of excited neurons
	int m_nExcitedNow, m_nExcitedLast;  // number of excited neurons at current and previous state of the network
	bool m_bCycleOccured; // redundant, as m_nExcitedNow>0 => m_bCycleOccured
	int m_nIter;
	float m_dAveCii2, m_dAveCij2, m_dAveAbsCij2;
	
	int m_nNoise;
	bool m_bCiiChanged;
	
};


char strMessage[80];
FILE *fout;

#define SHOW_OUTPUT 1
void printLog(FILE *f, char *str) 
{
	fprintf(f, "%s", str); 
#if SHOW_OUTPUT
	printf("%s", str); 
#endif
}

 

typedef struct MY_STATS
{
	double dMin;
	double dMax;
	double dTotal;
	int nCount;
} MY_STATS;


void resetStats(MY_STATS *stat)
{
	stat->dMin = 9999999999999999;
	stat->dMax = -9999999999999999;
	stat->dTotal = 0;
	stat->nCount = 0;
}


void updateStats(MY_STATS *stat, double d)
{
	if (d>stat->dMax)
		stat->dMax = d;
	else if (d<stat->dMin)
		stat->dMin = d;
	stat->dTotal+=d;
	stat->nCount++;
}

void main(void)
{
/* To do :

  1) C=C-a vs C=D C
  2) S->Y in learning. S->Y in testing (not to binorize face vectors)
  3) so-called dynamic desaturation : N=100
  M=10 prototypes. each w 15% noise and learn w. delta (or any other approximation) rule (neskol'ko raz, 
  kak w BackProp
  Potom Q: prototype s 20% -> k osnovnomu prototypu ili net ?
  3b) asynchronos dynamics: IN testing Update only that neuron which lowers E the most!

  4. SaveC. LoadC buttons

  5. net 1: vyrovnit' glaza: N= 20x5=100 ili 18x6=108. i tol'ko M=5?!. Dodelat' Slawin kod luchshe budet...
	 net 2. pose vyravnivaniq glaz. 6x6: nahozhdenie povorota golovy.
	 A potom uzhe frIDs!

a)  Wyvodit' fd->rect.width

  MOzhet w Face::loadFace() ne delat' resize (i BW) of detected face
  a delat' ego uzhe w fd->getVectorfromFace();
  (which will cause changing of sizes<- a my tam i tak eto delaem... no tak ne terqetsq data!..

b) KAK ARGUMENT W PINNtest.exe?

	*/

	MY_STATS statHemming, statIter;
	int i;
	CPINNmemory pinn;
	int M	=	40, 
		N	=	100; 
		
	FILE *f;
	char sFileName[] = "pinn-samples.txt";
	char sPattern[1002]; // make sure it is larger than N
//	sPattern = new char[N+1];
	signed char *V, *Y, *Y0;
	V = new signed char[N+1]; V[N]='\0';
	Y = new signed char[N+1]; Y[N]='\0'; 
	Y0 = new signed char[N+1]; Y0[N]='\0'; 

	fout=fopen("pinn-res.txt", "w");

	pinn.createMemory(N);

#if 1
	pinn.writeRandomBinaryVectors(N, M, sFileName, 0.3, 0);

// Memorize patterns

	printf("\nLearning patterns (N=%i, M=%i): \n", N,M);
	f=fopen(sFileName, "r");
	fgets(sPattern, 100, f);  /* first two comment lines */
	fgets(sPattern, 100, f);  /* first two comment lines */
	for (i=0;i<M;i++)
	{
		if ( ! fgets(sPattern, N+1, f) )
			break;
		fgetc(f);
		pinn.string2BinaryVector(&V, N, sPattern);
		pinn.putInMemory(V);
		pinn.analyzeMemory();
		printf("%i %.4f %.4f \n", pinn.m_M, pinn.m_dAveCii2, pinn.m_dAveCij2);
	}
	fclose (f);

	pinn.saveC("C.c_n");
#else
	pinn.loadC("C.c_n");

#endif

// Analyze memory and desaturate it if needed
	pinn.analyzeMemory();

/***
  Memory recognizes well when it is not saturated. 
  Whether the memory is saturated or not can be told by looking at ratio <Cii> / <Cij>, 
  which increases as the memory learns new patterns.
  It is advisable to desaturate the memory when  <Cii> / <Cij>  approaches   1/N.
***/


//  Uncomment these two lines to see the improvement due to the desaturation !
//
//	pinn.setDesaturationLevel(0.15);
//	pinn.desaturateMemory();

#define ASYNCHONOUS 0
#define UEDA 1

	int	nNoise= 15; // (int)N/10; // 0;
	
	sprintf(strMessage, "N=%i, M=%i, \t<Cii>^2=%.4f, <Cij^2>=%.4f,  RD(%i)=%.2f\n", 
		pinn.m_N,pinn.m_M,pinn.m_dAveCii2, pinn.m_dAveCij2, UEDA, pinn.m_D);  
	printLog(fout, strMessage);



//	for(UEDA=0;UEDA<=1;UEDA++)
	{
		int nUEDA=0;
		if (nUEDA)
		{
			pinn.m_D = 0.98; // 98;	
			for(i=0;i<pinn.m_N;i++)
				pinn.m_C[i][i]=pinn.m_C[i][i]-pinn.m_D;
		}
		else
		{
			pinn.m_D = 0.15; // 0.14; // 0.98;		 
			for(i=0;i<pinn.m_N;i++)
				pinn.m_C[i][i]=pinn.m_C[i][i]*pinn.m_D;//(1-pinn.m_D);
		}

//		for(ASYNCHONOUS=0;ASYNCHONOUS<=1;ASYNCHONOUS++)
			for (nNoise=0;nNoise<N/4;nNoise+=3)
			{
				sprintf(strMessage, "UEDA=%i, ASYNCHONOUS=%i, D=%.2f\n", nUEDA, ASYNCHONOUS, pinn.m_D);  
				printLog(fout, strMessage);
				
				
				
				pinn.setMemoryToDormantState();
				
				// Recognize(Retrieve) patterns from 
				
				sprintf(strMessage, "Noise=%i\n# \tHemming\t|CV-V|\t\t#Iter \t#Oscilating\n");  
				//	printf("# \tHemming\t|CV-V|\t\t#Iter \t#Oscilating\n");  
				printLog(fout, strMessage);  
				
				f=fopen(sFileName, "r");
				fgets(sPattern, 100, f);   /* first comment line */
				fgets(sPattern, 100, f);   /* first comment line */
				
				resetStats(&statHemming);
				resetStats(&statIter);

				for (i=0;i<M;i++)
				{
					if ( ! fgets(sPattern, N+1, f) )
						break;
					fgetc(f);
					pinn.string2BinaryVector(&V, N, sPattern);		
					pinn.add_noise(V, &Y0, N, nNoise); 
					
					pinn.retrieveFromMemory(Y0,&Y);
					int nHemming = pinn.computeHemming(Y,V,N);
					sprintf(strMessage, "%i \t%i \t%+.6f \t%i \t%i\n", 
						i, nHemming, pinn.m_E, pinn.m_nIter, pinn.m_nExcitedNow);  
					printLog(fout, strMessage);  

					updateStats(&statHemming, nHemming);
					updateStats(&statIter, pinn.m_nIter);
				}
				sprintf(strMessage, "%2i \t%.6f \t%.0f \t%.0f, %.0f\n\n", 
					nNoise, statHemming.dTotal/statHemming.nCount, statHemming.dMin, statHemming.dMax, statIter.dTotal/statIter.nCount);  
				printLog(fout, strMessage);  

				getchar();
				fclose (f);
			}

	}
	fclose (fout);
			

	delete Y0;		
	delete Y;		
//	delete sPattern;

	return;
} // main()




// PINNmemory.cpp: implementation of the CPINNmemory class.
//
//////////////////////////////////////////////////////////////////////

// #include "PINNmemory.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CPINNmemory::CPINNmemory()
{
	bCreated = false;
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
	}
}


void CPINNmemory::createMemory(int N)
{
	m_M = 0; // No pattern stored
	m_N = N;
    m_C = new float*[N];
	for (int i=0; i<N; i++)
	{
		m_C[i] = new float[N];
		for (int j=0; j<N; j++)
			m_C[i][j]=0;
	}
	m_Y = new signed char[N+1]; m_Y[N]='\0';
	m_S = new double[N];
	m_S0 = new double[N];
	m_aExcited = new int[N];
	m_nExcitedNow=0; 
	m_nExcitedLast=0;
	bCreated = true;
	m_D = 1.0; //0.15; // The most optimal desaturation value for most cases
//	m_D = 0; implies no desaturation (i.e. the same as m_D=1.0)
	m_nNoise = 0;
	m_dFade = 1.0;
	m_bCiiChanged = false;
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

#if 1 
	for(i=0;i<m_N;i++)
		for(j=0;j<m_N;j++)
			m_C[i][j]+=(V[i]-m_S[i])*(V[j]-m_S[j])/m_E; // Notice that m_C is symetric.

/*	for(i=0;i<m_N;i++)
		for(j=0;j<=i;j++)
		{
			m_C[i][j]+=(V[i]-m_S[i])*(V[j]-m_S[j])/m_E; 
			m_C[i][j] = m_C[j][i];
		}
*/
#else
// For dynamic desaturation:
	m_dFade = 1.0; 
    for(i=0;i<m_N;i++)
		for(j=0;j<m_N;j++)
			m_C[i][j] = m_dFade*m_C[i][j] + (V[i]-m_S[i])*(V[j]-m_S[j])/m_E; // Notice that m_C is symetric.
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
  m_dAveCii2 = X*X; 
  m_dAveCij2 = Y; 
  m_dAveAbsCij2 = Z;

 // Compute approximations of Average values of weights Cii and Cij by formula from Ref [2] 
  X=(double)m_M/(double)m_N; 
  Y=(double)m_M*(m_N-m_M)/(double)m_N/(double)m_N/(double)m_N;  
 // printf("vs   M/N  = %.5f,\tM(N-M)/N3 = %.5f \n", X,Y);

}     

void CPINNmemory::setDesaturationLevel(double dDesaturationLevel)
{
	m_D = dDesaturationLevel;
}
void CPINNmemory::desaturateMemory()
{
	if (m_D==0)
		return;
	if (m_bCiiChanged)
		return;

	for(int i=0;i<m_N;i++)
		m_C[i][i]=m_D*m_C[i][i];
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
		m_C[i][i]=m_C[i][i]/m_D;
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

int CPINNmemory::retrieveFromMemory(signed char *Y0, signed char **Yattractor)
{	

//make Y0 Yattractor - members!!
	int i;
	bool bAttractorAchieved;
	int nOscilatingNeurons;
	// Initialize state of the network to Y0, by setting the buffer of '+1' neurons
	for(m_nExcitedNow=0, i=0; i<m_N; i++)
	{
		m_S[i]=m_S0[i];
		m_Y[i] = Y0[i];
		if (Y0[i]>0) 
		    m_aExcited[m_nExcitedNow++]=+i+1; 
	}		

#if 1 //ASYNCHONOUS 
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
		nOscilatingNeurons=0; // where I use it ?
		for(int i=0; i<m_N; i++) // Update the state of the network. In "Parallel": i.e. go thru all neurons
		{
			if ( (m_S[i]<0) && (m_Y[i]>0) )
			{ 
				m_Y[i]=-1; 
				if (m_aExcited[m_nExcitedNow]==i+1)  // cycle check: see whether this neuron was excited in previous iteration
					nOscilatingNeurons++;	// if yes, update nOscilatingNeurons
				m_aExcited[m_nExcitedNow++]=-i-1; // update buffer
				}
			else if  ( (m_S[i]>=0) && (m_Y[i]<0) ) // another option: (m_S[i]>0)
			{
				m_Y[i]=+1; 
				if (m_aExcited[m_nExcitedNow]==-i-1)  // for check of a cycle 
					nOscilatingNeurons++;
				m_aExcited[m_nExcitedNow++]=+i+1;
			}
			else // 
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
	
#else
//	printf("ASYNCHRONOUS DYNAMICS");

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
	m_E -= m_D;

//	printf( "->  E = %.4f \n\n", m_E);
	printf( "\n");

	return 1;
}

bool CPINNmemory::saveC(char *filename)
{
	FILE *f=NULL;

	f = fopen(filename, "wb");
	if (f==NULL) 
		return false;

	fprintf(f, "%i %i %s\n", m_N, m_M, filename);
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
	fscanf(f, "%i %i %s\n", &N, &m_M, str);

	if (N!=m_N)
		;

	for (int i=0; i<m_N; i++)
		fread(m_C[i], m_N, sizeof(float), f);
	return true;
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
*** End of PINNmemory.cpp ***/

