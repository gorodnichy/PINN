// PINNmemory.h: interface for the CPINNmemory class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PINNMEMORY_H__04744B64_345D_499F_A829_08333811AE6B__INCLUDED_)
#define AFX_PINNMEMORY_H__04744B64_345D_499F_A829_08333811AE6B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


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
	void reset();
	void forgetAll() { reset(); };

	bool putInMemory(signed char *V);
	void setMemoryToDormantState(); // in which all neurons are in unexcited (-1) state
	int  retrieveFromMemory(signed char **Y0, signed char **Yattractor);
	int  associateFromMemory(signed char **Y0, signed char **Yattractor){
		setMemoryToDormantState(); 
		retrieveFromMemory(Y0, Yattractor);
	}


bool saveC(char *filename);
bool loadC(char *filename);

	void analyzeMemory(); // examine how filled the memory is 
	void setDesaturationLevel(double dDesaturationLevel);
	void desaturateMemory();
	void desaturateMemory(double dD);
	void restoreSaturationLevel();
	void addNoise(signed char **V);
// Utility functions
	void add_noise(signed char *Y, signed char **V, int N, int nNoise);
int computeHemming(signed char *Y, signed char *V, int N);
bool writeRandomBinaryVectors(int N, int M, char *sFileName, double dPercentageOfOnes, int nRndSeed);
void string2BinaryVector(signed char **V, int N, char *string);

	bool bCreated;
	int m_N; 
	int m_M; // number of stored prototypes in network's memory 
	float **m_C;	// matrix C of weightes (aka synaptic weights) of the network
	float m_D;
	float m_dFade;
	double	
		*m_S,	// S = CY, postsynaptic potentials (PSP) of the network
		*m_S0;
	double	m_E;	// E = || CY - Y ||
	signed char 
		*m_Y;	// state (aka potentials, or neurons, or neuron states) of the network

	int *m_aExcited; // buffer which stored indices of excited neurons
	int m_nExcitedNow, m_nExcitedLast;  // number of excited neurons at current and previous state of the network
	bool m_bCycleOccured; // redundant, as m_nExcitedNow>0 => m_bCycleOccured
	int m_nIter;
	double m_dAveCii, m_dAveCij2, m_dAveAbsCij2, m_dAveCii2;

	int m_nNoise;
	bool m_bCiiChanged;

};

#endif // !defined(AFX_PINNMEMORY_H__04744B64_345D_499F_A829_08333811AE6B__INCLUDED_)
