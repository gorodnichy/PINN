// PINNtest.cpp : 2003-2005
// Created by Dmitry Gorodnichy for FRinVideo batch mode tests.
// Takes commands (m/r/s) and lists of facial video-files (.avi) for M and R from scenario file(with faces)
//
 /// rand() <!--- removed  no error?!
#include "stdafx.h"
#include <stdio.h> 
#include <conio.h>
//#include <mmsystem.h>
#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <string.h>

#include "../PVS/PVSystem.h"


	FILE *m_fScenario, *m_fLog;

#define SHOW_OUTPUT 1
void printLog(char *str) 
{
	fprintf(m_fLog, "%s", str); 
#if SHOW_OUTPUT
	printf("%s", str); 
#endif
}

int selectFace(CPerceptualVisionSystem *pPvs);


void main( int argc, char *argv[])
{
	char m_strImagePath[200], sWhatToDo[20], sVideoClipName[220],
	str[900], str1[200], str2[200]; //, str3[10], str4[10], str5[10], 

	int m_nStoredFaces; // m_M; // 
	int nFramesM=0, nFramesR=0;
	int nFramesToUse = 10;

	int res, nClips=0;
//#define FACES_IN_DB 20
	int n10[20], n01[20], n11[20], n00[20], n12[20], n21[20];

	char cTmp, sTmp[10]; 
	char word[8][20];
	CFaceRecognizer fr;
//	CFaceDetector fd;
	CPerceptualVisionSystem pvs;

	char sVideoList[100];

    if( argc > 1 )
		strcpy(sVideoList, argv[1]);
	else
		strcpy(sVideoList, "cvg10-every7th.txt");
	
	m_fScenario = NULL;
	m_fScenario = fopen(sVideoList, "r");

	if (m_fScenario == NULL)
	{
		printf("File with list of videos: '%s' not found! ", sVideoList);
		return;
	}
	
/*** 1st line is the path ******************************************************/
	fgets(str, 800, m_fScenario); 
	sscanf(str, "%s", m_strImagePath);
//	fscanf(m_fScenario, "%s", m_strImagePath);

	printf("Face database: %s \n",  m_strImagePath);

// in 24x24 image
// char sWhatValuesToUse[5]; // "011" - don't use I, use dIx, dIy  
// char sHowToCrop[5]; // "22001" - cut from left and right 2 pixels , from top/bottom 0 pixels and scale by 1.
	// just remeber: The m_imgY will not be diplayed properly if m_nWidth is not muliple of 4 !

//	sHowToCrop[3]='0';


/*** 2nd line is the setting ******************************************************/

	char 
		sWhatValuesToUse[]="111", 
		sHowToCrop[]="00001";

	int nFaceIDs = 11;
	fgets(str, 800, m_fScenario);  
	sscanf(str, "nFaceIDs= %i sHowToCrop= %s sWhatValuesToUse= %s  ", &nFaceIDs, sHowToCrop, sWhatValuesToUse);

	fr.initialize(nFaceIDs, sWhatValuesToUse, sHowToCrop);

	fr.m_nTemporalFilter = 1; // 1 is normal
	fr.m_Sthresh = 0.0; // 0.0 is normal


	 struct tm *newtime;
     time_t long_time; 
		
     time( &long_time );                /* Get time as long integer. */
     newtime = localtime( &long_time ); /* Convert to local time. */
				
	sprintf (str, "log/_111-0000-%02i-%02i-%02i.log", 
			newtime->tm_mday, newtime->tm_hour, newtime->tm_min);

	m_fLog= fopen(str, "w");

	sprintf(str,  "Scenario file: %s. \nNetwork: N=%i (Cropped: %s, Values: %s), nIDs=%i. \n", 
		sVideoList, fr.m_N,	sHowToCrop, sWhatValuesToUse, fr.m_nIDs);
	printLog(str);
		


	fr.m_nFaceID = -1;
	
	strcpy(sWhatToDo, "nothing");
//	tStart = timeGetTime();


	while (	fgets(str, 800, m_fScenario) && 
		sscanf(str, "%s %s %s %s %s %s %s %s", word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7]) )
	{
		if (word[0][0]=='-')
		{
			strcpy(sWhatToDo, word[1]);
			sprintf(str, "** %s\n",  sWhatToDo);
			printLog(str);
			if (sWhatToDo[0] == 'q') 
				break; // finish (wrap up)
			continue;
		}
		

//		sprint(str, "C%s.nn", sVideoList);
//		sprintf(str, "C%s-%s-%i.nn", sWhatValuesToUse, sHowToCrop, fr.m_nIDs);
		switch (sWhatToDo[0]) 
		{

		case ('e'): strcpy(sWhatToDo, "do nothing"); 
			break;  // end
//		case ('s'): fr.m_pinn.saveC("C1.nn"); 			break; // (word[1]);   "C111-0000.nn")// save
//		case ('l'): fr.m_pinn.loadC("C1.nn"); 			break; // (word[1]);   "C111-0000.nn")// save


/*		case ('s'): fr.m_pinn.saveC(str); 
			break; // (word[1]);   "C111-0000.nn")// save
		case ('l'): 
			if (fr.m_pinn.loadC(str) == false)
			{
				printf("!!! Can't read %s   OR   wrong size m_N !!! ", str); 
				return;
			}
			break; // (word[1]);// load
*/
		case ('r'):
			n10[nClips]=0, n11[nClips]=0, n01[nClips]=0, n00[nClips]=0, n21[nClips]=0, n12[nClips]=0 ;
		case ('m'):
			sscanf(word[2],"%i", &fr.m_nFaceID);
			if (fr.m_nFaceID > fr.m_nIDs)
				fr.m_nFaceID=fr.m_nIDs;

/***  Open video file to Memorize or recognize faces *******************************************/
			sprintf(sVideoClipName, "%s%s", m_strImagePath, word[0]);

			if (!pvs.openVideoFile(sVideoClipName))
			{
				sprintf(str, "Cannot open '%s'",  sVideoClipName);
				printf("Cannot open '%s'",  sVideoClipName);
				printLog(str);
				continue; // go to next file
			}

			sprintf(str, "%s (as %i): #Frames: %i \n",  sVideoClipName, fr.m_nFaceID, pvs.m_nFramesTotal);
			printLog(str);

			pvs.playVideo();

/*** Get image from video  **********************************************************/
						
						
			sscanf(word[4], "%i", &nFramesToUse);

			int M1=0, M2=0;
			while (pvs.m_nFrame < pvs.m_nFramesTotal) // go to next video clip
			{
				pvs.initializeNextFrame();
				sscanf(word[4], "%i", &nFramesToUse);
				
/*** Detect face-looking regions in every Nth frame **********************************************************/
				if (pvs.m_nFrame % nFramesToUse != 0)
					continue;
				
				int nFacesDetected = pvs.m_fd.detectFacesIn(&pvs.m_imgBW);
				if (nFacesDetected == 0) 
				{
					printf("%4i: 0 faces\n", pvs.m_nFrame); continue;
				}
				else if (nFacesDetected > 1) 
					M2++; 
				else
					M1++; 
				printf("%4i: %i FACE(S) \n", pvs.m_nFrame, nFacesDetected); 
// Feb 2005 				continue;

/*** Select the best face-looking region **********************************************************/

				int nFaceSelected = 0; // take the fisrt one

#if 0 // USE_MOTION_COLOUR don't work in AVI-ed ?....
				nFaceSelected = selectFace(&pvs);
				if (nFaceSelected <0)
					continue; // to next frame
#endif	

/*** Load FR with vector obtained from face **********************************************************/



//				sprintf(str, "%4io", nFaceRotation); printLog(str);


				int nFaceRotation;
/*** Memorize **********************************************************/
				if (sWhatToDo[0]=='m') 
				{					

					nFaceRotation = fr.loadFaceVector(&pvs.m_imgBW, pvs.m_fd.getLocationOfFace(nFaceSelected)); 

					fr.setIDto1(fr.m_nFaceID);
					fr.m_pinn.putInMemory(fr.m_Y);

					sprintf(str, "%4i M=%i %.2f \tE=%.0f  \n", 
						pvs.m_nFrame, fr.m_pinn.m_M, (float)fr.m_pinn.m_M*100.0/fr.m_pinn.m_N,
						fr.m_pinn.m_E);
					
					printLog(str);
					nFramesM++;
				}
/*** Recognize **********************************************************/
				else 
				{
int nDegrees = 0;
int nBest;
double dSbest;
// for(int ii=0; ii<2;ii++, nDegrees=FR_AUTO_ROT_DETECTION)
{

//			nFaceRotation = fr.loadRotatedFaceVector(&pvs.m_imgBW, pvs.m_fd.getLocationOfFace(nFaceSelected), nDegrees); 
			nFaceRotation = fr.loadRotatedFaceVector(&pvs.m_imgBW, pvs.m_fd.getLocationOfFace(nFaceSelected), 0); 
			nFaceRotation = fr.loadRotatedFaceVector(&pvs.m_imgBW, pvs.m_fd.getLocationOfFace(nFaceSelected), FR_AUTO_ROT_DETECTION); 
					
/*** Desaturate ! **********************************************************/


/*	 March
				sscanf(word[5], "%i", &fr.m_nTemporalFilter);
					int nnn=999;
					sscanf(word[6], "%i", &nnn); 
					fr.m_Sthresh = (double)nnn/100.0;

					fr.m_pinn.m_D = 0.15;	// -0.05
					sscanf(word[7], "%i", &nnn); 
					fr.m_pinn.m_D = (double)nnn/100.0;
					fr.m_pinn.desaturateMemory();
//					printf ("D=%.2f !!", fr.m_pinn.m_D);

*/

/// START HERE .... WHERE DO I DESATURATE ?!!!!

/*		nUEDA = 0;
		if (nUEDA)
		{
			fr.pinn.m_D = 0.98; // 98;	
			for(i=0;i<pinn.m_N;i++)
				fr.pinn.m_C[i][i]=fr.pinn.m_C[i][i]-fr.pinn.m_D;
		}
		else
		{
			fr.pinn.m_D = 0.15; // 0.14; // 0.98;		 
			for(i=0;i<pinn.m_N;i++)
				fr.pinn.m_C[i][i]=fr.pinn.m_C[i][i]*fr.pinn.m_D;//(1-pinn.m_D);
		}

		sprintf(strMessage, "UEDA=%i, ASYNCHONOUS=%i, D=%.2f\n", nUEDA, ASYNCHONOUS, fr.pinn.m_D);  
		printLog(fout, strMessage);
				
*/




// 1. conventional
// 2. take the maximum
// 3. take maximum but only if  it is >= -0.5

// 4. use (S[last] as S[0] new
// 5a,b. take average over 3,4,frames
// 6a,b,c take average over 3,4,frames but only if it is "." '+'

					fr.m_pinn.setMemoryToDormantState();

					int t,i, j, nHemming;

#if 0 
// For recognition: 
// 1) prefilter Y by multiplying it by Y = CY ( the same as to use S instead of Y as input for NN

// TEST_S_VS_Y does not make it better!...
					for(i=0;i<fr.m_pinn.m_N;i++)
					{
						double S;
						for(j=0, S=0;j<fr.m_pinn.m_N;j++)
							S += fr.m_pinn.m_C[i][j] * fr.m_S[i]; // make sure S[i] is normalized to -1 +1
						fr.m_Y[i] = S > 0 ? +1 : -1;
					}

	for(nHemming=0,i=0;i<fr.m_pinn.m_N;i++) 
		if (fr.m_Y[i]*fr.m_S[i]<0) 
			nHemming++;
	sprintf(str, "Ynew: %3i  \t", nHemming);
	printLog(str);

#endif

					fr.m_pinn.retrieveFromMemory(&fr.m_Y, &fr.m_Y);
										
/*** Analyze results ******************************************************/
/*** Analyze results ******************************************************/
/*** Analyze results ******************************************************/

					int maxS = -999, imax=999; 

					strcpy(str2,"");
					strcpy(str1,"");
				
					int nPosY=0;

/*** Find the averages of S over time:
Current results (getSofID(i)) are stored in fr.m_SofID[0][i], the past results are shifted. 
	The average is in fr.m_SofID[fr.m_nTemporalFilter][i]				*****/
					
//					fr.addSToTemporalFilter();

					for(i=0; i<fr.m_nIDs; i++)
					{
/*** Find the averages of S over time:
Current results (getSofID(i)) are stored in fr.m_SofID[0][i], the past results are shifted. 
	The average is in fr.m_SofID[fr.m_nTemporalFilter][i]				*****/


						fr.m_SofID[fr.m_nTemporalFilter][i] = 0;
						for(t=0; t<fr.m_nTemporalFilter-1; t++)
						{
							fr.m_SofID[t+1][i] = fr.m_SofID[t][i];
							fr.m_SofID[fr.m_nTemporalFilter][i] += fr.m_SofID[t][i];
						}
						fr.m_SofID[0][i] = fr.getSofID(i);

						fr.m_SofID[fr.m_nTemporalFilter][i] += fr.m_SofID[0][i];
						fr.m_SofID[fr.m_nTemporalFilter][i] /=fr.m_nTemporalFilter;

/*** Find highest S *****************/

						if (fr.m_SofID[0][i] > maxS)
						{
							maxS = fr.m_SofID[0][i];
							imax = i;
						}

//						sprintf (word[1], "%+1i", (i+1)*fr.getYofID(i));

//						if (fr.m_SofID[fr.m_nTemporalFilter][i] > fr.m_Sthresh) // 0
						if (fr.m_SofID[0][i] > fr.m_Sthresh) // 0
						{
							sprintf (word[2], " %+3.1f", fr.getSofID(i));
							nPosY++;
						}
						else
							sprintf (word[2], " %+3.1f", fr.getSofID(i));
//						strcat(str2, word[1]);
						strcat(str1, word[2]);

					}

} // for (nDegrees)

// Statistics

// aa)
					if (imax == fr.m_nFaceID) // i.e. Greatest wins !
// a)				
//					if (fr.m_SofID[0][fr.m_nFaceID] > 0) //  same as if (fr.getYofID(fr.m_nFaceID) > 0)
// b)
////					if (fr.m_SofID[0][fr.m_nFaceID] > fr.m_Sthresh) 
// d)
//					if (fr.m_SofID[fr.m_nTemporalFilter][fr.m_nFaceID] > fr.m_Sthresh) 
					{
						if (nPosY>1) // many exceeded Sthresh
						{	cTmp='+';n11[nClips]++; }
						else if  (nPosY==1)
						{	cTmp='*';n10[nClips]++; } // best
						else // if  (nPosY==0) // none exceeded Sthresh
						{	cTmp='x';n12[nClips]++; }
//						{	cTmp='x';n10[nClips]++; }
					}
					else
					{
						if (nPosY==0) // none exceeded Sthresh
						{	cTmp='.';n00[nClips]++; }
						else if  (nPosY==1)
						{	cTmp=' ';n01[nClips]++; } // worst
						else // many exceeded Sthresh
						{	cTmp='_';n21[nClips]++; }
//						{	cTmp='_';n01[nClips]++; }
					}

					sprintf(str, "%c %4i Ssum[%i][%i]=%4.2f  %s  ( %4i %5.0f )\n", cTmp, pvs.m_nFrame, 
						fr.m_nTemporalFilter, imax, fr.m_SofID[fr.m_nTemporalFilter][imax], 
						// i.e. what's the average S (over T frames) of the currently winning neuron
						str1, fr.m_pinn.m_nIter, fr.m_pinn.m_E);
					printLog(str); 
					nFramesR++;


				} // Recognition 
			}// clip finished

			sprintf(str, "%s (ID: %i). Frames: %i. %i/%i times detected 1/2 face(s)  \n",  
				sVideoClipName, fr.m_nFaceID, pvs.m_nFramesTotal, M1, M2);
			printLog(str);


			if (sWhatToDo[0]=='r') 
			{
				sprintf(str1, "    Result: %4i %4i  %4i  %4i\n", n10[nClips], n11[nClips], n01[nClips], n00[nClips] );
				printLog(str1);
				nClips++;
				n10[nClips]=0, n11[nClips]=0, n01[nClips]=0, n00[nClips]=0, n21[nClips]=0, n12[nClips]=0; 
			}
			else 
			{
				
				fr.m_pinn.analyzeMemory();
				
				sprintf(str, "N=%i, M=%i, \t<Cii>^2=%.4f, <Cij^2>=%.4f\n", 
					fr.m_pinn.m_N,fr.m_pinn.m_M, fr.m_pinn.m_dAveCii2, fr.m_pinn.m_dAveCij2);  
				printLog(str);
				
			}
			
		} // if process clip (m or r) // case

			
		sprintf(str, "         %s finished!\n\n",  sWhatToDo);
		printLog(str);
	}

/*
			for (int i=0;i<512;i++)
			{
				sprintf(str, " %6i-%i ", fr.m_I3x3[i],  i);
				printLog(str);
				if (i%8==0)
					printLog("\n");
			}
*/

//	tNowR = timeGetTime() - tNowM;
				
	sprintf(str, "%i clips, %i frames in training. %i in testing. \n", 
		nClips, nFramesM,  nFramesR);
	printLog(str);
	sprintf(str, "Statistics:   10   11   01   00  |  right, but all<S |  wrong, but many>S \n");
	printLog(str);
	for(int j=0;j<nClips;j++)
	{
		sprintf(str, "ID %i &\t %4i & %4i & %4i & %4i \t \\\\ \\hline  % %4i  %4i %4i\n",  j, n10[j], n11[j], n01[j], n00[j], n12[j], n21[j]);
		printLog(str);
		n10[nClips]+=n10[j];
		n00[nClips]+=n00[j];
		n01[nClips]+=n01[j];
		n11[nClips]+=n11[j];
		n12[nClips]+=n12[j];
		n21[nClips]+=n21[j];
	}
	sprintf(str, "Total:&\t  %4i & %4i & %4i & %4i \t \\\\ \\hline  % %4i  %4i %4i\n",  n10[nClips], n11[nClips], n01[nClips], n00[nClips], n12[nClips], n21[nClips]);
	printLog(str);

	sprintf(str,  "Network: N=%i (%s-%s), nIDs=%i || D=%4.2f, T=%i, S0=%4.2f  \n  Scenario file: %s. \n", 
		fr.m_N,	sHowToCrop, sWhatValuesToUse, 
		fr.m_nIDs, fr.m_pinn.m_D, fr.m_nTemporalFilter, fr.m_Sthresh, sVideoList );
	printLog(str);
		


	Beep(440, 1000);
	fclose (m_fLog);
	fclose (m_fScenario);

//	pvs.destroy();
//	facerec.destroy();

//	printf("Done! Enter 'q' to finish ");
//	scanf("%s",str);
}



int selectFace(CPerceptualVisionSystem *pPVS)
{

	bool bFaceFound;
	CvRect *rectFaceDetected, rectAroundEyes;
	CvPoint ptLeftEye, ptRightEye, pt00, pt11; 
	int nEyeSize, nResolution;

	pPVS->m_chanColour.updateImages(&pPVS->m_imgIn);
	pPVS->m_chanMotion.updateImages(&pPVS->m_imgIn);
	pPVS->m_chanMotion.compute_dI();
	pPVS->m_chanMotion.compute_FG();


	int nFacesDetected = pPVS->m_fd.m_nFacesDetected;

	for(int nFace=0; nFace<nFacesDetected; nFace++)
//	for(int nFace=0; nFace<1; nFace++)
	{
//		rectFace = pPVS->m_fd.getLocationOfFace(nFace);
		CFace::fdRect2Eyes(pPVS->m_fd.getLocationOfFace(nFace), &ptLeftEye, &ptRightEye, &nEyeSize, &nResolution);

		CvPoint pt00, pt11; 
		 
		pt00=cvPoint(ptLeftEye.x, ptLeftEye.y-nEyeSize/2),	
		pt11=cvPoint(ptRightEye.x, ptRightEye.y+nEyeSize);

		rectAroundEyes=cvRect(pt00.x, pt00.y, pt11.x-pt00.x, pt11.y-pt00.y); 

		////////////////////////////////

		PVI_BLOB blobC1, blobC2, blobM1, blobM2;

		bool resC1 = isThereAnything(&pPVS->m_chanColour.m_imbSkinYCrCb, &rectAroundEyes, &blobC1);
		bool resC2 = isThereAnything(&pPVS->m_chanColour.m_imbSkinUCS, &rectAroundEyes, &blobC2);
		bool resM1 = isThereAnything(&pPVS->m_chanMotion.m_imbFG, &rectAroundEyes, &blobM1);
		bool resM2 = isThereAnything(&pPVS->m_chanMotion.m_imb_dI, &rectAroundEyes, &blobM2);
						
		if ( ! resC1 && ! resC2) 		// ( (blobC1.N > rect0->width) && (blobC1.N > rect0->width) ))
			continue;
//		if ( !resM2 )
//			continue;
		bFaceFound = true;
		break; // dont' search for other faces if one is found
	}

	// A uzh kakoj iz nih m_face[f] use decide sam	
	if (bFaceFound)
		return nFace-1;
	else
		return -1;
}



