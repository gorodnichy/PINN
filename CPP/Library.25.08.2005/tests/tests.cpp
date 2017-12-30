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

# include "tests.h"

//extern vector<int> deletedNeurons; (used for IJCNN paper check)


// calculates distribution of diagonal(0)/nondiagonal(1)/all_shifted(2) weight elements
// interval [-halfR, halfR], 2*halfC+1 nodes are used
// arg has x values
// func has y = f(x) values
// all_shifted means Cij - delta(i,j)*m/n
void StatMaker::calculateDistribution(CellularNet *net,
                double halfR, int halfC, Vector *arg, Vector *func, double *mean, double *disp, int type)
{
    int count = 2*halfC+1, i;
    double delta = 2*halfR/count, sum = 0, sumSq = 0;
    arg->assign(count, 0);
    func->assign(count, 0);
    for(i=0; i<count; i++)
        (*arg)[i] = (i-halfC)*delta;

    int sampleSize = 0;
    for(i=0; i<net->dim(); i++)
    {
        vector<int> &maski = net->mask[i];
        Vector &weightsi = net->weights[i];
        for(int j=0; j<maski.size(); j++)
        {
            if((type==0&&maski[j]!=i) || (type==1&&maski[j]==i))
                continue;
            double weight = weightsi[j];
            if(type==2 && maski[j]==i)
                weight -= net->numStored()/net->dim();

            sum += weight;
            sumSq += weight*weight;
            int index = weight/delta+halfC;
            if(index < 0)
                index = 0;
            if(index >= count)
                index = count-1;
            (*func)[index]++;
            sampleSize++;
        }
    }
    if(sampleSize == 0)
    {
        *mean = *disp = -1;
        return;
    }
    *mean = sum/sampleSize;
    *disp = sumSq/sampleSize - sum*sum/sampleSize/sampleSize;
    double factor = sampleSize*delta;
    for(i=0; i<count; i++)
        (*func)[i] /= factor;
}


// calculates correlation between weight and data elements
// pairs of elements with equal indices are used
// ?????????????????????????????????????????????????????????
double StatMaker::getCorrCoeff(CellularNet *net, IOData *data)
{
    assertion(net->dim() == data->inDim(), "[StatMaker::getCorrCoeff] net->dim() != data->dim()");
    assertion(data->outDim() == 0, "[StatMaker::getCorrCoeff] data->outDim() != 0");
    assertion(net->numStored() <= data->count(), "[StatMaker::getCorrCoeff] net->numStored() > data->size()");
    double sumWD = 0, sumW = 0, sumD = 0, sumWW = 0, sumDD = 0;
    int nWD = 0, nW = 0, nD = 0, i;
    Vector datad(net->dim()), outDatum(0);
    for(i=0; i<net->dim(); i++)
    {
        vector<int> &maski = net->mask[i];
        Vector &weightsi = net->weights[i];
        data->setToBeginning();
        for(int d=0; d<net->numStored(); d++)
        {
            data->getNext(&datad, &outDatum);
            for(int j=0; j<maski.size(); j++)
            {
                double weight = weightsi[j];
                double data = datad[maski[j]];
                sumWD += weight*data;
                nWD++;

                sumD += data;
                sumDD += data*data;
                nD++;

                if(d == 0)
                {
                    sumW += weight;
                    sumWW += weight*weight;
                    nW++;
                }
            }
        }
    }
    cout << " // " << net->numStored() << " " << sumWD/nWD;
    exit(0);  // ???????
    return (sumWD/nWD-(sumW/nW)*(sumD/nD))
                /sqrt(sumWW/nW-sumW*sumW/nW/nW)
                /sqrt(sumDD/nD-sumD*sumD/nD/nD);
}

//-------------------------- AdaptiveCellularSubj ------------------------------

void AdaptiveCellularSubj::doTask(Parameters params, Properties *prop)
{
    int dim = params.getInt(PARAM_NET_DIM);
    setDim(dim);

    int nStored = params.getInt("numStored");
    RandomBipolarRAMData data(dim, 0, nStored);

    Vector in(dim), out(0);
    Properties trainProp, testProp;

    float connDegreeStep = params.getFloat("connDegreeStep"), transitionConnectivity = 1;
    int numTestEpochs = params.getInt("numTestEpochs"), transitionRA = 0;

    // train full network to find fullRA
    setParameter(PARAM_CONNECTIVITY_DEG, 1);
    clear();
    for(int d=0; d<nStored; d++)
    {
        data.get(&in, &out, d);
        train(&in, &trainProp);
    }
    doEndTraining(&trainProp);
    int fullRA = getRAttraction(numTestEpochs, false, // singleIter
        this, &data, &testProp);

    // increase connDegree till non-zero RA is reached
    // skip this stage if !fullRA
    for(float connDegree = connDegreeStep; connDegree<=1 && fullRA; connDegree+=connDegreeStep)
    {
        // set connectivity & train
        setParameter(PARAM_CONNECTIVITY_DEG, connDegree);
        clear();
        for(int d=0; d<nStored; d++)
        {
            data.get(&in, &out, d);
            train(&in, &trainProp);
        }
        doEndTraining(&trainProp);

        // testing
        int rAttraction = getRAttraction(params.getInt("numTestEpochs"), false, // singleIter
            this, &data, &testProp);

        if(rAttraction > 0)
        {
            transitionConnectivity = connDegree;
            transitionRA = rAttraction;
            break;
        }
    }

    // return results
    (*prop)["transConn"] = transitionConnectivity;
    (*prop)["transRA"] = transitionRA;
    (*prop)["fullRA"] = fullRA;
    (*prop)["transDegree"] = fullRA? transitionRA*1./fullRA : 0;
}

//------------------------------------------------------------------------------

// creates new(!) network according to "params"
// must be deleted after usage
AssociativeNet* createNetwork(Parameters params)
{
    AssociativeNet *net;

    try
    {
        string netType = params.getString(PARAM_NET_TYPE);

        if(!netType.compare(PARAM_NT_CELLPROJ))
            net = new CellProjectiveNet();
        else if(!netType.compare(PARAM_NT_PSEUDOINVERSE))
            net = new PseudoInverseNet(params);
        else if(!netType.compare(PARAM_NT_FULLPROJ))
            net = new FullProjectiveNet(params.getInt(PARAM_NET_DIM));
        else if(!netType.compare(PARAM_NT_ADAPTIVE))
            net = new AdaptiveCellularNet(params);
        else if(!netType.compare(PARAM_NT_HEBBIAN))
            net = new HebbianCellularNet(params);
        else if(!netType.compare(PARAM_NT_DELTACELL))
            net = new DeltaCellularNet(params);
        else if(!netType.compare(PARAM_NT_SMALLWORLD))
            net = new SmallWorldNet(params);
        else
            throw GenException("[createNetwork] unknown netType = " + netType);

        // if the network has fixed architecture
        if(!netType.compare(PARAM_NT_CELLPROJ) ||
            !netType.compare(PARAM_NT_PSEUDOINVERSE) ||
            !netType.compare(PARAM_NT_HEBBIAN) ||
            !netType.compare(PARAM_NT_DELTACELL))
        {
            string topology = params.getString(PARAM_TOPOLOGY_TYPE);

            if(!topology.compare(PARAM_TP_1D))
                ((CellularNet*)net)->setLocal1DArchitecture(params.getInt(PARAM_NET_DIM),
                params.getInt(PARAM_CONN_R));
            else if(!topology.compare(PARAM_TP_2D))
                ((CellularNet*)net)->setLocal2DArchitecture(params.getInt(PARAM_TP_2D_SIZEY),
                params.getInt(PARAM_TP_2D_SIZEX), params.getInt(PARAM_CONN_R));
            else if(!topology.compare(PARAM_TP_RANDOM))
                ((CellularNet*)net)->setRandomArchitecture(params.getInt(PARAM_NET_DIM),
                params.getFloat(PARAM_CONNECTIVITY_DEG));
            else
                throw GenException("[createNetwork] unknown topology = " + topology);
        }

        net->setParameter(PARAM_DESAT_COEFF, params.getFloat(PARAM_DESAT_COEFF));
    }
    catch(GenException &exc)
	{
		throw GenException("[createNetwork]", exc);
	}

    return net;
}


void generalTestingScript(Parameters params)
{
    string reportFile = params.getString("reportFile");
    saveString(params.toString(), reportFile);

    //-------------------------- Network settings -----------------------------
    AssociativeNet *net = createNetwork(params);

    //---------------------------- Data settings -----------------------------
    RAMData *data = 0;
    if(params.getInt("testNum") != 5)
        data = new TextRAMData(params.getString("dataFile"));

    //---------------------------- Train and save ----------------------------
    if(params.getBool("onlyTrainAndSave"))
    {
        trainAndSave(net, data, params.getInt("numStored"), params.getString("reportFile") + toString(params.getInt("numStored")));
        return;
    }

    //---------------------------- Test settings -----------------------------
    assertion(params.getVarCount()==1, "[generalTestingScript] there must be exactly one variable parameter in ini-file");
    string testVariableName = params.getVarName(0);
    Parameter testVariableValue = params.getParameter(testVariableName);
    bool adjustBeforeTesting = params.getBool("adjustBeforeTesting");
    bool needSaveWeights = params.getBool("needSaveWeights");
    bool singleIter = params.getBool("singleIter");
    /*if(params.getInt("testNum") == 0 && !params.getString("netType").compare("fullprojective"))
    {
        ((FullProjectiveNet*)net)->retrainType = params.getInt("retrainType");
        testRetrain(params.getInt("numTestEpochs"),(FullProjectiveNet*)net,data,params.getInt("numStored"),params.getFloat("eraseValue"),params.getInt("eraseType"),params.getInt("retrainCount"),params.getInt("multCount"),reportName);
    }
    else */
    if(params.getInt("testNum") == 1)
    {
        assertion(!testVariableName.compare("numStored"), "[generalTestingScript] \"numStored\" must be a variable for test1");
        test1(params.getInt("numTestEpochs"), net, data, testVariableValue, reportFile, adjustBeforeTesting,
        needSaveWeights, singleIter);
    }
    /*else if(params.getInt("testNum") == 2)
        test2(params.getInt("numTestEpochs"), net, data, params.getString("parameterName"), parameterValue, reportName, adjustBeforeTesting, needSaveWeights, singleIter);*/
    else if(params.getInt("testNum") == 4)
        test4(params.getInt("numTestEpochs"), net, data,params.getInt("numStored"), testVariableName,
        testVariableValue, reportFile, adjustBeforeTesting, needSaveWeights, singleIter);
    else if(params.getInt("testNum") == 5)
        test5(net, params, params.getBool("contiguousNoise"));
    else
        throw GenException("unknown testNum");

    delete net;
    delete data;
}


// Performs numTestEpochs runs for part of data (unit.numStored) with random initialH
// returns average final H and auxilary info in prop
// errorPortion is set to num_of_errors/numTestEpochs
// errorDistribution, if specified, will contain average per-neuron errors
double getAveHamming(int numTestEpochs, AssociativeUnit *unit, IOData *data, int initialH, bool singleIter, double *errorPortion, Properties *prop, Vector *errorDistribution)
{
	assertion(unit->numStored() != 0, "[getAveHamming] empty unit");
	assertion(unit->dim() == data->inDim(), "[getAveHamming] unit->dim() != data->dim()");
    assertion(data->inDim() == data->outDim(), "[getAveHamming] data->inDim() != data->outDim()");
	assertion(unit->numStored()<=data->count(), "[getAveHamming] |net->stored| > |data|");

    initRNG();

    int checkCount = unit->numStored(); //<data->size()? unit->numStored() : data->size();

	int dim = unit->dim(), sumH = 0, sumIter = 0, nConverged = 0, nStable = 0, nError = 0;
    Vector in(dim), out(dim), inDatum(dim), outDatum(dim);
	bool isDynamicAttr;

	if(errorDistribution != 0)
		errorDistribution->assign(unit->dim(), 0);

    int nRun = 0;
	for(int t=0; t<numTestEpochs; t++)
	{
        data->setToBeginning();
		for(int d=0; d<checkCount; d++)
		{
            nRun++;
            data->getNext(&inDatum, &outDatum);
			in.setBipolarNoise(&inDatum, initialH);

			int nIter = unit->converge(&in, &out, &isDynamicAttr, singleIter);

            if(nIter > 0)
			{
				nConverged++;
				sumIter += nIter;
				nStable += (isDynamicAttr? 0 : 1);
			}
            int hamming = inDatum.hammingDistance(&out);

            # ifdef EXIT_AVEH_IF_ERROR
            if(!H_THRESHOLD && hamming) // skip the rest of vectors
            {
                cout << "[getAveHamming] error at h = " << initialH << "   ";
                if(prop != 0)
                {
                    (*prop)["<H_res>"] = NOT_A_NUMBER;
                    (*prop)["errorPortion"] = NOT_A_NUMBER;
                    (*prop)["converged"] = NOT_A_NUMBER;
                    (*prop)["<iter>"] = NOT_A_NUMBER;
                    (*prop)["stable"] = NOT_A_NUMBER;
                }
                return dim;
            }
            # endif
            
			sumH += hamming;
            if(hamming)
                nError++;
			if(errorDistribution != 0)
				for(int e=0; e<dim; e++)
					if(fabs(inDatum[e]-out[e])>0.5)
						(*errorDistribution)[e] += 1;
		}
	}

    double aveH = ((double)sumH)/nRun;
    double errors = ((double)nError)/nRun;
    if(errorPortion)
        *errorPortion = errors;

	if(errorDistribution != 0)
    {
        errorDistribution->setSize(dim);
        for(int e=0; e<dim; e++)
            (*errorDistribution)[e] /= nRun;
    }

    if(prop != 0)
    {
	    (*prop)["<H_res>"] = aveH;
        (*prop)["errorPortion"] = errors;
        (*prop)["converged"] = ((double)nConverged)/nRun;
        (*prop)["<iter>"] = nConverged>0? ((double)sumIter)/nConverged : -1;
        (*prop)["stable"] = nConverged>0? ((double)nStable)/nConverged : -1;
    }

	return aveH; 
}


// finds the highest h such as      probCriterion? errorPortion : aveH <= H_THRESHOLD
int getRAttraction(int numTestEpochs, bool singleIter, AssociativeUnit *unit, IOData *data, Properties *prop, bool probCriterion)
{
    map<int,Properties> testProp;
    int lowerH = 0, upperH = unit->dim();

    // bisection
    while(lowerH != upperH-1)
    {
        int h = (lowerH+upperH)/2;
        printf("\r%s %3.d  ", "[getRAttraction] checking h = ", h);
        double errorPortion;
        double aveH = getAveHamming(numTestEpochs, unit, data, h, singleIter, &errorPortion, &testProp[h]);
        double resValue = probCriterion? errorPortion : aveH;
        if(resValue <= H_THRESHOLD)
            lowerH = h;
        else
            upperH = h;
    }

    int resH = lowerH;

    // check h-values lower than resH till CHECK_POINT_COUNT values in a row satisfy aveH <= H_THRESHOLD
    deleteLine();
    if(CHECK_POINT_COUNT)
    {
        int nCheckedPoint = 0, h = resH-1;
        while(nCheckedPoint<CHECK_POINT_COUNT && h>0)
        {
            printf("\r%s %3.d  ", "[getRAttraction] varification, h = ", h);
            double errorPortion;
            double aveH = getAveHamming(numTestEpochs, unit, data, h, singleIter, &errorPortion, &testProp[h]);
            double resValue = probCriterion? errorPortion : aveH;
            if(resValue <= H_THRESHOLD)
                nCheckedPoint++;
            else
            {
                log(string("[getRAttraction] varifying failed at ")+toString(h), unit->getID());
                resH = h-1;
                nCheckedPoint = 0;
            }
            h--;
            cout.flush();
        }
    }

    Properties &resProp = testProp[resH? resH : 1];
    for(Properties::iterator iter = resProp.begin(); iter!=resProp.end(); iter++)
        (*prop)[iter->first] = iter->second;

    deleteLine();
    printf("%s %d\n", "[getRAttraction] Attraction Radius = ", resH);

    (*prop)["attrR"] = resH;
    return resH;
}

// Fills the net calculating rAttraction at each step
// rAttraction is a maximal initial Hamming resulting in <final Hemming> less or equal to H_THRESHOLD
// final Hamming is averaged over numTestEpochs runs for all stored images and
// each sequential initial Hamming value
// uses CHECK_POINT_COUNT
void test1(int numTestEpochs, AssociativeUnit *unit, RAMData *data, Parameter numStoredScale,
    string reportFile, bool adjustBeforeTesting, bool needSave, bool singleIter)
{
    assertion(data->inDim() == unit->dim(), "[test1] data->inDim() != unit->dim()");
    assertion(data->inDim() == data->outDim(), "[test1] data->inDim() != data->outDim()");
    Vector in(data->inDim()), out(data->outDim());

	Properties trainProp, testProp;
	bool firstRecord = true;
    assertion(numStoredScale.isVariable(), "[test1] numStoredScale not a variable");
    numStoredScale.init();
    numStoredScale.setNext(); // goto first value

    unit->clear();
    for(int i=0; i<data->count(); i++)
	{
        data->get(&in, &out, i);
		unit->train(&in, &trainProp);
		if(unit->numStored() < numStoredScale.getInt())
			continue;

		cout << "\nTest 1, numStored = " << unit->numStored() << endl;
        cout.flush();

        unit->doEndTraining(&trainProp);

        if(adjustBeforeTesting)
            unit->adjustBeforeTesting();
        getRAttraction(numTestEpochs, singleIter, unit, data, &testProp);

        if(firstRecord)
        {
            string str =
                string("\n--- --- --- --- --- --- TEST_1 --- --- --- --- --- ---\n")
                + "H_THRESHOLD = " + toString(H_THRESHOLD,3) + " numTestEpochs = " + toString(numTestEpochs)
                + " SingleIter = " + toStringB(singleIter) + "\n"
                + unit->getDescription() + "\n" + data->getDescription();
            saveString(str, reportFile);
            saveProperties(trainProp, testProp, true, REPORT_WIDTH_VAL, reportFile);
            firstRecord = false;
        }
        else
            saveProperties(trainProp, testProp, false, REPORT_WIDTH_VAL, reportFile);
            
        if(needSave)
            unit->saveDetails(reportFile + "_" + toString(unit->numStored()));

        if(unit->numStored() >= numStoredScale.getInt())
            if(!numStoredScale.setNext())
                break;
	}
	saveString(string("\n"), reportFile);
}

/*
// Calculates net capacity for each value of "parameterName"
// rAttraction is a maximal initial Hamming resulting in <final Hemming> less or equal to H_THRESHOLD
// final Hamming is averaged over numTestEpochs runs for all stored images       // each stored image // and
// each sequential initial Hamming value
// Net capacity is a maximum number of images stored so as rAttraction is still
// greater then some fixed value (R_ATTRACTION)
// uses START_PORTION, START_SHIFT, CHECK_POINT_COUNT
void test2(int numTestEpochs, AssociativeUnit *unit, Data *data, string parameterName, Parameter parameterValue, string reportFile,
            bool adjustBeforeTesting, bool needSave, bool singleIter)
{
	map<int,Properties> trainProp, testProp;
	int lastI = 0;
	bool firstRecord = true;
    assertion(parameterValue.isVariable(), "[test2] parameterScale not a variable");
    parameterValue.init();
	while(parameterValue.setNext())
	{
		unit->setParameter(parameterName, parameterValue.getFloat());
        unit->clear();
		cout << endl << "Test 2, " << parameterName << " = " << unit->getParameter(parameterName) << " checking image # ";
		cout.flush();

        bool mayNeedRetest = true;
		int nCheckPoint = 0;
		int startI = ((double)lastI)*START_PORTION - START_SHIFT - CHECK_POINT_COUNT, i;
        if(startI <= 0)
		{
			startI = 0;
			mayNeedRetest = false;
		}
		if(startI > unit->dim()-1)
			startI = unit->dim()-1;

		for(i=0; i<startI; i++)
		{
			unit->train(&(*data)[i]);
		}
		for(i=startI; i<data->size(); i++)
		{
            int nStored = unit->numStored()+1;
			unit->train(&(*data)[i],&trainProp[nStored]);
			cout << unit->numStored() << " ";
			cout.flush();
            double errorPortion;
            if(adjustBeforeTesting)
                unit->adjustBeforeTesting();
            double aveH = getAveHamming(numTestEpochs,unit,data,R_ATTRACTION,singleIter,&errorPortion,&testProp[nStored]);
			if(aveH <= H_THRESHOLD)
					nCheckPoint++;
				else
					break;
        }

		if(nCheckPoint<CHECK_POINT_COUNT && mayNeedRetest)
		{
			log(string("[test2] retesting from ")+toString(unit->numStored()),unit->getName());
            cout << " retesting ";
            cout.flush();
			unit->clear();
			for(i=0; i<data->size(); i++)
			{
                int nStored = unit->numStored()+1;
				unit->train(&(*data)[i],&trainProp[nStored]);
				cout << unit->numStored() << " ";
				cout.flush();
                double errorPortion;
                if(adjustBeforeTesting)
                    unit->adjustBeforeTesting();
                double aveH = getAveHamming(numTestEpochs,unit,data,R_ATTRACTION,singleIter,&errorPortion,&testProp[nStored]);
				if(aveH > H_THRESHOLD)
					break;
            }
		}

        int nStored = unit->numStored()>1 ? unit->numStored()-1 : 1;
        testProp[nStored][parameterName] = unit->getParameter(parameterName);
		if(firstRecord)
		{
			string str =
                string("\n--- --- --- --- --- --- TEST_2 --- --- --- --- --- ---\n")
                + "H_THRESHOLD = " + toString(H_THRESHOLD,3) + " numTestEpochs = " + toString(numTestEpochs)
                + " R_ATTRACTION = " + toString(R_ATTRACTION)
                + " SingleIter = " + (singleIter?"true\n":"false\n")
				+ unit->getDescription() + "\n" + data->getDescription();
            saveString(str,reportFile);
            saveProperties(trainProp[nStored],testProp[nStored],true,REPORT_WIDTH_VAL,reportFile);
            firstRecord = false;
		}
        else
            saveProperties(trainProp[nStored],testProp[nStored],false,REPORT_WIDTH_VAL,reportFile);

        if(needSave)
            unit->save(reportFile+"_"+toString(unit->getParameter(parameterName),2)+".wgt");

        lastI = unit->numStored();
	}

	saveString(string("\n"),reportFile);
}


// Calculates net capacity for each value of data dimension
// rAttraction is a maximal initial Hamming resulting in <final Hemming> less or equal to H_THRESHOLD
// final Hamming is averaged over numTestEpochs runs for all stored images       // each stored image // and
// each sequential initial Hamming value
// Net capacity is a maximum number of images stored so as rAttraction is still
// greater then some fixed value (R_ATTRACTION)
// uses START_PORTION, START_SHIFT, CHECK_POINT_COUNT
void test3(int numTestEpochs, CellularNet *net, vector<Data> *dataSet, string reportFile, bool singleIter)
{
	map<int,Properties> trainProp, testProp;
	int lastI = 0;
	int connR = round(net->getParameter(PARAM_CONN_R));
	bool firstRecord = true;
	for(int d=0; d<dataSet->size(); d++)
	{
        Data *data = &(*dataSet)[d];
        int dim = data->dim();
		net->init(dim,connR);
		cout << endl << "Test 3, dim = " << dim << " checking image # ";
		cout.flush();
		int startI = ((double)lastI)*START_PORTION - START_SHIFT - CHECK_POINT_COUNT, i;
		bool mayNeedRetest = true;
		int nCheckPoint = 0;
		string header, values;
		if(startI < 0)
		{
			startI = 0;
			mayNeedRetest = false;
		}
		if(startI > net->dim()-1)
			startI = net->dim()-1;

		for(i=0; i<startI; i++)
		{
			net->train(&(*data)[i]);
		}
		for(i=startI; i<data->size(); i++)
		{
            int nStored = net->numStored()+1;
			net->train(&(*data)[i],&trainProp[nStored]);
			cout << net->numStored() << " ";
			cout.flush();
            double errorPortion;
			double aveH = getAveHamming(numTestEpochs,net,data,R_ATTRACTION,singleIter,&errorPortion,&testProp[nStored]);
			if(aveH <= H_THRESHOLD)
					nCheckPoint++;
				else
					break;
        }

		if(nCheckPoint<CHECK_POINT_COUNT && mayNeedRetest)
		{
            log(string("[test3] retesting from ")+toString(net->numStored()),net->getName());
			cout << " retesting ";
			cout.flush();
			net->clear();
			for(i=0; i<data->size(); i++)
			{
                int nStored = net->numStored()+1;
				net->train(&(*data)[i],&trainProp[nStored]);
				cout << net->numStored() << " ";
				cout.flush();
                double errorPortion;
				double aveH = getAveHamming(numTestEpochs,net,data,R_ATTRACTION,singleIter,&errorPortion,&testProp[nStored]);
				if(aveH > H_THRESHOLD)
					break;
            }
		}

        int nStored = net->numStored()>1 ? net->numStored()-1 : 1;
		trainProp[nStored]["dim"] = net->dim();
        trainProp[nStored]["connR"] = net->getConnR();
		if(firstRecord)
		{
			string str =
				string("\n--- --- --- --- --- --- TEST_3 --- --- --- --- --- ---\n")
				+ "H_THRESHOLD = " + toString(H_THRESHOLD,3) + " numTestEpochs = " + toString(numTestEpochs)
                + " R_ATTRACTION = " + toString(R_ATTRACTION)
                + " SingleIter = " + (singleIter?"true\n":"false\n")
				+ net->getDescription() + "\n" + data->getDescription();
            saveString(str,reportFile);
            saveProperties(trainProp[nStored],testProp[nStored],true,REPORT_WIDTH_VAL,reportFile);
            firstRecord = false;
        }
        else
            saveProperties(trainProp[nStored],testProp[nStored],false,REPORT_WIDTH_VAL,reportFile);

        lastI = net->numStored();
	}
	saveString(string("\n"),reportFile);
}
*/

// Calculates rAttraction for the fixed net capacity and each value of "parameterName"
// rAttraction is a maximal initial Hamming resulting in <final Hemming> less or equal to H_THRESHOLD
// final Hamming is averaged over numTestEpochs runs for all stored images
void test4(int numTestEpochs, AssociativeUnit *unit, RAMData *data, int numStored, string variableName, Parameter variableValue, string reportFile, bool adjustBeforeTesting, bool needSave, bool singleIter)
{
    assertion(data->inDim() == unit->dim(), "[test4] data->inDim() != unit->dim()");
    assertion(data->inDim() == data->outDim(), "[test4] data->inDim() != data->outDim()");
    Vector in(data->inDim()), out(data->outDim());

    Properties trainProp, testProp;
	bool firstRecord = true;
    assertion(variableValue.isVariable(), "[test4] variableValue not a variable");
    variableValue.init();
    while(variableValue.setNext())
	{
		unit->setParameter(variableName, variableValue.getFloat());
        unit->clear();
        for(int i=0; i<numStored; i++)
        {
            data->get(&in, &out, i);
		    unit->train(&in, &trainProp);
        }
        unit->doEndTraining(&trainProp);

        //------------------------------------------------------------
        //if(numRetrain > 0)
        //    for(int i=0; i<numRetrain; i++)
        //        ((AdaptiveCellularNet*)unit)->retrain(1, &trainProp);
        //unit->getProperties(&trainProp);
        //------------------------------------------------------------

		cout << endl << "Test 4, " << variableName << " = " << unit->getParameter(variableName) << endl;
		cout.flush();

        if(adjustBeforeTesting)
            unit->adjustBeforeTesting();
        getRAttraction(numTestEpochs, singleIter, unit, data, &testProp);
        testProp[variableName] = unit->getParameter(variableName);
		
        if(firstRecord)
		{
			string str =
                string("\n--- --- --- --- --- --- TEST_4 --- --- --- --- --- ---\n")
                + "H_THRESHOLD = " + toString(H_THRESHOLD,3) + " numTestEpochs = " + toString(numTestEpochs)
                + " SingleIter = " + toStringB(singleIter) + "\n"
                //+ " numRetrain = " + toString(numRetrain) + "\n"
				+ unit->getDescription() + "\n" + data->getDescription();
            saveString(str, reportFile);
            saveProperties(trainProp, testProp, true, REPORT_WIDTH_VAL, reportFile);
            firstRecord = false;
		}
        else
            saveProperties(trainProp, testProp, false, REPORT_WIDTH_VAL, reportFile);

        if(needSave)
            unit->saveDetails(reportFile+"_"+toString(unit->getParameter(variableName), 3));
	}
	saveString(string("\n"),reportFile);
}


// Trains the network with "numStored" data vectors
// and then runs convergence process from a random (!) initial point "numInitialStates" times
// statistics about the convergence behaviour is logged to "reportFile"
void testStability(int numInitialStates, AssociativeUnit *unit, RAMData *data, int numStored, string reportFile)
{
    assertion(data->inDim() == unit->dim(), "[testStability] data->inDim() != unit->dim()");
    assertion(data->inDim() == data->outDim(), "[testStability] data->inDim() != data->outDim()");

    Properties trainProp;
    set<BipolarVector> trainData;
    BipolarVector in(data->inDim()), out(data->outDim());

    unit->clear();
    for(int i=0; i<numStored; i++)
    {
        data->get(&in, &out, i);
        unit->train(&in, &trainProp);
        trainData.insert(in);
    }
    unit->doEndTraining();

    string str =
        string("\n--- --- --- --- --- --- TEST_STABILITY --- --- --- --- --- ---\n")
        + " numInitialStates = " + toString(numInitialStates) + "\n"
        + unit->getDescription() + "\n" + data->getDescription();
    saveString(str, reportFile);

    int numUnconverged = 0, cycleLength;
    StatCounter iterCounter("iterations"), correctRecallCounter("correct recall");
    AdvancedStatCounter cycleLengthCounter(0, 10, "cycleLength");
    for(int i=0; i<numInitialStates; i++)
    {
        in.fillBipolarRand();
        int numIter = unit->extendedConverge(&in, &out, &cycleLength);
        if(numIter == 0)
            numUnconverged++;
        else
        {
            iterCounter.addValue(numIter);
            cycleLengthCounter.addValue(cycleLength);
        }

        set<BipolarVector>::iterator iter = trainData.find(out);
        if(iter != trainData.end())
            correctRecallCounter.addValue(1);

        printProgress("[testStability]", i, 100, numInitialStates);
    }
    cout << endl;

    Properties convergeProp;
    convergeProp["numDiverged"] = numUnconverged;
    unit->getProperties(&convergeProp);
    saveProperties(convergeProp, true, REPORT_WIDTH_VAL, reportFile);
    saveString(iterCounter.toString() + "\n" + correctRecallCounter.toString() + "\n"
        + cycleLengthCounter.toString(), reportFile);
}

// Trains net with 1..netFill elements of data
// saves the net after it
void trainAndSave(AssociativeUnit *unit, IOData * data, int netFill, string netFile)
{
    assertion(unit->dim() == data->inDim(), "[trainAndSave] unit->dim() != data->dim()");
    assertion(data->inDim() == data->outDim(), "[trainAndSave] data->inDim() != data->outDim()");

    Vector in(unit->dim()), out(unit->dim());
    unit->clear();
    data->setToBeginning();
    for(int d=0; d<netFill; d++)
    {
        assertion(d < data->count(), "[trainAndSave] not enough data size");
        data->getNext(&in, &out);
        unit->train(&in);
        printProgress("[trainAndSave] training", d, 1, netFill);
    }
    cout << endl;

    unit->doEndTraining();
    
    unit->saveDetails(netFile);
}

/*
// Calculates statistics for last (dimension-1) PsI eps for each dimension value [from...to) with step
// eps is averaged over nTests Matrix::pseudoInverseRecalc return value
// when (dim+1)-th random +/-1 vector addedd 
void testPIAlgorithm(int from, int to, int step, int nTests, string reportFile)
{
	bool firstRecord = true;
    Properties prop;
	for(int d=from; d<to; d+=step)
	{
		cout << endl << "[testPIAlgorithm] dim = " << d << " ";
		double sumEps = 0, sumSqEps = 0, minEps = -1, maxEps = -1, eps;
		int nIndep = 0, nIndepError = 0;
		bool isIndep;
		for(int t=0; t<nTests; t++)
		{
            cout << '.';
			Data data(d,d+1,abs((int)getTimeMsec()));
			Matrix mt(0,d), mtp(0,d);
			for(int v=0; v<d; v++)
			{
                eps = Matrix::pseudoInverseRecalc(false,&mt,&Matrix(&data[v]),&mtp,&isIndep);
				if(!isIndep)
					break;
				minEps = (minEps<eps && v!=0)? minEps : eps;
            }
			if(!isIndep)
				break;
			nIndep++;
            eps = Matrix::pseudoInverseRecalc(false,&mt,&Matrix(&data[d]),&mtp,&isIndep);
            maxEps = maxEps>eps? maxEps : eps;
            sumEps += eps;
            sumSqEps += eps*eps;
			if(isIndep)
				nIndepError++;
		}

        prop["Dim"] = d;
        prop["<eps>"] = sumEps/nIndep;
        prop["disp(eps)"] = sqrt(sumSqEps/nIndep-sumEps*sumEps/nIndep/nIndep);
        prop["max(eps)"] = maxEps;
        prop["nIndep"] = nIndep;
        prop["nIndepError"] = nIndepError;
        prop["min(eps_all)"] = minEps;
		if(firstRecord)
		{
			string str =
				string("\n--- --- --- --- --- --- TEST_PI --- --- --- --- --- ---\n")
				+ "nTests = " + toString(nTests);
            saveString(str,reportFile);
            saveProperties(prop,true,REPORT_WIDTH_VAL,reportFile);
            firstRecord = false;
        }
        else
            saveProperties(prop,false,REPORT_WIDTH_VAL,reportFile);
    }
}

// Fills the net calculating weigths distribution parameters and
// weights/data correlation at each step
void testStatistics(CellularNet *net, Data *data, int from, int to, int step, string reportFile)
{
    StatMaker sm;
	net->clear();
	Properties prop;
	int dim = net->dim();
	Vec in(dim), out(dim);
	bool firstRecord = true;
	for(int i=0; i<data->size(); i++)
	{
		net->train(&(*data)[i],(net->numStored()+1-from)%step==0? &prop : 0);
		if(net->numStored() < from)
			continue;
		if(net->numStored() >= to)
			break;

		if((net->numStored()-from)%step == 0)
		{
			cout << endl << "Test Statistics, numStored = " << net->numStored();
            cout.flush();

            double meanDiag, dispDiag, meanNonDiag, dispNonDiag, corr;
            Vec v1, v2;
            sm.calculateDistribution(net, 1, // halfR
                                          1, // halfC
                                          v1,&v2, &meanDiag,&dispDiag,0);
            sm.calculateDistribution(net, 1, // halfR
                                          1, // halfC
                                          &v1,&v2, &meanNonDiag,&dispNonDiag,2);
            corr = sm.getCorrCoeff(net,data);

            prop["mean(diag)"] = meanDiag;
            prop["disp(diag)"] = dispDiag;
            prop["mean(nondiag)"] = meanNonDiag;
            prop["disp(nondiag)"] = dispNonDiag;
            prop["corr(w/d)"] = corr;
            if(firstRecord)
            {
                string str =
                    string("\n--- --- --- --- --- --- TEST_STAT --- --- --- --- --- ---\n")
                    + net->getDescription().c_str() + "\n" + data->getDescription().c_str();
                saveString(str,reportFile);
                saveProperties(prop,true,REPORT_WIDTH_VAL,reportFile);
                firstRecord = false;
			}
            else
                saveProperties(prop,false,REPORT_WIDTH_VAL,reportFile);
        }
	}
	saveString(string("\n"),reportFile);
}


// Fills the net calculating per-neuron errors distribution at each step
// Initial Hamming is set to R_ATTRACTION
void testErrorDistribution(int numTestEpochs, AssociativeNet *net, Data *data, int from, int to, int step, string reportFile)
{
	net->clear();
	Properties prop;
	int dim = net->dim();
	Vec in(dim), out(dim);
	bool firstRecord = true;
	for(int i=0; i<data->size(); i++)
	{
		net->train(&(*data)[i],(net->numStored()+1-from)%step==0? &prop : 0);
		if(net->numStored() < from)
			continue;
		if(net->numStored() >= to)
			break;
		
		if((net->numStored()-from)%step == 0)
		{
			cout << endl << "Test ED, numStored = " << net->numStored()
                         << ", razl = " << net->getInputContentDiff(&(*data)[i]);
			cout.flush();
		
			string header, values;
			Vec errors;
            double errorPortion;
			getAveHamming(numTestEpochs,net,data,R_ATTRACTION,false,&errorPortion,&prop,&errors);
			ostrstream eoss;
			for(int e=0; e<errors.size(); e++)
				eoss << errors[e] << " ";
			eoss << ends;
			saveString(string(eoss.str()),reportFile+string("_errors"));
            eoss.freeze(false);
					
			if(firstRecord)
			{
				string str =
					string("\n--- --- --- --- --- --- TEST_ED --- --- --- --- --- ---\n")
					+ "numTestEpochs = " + toString(numTestEpochs) + " R_ATTRACTION = " + toString(R_ATTRACTION)
					+ net->getDescription() + "\n" + data->getDescription();
				saveString(str,reportFile);
                saveProperties(prop,true,REPORT_WIDTH_VAL,reportFile);
                firstRecord = false;
			}
            else
                saveProperties(prop,false,REPORT_WIDTH_VAL,reportFile);
        }
	}
	saveString(string("\n"),reportFile);
}

void testRetrain(int numTestEpochs, FullProjectiveNet *net, Data *data, int numStored, double eraseValue, int eraseType, int retrainCount, int multCount, string reportFile)
{
    string str = 	string("\n--- --- --- --- --- --- TEST_RETRAIN --- --- --- --- --- ---\n")
					+ "H_THRESHOLD = " + toString(H_THRESHOLD,3) + ", numTestEpochs " + toString(numTestEpochs) + ", numStored " + toString(numStored) +
                    + ", eraseValue " + toString(eraseValue,3) + ", eraseType " + toString(eraseType) + ", retrainCount " + toString(retrainCount) + "\n"
					+ net->getDescription() + "\n" + data->getDescription();
    saveString(str,reportFile);

    net->clear();
    for(int i=0; i<numStored; i++)
        net->train(&(*data)[i]);
    //net->saveWeights("_weights_original.txt");

    cout << "[testRetrain] original net: ";
    Properties testProp, netProp;
    testProp["retrain_num"] = -2;
    net->getProperties(&netProp);
    getRAttraction(numTestEpochs,false,net,data,&testProp);
    //testProp["retrain_diff"] = -1;
    //testProp.erase(testProp.find("converged"));
    saveProperties(testProp,netProp,true,REPORT_WIDTH_VAL,reportFile);

    srand(123);
    if(eraseType == 0)
        net->deleteNeurons(round(eraseValue));
    else if(eraseType == 1)
        net->deleteWeights(eraseValue);
    else
        throw GenException("[testRetrain] unknown erase type");
    //net->saveWeights("_weights_erased.txt");
    testProp["retrain_num"] = -1;
    net->getProperties(&netProp);
    cout << endl << "erased net : ";
    getRAttraction(numTestEpochs,false,net,data,&testProp);
    //testProp["retrain_diff"] = -1;
    //testProp.erase(testProp.find("converged"));
    saveProperties(testProp,netProp,false,REPORT_WIDTH_VAL,reportFile);

    for(int i=0; i<retrainCount; i++)
    {
        cout << endl << "retrain # " << i << ": ";
        if(!net->retrain(&(*data)[i%net->numStored()]))
            continue;
        testProp["retrain_num"] = i+1;
        net->getProperties(&netProp);
        getRAttraction(numTestEpochs,false,net,data,&testProp);
        //testProp["retrain_diff"] = net->getInputContentDiff(&(*data)[i%net->numStored()]);
        //testProp.erase(testProp.find("converged"));
        saveProperties(testProp,netProp,false,REPORT_WIDTH_VAL,reportFile);
    }
    //net->saveWeights("_weights_restored.txt");

    testProp.clear();
    if(multCount)
    {
        saveString("\nWeights multiplication:",reportFile);
        for(int i=0; i<multCount; i++)
        {
            cout << endl << "mult # " << i << ": ";
            net->multiplyWeights();
            testProp["rmult_num"] = i+1;
            net->getProperties(&netProp);
            getRAttraction(numTestEpochs,false,net,data,&testProp);
            //testProp["retrain_diff"] = net->getInputContentDiff(&(*data)[i%net->numStored()]);
            //testProp.erase(testProp.find("converged"));
            saveProperties(testProp,netProp,false,REPORT_WIDTH_VAL,reportFile);
        }
    }
}

void getPerVectorValues(int numTestEpochs, AssociativeNet *net, Data *data, vector<int> *rAttraction, Vec *difference)
{
    rAttraction->clear();
    difference->clear();
    for(int i=0; i<net->numStored(); i++)
    {
        Data singleVec;
        singleVec.push_back((*data)[i]);
        Properties prop;
        int rAttr = getRAttraction(numTestEpochs, false, // singleIter
                                             net, &singleVec, &prop,
                                             true // probCriterion
                                             );
        //double hamm = getAveHamming(1,net,&singleVec,0,true);
        rAttraction->push_back(rAttr);
        difference->push_back(net->getInputContentDiff(&singleVec[0]));
    }
}

void savePerVectorValues(vector<int> *rAttraction, Vec *difference)
{
    string rAttr, diff;
    for(int i=0; i<rAttraction->size(); i++)
    {
        rAttr += setWidth(toString((*rAttraction)[i]),12)+" ";
        diff += setWidth(toString((*difference)[i],4),12)+" ";
    }
    saveString(rAttr,"_rAttraction.txt");
    saveString(diff,"_difference.txt");
}

void retrainScript()
{
    int dim = 256;
    double desatCoeff = 0.1;
    int numStored = 120;
    int eraseNeuronNum = 40;
    //int multNum = 4;
    int numTestEpochs = 100;
    int afterTrainNum = 15;
    string reportFile = "_retrain_script_report.txt";

    FullProjectiveNet net(dim);
    net.setParameter(PARAM_DESAT_COEFF, desatCoeff);
    Data data("_434569017(2).dat");
    vector<int> rAttraction;
    Vec difference;

    string str = 	string("\n--- --- --- --- --- --- TEST_RETRAIN_SCRIPT --- --- --- --- --- ---\n")
					+ "H_THRESHOLD = " + toString(H_THRESHOLD,3) + ", numTestEpochs (for each vector) " + toString(numTestEpochs) + ", numStored " + toString(numStored) +
                    + ", erase neurons " + toString(eraseNeuronNum) + "\n"
					+ net.getDescription() + "\n" + data.getDescription();
    saveString(str,reportFile);

    // train
    for(int i=0; i<numStored; i++)
        net.train(&data[i]);
    net.save("_weights_original.txt");


    // test original net
    cout << endl << "[retrainScript] original net: ";
    Properties netProp;
    netProp["retrain_num"] = -2;
    net.getProperties(&netProp);
    saveProperties(netProp,true,REPORT_WIDTH_VAL,reportFile);
    //getPerVectorValues(numTestEpochs,&net,&data,&rAttraction,&difference);
    //savePerVectorValues(&rAttraction,&difference);


    // erase neurons and test
    cout << endl << "[retrainScript] erased net: ";
    srand(123);
    net.deleteNeurons(eraseNeuronNum);
    net.save("_weights_erased.txt");
    netProp["retrain_num"] = -1;
    net.getProperties(&netProp);
    saveProperties(netProp,false,REPORT_WIDTH_VAL,reportFile);
    //getPerVectorValues(numTestEpochs,&net,&data,&rAttraction,&difference);
    //savePerVectorValues(&rAttraction,&difference);

    // retrain and test

//    cout << endl << "[retrainScript] retrain: ";
//    netProp.clear();
//    int d = 0;
//    for(int r=0; r<eraseNeuronNum; r++)
//    {
//        if(d == numStored)
//        {
//            saveString("Not enough for retrain",reportFile);
//            return;
//        }
//        while(!net.retrain(&data[d++]) && d<numStored);
//        netProp["retrain_num"] = r+1;
//        net.getProperties(&netProp);
//        saveProperties(netProp,false,REPORT_WIDTH_VAL,reportFile);
//        if((r+1)%3==0 || r+1==10)
//        {
//            getPerVectorValues(numTestEpochs,&net,&data,&rAttraction,&difference);
//            savePerVectorValues(&rAttraction,&difference);
//            net.saveWeights(string("_weights_retrain_")+toString(r+1)+".txt");
//        }
//    }

    // after train
    cout << endl << "[retrainScript] aftertrain: ";
    saveString("After train:",reportFile);
    netProp.clear();
    for(int i=0; i<afterTrainNum; i++)
    {
        net.train(&data[i+numStored]);
        netProp["aftertrain"] = i+1;
        net.getProperties(&netProp);
        saveProperties(netProp,i==0,REPORT_WIDTH_VAL,reportFile);
        if((i+1)%5 == 0)
        {
            getPerVectorValues(numTestEpochs,&net,&data,&rAttraction,&difference);
            savePerVectorValues(&rAttraction,&difference);
            //net.saveWeights(string("_weights_aftertrain_")+toString(i+1)+".txt");
        }
    }

    // multiplication

//    saveString("Weights multiplication:",reportFile);
//    netProp.clear();
//    for(int m=0; m<multNum; m++)
//    {
//        net.multiplyWeights();
//        net.getProperties(&netProp);
//        netProp["multiplication"] = m+1;
//        saveProperties(netProp,false,REPORT_WIDTH_VAL,reportFile);
//    }
}

void print(AdvancedStatCounter &asc)
{
    Vec scale, density, prob;
    asc.getScale(&scale);
    asc.getProbabilityDensity(&density);
    asc.getProbabilityDistribution(&prob);
    Properties prop;
    asc.getData(&prop);
    cout << prop << Matrix(&prob).transpose() << Matrix(&density).transpose() << Matrix(&scale).transpose() << endl;
}

// some statistics used for modular network
void modularScript()
{
    srand(123);
    int dim = 256;
    int nStored = 102;
    int nTests = 1000;
    int granularity = 1;//00;
    int noise = 33;
    double kThreshold = 1-((double)nStored)/dim;
    double kDelta = 0.1;
    vector<Matrix> storedVectors;

    Matrix projective(dim,dim);
    projective.init(0);
    for(int i=0; i<nStored; i++)
    {
        Matrix vec(dim,1);
        vec.fillBipolarRand();
        storedVectors.push_back(vec);
        assertion(!projective.expandProjectiveMatrix(&vec), "!projective.expandProjectiveMatrix(&vec)");
    }

    AdvancedStatCounter   kC(0,    1,   granularity, "k");
    AdvancedStatCounter  dkC(-0.5, 0.5, granularity, "dk");
    AdvancedStatCounter dk0C(0,    1,   granularity, "dk0");


//    projective.init(0);
//    nStored = 0;
//    for(int i=0; i<dim; i++)
//    {
//        Matrix vec(dim,1);
//        vec.fillBipolarRand();
//        if(!projective.expandProjectiveMatrix(&vec))
//            throw GenException("!projective.expandProjectiveMatrix(&vec)");
//        nStored++;
//        StatCounter cij, cii, cijabs;
//        for(int i=0; i<dim; i++)
//            for(int j=0; j<dim; j++)
//                if(i==j)
//                    cii.addValue(projective.el(i,j)-((double)nStored)/dim);
//                else
//                {
//                    cij.addValue(projective.el(i,j));
//                    cijabs.addValue(fabs(projective.el(i,j)));
//                }
//
//        cout << nStored << " " << cij.getDisp() << " " << cii.getDisp() << " " << 1./dim*(((double)nStored)/dim - ((double)nStored)*nStored/dim/dim) << " " << cijabs.getAverage()*dim*(dim-1) << endl;
////
//    }
//    return;

//-----------------------------------------------

//    AdvancedStatCounter cC(0,1,160,"c");
//    for(int i=0; i<dim; i++)
//        for(int j=0; j<dim; j++)
//            if(i!=j)
//                cC.addValue(projective.el(i,j)/*-(i==j? ((double)nStored)/dim : 0));
//    print(cC);
//    return;



    Matrix vec(dim,1), vech(dim,1);
    int nPath = 0, nBelonging = 0, nSplit = 0;

    cout << "dim " << dim << ", nStored " << nStored << ", nTests " << nTests << ", noise " << noise
         << ", kThreshold " << kThreshold << ", kDelta " << kDelta << endl;
    for(int i=0; i<nTests; i++)
    {
        vec.fillBipolarRand();
        double diffR = projective.getDifference(&vec);
        kC.addValue(diffR);

        setNoise(&vec,&vech,noise);
        double diffRN = projective.getDifference(&vech);
        dkC.addValue(diffRN-diffR);

        Matrix &vecS = storedVectors[i%nStored];
        setNoise(&vecS,&vech,noise);
        double diffSN = projective.getDifference(&vech);
        dk0C.addValue(diffSN);

        if((diffR<kThreshold && diffRN>=kThreshold+kDelta) || (diffR>=kThreshold && diffRN<kThreshold-kDelta))
            nPath++;
        if(diffR<diffSN)
            nBelonging++;
        if(diffR>=kThreshold-kDelta && diffR<kThreshold+kDelta)
            nSplit++;
    }

    print(kC);
    print(dkC);
    print(dk0C);
    cout << endl << "Path error: " << ((double)nPath)/nTests << endl
                 << "Belonging error: " << ((double)nBelonging)/nTests << endl
                 << "Split probability: " << ((double)nSplit)/nTests;

    beep();
}

// experiments with Cellular Adaptive NN
void adaptiveScript()
{
    int dim = 256;
    bool isProjective = true;
    bool hasQuota = false;
    double connectivityPortion = 0.3;
    double alpha = 1;

    double desatCoeff = 0.05;
    int numStored = 51;
    int numTestEpochs = 10;//0;
    string reportFile = "_adaptive_script_report.txt";

    AdaptiveCellularNet net(dim, isProjective, hasQuota, connectivityPortion);
    net.setParameter(PARAM_DESAT_COEFF, desatCoeff);

    Data data("_434569017(2).dat");

    string str = 	string("\n--- --- --- --- --- --- TEST_ADAPTIVE_SCRIPT --- --- --- --- --- ---\n")
					+ "H_THRESHOLD = " + toString(H_THRESHOLD,3)
                    + ", numStored " + toString(numStored)
					+ "\n" + net.getDescription() + "\n" + data.getDescription();
    saveString(str, reportFile);

    // train
    cout << "Training";
    for(int i=0; i<numStored; i++)
    {
        net.train(&data[i]);
        cout << ".";
    }
    cout << endl;
    //net.save("_weights_original.txt");

    Properties prop;
    net.adjustDesaturationCoefficient();
    net.getProperties(&prop);
    getRAttraction(numTestEpochs, false, // singleIter
                             &net, &data, &prop, false, // probCriterion
                             );
    saveProperties(prop, true, REPORT_WIDTH_VAL, reportFile);


    prop.clear();
    for(int i=0; i<10; i++)
    {
        Properties prop2;
        cout << "\nRetraining...";
        net.retrain(alpha, &prop2);
        cout << "done\n";
        net.adjustDesaturationCoefficient();
        net.getProperties(&prop);
        getRAttraction(numTestEpochs, false, // singleIter
                                 &net, &data, &prop, false, // probCriterion
                                 );
        saveProperties(prop, prop2, i==0, REPORT_WIDTH_VAL, reportFile);
    }
}

// reads data
// performs weight selection for various connectivities using projective and correlation matrices
// prints the difference
void weightsSelectScript()
{
    Data data("_434569017(2).dat");
    int numStored = 77;
    string reportFile = "_weight_select_script.txt";

    string str = 	string("\n--- --- --- --- --- --- WEIGHT_SELECT_SCRIPT --- --- --- --- --- ---\n")
					+ data.getDescription() + "\n"
                    + "Num stored = " + toString(numStored);
    saveString(str, reportFile);


    Matrix projective(data.dim(), data.dim());
    CorrelationMatrix correlative;
    projective.init(0);
    for(int i=0; i<numStored; i++)
    {
        projective.expandProjectiveMatrix(&Matrix(&data[i]));
        correlative.addItem(&Matrix(&data[i]));
    }

    cout << correlative;

    Properties prop;
    bool firstRecord = true;
    Matrix scheme1(projective), scheme2(projective), difference(projective);
    for(double connectivity = 0.02; connectivity<=1; connectivity+=0.02)
    {
        cout << '.';   
        prop["connectivity"] = connectivity;
        prop["num_selected"] = (int) (connectivity * data.dim() * data.dim());
        prop["projRatio"] = AdaptiveCellularNet::createScheme(&projective, &scheme1, connectivity);
        prop["corrRatio"] = AdaptiveCellularNet::createScheme(&correlative, &scheme2, connectivity);
        scheme1.minus(&scheme2, &difference);
        prop["difference"] = difference.sqnorm();
        prop["difference%"] = difference.sqnorm()/(connectivity * data.dim() * data.dim());
        saveProperties(prop, firstRecord, REPORT_WIDTH_VAL, reportFile);
        firstRecord = false;
    }
} */

void adaptiveTransitionScript(Parameters params)
{
    AdaptiveCellularSubj net(params);
    runCrawler(&net, params, params.getString("reportFile"));
}


//--------------------------- New Tests (in accordance with Neil Davey et al) ---------------------------------

void getNormalizedRAttraction(int numDataSets, int numStored, AssociativeUnit *unit, bool contiguousNoise, Properties *prop)
{
    initRNG();

    Vector in(unit->dim()), startPoint(unit->dim()), out(0);
    PropertyStatCounter propCounter;
    StatCounter sc;
    double aveIter, convergedRatio;

    for(int t=0; t<numDataSets; t++) // cycle over different datasets
    {
        printProgress("[getNormalizedRAttraction]", t+1, 1, numDataSets);
        Properties trainProp;
        RandomBipolarRAMData data(unit->dim(), 0, numStored);
        unit->clear();
        __int64 timeMark = getTime_ms();
        for(int d=0; d<numStored; d++)
        {
            data.get(&in, &out, d);
            unit->train(&in, &trainProp);
        }
        unit->doEndTraining(&trainProp);
        trainProp["trainTime"] = getTime_ms() - timeMark;


        for(int d=0; d<numStored; d++)
        {
            data.get(&in, &out, d);
            double attrSimilarity = getAttrSimilarity(unit, &in, &startPoint,
                contiguousNoise, &aveIter, &convergedRatio);
            double crossSimilarity = getCrossSimilarity(&data, d, &startPoint);
            assertion(crossSimilarity<1, "[getNormalizedRAttraction] crossSimilarity==1");
            double normRA = (1-attrSimilarity) / (1-crossSimilarity);

            propCounter.addValue("..normRA", normRA);
            propCounter.addValue(".attrSim", attrSimilarity);
            propCounter.addValue(".crossSim", crossSimilarity);
            propCounter.addValue("convIter", aveIter);
            propCounter.addValue("converged", convergedRatio);
            sc.addValue(normRA);
        }

        propCounter.addValue(&trainProp);
    }
    deleteLine();
    cout << "[getNormalizedRAttraction] normRA = " << sc.getAverage() << endl;

    propCounter.getStatistics(prop, SC_AVERAGE | SC_SD);
}

// startting from (approximately) zero similarity (random vector) finds
// minimum similarity that provides convergence to the pattern
// if (contiguousNoise)
// then noise (random values) is set as a whole chank starting from the end of the vector
double getAttrSimilarity(AssociativeUnit *unit, Vector *pattern, Vector *startPoint, bool contiguousNoise,
    double *aveIter, double *convergedRatio)
{
    assertion(pattern->size()==unit->dim(), "[getMinSimilarity] different sizes");
    assertion(startPoint->size()==unit->dim(), "[getMinSimilarity] different sizes");

    StatCounter iterations, convRatio;
    Vector copiedElements(pattern), result(pattern);
    startPoint->fillBipolarRand();
    copiedElements.init(0);
    int dim = unit->dim();
    for(int i=0; i<=dim; i++) // i - number of assigned elements
    {
        while(i > 0)
        {
            int index = contiguousNoise? i-1 : rand(dim);
            if(copiedElements[index] == 0)
            {
                copiedElements[index] = 1;
                (*startPoint)[index] = (*pattern)[index];
                break;
            }
        }

        int numIter = unit->converge(startPoint, &result);
        if(numIter)
        {
            iterations.addValue(numIter);
            convRatio.addValue(numIter? 1 : 0);
        }
        else
        {
            convRatio.addValue(0);
        }

        int hammingDist = result.hammingDistance(pattern);
        if(hammingDist==0 || hammingDist==dim)
        {
            *aveIter = iterations.getCount()? iterations.getAverage() : NOT_A_NUMBER;
            *convergedRatio = convRatio.getAverage();
            
            //return ((double)i)/dim; - WRONG - underestimated, can cause normRA>1 !!!
            return startPoint->getSimilarity(pattern);
        }
    }

    return 1;
}


// maximum similarity of a pattern with respect
// to the rest of the data (all data vectors except patternIndex)
double getCrossSimilarity(RAMData *data, int patternIndex, Vector *pattern)
{
    assertion(data->outDim()==0, "[getMaxSimilarity] data->outDim()!=0");
    assertion(pattern->size()==data->inDim(), "[getMaxSimilarity] pattern->size()!=data->inDim()");
    assertion(patternIndex>=0 && patternIndex<data->count(), "[getMaxSimilarity] patternIndex out of range");

    if(data->count() == 1)
        return 0;

    Vector in(data->inDim()), out(0);
    StatCounter counter;
    for(int d=0; d<data->count(); d++)
    {
        if(d == patternIndex)
            continue;
        data->get(&in, &out, d);
        counter.addValue(in.getSimilarity(pattern));
    }

    return counter.getMax();
}

// finds normalized RA for each value of variable parameter
void test5(AssociativeUnit *unit, Parameters params, bool contiguousNoise)
{
    assertion(params.getVarCount()==1, "[test5] params must have exactly one variable");

    int numDataSets = params.getInt("numDataSets");
    string reportFile = params.getString("reportFile");

    bool firstRecord = true;
    string varName = params.getVarName(0);
    params.init(varName);
    while(params.setNext(varName))
    {
        cout << endl << "[test5] " << varName << " = " << params.getFloat(varName) << endl;

        if(varName.compare("numStored"))
            unit->setParameter(varName, params.getFloat(varName));

        Properties prop;
        getNormalizedRAttraction(numDataSets, params.getInt("numStored"), unit, contiguousNoise, &prop);
        prop[string("!")+varName] = params.getFloat(varName);

        if(firstRecord)
        {
            string str =
                string("\n--- --- --- --- --- --- TEST_5 --- --- --- --- --- ---\n")
                + "numDataSets = " + toString(numDataSets)
                + ", contiguousNoise = " + toStringB(contiguousNoise)
                + ", variableParamName = " + varName + "\n"
                + unit->getDescription() + "\n";
            saveString(str, reportFile);
            saveProperties(prop, true, REPORT_WIDTH_VAL+4, reportFile);
            firstRecord = false;
        }
        else
            saveProperties(prop, false, REPORT_WIDTH_VAL+4, reportFile);
    }
}

//-------------------------------------------------------------------------------------------------------------


// finds the probabilities of the following events:
// 1. split
// 2. jump
// 3. belonging
void modularStatScript(Parameters params)
{
    int jumpGranularity = 100;
    double jumpEpsilonMax = 0.2;

    int dim = params.getInt("dim");
    int numStored = params.getInt("numStored");
    double threshold = params.getFloat("threshold");
    int hamming = params.getInt("hamming");
    int numReplays = params.getInt("numReplays");
    int numTries = params.getInt("numTries");

    string reportFile = params.getString("reportFile");

    AdvancedStatCounter diff(0, 1, 100, "difference"), split(0, 0.2, 100, "split"),
        noise_stored(0, 1, 100, "noise_stored"), noise_shift(-0.5, 0.5, 100, "noise_shift");
    vector<StatCounter> jump(jumpGranularity);
    StatCounter belonging("belonging");

    for(int r=0; r<numReplays; r++)
    {
        printProgress("Replay #", r+1, 1, numReplays);

        Matrix proj(dim, dim);
        vector<Vector> data;
        Vector datum(dim);
        proj.init(0);
        for(int d=0; d<numStored; d++)
        {
            datum.fillBipolarRand();
            proj.expandProjectiveMatrix(&datum);
            data.push_back(datum);
        }

        Vector datum_noisy(dim), datum_stored_noisy(dim);
        for(int t=0; t<numTries; t++)
        {
            datum.fillBipolarRand();
            double difference = proj.getDifference(&datum);
            diff.addValue(difference);
            split.addValue(fabs(threshold - difference));

            datum_noisy.setBipolarNoise(&datum, hamming);
            double difference_noisy = proj.getDifference(&datum_noisy);
            noise_shift.addValue(difference_noisy - difference);

            Vector &datum_stored = data[rand(numStored)];
            datum_stored_noisy.setBipolarNoise(&datum_stored, hamming);
            double difference_stored_noisy = proj.getDifference(&datum_stored_noisy);
            noise_stored.addValue(difference_stored_noisy);

            belonging.addValue(difference_stored_noisy > difference ? 1 : 0); // if belonging error ocuured

            for(int e=0; e<jumpGranularity; e++)
            {
                double epsilon = e*jumpEpsilonMax/jumpGranularity;
                if(    (difference < threshold && difference_noisy > threshold+epsilon)
                    || (difference > threshold && difference_noisy < threshold-epsilon) )
                    jump[e].addValue(1); // jump occured
                else
                    jump[e].addValue(0); // no jump occured
            }
        }
    }

    saveString(params.toString(), reportFile, true);
    saveString(diff.toString(), reportFile);
    saveString(split.toString(), reportFile);
    saveString(noise_shift.toString(), reportFile);
    saveString(noise_stored.toString(), reportFile);
    saveString(belonging.toString(), reportFile);

    string jump_arg, jump_prob;
    for(int e=0; e<jumpGranularity; e++)
    {
        jump_arg += toString(e*jumpEpsilonMax/jumpGranularity, 4) + " ";
        jump_prob += toString(jump[e].getAverage(), 4) + " ";
    }
    saveString("\nJump probability:\n" + jump_arg + "\n" + jump_prob, reportFile);
}


void modularTestScript(Parameters params)
{
    //RandomBipolarRAMData data(256, 256, 3000, true);
    //data.saveToTextFile("256_3000.dat");
    //return;

    RAMData *data = new TextRAMData(params.getString("dataFile"));
    ModularNet net(params);

    string reportFile = params.getString("reportFile");
    string trainReport =  reportFile + "_train";
    string testReport =  reportFile + "_test";

    //saveString(net.getDescription()+"\n", trainReport, true);
    //saveString(data->getDescription()+"\n", trainReport);
    saveString(net.getDescription()+"\n", testReport, true);
    saveString(data->getDescription()+"\n", testReport);

    Properties trainProp, testProp;
    Vector in(data->inDim()), out(data->outDim());
    int testStep = params.getInt("testStep");
    int initialHamming = params.getInt("initialHamming");
    int testRuns = params.getInt("testRuns");

    saveString("Testing: initialHamming = " + toString(initialHamming)
        + ", testRuns = " + toString(testRuns) + "\n", testReport);

    int dataCount = data->count();
    for(int datumIndex=0; datumIndex<dataCount; datumIndex++)
    {
        data->get(&in, &out, datumIndex);
        net.train(&in, &trainProp);
        //saveProperties(trainProp, datumIndex==0, REPORT_WIDTH_VAL, trainReport);
        printProgress("[modularTestScript::train]", datumIndex, 10);

        if((datumIndex+1) % testStep == 0)
        {
            net.test(data, datumIndex+1, initialHamming, testRuns, &testProp);
            saveProperties(trainProp, testProp, datumIndex+1 == testStep, REPORT_WIDTH_VAL, testReport);
        }
    }
    cout << endl;

    Parameter testEpsilonVar = params.getParameter("testEpsilonVar");
    saveString("\n\n----------- Changing TestEpsilon: " + testEpsilonVar.toString() + " ---------\n", testReport);
    trainProp.clear();
    int epsilonIter = 0;
    while(testEpsilonVar.setNext())
    {
        double epsilon = testEpsilonVar.getFloat();
        net.setTestEpsilon(epsilon);
        net.test(data, dataCount, initialHamming, testRuns, &testProp);
        trainProp["_testEpsilon"] = epsilon;
        saveProperties(trainProp, testProp, epsilonIter++ ==0, REPORT_WIDTH_VAL, testReport);
        printProgress("[modularTestScript::test] checking epsilon", epsilonIter, 1, 0, toString(epsilon, 4));
    }
    cout << endl;

    delete data;
}


//------------------------------------------------------------------------------

void convergenceStabilityScript(Parameters params)
{
    AssociativeNet *net = createNetwork(params);
    RAMData *data = new TextRAMData(params.getString("dataFile"));

    testStability(params.getInt("numInitialStates"), net, data,
        params.getInt("numStored"), params.getString("reportFile"));

    delete net;
    delete data;
}























