//
//  Data.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Data_h
#define NeuralNetworks_Data_h

#include "NN.h"
#include "MNistParser.h"

namespace NN {

struct LabelData
{
    int                label;
    vector<float> data;
};
//
class DataContainer
{
public:
    DataContainer() {}
    
    size_t getDataSize() const {return trainLabelData[0].data.size();}
    
    const auto& getTrainLabelData() const { return trainLabelData;}
    const auto& getCrossLabelData() const { return crossLabelData;}
    const auto& getTestLabelData()  const { return testLabelData;}
    
    void constructLabelData(const vector<int>& trainLabels, const vector<int>& testLabels, const vector<vector<float> > trainData, const vector<vector<float> > testData, size_t fractionSize)
    {
        auto trainSize = trainLabels.size();
        auto crossSize = trainSize/fractionSize;
        auto testSize  = testLabels.size();
        
        trainLabelData.resize(trainSize);
        crossLabelData.resize(crossSize);
        testLabelData.resize(testSize);
        
        for (size_t i=0; i<trainSize; ++i)
        {
            trainLabelData[i].label = trainLabels[i];
            trainLabelData[i].data  = trainData[i];
        }
        random_shuffle(trainLabelData.begin(), trainLabelData.end());
        
        for (size_t i=0; i<crossSize; ++i)
            crossLabelData[i] = trainLabelData[trainSize-crossSize+i];
        trainLabelData.erase(trainLabelData.begin()+trainSize-crossSize, trainLabelData.end());
        
        for (size_t i=0; i<testSize; ++i)
        {
            testLabelData[i].label = testLabels[i];
            testLabelData[i].data  = testData[i];
        }
    }

protected:
    vector<LabelData> trainLabelData;
    vector<LabelData> crossLabelData;
    vector<LabelData> testLabelData;
};
//
class MNistDataContainer : public DataContainer
{
public:
    MNistDataContainer(string trainLabelFN, string testLabelFN, string trainDataFN, string testDataFN, size_t crossFraction=6)
    {
        vector<int>                 trainLabels, testLabels;
        vector<vector<float> > trainData, testData;
        parseLabels(trainLabelFN, trainLabels);
        parseLabels(testLabelFN , testLabels);
        parseImages(trainDataFN , trainData);
        parseImages(testDataFN  , testData);
        
        constructLabelData(trainLabels, testLabels, trainData, testData, crossFraction);
    }
};

}
#endif
