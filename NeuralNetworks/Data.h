//
//  Data.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Data_h
#define NeuralNetworks_Data_h

#include "includes.h"
#include "MNistParser.h"

struct LabelData
{
    int                 label;
    std::vector<float> data;
};
//
class DataContainer
{
public:
    DataContainer() {}
    
    size_t getDataSize() const {return trainLabelData[0].data.size();}
    
    const std::vector<LabelData>& getTrainLabelData() const { return trainLabelData;}
    const std::vector<LabelData>& getCrossLabelData() const { return crossLabelData;}
    const std::vector<LabelData>& getTestLabelData()  const { return testLabelData;}
    
    void constructLabelData(const std::vector<int>& trainLabels, const std::vector<int>& testLabels, const std::vector<std::vector<float> > trainData, const std::vector<std::vector<float> > testData, size_t fractionSize)
    {
        size_t trainSize = trainLabels.size();
        size_t crossSize = trainSize/fractionSize;
        size_t testSize  = testLabels.size();
        
        trainLabelData.resize(trainSize);
        crossLabelData.resize(crossSize);
        testLabelData.resize(testSize);
        
        for (size_t i=0; i<trainSize; ++i)
        {
            trainLabelData[i].label = trainLabels[i];
            trainLabelData[i].data  = trainData[i];
        }
        std::random_shuffle(trainLabelData.begin(), trainLabelData.end());
        
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
    std::vector<LabelData> trainLabelData;
    std::vector<LabelData> crossLabelData;
    std::vector<LabelData> testLabelData;
};
//
class MNistDataContainer : public DataContainer
{
public:
    MNistDataContainer(std::string trainLabelFN, std::string testLabelFN, std::string trainDataFN, std::string testDataFN, size_t crossFraction=6)
    {
        std::vector<int>                  trainLabels, testLabels;
        std::vector<std::vector<float> > trainData, testData;
        parseLabels(trainLabelFN, trainLabels);
        parseLabels(testLabelFN , testLabels);
        parseImages(trainDataFN , trainData);
        parseImages(testDataFN  , testData);
        
        constructLabelData(trainLabels, testLabels, trainData, testData, crossFraction);
    }
};

#endif
