//
//  MNistParser.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_MNistParser_h
#define NeuralNetworks_MNistParser_h

#include "NN.h"

namespace NN {
    
inline int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;
    
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
//
inline void parseLabels(string lblFileName, vector<int>& labels)
{
    ifstream file(lblFileName, ios::in | ios::binary);
    if (file.bad() || file.fail())
        throw "failed to open file:" + lblFileName;
    
    int magicNumber, nbItems;
    
    file.read((char*) &magicNumber, 4);
    file.read((char*) &nbItems    , 4);
    
    magicNumber = reverseInt(magicNumber);
    nbItems     = reverseInt(nbItems);

    if (magicNumber != 0x00000801 || nbItems <= 0)
        throw "MNIST label-file format error";
    
    for (size_t i=0; i<nbItems; ++i)
    {
        uint8_t label;
        file.read((char*) &label, 1);
        labels.push_back((int) label);
    }
}
//
inline void parseImage (ifstream& file, vector<float>& img, int nbRows, int nbCols, float scaleMin, float scaleMax)
{
    vector<uint8_t> imgVec(nbRows*nbCols);
    file.read((char *) &imgVec[0], nbRows*nbCols);
    
    img.resize(nbRows*nbCols);
    for (size_t i=0; i<nbRows; ++i)
        for (size_t j=0; j<nbCols; ++j)
            img[i*nbCols+j] = (imgVec[i*nbCols+j]/255.)*(scaleMax-scaleMin)+scaleMin;
}
//
inline void parseImages(string imgFileName, vector<vector<float> >& images, float scaleMin=0, float scaleMax=1.)
{
    ifstream file(imgFileName, ios::in | ios::binary);
    
    if (file.bad() || file.fail())
        throw "failed to open file:" + imgFileName;
    
    int magicNumber, nbItems, nbRows, nbCols;
    file.read((char*) &magicNumber, 4);
    file.read((char*) &nbItems    , 4);
    file.read((char*) &nbRows     , 4);
    file.read((char*) &nbCols     , 4);
    
    magicNumber = reverseInt(magicNumber);
    nbItems     = reverseInt(nbItems);
    nbRows      = reverseInt(nbRows);
    nbCols      = reverseInt(nbCols);
    
    images.resize(nbItems);
    for (size_t i=0; i<nbItems; i++)
        parseImage(file, images[i], nbRows, nbCols, scaleMin, scaleMax);
}

}
#endif
