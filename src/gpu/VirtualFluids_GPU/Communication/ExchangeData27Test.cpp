#include <gmock/gmock.h>

#include <filesystem>

#include "ExchangeData27.h"

#include <basics/config/ConfigurationFile.h>

auto RealEq = [](auto value) { 
#ifdef VF_DOUBLE_ACCURACY
    return testing::DoubleEq(value); 
#else 
    return testing::FloatEq(value);
#endif
};


SPtr<Parameter> initParameterClass(std::shared_ptr<Parameter> &para)
{
    std::filesystem::path filePath = __FILE__; //  assuming that the config file is stored parallel to this file.
    filePath.replace_filename("ExchangeData27Test.cfg");
    vf::basics::ConfigurationFile config;
    config.load(filePath.string());
    return std::make_shared<Parameter>(config, 1, 0);
}

TEST(ExchangeData27Test, copyEdgeNodes_XZ_CommunicationAfterFtoC)
{
    int level = 0;
    SPtr<Parameter> para = initParameterClass(para);
    para->setMaxLevel(level + 1); // setMaxLevel resizes parH
    para->initLBMSimulationParameter(); // init parH

    // indexInSend < 5 --> in AfterFToC
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,1,0,1);
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,6,0,6);
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,2,0,3);
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,7,0,8);

    int numNodes = 10;
    int numNodesAfterFtoC = 5;
    
    
    std::vector<ProcessNeighbor27> recvProcessNeighborHost(1);
    std::vector<real> recvFs(numNodes, 0.5);
    for (LBMSimulationParameter::EdgeNodePositions edgeNode : para->getParH(level)->edgeNodesXtoZ){
        recvFs[edgeNode.indexInRecvBuffer] = 0.1;        
    }
    for (uint direction = 0; direction <= dirEND; direction ++){
        recvFs.insert(recvFs.end(), recvFs.begin(), recvFs.begin()+10);
    }
    recvProcessNeighborHost[0].f[0] = recvFs.data();
    recvProcessNeighborHost[0].numberOfNodes = numNodes;


    
    std::vector<ProcessNeighbor27> sendProcessNeighborHost(1);
    std::vector<real> sendFs(27*numNodesAfterFtoC, 0.0);
    sendProcessNeighborHost[0].f[0] = sendFs.data();
    sendProcessNeighborHost[0].numberOfNodes = numNodesAfterFtoC;

    // expected
    std::vector<real> expectedFs(numNodesAfterFtoC, 0.0);
    expectedFs[1] = 0.1;
    expectedFs[3] = 0.1;
    std::vector<real> expectedFsAllDirections;
    for (uint direction = 0; direction <= dirEND; direction ++){
        expectedFsAllDirections.insert(expectedFsAllDirections.end(), expectedFs.begin(), expectedFs.end());
    }
    
    // act
    copyEdgeNodes(para->getParH(level)->edgeNodesXtoZ, recvProcessNeighborHost, sendProcessNeighborHost);

    // convert result to std::vector
    std::vector<real> result;
    result.assign(sendProcessNeighborHost[0].f[0],  sendProcessNeighborHost[0].f[0] + 27*numNodesAfterFtoC);

    EXPECT_THAT(result, testing::Eq(expectedFsAllDirections));
}

TEST(ExchangeData27Test, copyEdgeNodes_XZ_CommunicateAll)
{
    int level = 0;
    SPtr<Parameter> para = initParameterClass(para);
    para->setMaxLevel(level + 1); // setMaxLevel resizes parH
    para->initLBMSimulationParameter(); // init parH

    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,1,0,1);
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,6,0,6);
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,2,0,3);
    para->getParH(level)->edgeNodesXtoZ.emplace_back(0,7,0,8);

    int numNodes = 10;

    std::vector<ProcessNeighbor27> recvProcessNeighborHost(1);
    std::vector<real> recvFs(27*numNodes, 0.1);
    recvProcessNeighborHost[0].f[0] = recvFs.data();
    recvProcessNeighborHost[0].numberOfNodes = numNodes;
    
    std::vector<ProcessNeighbor27> sendProcessNeighborHost(1);
    std::vector<real> sendFs(27*numNodes, 0.0);
    sendProcessNeighborHost[0].f[0] = sendFs.data();
    sendProcessNeighborHost[0].numberOfNodes = numNodes;


    // expected
    std::vector<real> expectedFs(numNodes, 0.0);
    expectedFs[1] = 0.1;
    expectedFs[3] = 0.1;
    expectedFs[6] = 0.1;
    expectedFs[8] = 0.1;
    std::vector<real> expectedFsAllDirections;
    for (uint direction = 0; direction <= dirEND; direction ++){
        expectedFsAllDirections.insert(expectedFsAllDirections.end(), expectedFs.begin(), expectedFs.end());
    }
    
    // act
    copyEdgeNodes(para->getParH(level)->edgeNodesXtoZ, recvProcessNeighborHost, sendProcessNeighborHost);

    // convert result to std::vector
    std::vector<real> result;
    result.assign(sendProcessNeighborHost[0].f[0],  sendProcessNeighborHost[0].f[0] + 27*numNodes);

    EXPECT_THAT(result, testing::Eq(expectedFsAllDirections));
}