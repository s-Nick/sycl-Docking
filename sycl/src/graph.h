#ifndef GRAPH_H_
#define GRAPH_H_

#include <vector>

class Graph
{

private:
    std::vector<std::vector<bool>> adj_matrix;
    int numOfVert;

public:
    Graph(int num);
    void addEdge(unsigned int startingNode, unsigned int endingNode);
    void removeEdge(unsigned int startingNode, unsigned int endingNode);
    //~Graph();
    void to_string();
    void DFSlinkedNode(unsigned int startingNode, std::vector<unsigned int> &linkedNodes);
};

#endif