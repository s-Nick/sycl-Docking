#include "graph.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>

Graph::Graph(int num){
    numOfVert = num;
    std::vector<bool> row;
    for(int i = 0; i < numOfVert; i++){
        
        for(int j = 0; j < numOfVert; j++){
            row.push_back(false);
        }
        adj_matrix.push_back(row);
        row.clear();
    }
}

void Graph::addEdge(unsigned int startingNode, unsigned int endingNode){
    adj_matrix[startingNode][endingNode]= true;
    adj_matrix[endingNode][startingNode] = true;
}

void Graph::removeEdge(unsigned int startingNode, unsigned int endingNode){
    if(adj_matrix[startingNode][endingNode]){
        adj_matrix[startingNode][endingNode] = false;
        adj_matrix[endingNode][startingNode] = false;
    }
}

void Graph::to_string(){
    
    for (int i = 0; i < numOfVert; i++) {
        printf("%d   : ", i);
        for (int j = 0; j < numOfVert; j++){
            int p = adj_matrix[i][j];
            printf("%d  ", p);
      }
        printf("\n");
    }
}

/*
Graph::~Graph(){
    for(int i = 0; i < numOfVert; i ++){
        adj_matrix[i].clear();
    }
    adj_matrix.clear();
}*/

void Graph::DFSlinkedNode(unsigned int startingNode, std::vector<unsigned int>& linkedNodes){
    linkedNodes.push_back(startingNode);
    //std::vector<unsigned int>::iterator it = linkedNodes.begin();
    //std::vector<unsigned int>::iterator end = linkedNodes.end();
    auto it = startingNode;
    //std::cout << "Current Node: " << startingNode << " \n";
    for(unsigned int i = 0; i < numOfVert ; i++){
        if(adj_matrix[it][i]){
            //std::cout << i << std::endl;
            if(std::find(linkedNodes.begin(), linkedNodes.end(), i) == linkedNodes.end()){
                DFSlinkedNode(i,linkedNodes);
            }
        }
    }
    //std::cout << "end\n";
    return;
}
