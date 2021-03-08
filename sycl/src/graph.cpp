#include "graph.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>

/**
 * Initialize the graph with an empty adjacency matrix
 **/
Graph::Graph(int num)
{
    numOfVert = num;
    std::vector<bool> row;
    for (int i = 0; i < numOfVert; i++)
    {
        for (int j = 0; j < numOfVert; j++)
        {
            row.push_back(false);
        }
        adj_matrix.push_back(row);
        row.clear();
    }
}

/**
 * Add undirected edge to the graph.
 **/
void Graph::addEdge(unsigned int startingNode, unsigned int endingNode)
{
    adj_matrix[startingNode][endingNode] = true;
    adj_matrix[endingNode][startingNode] = true;
}

/**
 * Remove edge from the graph.
 **/
void Graph::removeEdge(unsigned int startingNode, unsigned int endingNode)
{
    if (adj_matrix[startingNode][endingNode])
    {
        adj_matrix[startingNode][endingNode] = false;
        adj_matrix[endingNode][startingNode] = false;
    }
}

/**
 * Useful to print the adjacency matrix of the graph for debugging.
 **/
void Graph::to_string()
{

    for (int i = 0; i < numOfVert; i++)
    {
        printf("%d   : ", i);
        for (int j = 0; j < numOfVert; j++)
        {
            int p = adj_matrix[i][j];
            printf("%d  ", p);
        }
        printf("\n");
    }
}

/**
 * Compute the node linked to the starting one using a Depth First Algorithm.
 * Store the result in a vector passed as argument.
 * 
 * @param startingNode
 * @param linkedNodes The vector containing all the nodes of the path. Store the result.
 **/
void Graph::DFSlinkedNode(unsigned int startingNode, std::vector<unsigned int> &linkedNodes)
{
    linkedNodes.push_back(startingNode);
    auto it = startingNode;
    for (unsigned int i = 0; i < numOfVert; i++)
    {
        if (adj_matrix[it][i])
        {
            if (std::find(linkedNodes.begin(), linkedNodes.end(), i) == linkedNodes.end())
            {
                DFSlinkedNode(i, linkedNodes);
            }
        }
    }
    return;
}
