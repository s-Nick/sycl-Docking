# Parallel Programming Project
## Summer semester of '20

### Used Libraries

The Project used 2 different extern libraries to handle different aspect:

- RDKit library --> used to handle chemical aspect of the molecule and parsing the file
- Boost library --> used by RDkit.

### Makefile

Within the repository I provide a Makefile that compile the variuous cuda and CPP file with the right compile flags.

In order to use it is necessary to have the Libraries and set the path to the rdkit as RDBASE and the path to the Boost as BOOST\_ROOT
