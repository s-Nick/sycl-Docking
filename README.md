# Parallel Programming Project
## Summer semester of '20

### Used Libraries

The Project used 2 different extern libraries to handle different aspect:

- RDKit library --> used to handle chemical aspect of the molecule and parsing the file
- Boost library --> used by RDkit.

#### Installation

In order to make the code compile and work is necessary to install the RDKit library.

All the information can be found at its repository: https://github.com/rdkit/rdkit, and must be installed following the instruction in their repository (https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md), which explain also how to install th Boost library for this project.

For simplicity it is reported the part relative to download and compile the C++ version, for further information consult the link above.

#### Building from Source

Starting with the `2018_03` release, the RDKit core C++ code is written in modern C++; for this release that means C++11.
This means that the compilers used to build it cannot be completely ancient. Here are the minimum tested versions:

- g++ v4.8: though note that the SLN parser code cannot be built with v4.8. It will automatically be disabled when this older compiler is used.
- clang v3.9: it may be that older versions of the compiler also work, but we haven't tested them.
- Visual Studio 2015: it may be that older versions of the compiler also work, but we haven't tested them.

##### Installing prerequisites from source

-   Required packages:
    - cmake. You need version 3.1 (or more recent). http://www.cmake.org if your linux distribution doesn't have an appropriate package.
    - The following are required if you are planning on using the Python wrappers
        -   The python headers. This probably means that you need to install the python-dev package (or whatever it's called) for your linux distribution.
        -   sqlite3. You also need the shared libraries. This may require that you install a sqlite3-dev package.
        -   You need to have [numpy](http://www.scipy.org/NumPy) installed.

> **note**
>
> for building with XCode4 on OS X there seems to be a problem with the version of numpy that comes with XCode4. Please see below in the (see faq) section for a workaround.


###### Installing Boost

If your linux distribution has a boost-devel package with a version >= 1.58 including the python and serialization libraries, you can use that and save yourself the steps below.

> **note**
>
> if you *do* have a version of the boost libraries pre-installed and you want to use your own version, be careful when you build the code. We've seen at least one example on a Fedora system where cmake compiled using a user-installed version of boost and then linked against the system version. This led to segmentation faults. There is a workaround for this below in the (see FAQ) section.

-   download the boost source distribution from [the boost web site](http://www.boost.org)
-   extract the source somewhere on your machine (e.g. `/usr/local/src/boost_1_58_0`)
-   build the required boost libraries. The boost site has [detailed instructions](http://www.boost.org/doc/libs/1_58_0/more/getting_started/index.html) for this, but here's an overview:
    -   `cd $BOOST`
    -   If you want to use the python wrappers: `./bootstrap.sh --with-libraries=python,serialization`
    -   If not using the python wrappers: `./bootstrap.sh --with-libraries=serialization`
    -   `./b2 install`

If you have any problems with this step, check the boost [installation instructions](http://www.boost.org/more/getting_started/unix-variants.html).

##### Building the RDKit

Fetch the source, here as tar.gz but you could use git as well:

```shellsession
$ wget https://github.com/rdkit/rdkit/archive/Release_XXXX_XX_X.tar.gz
```

-   Ensure that the prerequisites are installed
-   environment variables:
    -   `RDBASE`: the root directory of the RDKit distribution (e.g. `~/RDKit`)
    -   *Linux:* `LD_LIBRARY_PATH`: make sure it includes `$RDBASE/lib` and wherever the boost shared libraries were installed
    -   *OS X:* `DYLD_LIBRARY_PATH`: make sure it includes `$RDBASE/lib` and wherever the boost shared libraries were installed
    - The following are required if you are planning on using the Python wrappers:
        -   `PYTHONPATH`: make sure it includes `$RDBASE`
-   Building:
    -   `cd $RDBASE`
    -   `mkdir build`
    -   `cd build`
    -   `cmake ..` : See the section below on configuring the build if you need to specify a non-default version of python or if you have boost in a non-standard location
    -   `make` : this builds all libraries, regression tests, and wrappers (by default).
    -   `make install`

See below for a list of FAQ and solutions.

##### Testing the build (optional, but recommended)

-   `cd $RDBASE/build` and do `ctest`
-   you're done!

### Makefile

Within the repository I provide a Makefile that compile the variuous cuda and C++ files with the right compile flags.

In order to use it is necessary to have the Libraries and set the path to the rdkit as "RDBASE", the path to the Boost as "BOOST\_ROOT" and the path to cuda on your system naming it as "CUDA\_ROOT".

In case the driver and cuda version used are older than "Pascal" architecture and cuda 8  will be necessary to change also the architecture set at compile time.

### How to use it

Example of the command for running the code:
```shellsession
$ cd Project
$ ./main data/nameOfTheMolecule.mol2
```

### System and Driver Used

The code has been developped using and Nvidia GTX 1660 Super, using 440.100 drivers and cuda 10.2 .
