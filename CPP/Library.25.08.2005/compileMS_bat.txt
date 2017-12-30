rem The library is being created using Borland C++ Builder 5.0 with “Language
rem compliance” compiler option set to “Borland”, therefore it requires a couple
rem of tricks to compile it using Microsoft CL v. 12 (from MS Visual Studio 6.0). 
rem - # define for if (0) {} else for // scope of definition in “for” statements
rem - FORCE:MULTIPLE linker option

cl -DMS_COMPILER -GX main.cpp associative\AdaptiveCellular.cpp associative\BAMCellular.cpp associative\CellProjective.cpp associative\FullProjective.cpp associative\Hebbian.cpp associative\nets.cpp associative\pseudoinverse.cpp lib\crawler.cpp lib\Data.cpp lib\Math_aux.cpp lib\Utils.cpp tests\ClusterTest.cpp tests\tests.cpp associative\SmallWorld.cpp modular\modular.cpp associative\DeltaCellular.cpp /link -FORCE:MULTIPLE > compile.txt