 cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DOPENBLAS=ON -DOPENBLAS_PATH=/opt/OpenBLAS/lib/ 
 cmake --build build_debug -j