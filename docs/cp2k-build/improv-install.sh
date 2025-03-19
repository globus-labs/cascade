# Add modules
module reset
module load gcc/13.2.0
module load openmpi/5.0.1-gcc-13.2.0
module list

#export MPICC=`which mpicc`
#export MPIFC=`which mpif90`

# Make the dependencies
cd tools/toolchain
./install_cp2k_toolchain.sh --mpi-mode=openmpi --with-openmpi=system | tee install.log
cp install/arch/* ../../arch/
cd ../../

# Make the code
source ./tools/toolchain/install/setup
make -j 16 ARCH=local VERSION="ssmp psmp"
