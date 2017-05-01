# Installation instructions for ROCm
The ROCm Platform brings a rich foundation to advanced computing by seamlessly integrating the CPU and GPU with the goal of solving real-world problems.

To insatll rocm, please follow:
## Installing from AMD ROCm repositories
AMD is hosting both debian and rpm repositories for the ROCm 1.4 packages. The
packages in both repositories have been signed to ensure package integrity.
Directions for each repository are given below:

## Debian repository - apt-get

### Add the ROCm apt repository
For Debian based systems, like Ubuntu, configure the Debian ROCm repository as
follows:

```shell
wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
```
The gpg key might change, so it may need to be updated when installing a new 
release.

### Install or Update
Next, update the apt-get repository list and install/update the rocm package:

>**Warning**: Before proceeding, make sure to completely
>[uninstall any pre-release ROCm packages](https://github.com/RadeonOpenCompute/ROCm#removing-pre-release-packages):

```shell
sudo apt-get update
sudo apt-get install rocm
```
Then, make the ROCm kernel your default kernel. If using grub2 as your
bootloader, you can edit the `GRUB_DEFAULT` variable in the following file:

```shell
sudo vi /etc/default/grub
sudo update-grub
```

Once complete, reboot your system.

We recommend you [verify your installation](https://github.com/RadeonOpenCompute/ROCm#verify-installation) to make sure everything completed successfully.


# Installation instructions for Eigen

## Explanation before starting

Eigen consists only of header files, hence there is nothing to compile
before you can use it. Moreover, these header files do not depend on your
platform, they are the same for everybody.

### Method 1. Installing without using CMake

You can use right away the headers in the Eigen/ subdirectory. In order
to install, just copy this Eigen/ subdirectory to your favorite location.
If you also want the unsupported features, copy the unsupported/
subdirectory too.

### Method 2. Installing using CMake

Let's call this directory 'source_dir' (where this INSTALL file is).
Before starting, create another directory which we will call 'build_dir'.

Do:
```shell
  cd build_dir
  cmake source_dir
  make install
```
The `make install` step may require administrator privileges.

You can adjust the installation destination (the "prefix")
by passing the `-DCMAKE_INSTALL_PREFIX=myprefix` option to cmake, as is
explained in the message that cmake prints at the end.


# Build and Run hipeigen direct tests

To build the direct tests for hipeigen:
Do:
```shell
  cd build_dir
  make check -j 
```

The expected test results on ROCM1.5 is as follows:
```shell
96% tests passed, 3 tests failed out of 73
The following tests FAILED:
            1 - hip_basic (SEGFAULT)
            54 - sparse_extra_3 (OTHER_FAULT)
            63 - cxx11_tensor_contract_hip (SEGFAULT)
```
We are fixing those known issues.

