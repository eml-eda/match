## Network Compilation

To compile the ISOLDE model just run python3 compile.py, making sure MATCH is already installed and the environment is as expected(check main README.md).
The model and its default inputs are available in this directory, for other models the swap should work seamlessly.
After running the command an output directory should appear, this must be compiled with the Astral toolchains and can be run after.

## Code Compilation

To compile the test you have to set up first your environment correctly:
```
export BENDER=/home/fpgauser/git/bender/bender
export ASTRAL_PATH=/home/fpgauser/git/astral
export PATH=$ASTRAL_PATH/venv/bin:$ASTRAL_TOOLCHAIN_PATH/bin/:$PULP_TOOLCHAIN_PATH/bin/:$BENDER_PATH/:$RISCV_OPENOCD_PATH/jimtcl:$RISCV_OPENOCD_PATH/src:$VIVADO_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
```
The vivado path shouldn't be needed, but I copied here my exact configuration to build the application.
To run the build use the following command:
```
make build-host
```
To run the application you need a terminal in the astral-fpga directory and run:
```
make gdb-load-payload PAYLOAD=$APP_PATH/host.elf
```
To view the application output you can check the screen result with:
```
sudo screen /dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_0109106A-if01-port0 115200
```