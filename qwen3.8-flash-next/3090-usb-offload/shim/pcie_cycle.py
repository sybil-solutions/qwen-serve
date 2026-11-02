# PCIe power cycle through the ASM2464 (vendor request 0xF3): the same off->on step the bridge firmware takes at boot.
# Fundamental reset of the GPU and the bridge's transfer state; the card's own PSU is not touched.
import time
from tinygrad.runtime.support.usb import USB3, CustomASM24Controller
devs = USB3.list_devices(0x3801, 0x0001) + USB3.list_devices(0xADD1, 0x0001)
u = USB3(devs[0][0])
def ltssm(): return u.control_read(0xE4, 1, value=0xB450)[0]
print("LTSSM before", hex(ltssm()))
u.control_write(0xF3, value=0, timeout=10000); time.sleep(2); print("LTSSM after off", hex(ltssm()))
u.control_write(0xF3, value=1, timeout=10000)
for i in range(30):
  if ltssm() == 0x78: break
  time.sleep(1)
print("LTSSM after on", hex(ltssm()), f"({i} s)")
