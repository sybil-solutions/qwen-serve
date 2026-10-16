# drain stale bulk-IN data the ASM2464 still holds from a killed process (an armed 0xF2 read), then clear and reset
import ctypes, time
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support.usb import USB3
devs = USB3.list_devices(0x3801, 0x0001) + USB3.list_devices(0xADD1, 0x0001)
u = USB3(devs[0][0])
buf, got = (ctypes.c_ubyte * (1 << 20))(), ctypes.c_int(0)
total = 0
for i in range(64):
  rc = libusb.libusb_bulk_transfer(u.handle, 0x81, buf, len(buf), ctypes.byref(got), 300)
  total += got.value
  if rc == libusb.LIBUSB_ERROR_TIMEOUT and got.value == 0: break
  if rc < 0 and rc != libusb.LIBUSB_ERROR_TIMEOUT: print("bulk rc", rc); libusb.libusb_clear_halt(u.handle, 0x81)
print(f"drained {total} bytes in {i + 1} reads")
for ep in (0x02, 0x81): libusb.libusb_clear_halt(u.handle, ep)
print("reset", libusb.libusb_reset_device(u.handle)); time.sleep(3)
