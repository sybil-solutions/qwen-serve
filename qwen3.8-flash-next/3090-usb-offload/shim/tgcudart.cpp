// libcudart replacement for macOS: CUDA runtime API calls from clang --cuda-host-only objects are served by tgcu (Python,
// on tinygrad's NV device over the ASM2464 USB link). Streams are serialized onto one queue; graphs are captured launch lists.
#include <Python.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#define EXPORT extern "C" __attribute__((visibility("default")))

bool lazy_fault(uintptr_t a, bool write);   // below, outside the anonymous namespace
void lazy_reap();
namespace {

std::recursive_mutex g_lock;
PyObject* g_api = nullptr;
bool g_inited = false;
thread_local cudaError_t t_last = cudaSuccess;
bool g_trace = getenv("TGCU_TRACE") != nullptr;

struct FatHandle { const void* data; int mod = -1; };
struct Func { FatHandle* fat; std::string name; int kid = -1; int max_dyn_shared = 48 << 10; };
struct Var { FatHandle* fat; std::string name; size_t size; uint64_t addr = 0; };
std::map<const void*, Func> g_funcs;
std::map<const void*, Var> g_vars;
std::map<uint64_t, size_t> g_dev;                         // device allocations: va -> size
std::map<uintptr_t, std::pair<size_t, uint64_t>> g_host;  // host allocations: ptr -> (size, mirror va or 0)
std::vector<FatHandle*> g_fats;
int g_capturing = 0;
std::set<cudaStream_t> g_cap_streams;   // the capturing stream and those that waited on its events
std::set<cudaEvent_t> g_cap_events;     // events recorded inside the capture
bool captured(cudaStream_t s) { return g_capturing && g_cap_streams.count(s) != 0; }

struct Config { dim3 grid, block; size_t shmem; cudaStream_t stream; };
thread_local std::vector<Config> t_configs;
thread_local std::vector<uint8_t> t_args;

cudaError_t fail(cudaError_t e, const char* what) {
  if (g_trace || getenv("TGCU_VERBOSE")) fprintf(stderr, "[tgcu] %s -> error %d\n", what, (int)e);
  t_last = e;
  return e;
}

void py_fatal(const char* where) {
  fprintf(stderr, "[tgcu] python error in %s\n", where);
  PyErr_Print();
}

long long call_i(const char* fn, const char* fmt, ...);

// ---- large mapped regions (static mirrors): write-protected after each upload; a host write faults once per page,
// the handler marks the page dirty and unprotects it, and dirty pages are uploaded before the next submission/sync
struct Tracked { uintptr_t base; size_t size; uint64_t mirror; size_t npages; std::atomic<uint8_t>* dirty; };
std::vector<Tracked*> g_tracked;   // appended under g_lock; read lock-free by the signal handler
std::atomic<int> g_any_dirty{0};
size_t g_page = 16384;
struct sigaction g_prev_segv, g_prev_bus;
void track_fault(int sig, siginfo_t* si, void* uc) {
  uintptr_t a = (uintptr_t)si->si_addr;
  // a host access to a lazily downloaded range: its bytes are fetched (or, for a reused block, dropped), access retried
  const uint32_t esr = ((ucontext_t*)uc)->uc_mcontext->__es.__esr;
  if (lazy_fault(a, ((esr >> 26) & 0x3e) == 0x24 && ((esr >> 6) & 1))) return;
  for (Tracked* t : g_tracked) {
    if (a >= t->base && a < t->base + t->size) {
      size_t pg = (a - t->base) / g_page;
      t->dirty[pg].store(1, std::memory_order_relaxed);
      g_any_dirty.store(1, std::memory_order_release);
      mprotect((void*)(t->base + pg * g_page), g_page, PROT_READ | PROT_WRITE);
      return;  // the faulting store is retried
    }
  }
  struct sigaction* prev = sig == SIGBUS ? &g_prev_bus : &g_prev_segv;
  if (prev->sa_flags & SA_SIGINFO) prev->sa_sigaction(sig, si, uc);
  else if (prev->sa_handler == SIG_IGN) return;
  else if (prev->sa_handler != SIG_DFL) prev->sa_handler(sig);
  else { signal(sig, SIG_DFL); raise(sig); }
}
void install_fault_handler() {
  static bool installed = false;
  if (!installed) {
    g_page = (size_t)getpagesize();
    struct sigaction sa{};
    sa.sa_sigaction = track_fault;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_prev_segv);
    sigaction(SIGBUS, &sa, &g_prev_bus);
    installed = true;
  }
}
void track_region(uintptr_t base, size_t size, uint64_t mirror) {
  install_fault_handler();
  if (base % g_page || size % g_page) {  // only whole pages can be protected: such regions stay unsynced (warn)
    fprintf(stderr, "[tgcu] warning: mapped region %p+%zu is not page aligned; later host writes will not reach the GPU\n", (void*)base, size);
    return;
  }
  auto* t = new Tracked{base, size, mirror, size / g_page, nullptr};
  t->dirty = new std::atomic<uint8_t>[t->npages];
  for (size_t i = 0; i < t->npages; i++) t->dirty[i].store(0);
  g_tracked.push_back(t);
  mprotect((void*)base, size, PROT_READ);
}
void untrack_region(uintptr_t base) {
  for (Tracked* t : g_tracked)
    if (t->base == base && t->size) { mprotect((void*)t->base, t->size, PROT_READ | PROT_WRITE); t->size = 0; }
}
// called with the GIL held (inside API_ENTER): upload dirty pages in runs, then protect them again
void flush_dirty() {
  if (!g_any_dirty.exchange(0, std::memory_order_acquire)) return;
  for (Tracked* t : g_tracked) {
    if (!t->size) continue;
    size_t i = 0;
    while (i < t->npages) {
      if (!t->dirty[i].load(std::memory_order_relaxed)) { i++; continue; }
      size_t j = i;
      while (j < t->npages && t->dirty[j].load(std::memory_order_relaxed)) { t->dirty[j].store(0, std::memory_order_relaxed); j++; }
      mprotect((void*)(t->base + i * g_page), (j - i) * g_page, PROT_READ);   // protect first: writes during the upload re-dirty
      call_i("h2d_raw", "(KKK)", (unsigned long long)(t->mirror + i * g_page), (unsigned long long)(t->base + i * g_page),
             (unsigned long long)((j - i) * g_page));
      i = j;
    }
  }
}

bool ensure_init() {
  if (g_inited) return g_api != nullptr;
  g_inited = true;
  Py_InitializeEx(0);
  PyObject* mod = PyImport_ImportModule("tgcu_api");
  if (!mod) { py_fatal("import tgcu_api"); PyEval_SaveThread(); return false; }
  PyObject* r = PyObject_CallMethod(mod, "init", nullptr);
  if (!r) { py_fatal("tgcu_api.init"); PyEval_SaveThread(); return false; }
  Py_DECREF(r);
  g_api = mod;
  if (getenv("TGCU_CHECK_NAMES")) {  // resolve every registered kernel now, report the ones the fatbins do not have
    int missing = 0;
    for (auto& [host, fn] : g_funcs) {
      int m = fn.fat->mod < 0 ? (fn.fat->mod = (int)call_i("module_load", "(K)", (unsigned long long)(uintptr_t)fn.fat->data)) : fn.fat->mod;
      if (m < 0 || call_i("kernel_lookup", "(is)", m, fn.name.c_str()) < 0) { fprintf(stderr, "[tgcu] MISSING %s\n", fn.name.c_str()); missing++; }
    }
    fprintf(stderr, "[tgcu] checked %zu registered kernels, %d missing\n", g_funcs.size(), missing);
    exit(missing ? 1 : 0);
  }
  PyEval_SaveThread();  // later calls take the GIL with PyGILState_Ensure
  return true;
}

// call tgcu_api.<fn>(...) and return the result as an int64 (or a tuple through out)
struct Gil { PyGILState_STATE s; Gil() : s(PyGILState_Ensure()) {} ~Gil() { PyGILState_Release(s); } };

PyObject* callv(const char* fn, const char* fmt, va_list ap) {
  PyObject* f = PyObject_GetAttrString(g_api, fn);
  if (!f) { py_fatal(fn); return nullptr; }
  PyObject* args = fmt ? Py_VaBuildValue(fmt, ap) : PyTuple_New(0);
  if (args && !PyTuple_Check(args)) { PyObject* t = PyTuple_Pack(1, args); Py_DECREF(args); args = t; }
  PyObject* r = args ? PyObject_CallObject(f, args) : nullptr;
  Py_XDECREF(args); Py_DECREF(f);
  if (!r) py_fatal(fn);
  return r;
}
bool call_obj(PyObject** out, const char* fn, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  PyObject* r = callv(fn, fmt, ap);
  va_end(ap);
  *out = r;
  return r != nullptr;
}
long long call_i(const char* fn, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  PyObject* r = callv(fn, fmt, ap);
  va_end(ap);
  if (!r) return -1;
  long long v = PyLong_Check(r) ? PyLong_AsLongLong(r) : 0;
  Py_DECREF(r);
  return v;
}

#define API_ENTER(name) std::lock_guard<std::recursive_mutex> _lk(g_lock); \
  if (g_trace) fprintf(stderr, "[tgcu] %s\n", name); \
  if (!ensure_init()) return fail(cudaErrorInitializationError, name); \
  Gil _gil; \
  flush_dirty(); \
  lazy_reap();

bool is_dev(const void* p) {
  uint64_t a = (uint64_t)p;
  auto it = g_dev.upper_bound(a);
  if (it == g_dev.begin()) return false;
  --it;
  return a < it->first + it->second;
}
// mirror device address for a pointer inside a mapped host allocation, 0 if none
uint64_t host_mirror(const void* p) {
  uintptr_t a = (uintptr_t)p;
  auto it = g_host.upper_bound(a);
  if (it == g_host.begin()) return 0;
  --it;
  if (a >= it->first + it->second.first || !it->second.second) return 0;
  return it->second.second + (a - it->first);
}

int module_of(FatHandle* f) {
  if (f->mod < 0) f->mod = (int)call_i("module_load", "(K)", (unsigned long long)(uintptr_t)f->data);
  return f->mod;
}
Func* func_of(const void* host_fun) {
  auto it = g_funcs.find(host_fun);
  if (it == g_funcs.end()) return nullptr;
  Func& fn = it->second;
  if (fn.kid < 0) {
    int mod = module_of(fn.fat);
    if (mod >= 0) fn.kid = (int)call_i("kernel_lookup", "(is)", mod, fn.name.c_str());
    if (fn.kid < 0) fprintf(stderr, "[tgcu] kernel not found in fatbin: %s\n", fn.name.c_str());
  }
  return fn.kid >= 0 ? &fn : nullptr;
}
uint64_t var_addr(const void* host_var, size_t* size) {
  auto it = g_vars.find(host_var);
  if (it == g_vars.end()) return 0;
  Var& v = it->second;
  if (!v.addr) {
    int mod = module_of(v.fat);
    PyObject* r;
    if (mod >= 0 && call_obj(&r, "symbol", "(is)", mod, v.name.c_str())) {
      v.addr = PyLong_AsUnsignedLongLong(PyTuple_GetItem(r, 0));
      Py_DECREF(r);
    }
  }
  if (size) *size = v.size;
  return v.addr;
}

cudaError_t do_launch(Func* fn, dim3 g, dim3 b, size_t shmem, const void* params, size_t psize, cudaStream_t st, bool coop = false) {
  if (g.x == 0 || g.y == 0 || g.z == 0 || b.x * b.y * b.z == 0) return fail(cudaErrorInvalidConfiguration, "launch dims");
  if (b.x * b.y * b.z > 1024) return fail(cudaErrorInvalidConfiguration, "block too large");
  if (g_trace) fprintf(stderr, "[tgcu]   launch%s %s grid %u,%u,%u block %u,%u,%u shmem %zu params %zu\n", coop ? " cooperative" : "",
                       fn->name.c_str(), g.x, g.y, g.z, b.x, b.y, b.z, shmem, psize);
  long long r = call_i("launch", "(iIIIIIIKKKii)", fn->kid, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned long long)shmem,
                       (unsigned long long)(uintptr_t)params, (unsigned long long)psize, (int)captured(st), (int)coop);
  if (r == 2) return fail(cudaErrorCooperativeLaunchTooLarge, fn->name.c_str());
  return r == 0 ? cudaSuccess : fail(cudaErrorLaunchFailure, fn->name.c_str());
}

cudaError_t pack_and_launch(Func* fn, dim3 g, dim3 b, size_t shmem, void** args, cudaStream_t st, bool coop = false) {
  PyObject* r;
  if (!call_obj(&r, "kernel_params", "(i)", fn->kid)) return fail(cudaErrorLaunchFailure, "kernel_params");
  std::vector<uint8_t> buf;
  Py_ssize_t n = PyList_Size(r);
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject* t = PyList_GetItem(r, i);
    size_t off = PyLong_AsSize_t(PyTuple_GetItem(t, 0)), sz = PyLong_AsSize_t(PyTuple_GetItem(t, 1));
    if (buf.size() < off + sz) buf.resize(off + sz);
    memcpy(buf.data() + off, args[i], sz);
  }
  Py_DECREF(r);
  return do_launch(fn, g, b, shmem, buf.data(), buf.size(), st, coop);
}

const char* err_name(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorNotSupported: return "cudaErrorNotSupported";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
    case cudaErrorPeerAccessUnsupported: return "cudaErrorPeerAccessUnsupported";
    case cudaErrorStreamCaptureUnsupported: return "cudaErrorStreamCaptureUnsupported";
    case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
    case cudaErrorCooperativeLaunchTooLarge: return "cudaErrorCooperativeLaunchTooLarge";
    default: return "cudaErrorUnknown";
  }
}

void fill_props(cudaDeviceProp* p) {
  memset(p, 0, sizeof(*p));
  snprintf(p->name, sizeof(p->name), "NVIDIA GeForce RTX 3090 (tgcu USB)");
  p->totalGlobalMem = 24ull << 30;
  p->sharedMemPerBlock = 48 << 10;
  p->regsPerBlock = 65536;
  p->warpSize = 32;
  p->memPitch = INT32_MAX;
  p->maxThreadsPerBlock = 1024;
  p->maxThreadsDim[0] = 1024; p->maxThreadsDim[1] = 1024; p->maxThreadsDim[2] = 64;
  p->maxGridSize[0] = INT32_MAX; p->maxGridSize[1] = 65535; p->maxGridSize[2] = 65535;
  p->totalConstMem = 65536;
  p->major = 8; p->minor = 6;
  p->textureAlignment = 512;
  p->multiProcessorCount = 82;
  p->integrated = 0;
  p->canMapHostMemory = 1;
  p->computeMode = 0;
  p->concurrentKernels = 1;
  p->pciBusID = 2;
  p->asyncEngineCount = 1;
  p->unifiedAddressing = 1;
  p->memoryBusWidth = 384;
  p->l2CacheSize = 6 << 20;
  p->persistingL2CacheMaxSize = 0;
  p->maxThreadsPerMultiProcessor = 1536;
  p->sharedMemPerMultiprocessor = 100 << 10;
  p->regsPerMultiprocessor = 65536;
  p->managedMemory = 0;
  p->sharedMemPerBlockOptin = 99 << 10;
  p->reservedSharedMemPerBlock = 1024;
  p->maxBlocksPerMultiProcessor = 16;
  p->hostRegisterSupported = 1;
  p->globalL1CacheSupported = 1;
  p->localL1CacheSupported = 1;
  p->streamPrioritiesSupported = 1;
  p->cooperativeLaunch = 1;   // served by tgcu_api.launch(coop=1): grid workspace + co-residency check
}

}  // namespace

// *** registration (emitted by clang into every .cu host object) ***
EXPORT void** __cudaRegisterFatBinary(void* fatCubin) {
  struct Wrapper { int magic; int version; const void* data; void* filename; };
  auto* w = (Wrapper*)fatCubin;
  auto* h = new FatHandle{w->data};
  g_fats.push_back(h);
  return (void**)h;
}
EXPORT void __cudaRegisterFatBinaryEnd(void**) {}
EXPORT void __cudaUnregisterFatBinary(void**) {}
EXPORT void __cudaRegisterFunction(void** handle, const char* hostFun, char* deviceFun, const char* deviceName, int, uint3*, uint3*,
                                   dim3*, dim3*, int*) {
  g_funcs[(const void*)hostFun] = Func{(FatHandle*)handle, deviceName ? deviceName : deviceFun};
}
EXPORT void __cudaRegisterVar(void** handle, char* hostVar, char*, const char* deviceName, int, size_t size, int, int) {
  g_vars[(const void*)hostVar] = Var{(FatHandle*)handle, deviceName, size};
}
EXPORT void __cudaRegisterManagedVar(void** handle, void** hostVarPtrAddress, char*, const char* deviceName, int, size_t size, int, int) {
  g_vars[(const void*)hostVarPtrAddress] = Var{(FatHandle*)handle, deviceName, size};
}
EXPORT char __cudaInitModule(void**) { return 1; }
EXPORT void __cudaRegisterTexture(void**, const void*, const void**, const char*, int, int, int) {}
EXPORT void __cudaRegisterSurface(void**, const void*, const void**, const char*, int, int) {}

// *** launch ***
EXPORT cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
  t_configs.push_back({grid, block, shmem, stream});
  t_args.clear();
  return cudaSuccess;
}
EXPORT cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset) {
  if (t_args.size() < offset + size) t_args.resize(offset + size);
  memcpy(t_args.data() + offset, arg, size);
  return cudaSuccess;
}
EXPORT cudaError_t cudaLaunch(const void* func) {
  if (t_configs.empty()) return fail(cudaErrorInvalidConfiguration, "cudaLaunch without config");
  Config c = t_configs.back(); t_configs.pop_back();
  std::vector<uint8_t> args; args.swap(t_args);
  API_ENTER("cudaLaunch");
  Func* fn = func_of(func);
  if (!fn) return fail(cudaErrorInvalidDeviceFunction, "cudaLaunch");
  return do_launch(fn, c.grid, c.block, c.shmem, args.data(), args.size(), c.stream);
}
EXPORT unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shmem, void* stream) {
  t_configs.push_back({grid, block, shmem, (cudaStream_t)stream});
  return 0;
}
EXPORT cudaError_t __cudaPopCallConfiguration(dim3* grid, dim3* block, size_t* shmem, void* stream) {
  if (t_configs.empty()) return cudaErrorInvalidConfiguration;
  Config c = t_configs.back(); t_configs.pop_back();
  *grid = c.grid; *block = c.block; *shmem = c.shmem; *(cudaStream_t*)stream = c.stream;
  return cudaSuccess;
}
EXPORT cudaError_t cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shmem, cudaStream_t stream) {
  API_ENTER("cudaLaunchKernel");
  Func* fn = func_of(func);
  if (!fn) return fail(cudaErrorInvalidDeviceFunction, "cudaLaunchKernel");
  return pack_and_launch(fn, grid, block, shmem, args, stream);
}
EXPORT cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* cfg, const void* func, void** args) {
  API_ENTER("cudaLaunchKernelExC");
  bool coop = false;
  for (unsigned i = 0; i < cfg->numAttrs; i++) {
    if (cfg->attrs[i].id == cudaLaunchAttributeClusterDimension &&
        cfg->attrs[i].val.clusterDim.x * cfg->attrs[i].val.clusterDim.y * cfg->attrs[i].val.clusterDim.z > 1)
      return fail(cudaErrorNotSupported, "cluster launch");
    if (cfg->attrs[i].id == cudaLaunchAttributeCooperative) coop = cfg->attrs[i].val.cooperative != 0;
  }
  Func* fn = func_of(func);
  if (!fn) return fail(cudaErrorInvalidDeviceFunction, "cudaLaunchKernelExC");
  return pack_and_launch(fn, cfg->gridDim, cfg->blockDim, cfg->dynamicSmemBytes, args, cfg->stream, coop);
}
// grid.sync() kernels: every block co-resident (else cudaErrorCooperativeLaunchTooLarge), grid workspace in c[0][0x8c]/0x90
EXPORT cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 grid, dim3 block, void** args, size_t shmem, cudaStream_t stream) {
  API_ENTER("cudaLaunchCooperativeKernel");
  Func* fn = func_of(func);
  if (!fn) return fail(cudaErrorInvalidDeviceFunction, "cudaLaunchCooperativeKernel");
  return pack_and_launch(fn, grid, block, shmem, args, stream, true);
}

// *** memory ***
EXPORT cudaError_t cudaMalloc(void** p, size_t size) {
  API_ENTER("cudaMalloc");
  long long va = call_i("malloc", "(K)", (unsigned long long)size);
  if (va <= 0) return fail(cudaErrorMemoryAllocation, "cudaMalloc");
  g_dev[(uint64_t)va] = size;
  *p = (void*)va;
  return cudaSuccess;
}
EXPORT cudaError_t cudaMallocAsync(void** p, size_t size, cudaStream_t) { return cudaMalloc(p, size); }
EXPORT cudaError_t cudaFree(void* p) {
  if (!p) return cudaSuccess;
  API_ENTER("cudaFree");
  if (!g_dev.erase((uint64_t)p)) return fail(cudaErrorInvalidValue, "cudaFree");
  call_i("free", "(K)", (unsigned long long)(uintptr_t)p);
  return cudaSuccess;
}
EXPORT cudaError_t cudaFreeAsync(void* p, cudaStream_t) { return cudaFree(p); }

static const size_t kMappedMax = 8u << 20;   // tgcu_api.MAPPED_MAX: larger mapped regions are page-tracked static mirrors
static cudaError_t host_alloc(void** p, size_t size, unsigned flags) {
  void* h = nullptr;
  const size_t page = (size_t)getpagesize();
  const size_t alloc = (std::max<size_t>(size, 1) + page - 1) / page * page;   // whole pages, so they can be write-protected
  if (posix_memalign(&h, page, alloc)) return fail(cudaErrorMemoryAllocation, "host alloc");
  memset(h, 0, alloc);
  uint64_t mirror = 0;
  if (flags & cudaHostAllocMapped) {
    API_ENTER("cudaHostAlloc(mapped)");
    const size_t msize = size > kMappedMax ? alloc : size;
    mirror = (uint64_t)call_i("map_host", "(KK)", (unsigned long long)(uintptr_t)h, (unsigned long long)msize);
    if (mirror) g_dev[mirror] = msize;
    if (mirror && size > kMappedMax) track_region((uintptr_t)h, alloc, mirror);
  }
  { std::lock_guard<std::recursive_mutex> lk(g_lock); g_host[(uintptr_t)h] = {size, mirror}; }
  *p = h;
  return cudaSuccess;
}
EXPORT cudaError_t cudaHostAlloc(void** p, size_t size, unsigned flags) { return host_alloc(p, size, flags); }
EXPORT cudaError_t cudaMallocHost(void** p, size_t size) { return host_alloc(p, size, 0); }
EXPORT cudaError_t cudaFreeHost(void* p) {
  std::pair<size_t, uint64_t> e;
  {
    std::lock_guard<std::recursive_mutex> lk(g_lock);
    auto it = g_host.find((uintptr_t)p);
    if (it == g_host.end()) return fail(cudaErrorInvalidValue, "cudaFreeHost");
    e = it->second; g_host.erase(it);
  }
  if (e.second) {
    API_ENTER("cudaFreeHost");
    untrack_region((uintptr_t)p);
    g_dev.erase(e.second);
    call_i("unmap_host", "(K)", (unsigned long long)(uintptr_t)p);
  }
  free(p);
  return cudaSuccess;
}
EXPORT cudaError_t cudaHostRegister(void* p, size_t size, unsigned flags) {
  uint64_t mirror = 0;
  if (flags & cudaHostRegisterMapped) {
    API_ENTER("cudaHostRegister(mapped)");
    mirror = (uint64_t)call_i("map_host", "(KK)", (unsigned long long)(uintptr_t)p, (unsigned long long)size);
    if (mirror) g_dev[mirror] = size;
    if (mirror && size > kMappedMax) track_region((uintptr_t)p, size, mirror);
  }
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  g_host[(uintptr_t)p] = {size, mirror};
  return cudaSuccess;
}
EXPORT cudaError_t cudaHostUnregister(void* p) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  auto it = g_host.find((uintptr_t)p);
  if (it == g_host.end()) return fail(cudaErrorHostMemoryNotRegistered, "cudaHostUnregister");
  if (it->second.second) {
    API_ENTER("cudaHostUnregister");
    untrack_region((uintptr_t)p);
    g_dev.erase(it->second.second);
    call_i("unmap_host", "(K)", (unsigned long long)(uintptr_t)p);
  }
  g_host.erase(it);
  return cudaSuccess;
}
EXPORT cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  uint64_t m = host_mirror(h);
  if (!m) return fail(cudaErrorInvalidValue, "cudaHostGetDevicePointer (region not mirrored)");
  *d = (void*)m;
  return cudaSuccess;
}
EXPORT cudaError_t cudaHostGetFlags(unsigned* f, void* h) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  *f = host_mirror(h) ? cudaHostAllocMapped : 0;
  return cudaSuccess;
}

// ---- lazy downloads: a large device-to-host copy (a conversation checkpoint: ~118 MB of GDN state over a ~150 MB/s
// link) is copied into a VRAM shadow instead, and the page-aligned interior of the host range is made inaccessible.
// A copy back to the device from such a range is a VRAM copy; any host access faults and the handler downloads the
// bytes first (then the range is plain memory again). TGCU_LAZY_D2H_MB (default 768, 0 = off) caps the shadows; past
// it the oldest range is downloaded.
struct Lazy { uintptr_t lo, hi; uint64_t shadow; uint64_t clock; long long sem; };
std::map<uintptr_t, Lazy> g_lazy;   // by lo, under g_lock
size_t g_lazy_bytes = 0;
uint64_t g_lazy_clock = 0;
const size_t kLazyBudget = [] { const char* e = getenv("TGCU_LAZY_D2H_MB"); return (size_t)(e ? atoll(e) : 768) << 20; }();
const size_t kLazyMin = 1 << 20;
const bool g_lazy_verbose = getenv("TGCU_LAZY_VERBOSE") != nullptr;
static bool lazy_still_protected(const Lazy& z) {   // a freed-and-reused range is mapped afresh: not ours any more
  mach_vm_address_t at = z.lo;                       // every VM entry over [lo, hi) must still be inaccessible
  while (at < z.hi) {
    mach_vm_address_t a = at; mach_vm_size_t sz = 0;
    vm_region_basic_info_data_64_t info; mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64; mach_port_t obj = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &a, &sz, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS) return false;
    if (a > at || info.protection != VM_PROT_NONE) return false;
    at = a + sz;
  }
  return true;
}
std::vector<uint64_t> g_lazy_dead;   // shadows to free once no graph that spins on a flag can be running
static void lazy_release(std::map<uintptr_t, Lazy>::iterator it, bool fetch) {
  Lazy z = it->second;
  g_lazy.erase(it);
  g_lazy_bytes -= z.hi - z.lo;
  bool ours = lazy_still_protected(z);
  mprotect((void*)z.lo, z.hi - z.lo, PROT_READ | PROT_WRITE);   // harmless on a fresh mapping, fails on an unmapped one
  if (ours && fetch) call_i("lazy_fetch", "(KKKL)", (unsigned long long)z.lo, (unsigned long long)z.shadow, (unsigned long long)(z.hi - z.lo), z.sem);
  g_lazy_dead.push_back(z.shadow);
  if (g_lazy_verbose) fprintf(stderr, "[tgcu] lazy range %p+%zu %s\n", (void*)z.lo, (size_t)(z.hi - z.lo), !ours ? "stale" : fetch ? "downloaded" : "dropped");
}
// before host bytes [lo, hi) are read (fetch) or wholly overwritten: every lazy range they touch becomes plain memory
static void lazy_resolve(uintptr_t lo, uintptr_t hi, bool overwrite) {
  if (g_lazy.empty()) return;
  auto it = g_lazy.upper_bound(lo);
  if (it != g_lazy.begin()) --it;
  while (it != g_lazy.end() && it->second.lo < hi) {
    auto cur = it++;
    if (cur->second.hi <= lo) continue;
    bool inside = cur->second.lo >= lo && cur->second.hi <= hi;
    lazy_release(cur, !(overwrite && inside));
  }
}
// dead shadows are freed at an API call when no spinning graph may run (a free synchronizes, and in a signal handler
// there is no Python at all)
static void lazy_free_dead() {
  if (!call_i("lazy_ok", nullptr)) return;
  for (uint64_t sh : g_lazy_dead) call_i("free", "(K)", (unsigned long long)sh);
  g_lazy_dead.clear();
}
void lazy_reap() { if (!g_lazy_dead.empty()) lazy_free_dead(); }
bool lazy_fault(uintptr_t a, bool write) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  if (g_lazy.empty()) return false;
  auto it = g_lazy.upper_bound(a);
  if (it == g_lazy.begin()) return false;
  --it;
  if (a >= it->second.hi) return false;
  if (write && a - it->second.lo < g_page) {
    // A store to the first page: the program only ever fills these ranges with cudaMemcpy, so a host store means the
    // block was freed and handed out again (a new std::vector zero-filling it from the start) - its old bytes are dead
    Lazy z = it->second;
    g_lazy.erase(it);
    g_lazy_bytes -= z.hi - z.lo;
    mprotect((void*)z.lo, z.hi - z.lo, PROT_READ | PROT_WRITE);
    g_lazy_dead.push_back(z.shadow);
    if (g_lazy_verbose) fprintf(stderr, "[tgcu] lazy range %p+%zu reused (dropped)\n", (void*)z.lo, (size_t)(z.hi - z.lo));
    return true;
  }
  Gil g;
  lazy_release(it, true);
  return true;
}
// d2h into [dst, dst+n): a VRAM shadow for the page interior, real copies for the edges. false: not taken
static bool lazy_d2h(void* dst, const void* src, size_t n) {
  if (!kLazyBudget || n < kLazyMin || !call_i("lazy_ok", nullptr)) return false;
  install_fault_handler();
  uintptr_t d = (uintptr_t)dst, lo = (d + g_page - 1) / g_page * g_page, hi = (d + n) / g_page * g_page;
  if (hi <= lo || hi - lo < kLazyMin / 2 || hi - lo > kLazyBudget) return false;
  while (g_lazy_bytes + (hi - lo) > kLazyBudget && !g_lazy.empty()) {   // the oldest range goes to the host
    auto old = g_lazy.begin();
    for (auto it = g_lazy.begin(); it != g_lazy.end(); ++it) if (it->second.clock < old->second.clock) old = it;
    lazy_release(old, true);
  }
  uint64_t shadow = (uint64_t)call_i("malloc", "(K)", (unsigned long long)(hi - lo));
  if (!shadow || (long long)shadow == -1) { PyErr_Clear(); return false; }
  const uint64_t s = (uint64_t)(uintptr_t)src;
  if (call_i("d2d", "(KKKi)", (unsigned long long)shadow, (unsigned long long)(s + (lo - d)), (unsigned long long)(hi - lo), 0) != 0) {
    call_i("free", "(K)", (unsigned long long)shadow);
    return false;
  }
  if (lo > d) call_i("d2h", "(KKK)", (unsigned long long)d, (unsigned long long)s, (unsigned long long)(lo - d));
  if (d + n > hi) call_i("d2h", "(KKK)", (unsigned long long)hi, (unsigned long long)(s + (hi - d)), (unsigned long long)(d + n - hi));
  const long long sem = call_i("lazy_submit", nullptr);   // the shadow copy goes now; a fetch waits for it alone
  mprotect((void*)lo, hi - lo, PROT_NONE);
  g_lazy[lo] = Lazy{lo, hi, shadow, ++g_lazy_clock, sem};
  g_lazy_bytes += hi - lo;
  if (g_lazy_verbose) fprintf(stderr, "[tgcu] lazy range %p+%zu kept in VRAM\n", (void*)lo, (size_t)(hi - lo));
  return true;
}
// h2d from [src, src+n) that covers one lazy range: the edges from the host, the interior from its VRAM shadow
static bool lazy_h2d(void* dst, const void* src, size_t n) {
  if (g_lazy.empty()) return false;
  uintptr_t s = (uintptr_t)src, e = s + n;
  auto it = g_lazy.upper_bound(s);
  if (it != g_lazy.begin()) --it;
  if (it != g_lazy.end() && it->second.hi <= s) ++it;
  if (it == g_lazy.end() || it->second.lo >= e) return false;     // touches none
  Lazy z = it->second;
  auto nx = std::next(it);
  if (z.lo < s || z.hi > e || (nx != g_lazy.end() && nx->second.lo < e) || !lazy_still_protected(z)) {
    lazy_resolve(s, e, false);                                      // partial or several: plain memory first
    return false;
  }
  const uint64_t d = (uint64_t)(uintptr_t)dst;
  if (z.lo > s) call_i("h2d", "(KKK)", (unsigned long long)d, (unsigned long long)s, (unsigned long long)(z.lo - s));
  call_i("d2d", "(KKKi)", (unsigned long long)(d + (z.lo - s)), (unsigned long long)z.shadow, (unsigned long long)(z.hi - z.lo), 0);
  if (e > z.hi) call_i("h2d", "(KKK)", (unsigned long long)(d + (z.hi - s)), (unsigned long long)z.hi, (unsigned long long)(e - z.hi));
  it->second.clock = ++g_lazy_clock;
  return true;
}

static cudaError_t memcpy_impl(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t st = 0) {
  if (!n) return cudaSuccess;
  bool dd = is_dev(dst), sd = is_dev(src);
  if (kind == cudaMemcpyDefault) kind = dd ? (sd ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : (sd ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
  long long r = 0;
  switch (kind) {
    case cudaMemcpyHostToHost:
      lazy_resolve((uintptr_t)src, (uintptr_t)src + n, false); lazy_resolve((uintptr_t)dst, (uintptr_t)dst + n, true);
      memmove(dst, src, n); break;
    case cudaMemcpyHostToDevice:
      if (lazy_h2d(dst, src, n)) break;
      r = call_i("h2d", "(KKK)", (unsigned long long)(uintptr_t)dst, (unsigned long long)(uintptr_t)src, (unsigned long long)n); break;
    case cudaMemcpyDeviceToHost:
      lazy_resolve((uintptr_t)dst, (uintptr_t)dst + n, true);
      if (lazy_d2h(dst, src, n)) break;
      r = call_i("d2h", "(KKK)", (unsigned long long)(uintptr_t)dst, (unsigned long long)(uintptr_t)src, (unsigned long long)n); break;
    case cudaMemcpyDeviceToDevice: r = call_i("d2d", "(KKKi)", (unsigned long long)(uintptr_t)dst, (unsigned long long)(uintptr_t)src, (unsigned long long)n, (int)captured(st)); break;
    default: return fail(cudaErrorInvalidMemcpyDirection, "memcpy kind");
  }
  return r == 0 ? cudaSuccess : fail(cudaErrorUnknown, "memcpy");
}
EXPORT cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
  API_ENTER("cudaMemcpy");
  cudaError_t e = memcpy_impl(dst, src, n, kind);
  if (e == cudaSuccess && kind == cudaMemcpyDeviceToDevice) call_i("synchronize", nullptr);
  return e;
}
EXPORT cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t stream) {
  API_ENTER("cudaMemcpyAsync");
  if (captured(stream) && kind != cudaMemcpyDeviceToDevice && !(kind == cudaMemcpyDefault && is_dev(dst) && is_dev(src)))
    return fail(cudaErrorStreamCaptureUnsupported, "host memcpy during capture");
  return memcpy_impl(dst, src, n, kind, stream);
}
EXPORT cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                                     cudaMemcpyKind kind, cudaStream_t stream) {
  API_ENTER("cudaMemcpy2DAsync");
  bool dd = is_dev(dst), sd = is_dev(src);
  if (kind == cudaMemcpyDefault) kind = dd ? (sd ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : (sd ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
  if (kind == cudaMemcpyDeviceToDevice)
    return call_i("d2d_2d", "(KKKKKKi)", (unsigned long long)(uintptr_t)dst, (unsigned long long)dpitch, (unsigned long long)(uintptr_t)src,
                  (unsigned long long)spitch, (unsigned long long)width, (unsigned long long)height, (int)captured(stream)) == 0 ? cudaSuccess : fail(cudaErrorUnknown, "2d");
  for (size_t r = 0; r < height; r++) {
    cudaError_t e = memcpy_impl((char*)dst + r * dpitch, (const char*)src + r * spitch, width, kind);
    if (e) return e;
  }
  return cudaSuccess;
}
EXPORT cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
  cudaError_t e = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, 0);
  if (e) return e;
  API_ENTER("cudaMemcpy2D sync");
  call_i("synchronize", nullptr);
  return cudaSuccess;
}
EXPORT cudaError_t cudaMemcpyPeerAsync(void* dst, int, const void* src, int, size_t n, cudaStream_t s) {
  return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s);
}
EXPORT cudaError_t cudaMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count, cudaMemcpyAttributes*, size_t*, size_t,
                                        size_t* failIdx, cudaStream_t s) {
  for (size_t i = 0; i < count; i++) {
    cudaError_t e = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, s);
    if (e) { if (failIdx) *failIdx = i; return e; }
  }
  return cudaSuccess;
}
EXPORT cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t stream) {
  API_ENTER("cudaMemsetAsync");
  return call_i("memset", "(KiKi)", (unsigned long long)(uintptr_t)p, v, (unsigned long long)n, (int)captured(stream)) == 0 ? cudaSuccess : fail(cudaErrorUnknown, "memset");
}
EXPORT cudaError_t cudaMemset(void* p, int v, size_t n) {
  cudaError_t e = cudaMemsetAsync(p, v, n, 0);
  if (e) return e;
  API_ENTER("cudaMemset sync");
  call_i("synchronize", nullptr);
  return cudaSuccess;
}
EXPORT cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total) {
  API_ENTER("cudaMemGetInfo");
  PyObject* r;
  if (!call_obj(&r, "mem_info", nullptr)) return fail(cudaErrorUnknown, "mem_info");
  *free_b = PyLong_AsSize_t(PyTuple_GetItem(r, 0)); *total = PyLong_AsSize_t(PyTuple_GetItem(r, 1));
  Py_DECREF(r);
  return cudaSuccess;
}
EXPORT cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a, const void* p) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  memset(a, 0, sizeof(*a));
  if (is_dev(p)) { a->type = cudaMemoryTypeDevice; a->devicePointer = (void*)p; }
  else if (uint64_t m = host_mirror(p)) { a->type = cudaMemoryTypeHost; a->hostPointer = (void*)p; a->devicePointer = (void*)m; }
  else a->type = cudaMemoryTypeUnregistered;
  return cudaSuccess;
}

// *** symbols ***
EXPORT cudaError_t cudaGetSymbolAddress(void** d, const void* sym) {
  API_ENTER("cudaGetSymbolAddress");
  uint64_t a = var_addr(sym, nullptr);
  if (!a) return fail(cudaErrorInvalidSymbol, "cudaGetSymbolAddress");
  *d = (void*)a;
  return cudaSuccess;
}
EXPORT cudaError_t cudaGetSymbolSize(size_t* sz, const void* sym) {
  API_ENTER("cudaGetSymbolSize");
  if (!var_addr(sym, sz)) return fail(cudaErrorInvalidSymbol, "cudaGetSymbolSize");
  return cudaSuccess;
}
EXPORT cudaError_t cudaMemcpyToSymbolAsync(const void* sym, const void* src, size_t n, size_t off, cudaMemcpyKind kind, cudaStream_t) {
  API_ENTER("cudaMemcpyToSymbolAsync");
  uint64_t a = var_addr(sym, nullptr);
  if (!a) return fail(cudaErrorInvalidSymbol, "cudaMemcpyToSymbol");
  return memcpy_impl((void*)(a + off), src, n, kind == cudaMemcpyDefault ? (is_dev(src) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) : kind);
}
EXPORT cudaError_t cudaMemcpyToSymbol(const void* sym, const void* src, size_t n, size_t off, cudaMemcpyKind kind) {
  return cudaMemcpyToSymbolAsync(sym, src, n, off, kind, 0);
}
EXPORT cudaError_t cudaMemcpyFromSymbol(void* dst, const void* sym, size_t n, size_t off, cudaMemcpyKind kind) {
  API_ENTER("cudaMemcpyFromSymbol");
  uint64_t a = var_addr(sym, nullptr);
  if (!a) return fail(cudaErrorInvalidSymbol, "cudaMemcpyFromSymbol");
  return memcpy_impl(dst, (void*)(a + off), n, kind == cudaMemcpyDefault ? cudaMemcpyDeviceToHost : kind);
}

// *** sync, streams, events ***
EXPORT cudaError_t cudaDeviceSynchronize() {
  API_ENTER("cudaDeviceSynchronize");
  return call_i("synchronize", nullptr) == 0 ? cudaSuccess : fail(cudaErrorLaunchFailure, "cudaDeviceSynchronize");
}
EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t) {
  API_ENTER("cudaStreamSynchronize");
  return call_i("synchronize", nullptr) == 0 ? cudaSuccess : fail(cudaErrorLaunchFailure, "cudaStreamSynchronize");
}
EXPORT cudaError_t cudaStreamQuery(cudaStream_t) {
  API_ENTER("cudaStreamQuery");
  call_i("flush", nullptr);
  return call_i("query", nullptr) ? cudaSuccess : cudaErrorNotReady;
}
static uintptr_t g_next_stream = 0x5100;
EXPORT cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = (cudaStream_t)(g_next_stream += 16); return cudaSuccess; }
EXPORT cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { return cudaStreamCreate(s); }
EXPORT cudaError_t cudaStreamCreateWithPriority(cudaStream_t* s, unsigned, int) { return cudaStreamCreate(s); }
EXPORT cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
EXPORT cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  if (g_capturing && g_cap_events.count(e)) g_cap_streams.insert(s);   // a fork: s joins the capture
  return cudaSuccess;
}
EXPORT cudaError_t cudaStreamGetPriority(cudaStream_t, int* p) { *p = 0; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceGetStreamPriorityRange(int* lo, int* hi) { *lo = 0; *hi = -1; return cudaSuccess; }

struct Event { std::chrono::steady_clock::time_point t; int slot = -1; };
// events carry a GPU timestamp (cudaEventElapsedTime is GPU time, as on CUDA) and recording one submits nothing:
// every stream feeds one in-order queue, so a wait on an event is already ordered. TGCU_GPU_EVENTS=0: host time and
// a flush per record, as before
static const bool g_gpu_events = [] { const char* e = getenv("TGCU_GPU_EVENTS"); return !(e && e[0] == '0'); }();
EXPORT cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = (cudaEvent_t) new Event{}; return cudaSuccess; }
EXPORT cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { return cudaEventCreate(e); }
EXPORT cudaError_t cudaEventDestroy(cudaEvent_t e) { delete (Event*)e; return cudaSuccess; }
EXPORT cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
  if (captured(s)) { std::lock_guard<std::recursive_mutex> lk(g_lock); g_cap_events.insert(e); return cudaSuccess; }
  { API_ENTER("cudaEventRecord");
    if (g_gpu_events) ((Event*)e)->slot = (int) call_i("event_record", nullptr);
    else call_i("flush", nullptr); }
  ((Event*)e)->t = std::chrono::steady_clock::now();
  return cudaSuccess;
}
EXPORT cudaError_t cudaEventRecordWithFlags(cudaEvent_t e, cudaStream_t s, unsigned) { return cudaEventRecord(e, s); }
EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t e) {
  API_ENTER("cudaEventSynchronize");
  call_i("synchronize", nullptr);
  ((Event*)e)->t = std::max(((Event*)e)->t, std::chrono::steady_clock::now());
  return cudaSuccess;
}
EXPORT cudaError_t cudaEventQuery(cudaEvent_t) {
  API_ENTER("cudaEventQuery");
  call_i("flush", nullptr);
  return call_i("query", nullptr) ? cudaSuccess : cudaErrorNotReady;
}
EXPORT cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
  if (((Event*)a)->slot >= 0 && ((Event*)b)->slot >= 0) {
    API_ENTER("cudaEventElapsedTime");
    *ms = (float) ((double) call_i("event_elapsed_ns", "(ii)", ((Event*)a)->slot, ((Event*)b)->slot) / 1e6);
    return cudaSuccess;
  }
  *ms = std::chrono::duration<float, std::milli>(((Event*)b)->t - ((Event*)a)->t).count();
  return cudaSuccess;
}
EXPORT cudaError_t cudaLaunchHostFunc(cudaStream_t s, cudaHostFn_t fn, void* data) {
  {
    API_ENTER("cudaLaunchHostFunc");
    if (captured(s)) return fail(cudaErrorStreamCaptureUnsupported, "host func in capture");
    call_i("synchronize", nullptr);
  }
  fn(data);
  return cudaSuccess;
}

// *** graphs ***
EXPORT cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode) {
  API_ENTER("cudaStreamBeginCapture");
  call_i("flush", nullptr);
  call_i("begin_capture", nullptr);
  g_capturing = 1;
  g_cap_streams = {s};
  g_cap_events.clear();
  return cudaSuccess;
}
EXPORT cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) {
  API_ENTER("cudaStreamEndCapture");
  g_capturing = 0;
  g_cap_streams.clear();
  g_cap_events.clear();
  long long gi = call_i("end_capture", nullptr);
  if (gi < 0) return fail(cudaErrorStreamCaptureInvalidated, "cudaStreamEndCapture");
  *g = (cudaGraph_t)(uintptr_t)(gi + 1);
  return cudaSuccess;
}
EXPORT cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* st) {
  *st = captured(s) ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
  return cudaSuccess;
}
EXPORT cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t s, cudaStreamCaptureStatus* st, unsigned long long* id, cudaGraph_t* g,
                                               const cudaGraphNode_t** deps, size_t* ndeps) {
  *st = captured(s) ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
  if (id) *id = 1;
  if (g) *g = nullptr;
  if (deps) *deps = nullptr;
  if (ndeps) *ndeps = 0;
  return cudaSuccess;
}
EXPORT cudaError_t cudaStreamGetCaptureInfo_v3(cudaStream_t s, cudaStreamCaptureStatus* st, unsigned long long* id, cudaGraph_t* g,
                                               const cudaGraphNode_t** deps, const cudaGraphEdgeData** edges, size_t* ndeps) {
  if (edges) *edges = nullptr;
  return cudaStreamGetCaptureInfo_v2(s, st, id, g, deps, ndeps);
}
EXPORT cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode*) { return cudaSuccess; }
EXPORT cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long) {
  API_ENTER("cudaGraphInstantiate");
  long long ei = call_i("graph_instantiate", "(i)", (int)((uintptr_t)g - 1));
  if (ei < 0) return fail(cudaErrorMemoryAllocation, "cudaGraphInstantiate");
  *e = (cudaGraphExec_t)(uintptr_t)(ei + 1);
  return cudaSuccess;
}
EXPORT cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* e, cudaGraph_t g, unsigned long long f) { return cudaGraphInstantiate(e, g, f); }
EXPORT cudaError_t cudaGraphLaunch(cudaGraphExec_t e, cudaStream_t) {
  API_ENTER("cudaGraphLaunch");
  if (call_i("graph_launch", "(i)", (int)((uintptr_t)e - 1)) != 0) return fail(cudaErrorLaunchFailure, "cudaGraphLaunch");
  return cudaSuccess;
}
EXPORT cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
EXPORT cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
EXPORT cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
EXPORT cudaError_t cudaGraphGetNodes(cudaGraph_t g, cudaGraphNode_t* nodes, size_t* n) {
  API_ENTER("cudaGraphGetNodes");
  size_t cnt = (size_t)call_i("graph_nodes", "(i)", (int)((uintptr_t)g - 1));
  if (nodes) for (size_t i = 0; i < std::min(cnt, *n); i++) nodes[i] = (cudaGraphNode_t)(uintptr_t)(i + 1);
  *n = cnt;
  return cudaSuccess;
}
EXPORT cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType* t) { *t = cudaGraphNodeTypeKernel; return cudaSuccess; }
EXPORT cudaError_t cudaGraphGetEdges(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, size_t* n) { *n = 0; return cudaSuccess; }

// *** device ***
EXPORT cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
EXPORT cudaError_t cudaSetDevice(int d) { return d == 0 ? cudaSuccess : fail(cudaErrorInvalidDevice, "cudaSetDevice"); }
EXPORT cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
EXPORT cudaError_t cudaInitDevice(int, unsigned, unsigned) { API_ENTER("cudaInitDevice"); return cudaSuccess; }
EXPORT cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
EXPORT cudaError_t cudaGetDeviceFlags(unsigned* f) { *f = cudaDeviceMapHost; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceReset() { API_ENTER("cudaDeviceReset"); call_i("synchronize", nullptr); return cudaSuccess; }
EXPORT cudaError_t cudaFree0() { return cudaSuccess; }
EXPORT cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* p, int) { fill_props(p); return cudaSuccess; }
EXPORT cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
  cudaDeviceProp p; fill_props(&p);
  switch (a) {
    case cudaDevAttrComputeCapabilityMajor: *v = 8; break;
    case cudaDevAttrComputeCapabilityMinor: *v = 6; break;
    case cudaDevAttrMultiProcessorCount: *v = p.multiProcessorCount; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *v = (int)p.sharedMemPerBlockOptin; break;
    case cudaDevAttrMaxSharedMemoryPerBlock: *v = (int)p.sharedMemPerBlock; break;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *v = (int)p.sharedMemPerMultiprocessor; break;
    case cudaDevAttrMaxThreadsPerBlock: *v = 1024; break;
    case cudaDevAttrMaxThreadsPerMultiProcessor: *v = p.maxThreadsPerMultiProcessor; break;
    case cudaDevAttrWarpSize: *v = 32; break;
    case cudaDevAttrClockRate: *v = 1695000; break;
    case cudaDevAttrMemoryClockRate: *v = 9751000; break;
    case cudaDevAttrGlobalMemoryBusWidth: *v = 384; break;
    case cudaDevAttrL2CacheSize: *v = p.l2CacheSize; break;
    case cudaDevAttrIntegrated: *v = 0; break;
    case cudaDevAttrCanMapHostMemory: *v = 1; break;
    case cudaDevAttrUnifiedAddressing: *v = 1; break;
    case cudaDevAttrMaxRegistersPerBlock: *v = 65536; break;
    case cudaDevAttrMaxRegistersPerMultiprocessor: *v = 65536; break;
    case cudaDevAttrMaxGridDimX: *v = INT32_MAX; break;
    case cudaDevAttrMaxGridDimY: case cudaDevAttrMaxGridDimZ: *v = 65535; break;
    case cudaDevAttrMaxBlockDimX: case cudaDevAttrMaxBlockDimY: *v = 1024; break;
    case cudaDevAttrMaxBlockDimZ: *v = 64; break;
    case cudaDevAttrPciBusId: *v = 2; break;
    case cudaDevAttrPageableMemoryAccess: case cudaDevAttrConcurrentManagedAccess: case cudaDevAttrManagedMemory: *v = 0; break;
    case cudaDevAttrMaxBlocksPerMultiprocessor: *v = 16; break;
    case cudaDevAttrHostRegisterSupported: *v = 1; break;
    case cudaDevAttrMemoryPoolsSupported: *v = 0; break;
    case cudaDevAttrClusterLaunch: *v = 0; break;
    case cudaDevAttrCooperativeLaunch: *v = 1; break;
    default: *v = 0; break;
  }
  return cudaSuccess;
}
EXPORT cudaError_t cudaDeviceGetPCIBusId(char* s, int len, int) { snprintf(s, len, "0000:02:00.0"); return cudaSuccess; }
EXPORT cudaError_t cudaDriverGetVersion(int* v) { *v = 12080; return cudaSuccess; }
EXPORT cudaError_t cudaRuntimeGetVersion(int* v) { *v = 12080; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceCanAccessPeer(int* c, int, int) { *c = 0; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceEnablePeerAccess(int, unsigned) { return fail(cudaErrorPeerAccessUnsupported, "peer access"); }
EXPORT cudaError_t cudaDeviceDisablePeerAccess(int) { return cudaSuccess; }
EXPORT cudaError_t cudaDeviceSetLimit(cudaLimit, size_t) { return cudaSuccess; }
EXPORT cudaError_t cudaDeviceGetLimit(size_t* v, cudaLimit l) { *v = l == cudaLimitStackSize ? 1024 : 0; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache) { return cudaSuccess; }
EXPORT cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig) { return cudaSuccess; }

EXPORT cudaError_t cudaFuncSetAttribute(const void* f, cudaFuncAttribute attr, int value) {
  API_ENTER("cudaFuncSetAttribute");
  Func* fn = func_of(f);
  if (!fn) return fail(cudaErrorInvalidDeviceFunction, "cudaFuncSetAttribute");
  if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) {
    if (value > (99 << 10)) return fail(cudaErrorInvalidValue, "dynamic shared memory above 99 KB");
    fn->max_dyn_shared = value;
  }
  return cudaSuccess;
}
EXPORT cudaError_t cudaFuncSetCacheConfig(const void*, cudaFuncCache) { return cudaSuccess; }
EXPORT cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, const void* f) {
  API_ENTER("cudaFuncGetAttributes");
  Func* fn = func_of(f);
  if (!fn) return fail(cudaErrorInvalidDeviceFunction, "cudaFuncGetAttributes");
  PyObject* r;
  if (!call_obj(&r, "kernel_attrs", "(i)", fn->kid)) return fail(cudaErrorUnknown, "kernel_attrs");
  memset(a, 0, sizeof(*a));
  a->numRegs = (int)PyLong_AsLong(PyTuple_GetItem(r, 0));
  a->sharedSizeBytes = PyLong_AsSize_t(PyTuple_GetItem(r, 1));
  a->localSizeBytes = PyLong_AsSize_t(PyTuple_GetItem(r, 2));
  a->constSizeBytes = 0;
  Py_DECREF(r);
  a->maxThreadsPerBlock = a->numRegs ? std::min(1024, ((65536 / (((a->numRegs * 32 + 255) / 256) * 256)) / 4) * 4 * 32) : 1024;
  a->ptxVersion = 86; a->binaryVersion = 86;
  a->maxDynamicSharedSizeBytes = fn->max_dyn_shared;
  return cudaSuccess;
}
EXPORT cudaError_t cudaFuncGetName(const char** name, const void* f) {
  std::lock_guard<std::recursive_mutex> lk(g_lock);
  auto it = g_funcs.find(f);
  if (it == g_funcs.end()) return fail(cudaErrorInvalidDeviceFunction, "cudaFuncGetName");
  *name = it->second.name.c_str();
  return cudaSuccess;
}
EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* n, const void* f, int block, size_t shmem, unsigned) {
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, f);
  if (e) return e;
  int warps = (block + 31) / 32;
  int by_regs = a.numRegs ? 65536 / (warps * ((a.numRegs * 32 + 255) / 256) * 256) : 16;
  int by_threads = 1536 / std::max(1, warps * 32);
  size_t sh = (a.sharedSizeBytes + shmem + 1024 + 127) / 128 * 128;   // tgcu.Context.coresident_blocks: 128 B granules
  int by_shmem = (int)((100 << 10) / sh);
  *n = std::max(0, std::min({16, by_regs, by_threads, by_shmem}));
  return cudaSuccess;
}
EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void* f, int block, size_t shmem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(n, f, block, shmem, 0);
}
EXPORT cudaError_t cudaOccupancyMaxActiveClusters(int* n, const void*, const cudaLaunchConfig_t*) { *n = 0; return cudaErrorNotSupported; }

// *** errors and driver entry points ***
EXPORT cudaError_t cudaGetLastError() { cudaError_t e = t_last; t_last = cudaSuccess; return e; }
EXPORT cudaError_t cudaPeekAtLastError() { return t_last; }
EXPORT const char* cudaGetErrorString(cudaError_t e) { return err_name(e); }
EXPORT const char* cudaGetErrorName(cudaError_t e) { return err_name(e); }
// no driver API: VMM (cuMem*) users must fall back to cudaMalloc
EXPORT cudaError_t cudaGetDriverEntryPoint(const char*, void** fn, unsigned long long, cudaDriverEntryPointQueryResult* st) {
  *fn = nullptr; if (st) *st = cudaDriverEntryPointSymbolNotFound; return cudaErrorNotSupported;
}
EXPORT cudaError_t cudaGetDriverEntryPointByVersion(const char*, void** fn, unsigned, unsigned long long, cudaDriverEntryPointQueryResult* st) {
  *fn = nullptr; if (st) *st = cudaDriverEntryPointSymbolNotFound; return cudaErrorNotSupported;
}

// built-in kernels (memset, memcpy, cuBLAS stand-in) by name, with cudaLaunchKernel-style argument pointers
EXPORT int tgcu_builtin_launch(const char* name, unsigned gx, unsigned gy, unsigned bx, void** args, cudaStream_t st) {
  API_ENTER("tgcu_builtin_launch");
  static std::map<std::string, Func*> cache;
  Func*& fn = cache[name];
  if (!fn) {
    fn = new Func{nullptr, name};
    fn->kid = (int)call_i("builtin_kernel", "(s)", name);
    if (fn->kid < 0) { fprintf(stderr, "[tgcu] no builtin kernel %s\n", name); return 1; }
  }
  return pack_and_launch(fn, dim3(gx, gy, 1), dim3(bx, 1, 1), 0, args, st) == cudaSuccess ? 0 : 1;
}
EXPORT cudaError_t cudaGraphGetEdges_v2(cudaGraph_t, cudaGraphNode_t*, cudaGraphNode_t*, cudaGraphEdgeData*, size_t* n) { *n = 0; return cudaSuccess; }
EXPORT cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams*) { return cudaErrorNotSupported; }

// *** explicit mapped-memory sync for programs that hand data over mid-graph (Strata's verify windows) ***
static uintptr_t region_base(const void* p) {
  uintptr_t a = (uintptr_t)p;
  auto it = g_host.upper_bound(a);
  if (it == g_host.begin()) return 0;
  --it;
  return (a < it->first + it->second.first && it->second.second) ? it->first : 0;
}
EXPORT int tgcuMappedPush(const void* p, size_t n) {
  API_ENTER("tgcuMappedPush");
  uintptr_t b = region_base(p);
  if (!b) return fail(cudaErrorInvalidValue, "tgcuMappedPush: not a mapped region");
  return (int)call_i("mapped_push", "(KKK)", (unsigned long long)b, (unsigned long long)((uintptr_t)p - b), (unsigned long long)n);
}
EXPORT int tgcuMappedPull(void* p, size_t n) {
  API_ENTER("tgcuMappedPull");
  uintptr_t b = region_base(p);
  if (!b) return fail(cudaErrorInvalidValue, "tgcuMappedPull: not a mapped region");
  return (int)call_i("mapped_pull", "(KKK)", (unsigned long long)b, (unsigned long long)((uintptr_t)p - b), (unsigned long long)n);
}
// *** pre-armed handshake: segments are (host pointer, bytes) pairs inside mapped regions; see tgcu_api.hs_* ***
static PyObject* hs_segs(void* const* ps, const size_t* ns, int cnt) {
  PyObject* l = PyList_New(cnt);
  for (int i = 0; i < cnt; ++i) {
    uintptr_t b = region_base(ps[i]);
    if (!b) { Py_DECREF(l); return nullptr; }
    PyList_SET_ITEM(l, i, Py_BuildValue("(KKK)", (unsigned long long)b, (unsigned long long)((uintptr_t)ps[i] - b), (unsigned long long)ns[i]));
  }
  return l;
}
EXPORT int tgcuHsArm(void* const* ps, const size_t* ns, int cnt, const void* gate_p, unsigned want) {
  API_ENTER("tgcuHsArm");
  PyObject* l = hs_segs(ps, ns, cnt);
  uintptr_t gb = region_base(gate_p);
  if (!l || !gb) return -3;
  PyObject* r = nullptr;
  call_obj(&r, "hs_arm", "(OKKI)", l, (unsigned long long)gb, (unsigned long long)((uintptr_t)gate_p - gb), want);
  Py_DECREF(l);
  long long v = r && PyLong_Check(r) ? PyLong_AsLongLong(r) : -1;
  Py_XDECREF(r);
  return (int)v;
}
EXPORT int tgcuHsWait(int timeout_ms) {
  API_ENTER("tgcuHsWait");
  return (int)call_i("hs_wait", "(i)", timeout_ms);
}
EXPORT int tgcuHsPushPre(void* const* ps, const size_t* ns, int cnt) {
  API_ENTER("tgcuHsPushPre");
  PyObject* l = hs_segs(ps, ns, cnt);
  if (!l) return -3;
  PyObject* r = nullptr;
  call_obj(&r, "hs_push_pre", "(O)", l);
  Py_DECREF(l);
  long long v = r && PyLong_Check(r) ? PyLong_AsLongLong(r) : -1;
  Py_XDECREF(r);
  return (int)v;
}
EXPORT int tgcuHsPushPreNext(void* const* ps, const size_t* ns, int cnt, void* const* nps, const size_t* nns, int ncnt,
                             const void* gate_p, unsigned want) {
  API_ENTER("tgcuHsPushPreNext");
  PyObject* l = hs_segs(ps, ns, cnt);
  PyObject* l2 = hs_segs(nps, nns, ncnt);
  uintptr_t gb = region_base(gate_p);
  if (!l || !l2 || !gb) { Py_XDECREF(l); Py_XDECREF(l2); return -3; }
  PyObject* r = nullptr;
  call_obj(&r, "hs_push_pre_next", "(OOKKI)", l, l2, (unsigned long long)gb, (unsigned long long)((uintptr_t)gate_p - gb), want);
  Py_DECREF(l); Py_DECREF(l2);
  long long v = r && PyLong_Check(r) ? PyLong_AsLongLong(r) : -1;
  Py_XDECREF(r);
  return (int)v;
}
EXPORT int tgcuHsRelease() {
  API_ENTER("tgcuHsRelease");
  return (int)call_i("hs_release", nullptr);
}
EXPORT int tgcuHsPushGo() {
  API_ENTER("tgcuHsPushGo");
  return (int)call_i("hs_push_go", nullptr);
}
EXPORT int tgcuMappedManage(const void* p) {
  API_ENTER("tgcuMappedManage");
  uintptr_t b = region_base(p);
  return b ? (int)call_i("mapped_manage", "(K)", (unsigned long long)b) : 1;
}
