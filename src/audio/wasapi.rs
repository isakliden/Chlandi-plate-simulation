use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Host};

#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::CloseHandle;
#[cfg(target_os = "windows")]
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> *mut std::ffi::c_void;
    fn GetProcAddress(hModule: *mut std::ffi::c_void, lpProcName: *const u8) -> *mut std::ffi::c_void;
    fn FreeLibrary(hModule: *mut std::ffi::c_void) -> i32;
}

#[cfg(target_os = "windows")]
pub struct ProAudioThreadGuard {
    handle: *mut std::ffi::c_void,
    revert_fn: Option<unsafe extern "system" fn(*mut std::ffi::c_void) -> i32>,
    avrt_module: *mut std::ffi::c_void,
}

#[cfg(target_os = "windows")]
impl ProAudioThreadGuard {
    pub fn enter() -> Option<Self> {
        unsafe {
            let avrt = LoadLibraryA(c"avrt.dll".as_ptr().cast());
            if avrt.is_null() {
                return None;
            }
            let set_proc =
                GetProcAddress(avrt, c"AvSetMmThreadCharacteristicsW".as_ptr().cast());
            let revert_proc =
                GetProcAddress(avrt, c"AvRevertMmThreadCharacteristics".as_ptr().cast());
            if set_proc.is_null() || revert_proc.is_null() {
                FreeLibrary(avrt);
                return None;
            }
            let set_fn: unsafe extern "system" fn(*const u16, *mut u32) -> *mut std::ffi::c_void =
                std::mem::transmute(set_proc);
            let revert_fn: unsafe extern "system" fn(*mut std::ffi::c_void) -> i32 =
                std::mem::transmute(revert_proc);

            let mut task_index = 0u32;
            let pro_audio: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
            let handle = set_fn(pro_audio.as_ptr(), &mut task_index);
            if handle.is_null() {
                FreeLibrary(avrt);
                None
            } else {
                Some(Self {
                    handle,
                    revert_fn: Some(revert_fn),
                    avrt_module: avrt,
                })
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for ProAudioThreadGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(rev) = self.revert_fn {
                if !self.handle.is_null() {
                    rev(self.handle);
                }
            }
            if !self.avrt_module.is_null() {
                FreeLibrary(self.avrt_module);
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub struct ProAudioThreadGuard;

#[cfg(not(target_os = "windows"))]
impl ProAudioThreadGuard {
    pub fn enter() -> Option<Self> {
        Some(Self)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
    pub is_output: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessItem {
    pub pid: u32,
    pub name: String,
}

#[cfg(target_os = "windows")]
pub fn list_running_processes() -> Vec<ProcessItem> {
    let mut processes = Vec::new();
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snapshot == -1isize as windows_sys::Win32::Foundation::HANDLE {
            return processes;
        }

        let mut entry: PROCESSENTRY32W = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;

        if Process32FirstW(snapshot, &mut entry) != 0 {
            loop {
                let name_len = entry
                    .szExeFile
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(entry.szExeFile.len());
                let name = String::from_utf16_lossy(&entry.szExeFile[..name_len]);
                if !name.is_empty()
                    && !name.eq_ignore_ascii_case("[System Process]")
                    && !name.eq_ignore_ascii_case("System")
                    && !name.eq_ignore_ascii_case("Registry")
                {
                    processes.push(ProcessItem {
                        pid: entry.th32ProcessID,
                        name,
                    });
                }
                if Process32NextW(snapshot, &mut entry) == 0 {
                    break;
                }
            }
        }
        CloseHandle(snapshot);
    }

    processes.sort_by_key(|a| a.name.to_lowercase());
    processes.dedup_by(|a, b| a.name.eq_ignore_ascii_case(&b.name));
    processes
}

#[cfg(not(target_os = "windows"))]
pub fn list_running_processes() -> Vec<ProcessItem> {
    Vec::new()
}

pub fn enumerate_audio_devices(host: &Host) -> (Vec<AudioDeviceInfo>, Vec<AudioDeviceInfo>) {
    let mut output_devices = Vec::new();
    let mut input_devices = Vec::new();

    let default_output_name = host.default_output_device().map(|d| {
        d.description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| d.to_string())
    });

    let default_input_name = host.default_input_device().map(|d| {
        d.description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| d.to_string())
    });

    if let Ok(devices) = host.output_devices() {
        for dev in devices {
            let name = dev
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| dev.to_string());
            let is_default = default_output_name.as_ref() == Some(&name);
            output_devices.push(AudioDeviceInfo {
                name,
                is_default,
                is_output: true,
            });
        }
    }

    if let Ok(devices) = host.input_devices() {
        for dev in devices {
            let name = dev
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| dev.to_string());
            let is_default = default_input_name.as_ref() == Some(&name);
            input_devices.push(AudioDeviceInfo {
                name,
                is_default,
                is_output: false,
            });
        }
    }

    (output_devices, input_devices)
}

pub fn find_output_device(host: &Host, name_opt: Option<&str>) -> Option<Device> {
    if let Some(name) = name_opt {
        if let Ok(mut devs) = host.output_devices() {
            if let Some(d) = devs.find(|d| {
                d.description()
                    .map(|desc| desc.name() == name)
                    .unwrap_or_else(|_| d.to_string() == name)
            }) {
                return Some(d);
            }
        }
    }
    host.default_output_device()
}

pub fn find_input_device(host: &Host, name_opt: Option<&str>) -> Option<Device> {
    if let Some(name) = name_opt {
        if let Ok(mut devs) = host.input_devices() {
            if let Some(d) = devs.find(|d| {
                d.description()
                    .map(|desc| desc.name() == name)
                    .unwrap_or_else(|_| d.to_string() == name)
            }) {
                return Some(d);
            }
        }
    }
    host.default_input_device()
}
