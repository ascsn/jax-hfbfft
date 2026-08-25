"""
System API routes.

This module provides endpoints for system status, available forces,
and JIT warmup management.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from jax_hfbfft.gui.models import SystemStatus, ForceInfo
from jax_hfbfft.gui.services.warmup import get_warmup_manager
from jax_hfbfft.gui.services.hfb_service import get_hfb_service

router = APIRouter()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get system status including backend, JIT warmup, and resource usage.
    
    Returns comprehensive system information for the status display.
    """
    warmup = get_warmup_manager()
    service = get_hfb_service()
    
    # Get JAX backend info
    jax_version = None
    try:
        import jax
        backend = jax.default_backend()
        devices = jax.devices()
        device_name = str(devices[0]) if devices else "unknown"
        jax_version = jax.__version__
        
        # Check precision
        from jax_hfbfft.jax_config import check_precision
        precision = "float64" if check_precision() else "float32"
    except ImportError:
        backend = "unknown"
        device_name = "JAX not loaded"
        precision = "unknown"
    
    # Get memory usage if available
    memory_used_gb = None
    memory_total_gb = None
    gpu_memory_used_gb = None
    gpu_memory_total_gb = None
    
    try:
        import psutil
        process = psutil.Process()
        memory_used_gb = process.memory_info().rss / (1024**3)
        memory_total_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    # Get GPU memory if available
    try:
        if backend == "gpu":
            # Try to get GPU memory from JAX
            try:
                import jax
                
                # Get device memory stats
                devices = jax.devices()
                if devices:
                    device = devices[0]
                    # Try multiple methods to get memory stats
                    try:
                        # Method 1: device.memory_stats() - newer JAX versions
                        if hasattr(device, 'memory_stats'):
                            stats = device.memory_stats()
                            if stats:
                                gpu_memory_used_gb = stats.get('bytes_in_use', 0) / (1024**3)
                                gpu_memory_total_gb = stats.get('bytes_limit', 0) / (1024**3)
                    except Exception:
                        pass
                    
                    # Method 2: Try jax.lib.xla_bridge for older versions
                    if gpu_memory_total_gb is None:
                        try:
                            from jax.lib import xla_bridge
                            backend_obj = xla_bridge.get_backend()
                            if hasattr(backend_obj, 'get_memory_info'):
                                mem_info = backend_obj.get_memory_info(device.id)
                                if mem_info:
                                    gpu_memory_used_gb = mem_info.get('bytes_in_use', 0) / (1024**3)
                                    gpu_memory_total_gb = mem_info.get('bytes_limit', 0) / (1024**3)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Warning: Could not get JAX GPU memory stats: {e}")
            
            # Fallback: try nvidia-smi if JAX methods failed
            if gpu_memory_total_gb is None:
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if lines and lines[0]:
                            parts = lines[0].split(',')
                            if len(parts) == 2:
                                gpu_memory_used_gb = float(parts[0].strip()) / 1024
                                gpu_memory_total_gb = float(parts[1].strip()) / 1024
                except Exception as e:
                    print(f"Warning: Could not get nvidia-smi GPU memory: {e}")
    except Exception as e:
        print(f"Warning: Error in GPU memory detection: {e}")
    
    # Count active calculations
    calcs = await service.list_calculations()
    active_count = sum(
        1 for c in calcs 
        if c.phase.value in ("pending", "warmup", "initializing", "iterating")
    )
    
    # Get warmup status with error details
    warmup_status = warmup.get_status()
    
    return SystemStatus(
        backend=backend,
        device_name=device_name,
        device_info=device_name,
        precision=precision,
        jax_version=jax_version,
        jit_warmed_up=warmup.is_ready,
        warmup_progress=warmup.progress,
        warmup_status=warmup_status.get('status', 'not_started'),
        warmup_error=warmup_status.get('error'),
        active_calculations=active_count,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        gpu_memory_used_gb=gpu_memory_used_gb,
        gpu_memory_total_gb=gpu_memory_total_gb,
    )


@router.get("/diagnostics")
async def get_diagnostics():
    """
    Get detailed system diagnostics for debugging.
    
    Returns comprehensive system information including:
    - JAX configuration and devices
    - Memory usage (CPU and GPU)
    - Python environment details
    - GPU driver information
    - JIT compilation status
    """
    diagnostics = {}
    
    # Python environment
    import sys
    import platform
    diagnostics["python"] = {
        "version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    
    # JAX information
    try:
        import jax
        import jax.numpy as jnp
        
        diagnostics["jax"] = {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "device_count": len(jax.devices()),
            "x64_enabled": jnp.array(1.0).dtype == jnp.float64,
        }
        
        # Device details
        devices_info = []
        for device in jax.devices():
            device_info = {
                "id": device.id,
                "platform": device.platform,
                "device_kind": str(device.device_kind),
            }
            
            # Try to get memory stats
            try:
                if hasattr(device, 'memory_stats'):
                    mem_stats = device.memory_stats()
                    if mem_stats:
                        device_info["memory_stats"] = {
                            "bytes_in_use_gb": mem_stats.get('bytes_in_use', 0) / (1024**3),
                            "bytes_limit_gb": mem_stats.get('bytes_limit', 0) / (1024**3),
                            "peak_bytes_in_use_gb": mem_stats.get('peak_bytes_in_use', 0) / (1024**3),
                        }
            except Exception as e:
                device_info["memory_stats_error"] = str(e)
            
            devices_info.append(device_info)
        
        diagnostics["jax"]["device_details"] = devices_info
        
    except ImportError as e:
        diagnostics["jax"] = {"error": f"JAX not available: {e}"}
    except Exception as e:
        diagnostics["jax"] = {"error": str(e)}
    
    # System memory
    try:
        import psutil
        vm = psutil.virtual_memory()
        diagnostics["system_memory"] = {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent": vm.percent,
        }
    except ImportError:
        diagnostics["system_memory"] = {"error": "psutil not available"}
    except Exception as e:
        diagnostics["system_memory"] = {"error": str(e)}
    
    # GPU information (nvidia-smi)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpu_info.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "driver_version": parts[2],
                            "memory_total_mb": float(parts[3]),
                            "memory_used_mb": float(parts[4]),
                            "memory_free_mb": float(parts[5]),
                            "temperature_c": float(parts[6]) if parts[6] else None,
                            "utilization_percent": float(parts[7]) if parts[7] else None,
                        })
            diagnostics["nvidia_gpus"] = gpu_info
    except FileNotFoundError:
        diagnostics["nvidia_gpus"] = {"error": "nvidia-smi not found"}
    except Exception as e:
        diagnostics["nvidia_gpus"] = {"error": str(e)}
    
    # Warmup status
    warmup = get_warmup_manager()
    diagnostics["warmup"] = warmup.get_status()
    
    # Installed packages
    try:
        import pkg_resources
        packages = {
            pkg.key: pkg.version 
            for pkg in pkg_resources.working_set
        }
        diagnostics["installed_packages"] = packages
    except Exception as e:
        diagnostics["installed_packages"] = {"error": str(e)}
    
    return diagnostics


@router.post("/warmup")
async def trigger_warmup(force: bool = False):
    """
    Trigger JIT warmup.
    
    Args:
        force: If True, re-run warmup even if already complete.
        
    Returns:
        Warmup status.
    """
    warmup = get_warmup_manager()
    
    if warmup.is_ready and not force:
        return {
            "message": "JIT already warmed up",
            "status": warmup.get_status(),
        }
    
    # Start warmup in background
    warmup.start_background_warmup()
    
    return {
        "message": "JIT warmup started",
        "status": warmup.get_status(),
    }


@router.get("/warmup/status")
async def get_warmup_status():
    """
    Get current JIT warmup status.
    
    Returns detailed warmup progress information.
    """
    warmup = get_warmup_manager()
    return warmup.get_status()


@router.get("/forces", response_model=List[ForceInfo])
async def list_forces():
    """
    List all available Skyrme forces.
    
    Returns a list of force names with their parameters.
    """
    try:
        from jax_hfbfft.core.force import Force
        from pathlib import Path
        import yaml
        
        # Load forces from YAML file
        forces_file = Path(__file__).parent.parent.parent / "data" / "_forces.yml"
        if not forces_file.exists():
            forces_file = Path(__file__).parent.parent.parent.parent.parent / "_forces.yml"
        
        with open(forces_file, 'r') as f:
            forces_data = yaml.safe_load(f)
        
        forces = []
        for name in forces_data.keys():
            try:
                force = Force.from_name(name)
                forces.append(ForceInfo(
                    name=name,
                    description=forces_data[name].get('description', None),
                    t0=float(force.t0),
                    t1=float(force.t1),
                    t2=float(force.t2),
                    t3=float(force.t3),
                    x0=float(force.x0),
                    x1=float(force.x1),
                    x2=float(force.x2),
                    x3=float(force.x3),
                    b4=float(getattr(force, 'b4', 0.0)),
                    b4p=float(force.b4p),
                    power=float(force.power),
                ))
            except Exception:
                # Skip forces that fail to load
                continue
        
        return forces
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Could not load force definitions"
        )


@router.get("/forces/{name}", response_model=ForceInfo)
async def get_force(name: str):
    """
    Get parameters for a specific Skyrme force.
    
    Args:
        name: Force name (e.g., "SLy4", "SkM*").
        
    Returns:
        Force parameters.
    """
    try:
        from jax_hfbfft.core.force import Force
        
        force = Force.from_name(name)
        return ForceInfo(
            name=name,
            description=getattr(force, 'description', None),
            t0=float(force.t0),
            t1=float(force.t1),
            t2=float(force.t2),
            t3=float(force.t3),
            x0=float(force.x0),
            x1=float(force.x1),
            x2=float(force.x2),
            x3=float(force.x3),
            b4=float(getattr(force, 'b4', 0.0)),
            b4p=float(force.b4p),
            power=float(force.power),
        )
    
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Force '{name}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading force: {e}"
        )


@router.get("/elements")
async def list_elements():
    """
    List common elements for the nucleus selector.
    
    Returns a list of element symbols with atomic numbers.
    """
    # Common elements used in nuclear physics
    elements = [
        {"symbol": "H", "z": 1, "name": "Hydrogen"},
        {"symbol": "He", "z": 2, "name": "Helium"},
        {"symbol": "Li", "z": 3, "name": "Lithium"},
        {"symbol": "Be", "z": 4, "name": "Beryllium"},
        {"symbol": "B", "z": 5, "name": "Boron"},
        {"symbol": "C", "z": 6, "name": "Carbon"},
        {"symbol": "N", "z": 7, "name": "Nitrogen"},
        {"symbol": "O", "z": 8, "name": "Oxygen"},
        {"symbol": "F", "z": 9, "name": "Fluorine"},
        {"symbol": "Ne", "z": 10, "name": "Neon"},
        {"symbol": "Na", "z": 11, "name": "Sodium"},
        {"symbol": "Mg", "z": 12, "name": "Magnesium"},
        {"symbol": "Al", "z": 13, "name": "Aluminum"},
        {"symbol": "Si", "z": 14, "name": "Silicon"},
        {"symbol": "P", "z": 15, "name": "Phosphorus"},
        {"symbol": "S", "z": 16, "name": "Sulfur"},
        {"symbol": "Cl", "z": 17, "name": "Chlorine"},
        {"symbol": "Ar", "z": 18, "name": "Argon"},
        {"symbol": "K", "z": 19, "name": "Potassium"},
        {"symbol": "Ca", "z": 20, "name": "Calcium"},
        {"symbol": "Sc", "z": 21, "name": "Scandium"},
        {"symbol": "Ti", "z": 22, "name": "Titanium"},
        {"symbol": "V", "z": 23, "name": "Vanadium"},
        {"symbol": "Cr", "z": 24, "name": "Chromium"},
        {"symbol": "Mn", "z": 25, "name": "Manganese"},
        {"symbol": "Fe", "z": 26, "name": "Iron"},
        {"symbol": "Co", "z": 27, "name": "Cobalt"},
        {"symbol": "Ni", "z": 28, "name": "Nickel"},
        {"symbol": "Cu", "z": 29, "name": "Copper"},
        {"symbol": "Zn", "z": 30, "name": "Zinc"},
        {"symbol": "Ga", "z": 31, "name": "Gallium"},
        {"symbol": "Ge", "z": 32, "name": "Germanium"},
        {"symbol": "As", "z": 33, "name": "Arsenic"},
        {"symbol": "Se", "z": 34, "name": "Selenium"},
        {"symbol": "Br", "z": 35, "name": "Bromine"},
        {"symbol": "Kr", "z": 36, "name": "Krypton"},
        {"symbol": "Rb", "z": 37, "name": "Rubidium"},
        {"symbol": "Sr", "z": 38, "name": "Strontium"},
        {"symbol": "Y", "z": 39, "name": "Yttrium"},
        {"symbol": "Zr", "z": 40, "name": "Zirconium"},
        {"symbol": "Nb", "z": 41, "name": "Niobium"},
        {"symbol": "Mo", "z": 42, "name": "Molybdenum"},
        {"symbol": "Tc", "z": 43, "name": "Technetium"},
        {"symbol": "Ru", "z": 44, "name": "Ruthenium"},
        {"symbol": "Rh", "z": 45, "name": "Rhodium"},
        {"symbol": "Pd", "z": 46, "name": "Palladium"},
        {"symbol": "Ag", "z": 47, "name": "Silver"},
        {"symbol": "Cd", "z": 48, "name": "Cadmium"},
        {"symbol": "In", "z": 49, "name": "Indium"},
        {"symbol": "Sn", "z": 50, "name": "Tin"},
        {"symbol": "Sb", "z": 51, "name": "Antimony"},
        {"symbol": "Te", "z": 52, "name": "Tellurium"},
        {"symbol": "I", "z": 53, "name": "Iodine"},
        {"symbol": "Xe", "z": 54, "name": "Xenon"},
        {"symbol": "Cs", "z": 55, "name": "Cesium"},
        {"symbol": "Ba", "z": 56, "name": "Barium"},
        {"symbol": "La", "z": 57, "name": "Lanthanum"},
        {"symbol": "Ce", "z": 58, "name": "Cerium"},
        {"symbol": "Pr", "z": 59, "name": "Praseodymium"},
        {"symbol": "Nd", "z": 60, "name": "Neodymium"},
        {"symbol": "Pm", "z": 61, "name": "Promethium"},
        {"symbol": "Sm", "z": 62, "name": "Samarium"},
        {"symbol": "Eu", "z": 63, "name": "Europium"},
        {"symbol": "Gd", "z": 64, "name": "Gadolinium"},
        {"symbol": "Tb", "z": 65, "name": "Terbium"},
        {"symbol": "Dy", "z": 66, "name": "Dysprosium"},
        {"symbol": "Ho", "z": 67, "name": "Holmium"},
        {"symbol": "Er", "z": 68, "name": "Erbium"},
        {"symbol": "Tm", "z": 69, "name": "Thulium"},
        {"symbol": "Yb", "z": 70, "name": "Ytterbium"},
        {"symbol": "Lu", "z": 71, "name": "Lutetium"},
        {"symbol": "Hf", "z": 72, "name": "Hafnium"},
        {"symbol": "Ta", "z": 73, "name": "Tantalum"},
        {"symbol": "W", "z": 74, "name": "Tungsten"},
        {"symbol": "Re", "z": 75, "name": "Rhenium"},
        {"symbol": "Os", "z": 76, "name": "Osmium"},
        {"symbol": "Ir", "z": 77, "name": "Iridium"},
        {"symbol": "Pt", "z": 78, "name": "Platinum"},
        {"symbol": "Au", "z": 79, "name": "Gold"},
        {"symbol": "Hg", "z": 80, "name": "Mercury"},
        {"symbol": "Tl", "z": 81, "name": "Thallium"},
        {"symbol": "Pb", "z": 82, "name": "Lead"},
        {"symbol": "Bi", "z": 83, "name": "Bismuth"},
        {"symbol": "Po", "z": 84, "name": "Polonium"},
        {"symbol": "At", "z": 85, "name": "Astatine"},
        {"symbol": "Rn", "z": 86, "name": "Radon"},
        {"symbol": "Fr", "z": 87, "name": "Francium"},
        {"symbol": "Ra", "z": 88, "name": "Radium"},
        {"symbol": "Ac", "z": 89, "name": "Actinium"},
        {"symbol": "Th", "z": 90, "name": "Thorium"},
        {"symbol": "Pa", "z": 91, "name": "Protactinium"},
        {"symbol": "U", "z": 92, "name": "Uranium"},
        {"symbol": "Np", "z": 93, "name": "Neptunium"},
        {"symbol": "Pu", "z": 94, "name": "Plutonium"},
    ]
    
    return elements


@router.get("/presets")
async def list_nucleus_presets():
    """
    List preset nuclei for quick selection.
    
    Returns common nuclei used in nuclear physics studies.
    """
    presets = [
        {"symbol": "O", "a": 16, "z": 8, "n": 8, "label": "O-16 (Doubly Magic)"},
        {"symbol": "Ca", "a": 40, "z": 20, "n": 20, "label": "Ca-40 (Doubly Magic)"},
        {"symbol": "Ca", "a": 48, "z": 20, "n": 28, "label": "Ca-48 (Neutron Rich)"},
        {"symbol": "Ni", "a": 56, "z": 28, "n": 28, "label": "Ni-56 (Doubly Magic)"},
        {"symbol": "Ni", "a": 78, "z": 28, "n": 50, "label": "Ni-78 (Doubly Magic)"},
        {"symbol": "Zr", "a": 90, "z": 40, "n": 50, "label": "Zr-90 (Semi-Magic)"},
        {"symbol": "Sn", "a": 100, "z": 50, "n": 50, "label": "Sn-100 (Doubly Magic)"},
        {"symbol": "Sn", "a": 132, "z": 50, "n": 82, "label": "Sn-132 (Doubly Magic)"},
        {"symbol": "Pb", "a": 208, "z": 82, "n": 126, "label": "Pb-208 (Doubly Magic)"},
        {"symbol": "Mg", "a": 24, "z": 12, "n": 12, "label": "Mg-24 (Deformed)"},
        {"symbol": "Si", "a": 28, "z": 14, "n": 14, "label": "Si-28 (Oblate)"},
        {"symbol": "Sm", "a": 152, "z": 62, "n": 90, "label": "Sm-152 (Deformed)"},
        {"symbol": "Gd", "a": 154, "z": 64, "n": 90, "label": "Gd-154 (Deformed)"},
        {"symbol": "Er", "a": 166, "z": 68, "n": 98, "label": "Er-166 (Deformed)"},
    ]
    
    return presets
