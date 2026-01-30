"""
HFB Service for managing calculations.

This module provides an async wrapper around the HFBFFT calculation engine
for use in the GUI backend.
"""

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable, Awaitable, Any
from concurrent.futures import ThreadPoolExecutor
import threading

from jax_hfbfft.gui.models import (
    CalculationRequest,
    CalculationStatus,
    CalculationProgress,
    CalculationResults,
    CalculationPhase,
    EnergyBreakdown,
    RadiiResults,
    DeformationResults,
    PairingResults,
    SingleParticleLevel,
    NucleusInput,
)
from jax_hfbfft.gui.services.warmup import get_warmup_manager, ensure_warmup


@dataclass
class ActiveCalculation:
    """Tracks an active calculation."""
    id: str
    request: CalculationRequest
    status: CalculationStatus
    cancel_event: threading.Event = field(default_factory=threading.Event)
    start_time: float = field(default_factory=time.time)
    hfbfft_instance: Any = None  # Store HFBFFT instance for density extraction
    

class HFBService:
    """
    Service for managing HFB calculations.
    
    This service handles:
    - Starting new calculations
    - Tracking calculation progress
    - Cancelling calculations
    - Converting results to API models
    
    Example:
        service = HFBService()
        
        # Start a calculation
        calc_id = await service.start_calculation(request)
        
        # Monitor progress
        async for progress in service.stream_progress(calc_id):
            print(f"Iteration {progress.iteration}")
        
        # Get results
        status = await service.get_calculation(calc_id)
    """
    
    def __init__(self, max_concurrent: int = 4):
        """
        Initialize the HFB service.
        
        Args:
            max_concurrent: Maximum concurrent calculations.
        """
        self._calculations: Dict[str, ActiveCalculation] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def start_calculation(
        self,
        request: CalculationRequest,
        progress_callback: Optional[Callable[[CalculationProgress], Awaitable[None]]] = None,
    ) -> str:
        """
        Start a new HFB calculation.
        
        Args:
            request: Calculation request parameters.
            progress_callback: Async callback for progress updates.
            
        Returns:
            Calculation ID.
        """
        calc_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create initial status
        status = CalculationStatus(
            id=calc_id,
            nucleus=request.nucleus,
            force_name=request.force_name,
            phase=CalculationPhase.PENDING,
            progress=CalculationProgress(
                calculation_id=calc_id,
                phase=CalculationPhase.PENDING,
                message="Queued for execution",
            ),
            started_at=now,
        )
        
        # Create active calculation tracker
        active = ActiveCalculation(
            id=calc_id,
            request=request,
            status=status,
        )
        
        async with self._lock:
            self._calculations[calc_id] = active
            if progress_callback:
                self._progress_callbacks[calc_id] = [progress_callback]
        
        # Start the calculation in background
        asyncio.create_task(self._run_calculation(active))
        
        return calc_id
    
    async def _run_calculation(self, active: ActiveCalculation):
        """Run a calculation in the background."""
        calc_id = active.id
        request = active.request
        
        try:
            # Phase 1: Ensure JIT warmup
            await self._update_progress(
                calc_id,
                CalculationProgress(
                    calculation_id=calc_id,
                    phase=CalculationPhase.WARMUP,
                    message="Warming up JIT compiler...",
                ),
            )
            
            warmup_manager = get_warmup_manager()
            if not warmup_manager.is_ready:
                await ensure_warmup(timeout=180.0)
            
            if active.cancel_event.is_set():
                await self._mark_cancelled(calc_id)
                return
            
            # Phase 2: Initialize
            await self._update_progress(
                calc_id,
                CalculationProgress(
                    calculation_id=calc_id,
                    phase=CalculationPhase.INITIALIZING,
                    message="Initializing calculation...",
                ),
            )
            
            # Run the actual calculation in thread pool
            # Pass the event loop so the sync function can schedule callbacks
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                lambda: self._run_hfb_sync(active, loop),
            )
            
            if active.cancel_event.is_set():
                await self._mark_cancelled(calc_id)
                return
            
            # Phase 3: Complete
            async with self._lock:
                if calc_id in self._calculations:
                    calc = self._calculations[calc_id]
                    calc.status.phase = CalculationPhase.CONVERGED
                    calc.status.results = results
                    calc.status.completed_at = datetime.now()
                    calc.status.progress = CalculationProgress(
                        calculation_id=calc_id,
                        phase=CalculationPhase.CONVERGED,
                        iteration=results.iterations,
                        max_iterations=request.iteration.max_iterations,
                        fluctuation=results.final_fluctuation,
                        energy=results.energies.total,
                        message="Calculation converged",
                    )
            
            # Save to storage for history
            try:
                from jax_hfbfft.gui.services.storage import get_storage
                storage = await get_storage()
                await storage.save_calculation(self._calculations[calc_id].status)
            except Exception as e:
                print(f"Warning: Failed to save calculation to history: {e}")
            
            await self._notify_callbacks(
                calc_id,
                self._calculations[calc_id].status.progress,
            )
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"Calculation {calc_id} failed:\n{traceback_str}")
            await self._mark_failed(calc_id, error_msg, traceback_str)
    
    def _run_hfb_sync(self, active: ActiveCalculation, main_loop: asyncio.AbstractEventLoop) -> CalculationResults:
        """
        Run HFB calculation synchronously.
        
        This runs in a thread pool worker.
        
        Args:
            active: The active calculation context.
            main_loop: The main event loop for scheduling async callbacks.
        """
        request = active.request
        
        # Import here to avoid loading JAX at module import
        from jax_hfbfft import HFBFFT, Nucleus, Force, Constraint
        
        # Create nucleus
        nucleus = Nucleus(
            protons=request.nucleus.protons,
            neutrons=request.nucleus.neutrons,
        )
        
        # Create force
        force = Force.from_name(
            request.force_name,
            ipair=self._pairing_type_to_int(request.pairing.type),
            v0neut=request.pairing.v0_neutron,
            v0prot=request.pairing.v0_proton,
        )
        
        # Grid config
        grid_config = {}
        if not request.grid.auto:
            grid_config = {
                "nx": request.grid.nx,
                "ny": request.grid.ny,
                "nz": request.grid.nz,
                "dx": request.grid.dx,
                "dy": request.grid.dy,
                "dz": request.grid.dz,
            }
        else:
            # Auto grid based on nucleus size
            grid_size = self._auto_grid_size(nucleus.mass_number)
            grid_config = {
                "nx": grid_size,
                "ny": grid_size,
                "nz": grid_size,
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
            }
        
        # Create constraint
        constraint = None
        if request.constraint.type.value != "none":
            if request.constraint.type.value == "multipole":
                multipoles = {}
                if request.constraint.q20 is not None:
                    multipoles["Q20"] = request.constraint.q20
                if request.constraint.q30 is not None:
                    multipoles["Q30"] = request.constraint.q30
                if request.constraint.q40 is not None:
                    multipoles["Q40"] = request.constraint.q40
                if multipoles:
                    constraint = Constraint.from_multipoles(multipoles)
            elif request.constraint.type.value == "beta_gamma":
                if request.constraint.beta2 is not None:
                    constraint = Constraint.from_beta_gamma(
                        mass_number=nucleus.mass_number,
                        beta2=request.constraint.beta2,
                        gamma=request.constraint.gamma or 0.0,
                    )
        
        # Create calculation
        calc = HFBFFT(
            nucleus=nucleus,
            force=force,
            constraint=constraint,
            **grid_config,
        )
        
        # Initialize
        calc.initialize_wavefunctions(method="harmonic_oscillator")
        
        # Set up progress callback
        start_time = time.time()
        jit_time = 0.0
        
        def iteration_callback(iteration, energy, fluctuation):
            """Called after each iteration."""
            if active.cancel_event.is_set():
                raise InterruptedError("Calculation cancelled")
            
            # Update progress
            progress = CalculationProgress(
                calculation_id=active.id,
                phase=CalculationPhase.ITERATING,
                iteration=iteration,
                max_iterations=request.iteration.max_iterations,
                fluctuation=fluctuation,
                energy=energy,
            )
            active.status.progress = progress
            active.status.phase = CalculationPhase.ITERATING
            
            # Notify callbacks (for websocket updates) - use thread-safe call
            if main_loop is not None:
                try:
                    # Use call_soon_threadsafe to schedule callback on main loop
                    # Need to capture progress in a closure to avoid late binding issues
                    def schedule_callback(p=progress):
                        asyncio.ensure_future(
                            self._notify_callbacks(active.id, p),
                            loop=main_loop
                        )
                    main_loop.call_soon_threadsafe(schedule_callback)
                except Exception:
                    pass  # Ignore callback errors in thread
        
        # Run calculation
        calc._callbacks.append(iteration_callback)
        
        hfb_results = calc.run(
            max_iterations=request.iteration.max_iterations,
            convergence_threshold=request.iteration.convergence_threshold,
            print_interval=request.iteration.print_interval,
        )
        
        end_time = time.time()
        
        # Store the HFBFFT instance for density extraction
        active.hfbfft_instance = calc
        
        # Clear JAX memory to prevent OOM on subsequent runs (but keep calc in memory for now)
        try:
            import jax
            import gc
            # Clear compilation cache
            jax.clear_caches()
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not clear JAX memory: {e}")
        
        # Convert to API results
        return self._convert_results(calc, hfb_results, end_time - start_time, jit_time)
    
    def _convert_results(self, calc, hfb_results, total_time: float, jit_time: float) -> CalculationResults:
        """Convert HFBFFT results to API model."""
        
        # Energy breakdown
        energies = EnergyBreakdown(
            total=float(hfb_results.total_energy),
            kinetic=float(hfb_results.kinetic_energy),
            potential=float(hfb_results.potential_energy),
            coulomb=float(hfb_results.coulomb_energy),
            pairing=float(hfb_results.pairing_energy),
            cm_correction=float(hfb_results.cm_correction),
            ehf0=float(hfb_results.ehf0),
            ehf1=float(hfb_results.ehf1),
            ehf2=float(hfb_results.ehf2),
            ehf3=float(hfb_results.ehf3),
            ehfls=float(hfb_results.ehfls),
        )
        
        # Radii
        radii = RadiiResults(
            neutron=float(hfb_results.rms_radius_n),
            proton=float(hfb_results.rms_radius_p),
            total=float(hfb_results.rms_radius_total),
            charge=float(hfb_results.charge_radius),
        )
        
        # Deformation
        deformation = DeformationResults(
            beta2=float(hfb_results.beta2),
            gamma=float(hfb_results.gamma),
            q20=float(hfb_results.q20),
            q22=float(hfb_results.q22),
        )
        
        # Pairing
        pairing = PairingResults(
            gap_neutron=float(hfb_results.pairing_gap_n),
            gap_proton=float(hfb_results.pairing_gap_p),
            fermi_neutron=float(hfb_results.fermi_energy_n),
            fermi_proton=float(hfb_results.fermi_energy_p),
        )
        
        # Single-particle spectrum
        sp_spectrum = []
        if calc.state.sp_energy is not None:
            import jax.numpy as jnp
            
            for i in range(calc._nstmax):
                isospin = "neutron" if calc.state.isospin[i] == 0 else "proton"
                occupation = float(calc.state.wocc[i])
                
                # Only include occupied or near-occupied states
                if occupation > 0.01:
                    sp_spectrum.append(SingleParticleLevel(
                        index=i,
                        isospin=isospin,
                        energy=float(calc.state.sp_energy[i]),
                        occupation=occupation,
                        parity=int(calc.state.sp_parity[i]) if calc.state.sp_parity is not None else 1,
                    ))
        
        return CalculationResults(
            energies=energies,
            radii=radii,
            deformation=deformation,
            pairing=pairing,
            single_particle_spectrum=sp_spectrum,
            converged=hfb_results.converged,
            iterations=hfb_results.iterations,
            final_fluctuation=float(hfb_results.final_fluctuation),
            total_time_seconds=total_time,
            jit_time_seconds=jit_time,
        )
    
    def _pairing_type_to_int(self, pairing_type) -> int:
        """Convert pairing type enum to integer."""
        from jax_hfbfft.gui.models import PairingType
        mapping = {
            PairingType.NONE: 0,
            PairingType.VDI: 5,
            PairingType.DDDI: 6,
        }
        return mapping.get(pairing_type, 0)
    
    def _auto_grid_size(self, mass_number: int) -> int:
        """Auto-calculate grid size based on nucleus size."""
        if mass_number <= 20:
            return 20
        elif mass_number <= 60:
            return 24
        elif mass_number <= 150:
            return 28
        else:
            return 32
    
    async def _update_progress(self, calc_id: str, progress: CalculationProgress):
        """Update calculation progress."""
        async with self._lock:
            if calc_id in self._calculations:
                self._calculations[calc_id].status.progress = progress
                self._calculations[calc_id].status.phase = progress.phase
        
        await self._notify_callbacks(calc_id, progress)
    
    async def _notify_callbacks(self, calc_id: str, progress: CalculationProgress):
        """Notify all registered callbacks."""
        callbacks = self._progress_callbacks.get(calc_id, [])
        for callback in callbacks:
            try:
                await callback(progress)
            except Exception:
                pass  # Ignore callback errors
    
    async def _mark_cancelled(self, calc_id: str):
        """Mark a calculation as cancelled."""
        async with self._lock:
            if calc_id in self._calculations:
                calc = self._calculations[calc_id]
                calc.status.phase = CalculationPhase.CANCELLED
                calc.status.completed_at = datetime.now()
                calc.status.progress = CalculationProgress(
                    calculation_id=calc_id,
                    phase=CalculationPhase.CANCELLED,
                    message="Calculation cancelled by user",
                )
    
    async def _mark_failed(self, calc_id: str, error: str, traceback: str = ""):
        """Mark a calculation as failed."""
        async with self._lock:
            if calc_id in self._calculations:
                calc = self._calculations[calc_id]
                calc.status.phase = CalculationPhase.FAILED
                calc.status.completed_at = datetime.now()
                # Store full error with traceback for debugging
                full_error = f"{error}\n\nTraceback:\n{traceback}" if traceback else error
                calc.status.error_message = full_error
                calc.status.progress = CalculationProgress(
                    calculation_id=calc_id,
                    phase=CalculationPhase.FAILED,
                    message=f"Calculation failed: {error}",
                )
        
        # Save to storage for history
        try:
            from jax_hfbfft.gui.services.storage import get_storage
            storage = await get_storage()
            if calc_id in self._calculations:
                await storage.save_calculation(self._calculations[calc_id].status)
        except Exception as e:
            print(f"Warning: Failed to save failed calculation to history: {e}")
    
    async def get_calculation(self, calc_id: str) -> Optional[CalculationStatus]:
        """Get the status of a calculation."""
        async with self._lock:
            if calc_id in self._calculations:
                return self._calculations[calc_id].status
        return None
    
    async def cancel_calculation(self, calc_id: str) -> bool:
        """Cancel a running calculation."""
        async with self._lock:
            if calc_id in self._calculations:
                self._calculations[calc_id].cancel_event.set()
                return True
        return False
    
    async def list_calculations(self) -> List[CalculationStatus]:
        """List all calculations."""
        async with self._lock:
            return [calc.status for calc in self._calculations.values()]
    
    async def add_progress_callback(
        self,
        calc_id: str,
        callback: Callable[[CalculationProgress], Awaitable[None]],
    ):
        """Add a progress callback for a calculation."""
        async with self._lock:
            if calc_id not in self._progress_callbacks:
                self._progress_callbacks[calc_id] = []
            self._progress_callbacks[calc_id].append(callback)
    
    async def remove_progress_callback(
        self,
        calc_id: str,
        callback: Callable[[CalculationProgress], Awaitable[None]],
    ):
        """Remove a progress callback."""
        async with self._lock:
            if calc_id in self._progress_callbacks:
                try:
                    self._progress_callbacks[calc_id].remove(callback)
                except ValueError:
                    pass
    
    async def get_density_data(
        self,
        calc_id: str,
        density_type: str = "total",
        downsample: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract density data from a completed calculation.
        
        Args:
            calc_id: Calculation ID
            density_type: Type of density to extract
            downsample: Optional grid size to downsample to
            
        Returns:
            Dictionary with density data or None if not available
        """
        async with self._lock:
            if calc_id not in self._calculations:
                return None
            
            calc = self._calculations[calc_id]
            hfb = calc.hfbfft_instance
            
            if hfb is None or hfb.state.rho is None:
                return None
            
            # Extract the requested density
            import jax.numpy as jnp
            
            if density_type == "neutron":
                density = hfb.state.rho[0]  # Neutron density
            elif density_type == "proton":
                density = hfb.state.rho[1]  # Proton density
            elif density_type == "total":
                density = hfb.state.rho[0] + hfb.state.rho[1]  # Total
            elif density_type == "tau_n":
                density = hfb.state.tau[0]  # Neutron kinetic density
            elif density_type == "tau_p":
                density = hfb.state.tau[1]  # Proton kinetic density
            elif density_type == "tau_total":
                density = hfb.state.tau[0] + hfb.state.tau[1]  # Total kinetic
            else:
                return None
            
            # Convert to numpy
            density_np = jnp.array(density)
            
            # Downsample if requested
            if downsample is not None and downsample < hfb.grid.nx:
                from scipy.ndimage import zoom
                factor = downsample / hfb.grid.nx
                density_np = zoom(density_np, factor, order=1)
                dx = hfb.grid.dx / factor
                dy = hfb.grid.dy / factor
                dz = hfb.grid.dz / factor
                nx, ny, nz = downsample, downsample, downsample
            else:
                dx, dy, dz = hfb.grid.dx, hfb.grid.dy, hfb.grid.dz
                nx, ny, nz = hfb.grid.nx, hfb.grid.ny, hfb.grid.nz
            
            # Convert to Python lists for JSON serialization
            density_list = density_np.tolist()
            
            # Calculate metadata
            min_val = float(jnp.min(density_np))
            max_val = float(jnp.max(density_np))
            
            return {
                "density": density_list,
                "grid": {
                    "nx": int(nx),
                    "ny": int(ny),
                    "nz": int(nz),
                    "dx": float(dx),
                    "dy": float(dy),
                    "dz": float(dz),
                },
                "type": density_type,
                "metadata": {
                    "min_value": min_val,
                    "max_value": max_val,
                    "units": "fm^-3",
                },
            }


# Global service instance
_hfb_service: Optional[HFBService] = None


def get_hfb_service() -> HFBService:
    """Get the global HFB service instance."""
    global _hfb_service
    if _hfb_service is None:
        _hfb_service = HFBService()
    return _hfb_service
