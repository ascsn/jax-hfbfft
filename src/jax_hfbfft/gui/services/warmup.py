"""
JIT Warmup Manager for interactive calculations.

This module handles pre-warming the JIT cache to minimize latency
for interactive GUI calculations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from enum import Enum


class WarmupStatus(str, Enum):
    """Status of JIT warmup."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WarmupConfig:
    """Configuration for JIT warmup."""
    # Use smaller typical nucleus for warmup (O-16)
    max_neutron_states: int = 16  # O-16 uses 8n, but buffer for larger
    max_proton_states: int = 16   # O-16 uses 8p, but buffer for larger  
    grid_size: int = 20  # Matches auto grid for mass <= 20
    grid_spacing: float = 1.0
    warmup_iterations: int = 3  # Ensure all code paths compiled


@dataclass
class WarmupState:
    """Current state of warmup process."""
    status: WarmupStatus = WarmupStatus.NOT_STARTED
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None


class JITWarmupManager:
    """
    Manages JIT warmup for interactive HFB calculations.
    
    The warmup process runs a calculation with maximum-size arrays
    to ensure all JAX functions are JIT-compiled before user
    calculations begin.
    
    Example:
        manager = JITWarmupManager()
        await manager.warmup(progress_callback=update_ui)
        
        # Now calculations will be fast
        if manager.is_ready:
            result = run_calculation(...)
    """
    
    def __init__(self, config: Optional[WarmupConfig] = None):
        """
        Initialize the warmup manager.
        
        Args:
            config: Warmup configuration. If None, uses defaults.
        """
        self.config = config or WarmupConfig()
        self.state = WarmupState()
        self._warmup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    @property
    def is_ready(self) -> bool:
        """Check if JIT warmup is complete."""
        return self.state.status == WarmupStatus.COMPLETED
    
    @property
    def progress(self) -> float:
        """Get warmup progress (0.0 to 1.0)."""
        return self.state.progress
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed warmup time in seconds."""
        if self.state.start_time is None:
            return None
        end = self.state.end_time or time.time()
        return end - self.state.start_time
    
    async def warmup(
        self,
        progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
        force: bool = False,
    ) -> bool:
        """
        Run JIT warmup asynchronously.
        
        Args:
            progress_callback: Async callback for progress updates.
                              Called with (progress: float, message: str)
            force: If True, re-run warmup even if already completed.
            
        Returns:
            True if warmup completed successfully, False otherwise.
        """
        async with self._lock:
            # Check if already complete or in progress
            if self.state.status == WarmupStatus.COMPLETED and not force:
                return True
            
            if self.state.status == WarmupStatus.IN_PROGRESS:
                # Wait for existing warmup
                if self._warmup_task:
                    await self._warmup_task
                return self.is_ready
            
            # Start warmup
            self.state = WarmupState(
                status=WarmupStatus.IN_PROGRESS,
                progress=0.0,
                message="Starting JIT warmup...",
                start_time=time.time(),
            )
            
            if progress_callback:
                await progress_callback(0.0, "Starting JIT warmup...")
        
        try:
            success = await self._run_warmup(progress_callback)
            
            async with self._lock:
                if success:
                    self.state.status = WarmupStatus.COMPLETED
                    self.state.progress = 1.0
                    self.state.message = "JIT warmup complete"
                else:
                    self.state.status = WarmupStatus.FAILED
                    self.state.message = "JIT warmup failed"
                
                self.state.end_time = time.time()
            
            if progress_callback:
                await progress_callback(self.state.progress, self.state.message)
            
            return success
            
        except Exception as e:
            async with self._lock:
                self.state.status = WarmupStatus.FAILED
                self.state.error = str(e)
                self.state.message = f"JIT warmup failed: {e}"
                self.state.end_time = time.time()
            
            if progress_callback:
                await progress_callback(self.state.progress, self.state.message)
            
            return False
    
    async def _run_warmup(
        self,
        progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ) -> bool:
        """
        Execute the warmup calculation.
        
        This runs in a thread pool to avoid blocking the async loop.
        """
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        
        # Run the heavy computation in a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                self._warmup_sync,
            )
            
            # Poll for progress while waiting
            warmup_iterations = self.config.warmup_iterations
            check_interval = 0.5  # seconds
            
            while not future.done():
                await asyncio.sleep(check_interval)
                
                # Update progress based on time elapsed (rough estimate)
                # Actual progress is hard to track from async context
                elapsed = time.time() - (self.state.start_time or time.time())
                estimated_total = 90.0  # Rough estimate: 90 seconds for warmup
                progress = min(0.95, elapsed / estimated_total)
                
                async with self._lock:
                    self.state.progress = progress
                    self.state.message = f"Compiling JAX functions... ({elapsed:.0f}s)"
                
                if progress_callback:
                    await progress_callback(progress, self.state.message)
            
            # Get result
            try:
                result = await asyncio.wrap_future(future)
                return result
            except Exception as e:
                async with self._lock:
                    self.state.error = str(e)
                return False
    
    def _warmup_sync(self) -> bool:
        """
        Synchronous warmup calculation.
        
        This actually imports JAX and runs the warmup. It must be run
        in a separate thread to avoid blocking the async event loop.
        """
        try:
            # Import here to avoid loading JAX at module import time
            from jax_hfbfft import HFBFFT, Nucleus, Force
            
            # Create a moderate-sized calculation to warm up all code paths
            # Using O-16 with typical parameters users will actually use
            calc = HFBFFT(
                nucleus=Nucleus(protons=8, neutrons=8),  # O-16
                force=Force.from_name("SLy4"),
                nx=self.config.grid_size,
                ny=self.config.grid_size,
                nz=self.config.grid_size,
                dx=self.config.grid_spacing,
                dy=self.config.grid_spacing,
                dz=self.config.grid_spacing,
                npsi=(self.config.max_neutron_states, self.config.max_proton_states),
                ipair=0,  # No pairing for simplest warmup
            )
            
            # Initialize wavefunctions
            calc.initialize_wavefunctions(method="harmonic_oscillator")
            
            # Run a few iterations to compile all functions
            # Use run() with limited iterations - this triggers JIT compilation
            calc.run(
                max_iterations=self.config.warmup_iterations,
                convergence_threshold=1e-6,
                print_interval=1,
            )
            
            return True
            
        except Exception as e:
            print(f"Warmup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_background_warmup(
        self,
        progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
    ):
        """
        Start warmup as a background task.
        
        This returns immediately and runs warmup in the background.
        Check is_ready or progress to monitor status.
        """
        async def run():
            await self.warmup(progress_callback)
        
        self._warmup_task = asyncio.create_task(run())
        return self._warmup_task
    
    async def wait_for_warmup(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for warmup to complete.
        
        Args:
            timeout: Maximum time to wait in seconds. None = wait forever.
            
        Returns:
            True if warmup completed, False if timed out or failed.
        """
        if self.is_ready:
            return True
        
        if self._warmup_task is None:
            return False
        
        try:
            await asyncio.wait_for(self._warmup_task, timeout=timeout)
            return self.is_ready
        except asyncio.TimeoutError:
            return False
    
    def get_status(self) -> dict:
        """Get current warmup status as a dictionary."""
        return {
            "status": self.state.status.value,
            "progress": self.state.progress,
            "message": self.state.message,
            "elapsed_seconds": self.elapsed_time,
            "is_ready": self.is_ready,
            "error": self.state.error,
        }


# Global warmup manager instance
_warmup_manager: Optional[JITWarmupManager] = None


def get_warmup_manager() -> JITWarmupManager:
    """Get the global warmup manager instance."""
    global _warmup_manager
    if _warmup_manager is None:
        _warmup_manager = JITWarmupManager()
    return _warmup_manager


async def ensure_warmup(
    progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
    timeout: Optional[float] = 120.0,
) -> bool:
    """
    Ensure JIT warmup is complete, starting it if necessary.
    
    Args:
        progress_callback: Callback for progress updates.
        timeout: Maximum time to wait for warmup.
        
    Returns:
        True if warmup is complete, False otherwise.
    """
    manager = get_warmup_manager()
    
    if manager.is_ready:
        return True
    
    if manager.state.status == WarmupStatus.NOT_STARTED:
        manager.start_background_warmup(progress_callback)
    
    return await manager.wait_for_warmup(timeout)
