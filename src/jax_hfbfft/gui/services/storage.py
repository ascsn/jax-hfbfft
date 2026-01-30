"""
Storage service for run history persistence.

This module provides SQLite-based storage for calculation history,
allowing users to browse and filter past runs.
"""

import asyncio
import json
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from jax_hfbfft.gui.models import (
    CalculationStatus,
    CalculationSummary,
    CalculationResults,
    CalculationPhase,
    HistoryFilter,
    HistoryResponse,
    NucleusInput,
    BetaSurfaceResult,
    RunType,
)


@dataclass
class StorageConfig:
    """Configuration for run storage."""
    db_path: str = "~/.hfbfft/history.db"
    max_results: int = 1000


class RunStorage:
    """
    SQLite-based storage for calculation history.
    
    This service persists calculation results to disk, allowing users
    to browse historical runs and compare results.
    
    Example:
        storage = RunStorage()
        await storage.initialize()
        
        # Save a completed calculation
        await storage.save_calculation(calc_status)
        
        # Query history
        history = await storage.get_history(filter=HistoryFilter(element="Ca"))
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the storage service.
        
        Args:
            config: Storage configuration.
        """
        self.config = config or StorageConfig()
        self._db_path = Path(self.config.db_path).expanduser()
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the database, creating tables if needed."""
        if self._initialized:
            return
        
        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and create tables
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS calculations (
                id TEXT PRIMARY KEY,
                protons INTEGER NOT NULL,
                neutrons INTEGER NOT NULL,
                element_symbol TEXT NOT NULL,
                mass_number INTEGER NOT NULL,
                force_name TEXT NOT NULL,
                run_type TEXT NOT NULL DEFAULT 'calculation',
                phase TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                total_energy REAL,
                final_fluctuation REAL,
                converged INTEGER,
                iterations INTEGER,
                error_message TEXT,
                results_json TEXT,
                request_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate older databases to include run_type/request_json if missing
        async with self._db.execute("PRAGMA table_info(calculations)") as cursor:
            columns = {row[1] async for row in cursor}
        if "run_type" not in columns:
            await self._db.execute("ALTER TABLE calculations ADD COLUMN run_type TEXT NOT NULL DEFAULT 'calculation'")
        if "request_json" not in columns:
            await self._db.execute("ALTER TABLE calculations ADD COLUMN request_json TEXT")
        
        # Create indices for common queries
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_element 
            ON calculations(element_symbol)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_force 
            ON calculations(force_name)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_started 
            ON calculations(started_at)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_phase 
            ON calculations(phase)
        """)
        
        await self._db.commit()
        self._initialized = True
    
    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False
    
    async def save_calculation(self, status: CalculationStatus):
        """
        Save a calculation to the database.
        
        Args:
            status: The calculation status to save.
        """
        await self.initialize()
        
        nucleus = status.nucleus
        results_json = None
        if status.results:
            results_json = status.results.model_dump_json()
        
        await self._db.execute("""
            INSERT OR REPLACE INTO calculations (
                id, protons, neutrons, element_symbol, mass_number,
                force_name, run_type, phase, started_at, completed_at,
                total_energy, final_fluctuation, converged, iterations,
                error_message, results_json, request_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            status.id,
            nucleus.protons,
            nucleus.neutrons,
            nucleus.symbol,
            nucleus.mass_number,
            status.force_name,
            status.run_type.value,
            status.phase.value,
            status.started_at.isoformat(),
            status.completed_at.isoformat() if status.completed_at else None,
            status.results.energies.total if status.results else None,
            status.results.final_fluctuation if status.results else None,
            1 if status.results and status.results.converged else 0,
            status.results.iterations if status.results else None,
            status.error_message,
            results_json,
            status.progress.model_dump_json() if status.progress else None,
        ))
        
        await self._db.commit()

    async def save_surface(self, surface_id: str, nucleus: NucleusInput, force_name: str, result: BetaSurfaceResult):
        """Save a surface scan result to history."""
        await self.initialize()

        results_json = result.model_dump_json()
        now = datetime.now().isoformat()

        await self._db.execute("""
            INSERT OR REPLACE INTO calculations (
                id, protons, neutrons, element_symbol, mass_number,
                force_name, run_type, phase, started_at, completed_at,
                total_energy, final_fluctuation, converged, iterations,
                error_message, results_json, request_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            surface_id,
            nucleus.protons,
            nucleus.neutrons,
            nucleus.symbol,
            nucleus.mass_number,
            force_name,
            RunType.SURFACE.value,
            CalculationPhase.CONVERGED.value,
            now,
            now,
            None,
            None,
            None,
            None,
            None,
            results_json,
            None,
        ))

        await self._db.commit()
    
    async def get_calculation(self, calc_id: str) -> Optional[CalculationStatus]:
        """
        Get a calculation by ID.
        
        Args:
            calc_id: The calculation ID.
            
        Returns:
            The calculation status, or None if not found.
        """
        await self.initialize()
        
        async with self._db.execute(
            "SELECT * FROM calculations WHERE id = ?",
            (calc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_status(row)
        return None
    
    async def delete_calculation(self, calc_id: str) -> bool:
        """
        Delete a calculation from history.
        
        Args:
            calc_id: The calculation ID.
            
        Returns:
            True if deleted, False if not found.
        """
        await self.initialize()
        
        cursor = await self._db.execute(
            "DELETE FROM calculations WHERE id = ?",
            (calc_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
    
    async def get_history(
        self,
        filter: Optional[HistoryFilter] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "started_at",
        sort_desc: bool = True,
    ) -> HistoryResponse:
        """
        Get calculation history with filtering and pagination.
        
        Args:
            filter: Optional filters to apply.
            page: Page number (1-indexed).
            page_size: Number of results per page.
            sort_by: Column to sort by.
            sort_desc: Sort descending if True.
            
        Returns:
            Paginated history response.
        """
        await self.initialize()
        
        # Build query
        conditions = []
        params = []
        
        if filter:
            if filter.element:
                conditions.append("element_symbol = ?")
                params.append(filter.element)
            if filter.min_a is not None:
                conditions.append("mass_number >= ?")
                params.append(filter.min_a)
            if filter.max_a is not None:
                conditions.append("mass_number <= ?")
                params.append(filter.max_a)
            if filter.force_name:
                conditions.append("force_name = ?")
                params.append(filter.force_name)
            if filter.status:
                conditions.append("phase = ?")
                params.append(filter.status.value)
            if filter.run_type:
                conditions.append("run_type = ?")
                params.append(filter.run_type.value)
            if filter.from_date:
                conditions.append("started_at >= ?")
                params.append(filter.from_date.isoformat())
            if filter.to_date:
                conditions.append("started_at <= ?")
                params.append(filter.to_date.isoformat())
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        # Validate sort column
        valid_columns = ["started_at", "completed_at", "total_energy", "mass_number", "element_symbol"]
        if sort_by not in valid_columns:
            sort_by = "started_at"
        
        sort_order = "DESC" if sort_desc else "ASC"
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM calculations {where_clause}"
        async with self._db.execute(count_query, params) as cursor:
            row = await cursor.fetchone()
            total = row[0]
        
        # Get paginated results
        offset = (page - 1) * page_size
        query = f"""
            SELECT * FROM calculations {where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])
        
        calculations = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                calculations.append(self._row_to_summary(row))
        
        total_pages = (total + page_size - 1) // page_size
        
        return HistoryResponse(
            calculations=calculations,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    
    async def get_results(self, calc_id: str) -> Optional[CalculationResults]:
        """
        Get the full results for a calculation.
        
        Args:
            calc_id: The calculation ID.
            
        Returns:
            The calculation results, or None if not found.
        """
        await self.initialize()
        
        async with self._db.execute(
            "SELECT results_json, run_type FROM calculations WHERE id = ?",
            (calc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["results_json"]:
                if row["run_type"] == RunType.SURFACE.value:
                    return None
                return CalculationResults.model_validate_json(row["results_json"])
        return None

    async def get_surface_results(self, calc_id: str) -> Optional[BetaSurfaceResult]:
        """Get surface scan results for a history item."""
        await self.initialize()

        async with self._db.execute(
            "SELECT results_json, run_type FROM calculations WHERE id = ?",
            (calc_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["results_json"] and row["run_type"] == RunType.SURFACE.value:
                return BetaSurfaceResult.model_validate_json(row["results_json"])
        return None
    
    def _row_to_status(self, row) -> CalculationStatus:
        """Convert a database row to CalculationStatus."""
        from jax_hfbfft.gui.models import CalculationProgress
        
        results = None
        surface_results = None
        if row["results_json"]:
            if row["run_type"] == RunType.SURFACE.value:
                surface_results = BetaSurfaceResult.model_validate_json(row["results_json"])
            else:
                results = CalculationResults.model_validate_json(row["results_json"])
        
        nucleus = NucleusInput(
            protons=row["protons"],
            neutrons=row["neutrons"],
        )
        
        phase = CalculationPhase(row["phase"])
        
        return CalculationStatus(
            id=row["id"],
            nucleus=nucleus,
            force_name=row["force_name"],
            phase=phase,
            progress=CalculationProgress(
                calculation_id=row["id"],
                phase=phase,
                iteration=row["iterations"] or 0,
                fluctuation=row["final_fluctuation"] or 0.0,
                energy=row["total_energy"] or 0.0,
            ),
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            results=results,
            error_message=row["error_message"],
            run_type=RunType(row["run_type"]) if row["run_type"] else RunType.CALCULATION,
            surface_results=surface_results,
        )
    
    def _row_to_summary(self, row) -> CalculationSummary:
        """Convert a database row to CalculationSummary."""
        return CalculationSummary(
            id=row["id"],
            nucleus_symbol=row["element_symbol"],
            nucleus_a=row["mass_number"],
            force_name=row["force_name"],
            phase=CalculationPhase(row["phase"]),
            energy=row["total_energy"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            run_type=RunType(row["run_type"]) if row["run_type"] else RunType.CALCULATION,
        )


# Global storage instance
_storage: Optional[RunStorage] = None


async def get_storage() -> RunStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = RunStorage()
        await _storage.initialize()
    return _storage
