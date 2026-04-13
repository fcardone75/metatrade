"""Module and schema versioning.

Every analysis module, contract, and serialized artefact carries a version.
This allows the system to detect incompatibilities at runtime rather than
silently processing mismatched data.

ModuleVersion — semantic version (MAJOR.MINOR.PATCH) for code modules.
SchemaVersion — integer version for data contracts (AnalysisSignal, etc.).
               Incremented only on breaking changes to the contract shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class ModuleVersion:
    """Semantic version for a code module.

    Compatibility rule: two versions are compatible iff they share the same
    MAJOR number. A MAJOR bump signals a breaking API change.
    """

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name, value in (("major", self.major), ("minor", self.minor), ("patch", self.patch)):
            if value < 0:
                raise ValueError(f"ModuleVersion.{name} must be >= 0, got {value}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: ModuleVersion) -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: ModuleVersion) -> bool:
        return self == other or self < other

    def is_compatible_with(self, other: ModuleVersion) -> bool:
        """True if same MAJOR version (backward-compatible range)."""
        return self.major == other.major

    @classmethod
    def parse(cls, version_str: str) -> ModuleVersion:
        """Parse a semver string like '1.2.3'."""
        match = _SEMVER_RE.match(version_str.strip())
        if not match:
            raise ValueError(
                f"Invalid semantic version: {version_str!r}. Expected MAJOR.MINOR.PATCH"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )


@dataclass(frozen=True)
class SchemaVersion:
    """Integer version for a data contract schema.

    Increment when the shape of a serialized contract changes in a
    backward-incompatible way (e.g. renamed field, changed type).
    """

    version: int

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"SchemaVersion must be >= 1, got {self.version}")

    def __str__(self) -> str:
        return str(self.version)

    def __int__(self) -> int:
        return self.version

    def is_compatible_with(self, other: SchemaVersion) -> bool:
        """Schemas are compatible only when identical."""
        return self.version == other.version

    def next(self) -> SchemaVersion:
        """Return the next schema version."""
        return SchemaVersion(self.version + 1)
