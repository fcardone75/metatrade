"""Core data contracts.

These are the shared data types that flow between every module.
Contracts are immutable value objects (frozen dataclasses).
All timestamps are UTC-aware datetimes.
All financial values are Decimal (never float).
"""
