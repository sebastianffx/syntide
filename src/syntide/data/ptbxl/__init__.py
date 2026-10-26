"""Connect PTB-XL to TimeNet using the standard connector interface."""

from syntide.data.ptbxl.connector import PtbXlConnector

CONNECTOR = PtbXlConnector

__all__ = ["CONNECTOR", "PtbXlConnector"]
