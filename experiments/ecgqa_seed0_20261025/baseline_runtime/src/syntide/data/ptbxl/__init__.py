"""PTB-XL connector package; ``CONNECTOR`` follows the timenet-connectors convention."""

from syntide.data.ptbxl.connector import PtbXlConnector

CONNECTOR = PtbXlConnector

__all__ = ["CONNECTOR", "PtbXlConnector"]
