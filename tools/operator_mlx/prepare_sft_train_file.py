"""Compatibility wrapper. Prefer ``aptus prepare-train``."""

from aptus.prepare_train import main, mlx_valid_count, order_rows_for_mlx_split

__all__ = ["main", "mlx_valid_count", "order_rows_for_mlx_split"]


if __name__ == "__main__":
    raise SystemExit(main())
