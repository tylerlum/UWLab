"""Thin wrapper for evaluation-oriented distillation modes.

Usage examples:
    python isaacsim_conversion/distill_eval.py --mode teacher_eval ...
    python isaacsim_conversion/distill_eval.py --mode student_eval ...
"""

from isaacsim_conversion.distill import main


if __name__ == "__main__":
    main()
