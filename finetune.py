"""Compatibility wrapper for the released D-Heart-Reg CAS fine-tuning entry.

The reported objective is vertex reconstruction plus a raw PCA-coefficient L2
regularizer. Segmentation-mask supervision is intentionally not used here.
"""

from scripts.finetune_cta1100_cas import main


if __name__ == "__main__":
    main()
