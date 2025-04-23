import typing as tp

if __name__ == "__main__":
    main(
        device="cuda:0",
        paths=[
            "image_refl_35_40_25",
            "image_refl_35_40_30",
            "image_refl_35_40_35",
            "image_refl_35_40_36",
            "image_refl_35_40_38",
            "image_refl_35_40_39",
        ],
        data_path="data/lpips/",
        grouped_by=None,
    )
