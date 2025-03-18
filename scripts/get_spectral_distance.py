import typing as tp




if __name__ == "__main__":
    main(
        device="cuda:0",
        paths=[
            'image_refl_35_40_25',
            'image_refl_35_40_30',
            'image_refl_35_40_35',
            'image_refl_35_40_36',
            'image_refl_35_40_38',
            'image_refl_35_40_39',
            # 'fid_refl_fid_0',
            # 'fid_refl_fid_20',
            # 'fid_refl_fid_25',
            # 'fid_refl_fid_30',
            # 'fid_refl_fid_32',
            # 'fid_refl_fid_35',
            # 'fid_refl_fid_38',
            #
            # 'fid_refl_0.01_0',
            # 'fid_refl_0.01_20',
            # 'fid_refl_0.01_25',
            # 'fid_refl_0.01_32',
            # 'fid_refl_0.01_35',
            # 'fid_refl_0.01_36',
            # 'fid_refl_0.01_37',
            # 'fid_refl_0.01_38',
            #
            # 'fid_refl_0.01_1_4_0',
            # 'fid_refl_0.01_1_4_20',
            # 'fid_refl_0.01_1_4_25',
            # 'fid_refl_0.01_1_4_32',
            # 'fid_refl_0.01_1_4_35',
            # 'fid_refl_0.01_1_4_38',
            #
            # "fid_refl_coco_0",
            # "fid_refl_coco_20",
            # "fid_refl_coco_25",
            # "fid_refl_coco_32",
        ],
        data_path="data/lpips/",
        grouped_by=None,
    )
