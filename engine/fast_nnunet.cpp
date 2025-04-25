//
// Created by justin on 25-3-9.
//

#include "fast_nnunet_tools.h"
#include "fast_nnunet_load_image.h"
#include "fast_nnunet_dicom_convertor.h"
#include "fast_nnunet_engine.h"
#include "fast_nnunet_evaluator.h"

namespace FastnnUNet {
    Eva::Eva() = default;

    void Eva::initializer()
    {
        // initialization
        const auto Engine = std::make_shared<FastnnUNet::Engine>();

        Engine->set_config("configs/nnunet_bone_low_config.ini");
        Engine->set_workspace("models/batch" ,false, true);

        // load
        const std::string nii_file = "test_image/headneck.nii.gz";
        const auto [image, inimg_raw, original_orientation] = Data::LoadData(nii_file);

        // infer
        const auto output_mask = Engine->infer(inimg_raw, image, true, false, true);

        // save
        Tools::save_mask(output_mask, image, "output_mask.nii.gz");
    }
}