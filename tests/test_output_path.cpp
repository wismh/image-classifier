#include "model/output_path.h"

#include <gtest/gtest.h>

namespace game {

TEST(PlannedOutputPath, IsEmptyWithoutAPhotosFolder) {
    EXPECT_TRUE(planned_output_path({}, "C:/out").empty());
}

TEST(PlannedOutputPath, GoesBesideThePhotosFolderByDefault) {
    EXPECT_EQ(planned_output_path("C:/data/pets", {}), std::filesystem::path{"C:/data/pets.cvds"});
}

TEST(PlannedOutputPath, GoesToTheChosenFolder) {
    EXPECT_EQ(planned_output_path("C:/data/pets", "D:/out"), std::filesystem::path{"D:/out/pets.cvds"});
}

TEST(PlannedOutputPath, ATrailingSeparatorStillNamesTheFileAfterTheLastFolder) {
    EXPECT_EQ(planned_output_path("C:/data/pets/", "D:/out"), std::filesystem::path{"D:/out/pets.cvds"});
}

}
