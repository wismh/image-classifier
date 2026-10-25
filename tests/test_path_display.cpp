#include "model/path_display.h"

#include <gtest/gtest.h>

namespace game {

TEST(DisplayPath, ShortPathsStayWhole) {
    EXPECT_EQ(display_path("C:/photos/cats", 40), "C:/photos/cats");
    EXPECT_EQ(display_path("", 10), "");
    EXPECT_EQ(display_path("12345", 5), "12345");
}

TEST(DisplayPath, LongPathsKeepTheEndCutAtASeparator) {
    const std::string shown = display_path("C:/Users/someone/Pictures/pets/cats", 24);
    EXPECT_LE(shown.size(), 24u);
    // 21 characters fit after the "..."; the cut would land inside "someone", so it moves on to the next separator.
    EXPECT_EQ(shown, ".../Pictures/pets/cats");
}

TEST(DisplayPath, BackslashesAreSeparatorsToo) {
    const std::string shown = display_path(std::string{R"(C:\Users\someone\Pictures\pets\cats)"}, 22);
    EXPECT_LE(shown.size(), 22u);
    EXPECT_TRUE(shown.ends_with(R"(\pets\cats)"));
}

TEST(DisplayPath, OneLongNameIsCutInsideIt) {
    const std::string shown = display_path(std::string(50, 'x'), 20);
    EXPECT_EQ(shown.size(), 20u);
    EXPECT_TRUE(shown.starts_with("..."));
}

TEST(DisplayPath, NeverCutsInsideAMultiByteCharacter) {
    const std::string shown = display_path("C:/Фото/Коти/абиссинські", 14);
    for (std::size_t i = 3; i < shown.size(); ++i) {
        // A continuation byte may only follow a lead byte, never directly follow the "...".
        if (i == 3) {
            EXPECT_NE(static_cast<unsigned char>(shown[i]) & 0xC0, 0x80);
        }
    }
}

}
