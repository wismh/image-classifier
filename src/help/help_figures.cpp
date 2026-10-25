#include "help/help_figures.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <vector>
#include <string>

namespace game {

namespace {

using engine::render::Rect;
using engine::ui::IDrawList;

constexpr glm::vec4 kInk{0.20f, 0.25f, 0.33f, 1.0f};
constexpr glm::vec4 kMuted{0.58f, 0.64f, 0.72f, 1.0f};
constexpr glm::vec4 kGrid{0.89f, 0.91f, 0.94f, 1.0f};
constexpr glm::vec4 kBlue{0.15f, 0.39f, 0.92f, 1.0f};
constexpr glm::vec4 kRed{0.86f, 0.15f, 0.15f, 1.0f};
constexpr glm::vec4 kPaper{1.0f, 1.0f, 1.0f, 1.0f};
constexpr glm::vec4 kBlack{0.0f, 0.0f, 0.0f, 1.0f};

// A small drawing board: text, dots and polylines in the area of one figure.
class Board {
public:
    Board(IDrawList& list, const Rect& area, const Tr& tr)
        : list_(list)
        , area_(area)
        , tr_(tr) {}

    [[nodiscard]] const Rect& area() const { return area_; }
    [[nodiscard]] std::string tr(const char* key) const { return tr_(key); }

    void line(glm::vec2 a, glm::vec2 b, glm::vec4 color, float width = 1.5f) {
        list_.line(abs(a), abs(b), color, width);
    }

    void dashed(glm::vec2 a, glm::vec2 b, glm::vec4 color, float width = 1.0f) {
        const glm::vec2 from = abs(a);
        const glm::vec2 to = abs(b);
        const float length = glm::length(to - from);
        const int pieces = std::max(1, static_cast<int>(length / 8.0f));
        for (int i = 0; i < pieces; ++i) {
            const float t0 = static_cast<float>(i) / static_cast<float>(pieces);
            const float t1 = (static_cast<float>(i) + 0.55f) / static_cast<float>(pieces);
            list_.line(from + (to - from) * t0, from + (to - from) * t1, color, width);
        }
    }

    void polyline(const std::vector<glm::vec2>& points, glm::vec4 color, float width = 2.0f) {
        for (std::size_t i = 1; i < points.size(); ++i) {
            list_.line(abs(points[i - 1]), abs(points[i]), color, width);
        }
    }

    void dot(glm::vec2 centre, float radius, glm::vec4 color) {
        const glm::vec2 c = abs(centre);
        list_.fill_rect({c.x - radius, c.y - radius, radius * 2.0f, radius * 2.0f}, color, radius);
    }

    void box(const Rect& rect, glm::vec4 fill, glm::vec4 border, float width = 1.0f) {
        const Rect r{area_.x + rect.x, area_.y + rect.y, rect.w, rect.h};
        list_.fill_rect(r, fill);
        list_.stroke_rect(r, border, width);
    }

    void fill(const Rect& rect, glm::vec4 color, float radius = 0.0f) {
        list_.fill_rect({area_.x + rect.x, area_.y + rect.y, rect.w, rect.h}, color, radius);
    }

    // `at` is the top-left of the text; `align` 0 left, 0.5 centred, 1 right of that point (width is estimated).
    void text(const std::string& value, glm::vec2 at, glm::vec4 color, float size = 13.0f, float align = 0.0f) {
        list_.set_font(engine::builtin::font_ui, size);
        const float width = static_cast<float>(value.size()) * size * 0.52f;
        list_.text(value, abs({at.x - width * align, at.y}), color);
    }

private:
    [[nodiscard]] glm::vec2 abs(glm::vec2 p) const { return {area_.x + p.x, area_.y + p.y}; }

    IDrawList& list_;
    Rect area_;
    const Tr& tr_;
};

// A plot with its own coordinates: world (x, y) maps into a margin-inset part of the board, y pointing up.
struct Plot {
    Board& board;
    Rect frame;  // in board pixels
    float x0, x1, y0, y1;

    [[nodiscard]] glm::vec2 at(float x, float y) const {
        return {frame.x + (x - x0) / (x1 - x0) * frame.w, frame.y + frame.h - (y - y0) / (y1 - y0) * frame.h};
    }

    void grid(float step_x, float step_y) const {
        for (float x = std::ceil(x0 / step_x) * step_x; x <= x1 + 1e-4f; x += step_x) {
            board.line(at(x, y0), at(x, y1), kGrid, 1.0f);
        }
        for (float y = std::ceil(y0 / step_y) * step_y; y <= y1 + 1e-4f; y += step_y) {
            board.line(at(x0, y), at(x1, y), kGrid, 1.0f);
        }
    }

    void axes() const {
        const float ax = std::clamp(0.0f, x0, x1);
        const float ay = std::clamp(0.0f, y0, y1);
        board.line(at(x0, ay), at(x1, ay), kMuted, 1.5f);
        board.line(at(ax, y0), at(ax, y1), kMuted, 1.5f);
    }

    [[nodiscard]] std::vector<glm::vec2> curve(float from, float to, int steps, float (*f)(float)) const {
        std::vector<glm::vec2> points;
        for (int i = 0; i <= steps; ++i) {
            const float x = from + (to - from) * static_cast<float>(i) / static_cast<float>(steps);
            points.push_back(at(x, f(x)));
        }
        return points;
    }
};

// A grid of numbers, one cell per value.
void number_grid(Board& board, glm::vec2 origin, int columns, int rows, float cell, const std::vector<float>& values,
        glm::vec4 fill) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            const float v = values[static_cast<std::size_t>(r * columns + c)];
            const Rect rect{origin.x + static_cast<float>(c) * cell, origin.y + static_cast<float>(r) * cell, cell - 1.0f,
                    cell - 1.0f};
            board.box(rect, fill, kGrid);
            board.text(std::format("{:.0f}", v), {rect.x + rect.w * 0.5f, rect.y + cell * 0.5f - 7.0f}, kInk, 12.0f, 0.5f);
        }
    }
}

void outline(Board& board, const Rect& rect, glm::vec4 color, float width) {
    board.line({rect.x, rect.y}, {rect.x + rect.w, rect.y}, color, width);
    board.line({rect.x + rect.w, rect.y}, {rect.x + rect.w, rect.y + rect.h}, color, width);
    board.line({rect.x + rect.w, rect.y + rect.h}, {rect.x, rect.y + rect.h}, color, width);
    board.line({rect.x, rect.y + rect.h}, {rect.x, rect.y}, color, width);
}

void arrow(Board& board, glm::vec2 from, glm::vec2 to) {
    board.line(from, to, kMuted, 2.5f);
    board.line(to + glm::vec2{-10.0f, -7.0f}, to, kMuted, 2.5f);
    board.line(to + glm::vec2{-10.0f, 7.0f}, to, kMuted, 2.5f);
}

void conv_figure(Board& board) {
    constexpr int kSide = 6;
    constexpr int kOut = kSide - 2;
    constexpr float kCell = 28.0f;
    constexpr float kTop = 30.0f;
    // A vertical edge: dark on the left, bright on the right.
    std::vector<float> input(kSide * kSide);
    for (int r = 0; r < kSide; ++r) {
        for (int c = 0; c < kSide; ++c) {
            input[static_cast<std::size_t>(r * kSide + c)] = c < 3 ? 1.0f : 8.0f;
        }
    }
    const std::vector<float> kernel{1, 0, -1, 1, 0, -1, 1, 0, -1};
    std::vector<float> output(kOut * kOut);
    for (int i = 0; i < kOut; ++i) {
        for (int j = 0; j < kOut; ++j) {
            float sum = 0.0f;
            for (int u = 0; u < 3; ++u) {
                for (int v = 0; v < 3; ++v) {
                    sum += kernel[static_cast<std::size_t>(u * 3 + v)] * input[static_cast<std::size_t>((i + u) * kSide + j + v)];
                }
            }
            output[static_cast<std::size_t>(i * kOut + j)] = sum;
        }
    }
    constexpr int kRow = 1;
    constexpr int kColumn = 1;

    const glm::vec2 input_at{20.0f, kTop};
    number_grid(board, input_at, kSide, kSide, kCell, input, kPaper);
    const Rect window{input_at.x + kColumn * kCell, input_at.y + kRow * kCell, 3.0f * kCell - 1.0f, 3.0f * kCell - 1.0f};
    board.fill({window.x, window.y, window.w, window.h}, {0.15f, 0.39f, 0.92f, 0.12f});
    outline(board, window, kBlue, 2.5f);
    board.text(board.tr("fig.patch"), {input_at.x + kSide * kCell * 0.5f, kTop - 22.0f}, kInk, 12.0f, 0.5f);

    const float kernel_x = input_at.x + kSide * kCell + 46.0f;
    board.text("x", {kernel_x - 26.0f, kTop + 3.0f * kCell - 12.0f}, kMuted, 22.0f, 0.5f);
    const glm::vec2 kernel_at{kernel_x, kTop + 1.5f * kCell};
    number_grid(board, kernel_at, 3, 3, kCell, kernel, {0.93f, 0.96f, 1.0f, 1.0f});
    outline(board, {kernel_at.x, kernel_at.y, 3.0f * kCell - 1.0f, 3.0f * kCell - 1.0f}, kBlue, 2.0f);
    board.text(board.tr("fig.filter"), {kernel_at.x + 1.5f * kCell, kTop - 22.0f + 1.5f * kCell}, kInk, 12.0f, 0.5f);

    const float output_x = kernel_x + 3.0f * kCell + 66.0f;
    arrow(board, {kernel_x + 3.0f * kCell + 10.0f, kTop + 3.0f * kCell}, {output_x - 12.0f, kTop + 3.0f * kCell});
    const glm::vec2 output_at{output_x, kTop + 1.0f * kCell};
    number_grid(board, output_at, kOut, kOut, kCell, output, kPaper);
    const Rect result{output_at.x + kColumn * kCell, output_at.y + kRow * kCell, kCell - 1.0f, kCell - 1.0f};
    board.fill(result, {0.15f, 0.39f, 0.92f, 0.18f});
    outline(board, result, kBlue, 2.5f);
    board.text(board.tr("fig.map"), {output_at.x + kOut * kCell * 0.5f, kTop + kCell - 22.0f}, kInk, 12.0f, 0.5f);
}

void pool_figure(Board& board) {
    constexpr float kCell = 34.0f;
    constexpr float kTop = 28.0f;
    const std::vector<float> input{1, 3, 2, 0, 5, 2, 7, 1, 0, 4, 1, 6, 2, 8, 3, 2};
    const std::array<glm::vec4, 4> tints{glm::vec4{0.86f, 0.92f, 1.0f, 1.0f}, glm::vec4{0.86f, 0.97f, 0.88f, 1.0f},
            glm::vec4{1.0f, 0.93f, 0.84f, 1.0f}, glm::vec4{0.97f, 0.88f, 0.95f, 1.0f}};
    const glm::vec2 input_at{40.0f, kTop};
    std::array<float, 4> biggest{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const std::size_t quarter = static_cast<std::size_t>((r / 2) * 2 + c / 2);
            const float v = input[static_cast<std::size_t>(r * 4 + c)];
            biggest[quarter] = std::max(biggest[quarter], v);
            const Rect rect{input_at.x + static_cast<float>(c) * kCell, input_at.y + static_cast<float>(r) * kCell,
                    kCell - 1.0f, kCell - 1.0f};
            board.box(rect, tints[quarter], kGrid);
            board.text(std::format("{:.0f}", v), {rect.x + rect.w * 0.5f, rect.y + kCell * 0.5f - 7.0f}, kInk, 13.0f, 0.5f);
        }
    }
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            outline(board, {input_at.x + static_cast<float>(c) * 2.0f * kCell, input_at.y + static_cast<float>(r) * 2.0f * kCell,
                           2.0f * kCell - 1.0f, 2.0f * kCell - 1.0f},
                    kMuted, 1.5f);
        }
    }
    board.text(board.tr("fig.pool_in"), {input_at.x + 2.0f * kCell, kTop - 22.0f}, kInk, 12.0f, 0.5f);

    const float output_x = input_at.x + 4.0f * kCell + 110.0f;
    arrow(board, {input_at.x + 4.0f * kCell + 24.0f, kTop + 2.0f * kCell}, {output_x - 24.0f, kTop + 2.0f * kCell});
    board.text("max", {input_at.x + 4.0f * kCell + 52.0f, kTop + 2.0f * kCell - 26.0f}, kInk, 12.0f, 0.5f);
    const glm::vec2 output_at{output_x, kTop + kCell};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            const std::size_t quarter = static_cast<std::size_t>(r * 2 + c);
            const Rect rect{output_at.x + static_cast<float>(c) * kCell * 1.5f, output_at.y + static_cast<float>(r) * kCell * 1.5f,
                    kCell * 1.5f - 1.0f, kCell * 1.5f - 1.0f};
            board.box(rect, tints[quarter], kMuted, 1.5f);
            board.text(std::format("{:.0f}", biggest[quarter]), {rect.x + rect.w * 0.5f, rect.y + rect.h * 0.5f - 9.0f}, kInk,
                    17.0f, 0.5f);
        }
    }
    board.text(board.tr("fig.pool_out"), {output_at.x + kCell * 1.5f, kTop - 22.0f + kCell}, kInk, 12.0f, 0.5f);
}

void loss_figure(Board& board) {
    const Rect area = board.area();
    const Plot plot{board, {40.0f, 14.0f, area.w - 70.0f, area.h - 56.0f}, 0.0f, 1.0f, 0.0f, 3.6f};
    plot.grid(0.25f, 0.5f);
    plot.axes();
    std::vector<glm::vec2> points;
    for (int i = 0; i <= 80; ++i) {
        const float p = 0.027f + (1.0f - 0.027f) * static_cast<float>(i) / 80.0f;
        points.push_back(plot.at(p, -std::log(p)));
    }
    board.polyline(points, kBlue, 3.0f);
    for (const float p : {0.1f, 0.5f, 1.0f}) {
        board.dot(plot.at(p, -std::log(p)), 4.5f, kRed);
        const float shift = p < 0.9f ? 8.0f : -36.0f;
        board.text(std::format("{:.2f}", std::fabs(std::log(p))), plot.at(p, -std::log(p)) + glm::vec2{shift, -18.0f}, kRed,
                12.0f);
    }
    board.text(board.tr("fig.prob_correct"), {40.0f + (area.w - 70.0f) * 0.5f, area.h - 22.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.loss"), {44.0f, 2.0f}, kInk, 12.0f);
}

void overfit_figure(Board& board) {
    const Rect area = board.area();
    const Plot plot{board, {40.0f, 14.0f, area.w - 70.0f, area.h - 56.0f}, 0.0f, 10.0f, 0.0f, 2.5f};
    plot.grid(2.0f, 0.5f);
    plot.axes();
    board.polyline(plot.curve(0.0f, 10.0f, 50, [](float t) { return 0.12f + 2.1f * std::exp(-0.55f * t); }), kBlue, 3.0f);
    board.polyline(plot.curve(0.0f, 10.0f, 50,
                           [](float t) { return 0.55f + 1.7f * std::exp(-0.7f * t) + 0.012f * t * t; }),
            kRed, 3.0f);
    board.dashed(plot.at(4.6f, 0.0f), plot.at(4.6f, 2.5f), kMuted, 1.5f);
    board.text(board.tr("fig.train"), plot.at(6.4f, 0.12f) + glm::vec2{0.0f, 10.0f}, kBlue, 12.0f);
    board.text(board.tr("fig.validation"), plot.at(6.2f, 1.75f), kRed, 12.0f);
    board.text(board.tr("fig.best"), plot.at(4.7f, 2.3f), kMuted, 12.0f);
    board.text(board.tr("fig.epochs"), {40.0f + (area.w - 70.0f) * 0.5f, area.h - 22.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.loss"), {44.0f, 2.0f}, kInk, 12.0f);
}

void heat_figure(Board& board) {
    const Rect area = board.area();
    constexpr float kFrame = 150.0f;
    const float left = area.w * 0.5f - kFrame - 40.0f;
    const float top = 14.0f;
    const glm::vec4 fur{0.62f, 0.58f, 0.52f, 1.0f};
    const auto place = [&](float u, float v) { return glm::vec2{left + u * kFrame, top + v * kFrame}; };

    // A photo: a sofa, and an animal with a round head and two ears.
    board.box({left, top, kFrame, kFrame}, {0.22f, 0.30f, 0.38f, 1.0f}, kMuted, 2.0f);
    board.fill({left, top + kFrame * 0.72f, kFrame, kFrame * 0.28f}, {0.36f, 0.27f, 0.22f, 1.0f});
    board.fill({left + kFrame * 0.2f, top + kFrame * 0.5f, kFrame * 0.6f, kFrame * 0.34f}, fur, 14.0f);
    board.dot(place(0.5f, 0.36f), kFrame * 0.17f, fur);
    board.dot(place(0.38f, 0.2f), kFrame * 0.05f, fur);
    board.dot(place(0.62f, 0.2f), kFrame * 0.05f, fur);
    board.dot(place(0.44f, 0.35f), 3.0f, kBlack);
    board.dot(place(0.56f, 0.35f), 3.0f, kBlack);

    // The same photo with the evidence over it: strongest at the head.
    const float right = area.w * 0.5f + 40.0f;
    board.box({right, top, kFrame, kFrame}, {0.17f, 0.22f, 0.28f, 1.0f}, kMuted, 2.0f);
    const auto at = [&](float u, float v) { return glm::vec2{right + u * kFrame, top + v * kFrame}; };
    board.fill({right, top + kFrame * 0.72f, kFrame, kFrame * 0.28f}, {0.27f, 0.2f, 0.17f, 1.0f});
    board.fill({right + kFrame * 0.2f, top + kFrame * 0.5f, kFrame * 0.6f, kFrame * 0.34f}, {0.46f, 0.43f, 0.39f, 1.0f}, 14.0f);
    constexpr int kRings = 7;
    for (int ring = 0; ring < kRings; ++ring) {
        const float t = static_cast<float>(ring) / (kRings - 1);
        const float radius = kFrame * (0.28f - 0.24f * t);
        board.dot(at(0.5f, 0.34f), radius, {1.0f, 1.0f - t, 0.0f, 0.22f + 0.1f * t});
    }
    board.dot(at(0.5f, 0.62f), kFrame * 0.07f, {1.0f, 0.75f, 0.0f, 0.35f});
    arrow(board, {left + kFrame + 8.0f, top + kFrame * 0.5f}, {right - 8.0f, top + kFrame * 0.5f});

    // The scale of the colors.
    const float bar_left = right + kFrame + 24.0f;
    for (int i = 0; i < 20; ++i) {
        const float t = static_cast<float>(i) / 19.0f;
        board.fill({bar_left, top + kFrame - static_cast<float>(i + 1) * kFrame / 20.0f, 14.0f, kFrame / 20.0f + 0.5f},
                {1.0f, 1.0f - t, 0.0f, 1.0f});
    }
    board.text(board.tr("fig.strong"), {bar_left + 20.0f, top}, kInk, 12.0f);
    board.text(board.tr("fig.weak"), {bar_left + 20.0f, top + kFrame - 14.0f}, kInk, 12.0f);
    board.text(board.tr("fig.photo"), {left + kFrame * 0.5f, top + kFrame + 8.0f}, kInk, 12.0f, 0.5f);
    board.text(board.tr("fig.evidence"), {right + kFrame * 0.5f, top + kFrame + 8.0f}, kInk, 12.0f, 0.5f);
}

}

void draw_figure(FigureId figure, IDrawList& list, const Rect& area, const Tr& tr) {
    Board board(list, area, tr);
    board.box({0.0f, 0.0f, area.w, area.h}, kPaper, kGrid);
    switch (figure) {
    case FigureId::None: break;
    case FigureId::Conv: conv_figure(board); break;
    case FigureId::Pool: pool_figure(board); break;
    case FigureId::Loss: loss_figure(board); break;
    case FigureId::Overfit: overfit_figure(board); break;
    case FigureId::Heat: heat_figure(board); break;
    }
}

}
