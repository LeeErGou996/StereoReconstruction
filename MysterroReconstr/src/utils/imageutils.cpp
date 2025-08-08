#include "imageutils.h"
#include <png.h>
#include <cstdio>
#include <stdexcept>
#include <array>
#include <cassert>
#include <cmath> // For fabs and std::round

bool myImReadPNG(const std::string& filename, MyImage& img) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) return false;

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); fclose(fp); return false; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return false;
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    img.width = png_get_image_width(png, info);
    img.height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    // 只支持8位深度
    if (bit_depth != 8) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return false;
    }

    // 转换为8位RGB或灰度
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER); // 添加alpha通道
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);
    img.channels = png_get_channels(png, info);
    img.data.resize(img.width * img.height * img.channels);
    std::vector<png_bytep> row_pointers(img.height);
    for (int y = 0; y < img.height; ++y)
        row_pointers[y] = img.data.data() + y * img.width * img.channels;
    png_read_image(png, row_pointers.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return true;
}

bool myImWritePNG(const std::string& filename, const MyImage& img) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) return false;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return false; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }
    png_init_io(png, fp);
    int color_type = (img.channels == 1) ? PNG_COLOR_TYPE_GRAY :
                     (img.channels == 3) ? PNG_COLOR_TYPE_RGB :
                     (img.channels == 4) ? PNG_COLOR_TYPE_RGBA : -1;
    if (color_type == -1) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }
    png_set_IHDR(png, info, img.width, img.height, 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> row_pointers(img.height);
    for (int y = 0; y < img.height; ++y)
        row_pointers[y] = const_cast<unsigned char*>(img.data.data() + y * img.width * img.channels);
    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

// 对MyImage应用3x3单应变换（最近邻插值）
void warpImage(const MyImage& src, MyImage& dst, const std::array<std::array<double, 3>, 3>& H) {
    dst.width = src.width;
    dst.height = src.height;
    dst.channels = src.channels;
    dst.data.resize(dst.width * dst.height * dst.channels, 0);
    // 计算H的逆
    double a = H[0][0], b = H[0][1], c = H[0][2];
    double d = H[1][0], e = H[1][1], f = H[1][2];
    double g = H[2][0], h = H[2][1], i = H[2][2];
    double det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h;
    if (fabs(det) < 1e-12) return;
    double invH[3][3];
    invH[0][0] = (e*i - f*h) / det;
    invH[0][1] = (c*h - b*i) / det;
    invH[0][2] = (b*f - c*e) / det;
    invH[1][0] = (f*g - d*i) / det;
    invH[1][1] = (a*i - c*g) / det;
    invH[1][2] = (c*d - a*f) / det;
    invH[2][0] = (d*h - e*g) / det;
    invH[2][1] = (b*g - a*h) / det;
    invH[2][2] = (a*e - b*d) / det;
    for (int y = 0; y < dst.height; ++y) {
        for (int x = 0; x < dst.width; ++x) {
            double X = invH[0][0]*x + invH[0][1]*y + invH[0][2];
            double Y = invH[1][0]*x + invH[1][1]*y + invH[1][2];
            double W = invH[2][0]*x + invH[2][1]*y + invH[2][2];
            if (fabs(W) < 1e-8) continue;
            X /= W; Y /= W;
            int ix = static_cast<int>(std::round(X));
            int iy = static_cast<int>(std::round(Y));
            if (ix >= 0 && ix < src.width && iy >= 0 && iy < src.height) {
                for (int c = 0; c < src.channels; ++c) {
                    dst.data[(y*dst.width + x)*dst.channels + c] = src.data[(iy*src.width + ix)*src.channels + c];
                }
            }
        }
    }
} 

void colorize_disparity(const MyImage& gray, MyImage& color) {
    color.width = gray.width;
    color.height = gray.height;
    color.channels = 3;
    color.data.resize(color.width * color.height * 3);

    for (int i = 0; i < gray.width * gray.height; ++i) {
        float v = gray.data[i] / 255.0f;
        v = std::max(0.0f, std::min(1.0f, v));
        float r, g, b;
        if (v < 0.25f) {
            r = 0; g = 4 * v; b = 1;
        } else if (v < 0.5f) {
            r = 0; g = 1; b = 1 - 4 * (v - 0.25f);
        } else if (v < 0.75f) {
            r = 4 * (v - 0.5f); g = 1; b = 0;
        } else {
            r = 1; g = 1 - 4 * (v - 0.75f); b = 0;
        }
        color.data[i*3+0] = static_cast<unsigned char>(r * 255);
        color.data[i*3+1] = static_cast<unsigned char>(g * 255);
        color.data[i*3+2] = static_cast<unsigned char>(b * 255);
    }
}

void hstack3(const MyImage& a, const MyImage& b, const MyImage& c, MyImage& out) {
    assert(a.height == b.height && b.height == c.height);
    assert(a.channels == b.channels && b.channels == c.channels);

    out.width = a.width + b.width + c.width;
    out.height = a.height;
    out.channels = a.channels;
    out.data.resize(out.width * out.height * out.channels);

    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            int src_x = x;
            const MyImage* src = nullptr;
            if (x < a.width) {
                src = &a;
                src_x = x;
            } else if (x < a.width + b.width) {
                src = &b;
                src_x = x - a.width;
            } else {
                src = &c;
                src_x = x - a.width - b.width;
            }

            for (int c_idx = 0; c_idx < out.channels; ++c_idx) {
                out.data[(y * out.width + x) * out.channels + c_idx] =
                    src->data[(y * src->width + src_x) * src->channels + c_idx];
            }
        }
    }
}

bool myImReadPFM(const std::string& filename, MyImage& img) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) return false;

    char type[3];
    if (fscanf(fp, "%2s\n", type) != 1 || (type[0] != 'P' || type[1] != 'f')) {
        fclose(fp);
        return false;
    }

    int width, height;
    float scale;
    if (fscanf(fp, "%d %d\n%f\n", &width, &height, &scale) != 3) {
        fclose(fp);
        return false;
    }

    img.width = width;
    img.height = height;
    img.channels = 1;
    std::vector<float> float_data(width * height);

    if (fread(float_data.data(), sizeof(float), width * height, fp) != size_t(width * height)) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    // 找到有效最小最大值（排除无效值）
    float minval = 1e9f, maxval = -1e9f;
    for (float v : float_data) {
        if (v > -1e6f && v < 1e6f) {
            minval = std::min(minval, v);
            maxval = std::max(maxval, v);
        }
    }

    if (maxval == minval) {
        maxval = minval + 1e-5f; // 避免除零
    }

    img.data.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        float v = float_data[i];
        if (v < minval) v = minval;
        if (v > maxval) v = maxval;
        float norm = (v - minval) / (maxval - minval);
        img.data[i] = static_cast<unsigned char>(norm * 255.0f);
    }

    // 修正：Middlebury 的 PFM 是 bottom-up 需要 flip
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width; ++x) {
            std::swap(img.data[y * width + x], img.data[(height - 1 - y) * width + x]);
        }
    }

    return true;
}

bool myImReadPGM(const std::string& filename, MyImage& img) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) return false;

    char magic[3];
    if (fscanf(fp, "%2s\n", magic) != 1 || (magic[0] != 'P' || magic[1] != '5')) {
        fclose(fp);
        return false;
    }

    int width, height, maxval;
    if (fscanf(fp, "%d %d\n%d\n", &width, &height, &maxval) != 3) {
        fclose(fp);
        return false;
    }

    img.width = width;
    img.height = height;
    img.channels = 1;
    img.data.resize(width * height);

    if (fread(img.data.data(), 1, width * height, fp) != size_t(width * height)) {
        fclose(fp);
        return false;
    }
    fclose(fp);

    return true;
}
