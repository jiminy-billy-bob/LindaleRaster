/*
    This is a modified version of uraster : https://github.com/Steve132/uraster
*/

#pragma once

#include <iostream>
#define cimg_use_jpeg
#define cimg_use_xshm
#include "CImg.h"
#include <Eigen/Dense>
#include <Eigen/LU>	// for .inverse(). Probably not needed
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include "stb_image_write.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION

namespace LindaleRaster
{

    typedef struct {
        float x, y, z,
            nx, ny, nz, // Normal
            u, v;
    } Vert;

    struct VertShaderOut
    {
        Eigen::Vector4f pos;
        Eigen::Vector3f normal;
        Eigen::Vector2f uv;

        VertShaderOut() :
            pos(0.0f, 0.0f, 0.0f, 0.0f), normal(0.0f, 0.0f, 0.0f), uv(0.0f, 0.0f)
        {}
        const Eigen::Vector4f& position() const
        {
            return pos;
        }
        VertShaderOut& operator+=(const VertShaderOut& tp)
        {
            pos += tp.pos;
            normal += tp.normal;
            uv += tp.uv;
            return *this;
        }
        VertShaderOut& operator*=(const float& f)
        {
            pos *= f;
            normal *= f;
            uv *= f;
            return *this;
        }
    };

    class Pixel
    {
    public:
        Eigen::Vector4f color;
        float& depth() { return color[3]; }
        Pixel() :color(0.0f, 0.0f, 0.0f, -1e10f)
        {}
    };


    // This is the framebuffer class.
    class Framebuffer
    {
    protected:
        std::vector<Pixel> data;
    public:
        const std::size_t width;
        const std::size_t height;
        // constructor initializes the array
        Framebuffer(std::size_t w, std::size_t h, const Pixel& pt = Pixel()) :
            data(w*h, pt),
            width(w), height(h)
        {}
        // 2D pixel access
        Pixel& operator()(std::size_t x, std::size_t y)
        {
            return data[y*width + x];
        }
        // const version
        const Pixel& operator()(std::size_t x, std::size_t y) const
        {
            return data[y*width + x];
        }
        void clear(const Pixel& pt = Pixel())
        {
            std::fill(data.begin(), data.end(), pt);
        }
    };

    VertShaderOut vertex_shader(const Vert& vertIn, const Eigen::Matrix4f& mvp = Eigen::Matrix4f::Identity())
    {
        VertShaderOut vout;
        vout.pos = mvp*Eigen::Vector4f(vertIn.x, vertIn.y, vertIn.z, 1.0f);
        vout.uv = Eigen::Vector2f(vertIn.u, vertIn.v).array().abs();
        vout.uv[0] = vout.uv[0] - floor(vout.uv[0]);
        vout.uv[1] = vout.uv[1] - floor(vout.uv[1]);
        vout.normal = Eigen::Vector3f(vertIn.nx, vertIn.ny, vertIn.nz);
        return vout;
    }

    Pixel fragment_shader(const VertShaderOut& fsin, const cimg_library::CImg<uint8_t>& texture)
    {
        Pixel pixel;

        //Eigen::Vector2f uv(fsin.uv[0] * tex1.width()*0.4f, fsin.uv[1] * tex1.height()*0.4f);
        Eigen::Vector2f uv(fsin.uv[0] * texture.width(), fsin.uv[1] * texture.height());
        Eigen::Vector3f diffuse;
        for (int c = 0; c < 3; c++)
        {
            diffuse[c] = texture.linear_atXY(uv[0], uv[1], 0, c) / 255.0f;
        }
        float theta = 1.2f*sin(0);
        Eigen::Vector3f ld(1.0, sin(theta), cos(theta));
        float intensity = ld.normalized().dot(fsin.normal);
        pixel.color.head<3>() = diffuse*intensity;
        return pixel;
    }


    // This function runs the vertex shader on all the vertices, producing the varyings that will be interpolated by the rasterizer.
    // Vert can be anything, VertShaderOut MUST have a position() method that returns a 4D vector, and it must have an overloaded *= and += operator for the interpolation
    // The right way to think of VertShaderOut is that it is the class you write containing the varying outputs from the vertex shader.
    void run_vertex_shader(std::vector<Vert>* vertexbuffer, std::vector<VertShaderOut>* vertShaders)
    {
#pragma omp parallel for
        for (std::size_t i = 0; i < vertexbuffer->size(); i++)
        {
            vertShaders->at(i) = vertex_shader(vertexbuffer->at(i));
        }
    }

    struct BarycentricTransform
    {
    private:
        Eigen::Vector2f offset;
        Eigen::Matrix2f Ti;
    public:
        BarycentricTransform(const Eigen::Vector2f& s1, const Eigen::Vector2f& s2, const Eigen::Vector2f& s3) :
            offset(s3)
        {
            Eigen::Matrix2f T;
            T << (s1 - s3), (s2 - s3);
            Ti = T.inverse();
        }
        Eigen::Vector3f operator()(const Eigen::Vector2f& v) const
        {
            Eigen::Vector2f b;
            b = Ti*(v - offset);
            return Eigen::Vector3f(b[0], b[1], 1.0f - b[0] - b[1]);
        }
    };


    // This function takes in 3 varyings vertices from the fragment shader that make up a triangle,
    // rasterizes the triangle and runs the fragment shader on each resulting pixel.
    void rasterize_triangle(Framebuffer& fb, const std::array<VertShaderOut, 3>& verts)
    {
        std::array<Eigen::Vector4f, 3> points{ {verts[0].position(),verts[1].position(),verts[2].position()} };
        //Do the perspective divide by w to get screen space coordinates.
        std::array<Eigen::Vector4f, 3> epoints{ {points[0] / points[0][3],points[1] / points[1][3],points[2] / points[2][3]} };
        auto ss1 = epoints[0].head<2>().array(), ss2 = epoints[1].head<2>().array(), ss3 = epoints[2].head<2>().array();

        // calculate the bounding box of the triangle in screen space floating point.
        Eigen::Array2f bb_ul = ss1.min(ss2).min(ss3);
        Eigen::Array2f bb_lr = ss1.max(ss2).max(ss3);
        Eigen::Array2i isz(fb.width, fb.height);

        // convert bounding box to fixed point.
        // move bounding box from (-1.0,1.0)->(0,imgdim)
        Eigen::Array2i ibb_ul = ((bb_ul*0.5f + 0.5f)*isz.cast<float>()).cast<int>();
        Eigen::Array2i ibb_lr = ((bb_lr*0.5f + 0.5f)*isz.cast<float>()).cast<int>();
        ibb_lr += 1;	//add one pixel of coverage

        // clamp the bounding box to the framebuffer size if necessary (this is clipping.  Not quite how the GPU actually does it but same effect sorta).
        ibb_ul = ibb_ul.max(Eigen::Array2i(0, 0));
        ibb_lr = ibb_lr.min(isz);

        cimg_library::CImg<uint8_t> texture("C:/woodgrain.jpg");

        BarycentricTransform bt(ss1.matrix(), ss2.matrix(), ss3.matrix());

        // for all the pixels in the bounding box
        for (int y = ibb_ul[1]; y < ibb_lr[1]; y++)
            for (int x = ibb_ul[0]; x < ibb_lr[0]; x++)
            {
                Eigen::Vector2f ssc(x, y);
                ssc.array() /= isz.cast<float>();	//move pixel to relative coordinates
                ssc.array() -= 0.5f;
                ssc.array() *= 2.0f;

                // Compute barycentric coordinates of the pixel center
                Eigen::Vector3f bary = bt(ssc);

                // if the pixel has valid barycentric coordinates, the pixel is in the triangle
                if ((bary.array() < 1.0f).all() && (bary.array() > 0.0f).all())
                {
                    float d = bary[0] * epoints[0][2] + bary[1] * epoints[1][2] + bary[2] * epoints[2][2];
                    // Reference the current pixel at that coordinate
                    Pixel& po = fb(x, y);
                    // if the interpolated depth passes the depth test
                    if (po.depth() < d && d < 1.0)
                    {
                        // interpolate varying parameters
                        VertShaderOut v = verts[0];
                        v *= bary[0];
                        VertShaderOut vt = verts[1];
                        vt *= bary[1];
                        v += vt;
                        vt = verts[2];
                        vt *= bary[2];
                        v += vt;

                        // call the fragment shader
                        po = fragment_shader(v, texture);
                        po.depth() = d; //write the depth buffer
                    }
                }
            }
    }


    // This function rasterizes a set of triangles determined by an index buffer and a buffer of output verts.
    void rasterize(Framebuffer& fb, std::vector<int>* indices, std::vector<VertShaderOut>* verts)
    {
#pragma omp parallel for
        for (std::size_t i = 0; i < indices->size(); i += 3)
        {
            std::array<VertShaderOut, 3> tri{ { verts->at(indices->at(i)), verts->at(indices->at(i + 1)), verts->at(indices->at(i + 2)) } };
            rasterize_triangle(fb, tri);
        }
    }


    // This function does a draw call from an indexed buffer
    void draw(Framebuffer& fb,
        std::vector<Vert>* vertexbuffer,
        std::vector<int>* facebuffer)
    {
        std::vector<VertShaderOut> vertShaders;
        vertShaders.resize(vertexbuffer->size());
        run_vertex_shader(vertexbuffer, &vertShaders);
        rasterize(fb, facebuffer, &vertShaders);
    }


    bool write_framebuffer(const Framebuffer& fb, const std::string& filename)
    {
        uint8_t* pixels = new uint8_t[fb.width*fb.height * 3];
        std::unique_ptr<uint8_t[]> data(pixels);

        const float* fbdata = &(fb(0, 0).color[0]);
        for (size_t i = 0; i<fb.width*fb.height; i++)
        {
            for (int c = 0; c<3; c++)
            {
                pixels[3 * i + c] = std::max(0.0f, std::min(fbdata[4 * i + c] * 255.0f, 255.0f));
            }
        }

        if (0 == stbi_write_png(filename.c_str(), fb.width, fb.height, 3, pixels, 0))
        {
            return false;
        }
        else {
            return true;
        }
    }

}
