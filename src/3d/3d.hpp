#pragma once

#include "src/3d/gl.hpp"
#include "src/assertions.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x3.hpp>

#include <vector>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <memory>
#include <unordered_map>
#include <array>

namespace osc {
    struct Untextured_vert final {
        glm::vec3 pos;
        glm::vec3 normal;
    };

    struct Textured_vert final {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 texcoord;
    };

    // important: puts an upper limit on the number of verts that a single
    // mesh may contain
    using elidx_t = GLushort;

    template<typename TVert>
    struct CPU_mesh {
        std::vector<TVert> verts;
        std::vector<elidx_t> indices;

        void clear() {
            verts.clear();
            indices.clear();
        }
    };

    struct Untextured_mesh : public CPU_mesh<Untextured_vert> {};
    struct Textured_mesh : public CPU_mesh<Textured_vert> {};

    template<typename TVert>
    void generate_1to1_indices_for_verts(CPU_mesh<TVert>& mesh) {
        if (mesh.verts.size() > std::numeric_limits<elidx_t>::max()) {
            throw std::runtime_error{"cannot generate indices for a mesh: the mesh has too many vertices: if you need to support this many vertices then contact the developers"};
        }

        size_t n = mesh.verts.size();
        mesh.indices.resize(n);
        for (size_t i = 0; i < n; ++i) {
            mesh.indices[i] = static_cast<elidx_t>(i);
        }
    }

    struct Rgba32 final {
        GLubyte r;
        GLubyte g;
        GLubyte b;
        GLubyte a;

        [[nodiscard]] static constexpr Rgba32 from_vec4(glm::vec4 const& v) noexcept {
            Rgba32 rv{};
            rv.r = static_cast<GLubyte>(255.0f * v.r);
            rv.g = static_cast<GLubyte>(255.0f * v.g);
            rv.b = static_cast<GLubyte>(255.0f * v.b);
            rv.a = static_cast<GLubyte>(255.0f * v.a);
            return rv;
        }

        [[nodiscard]] static constexpr Rgba32 from_d4(double r, double g, double b, double a) noexcept {
            Rgba32 rv{};
            rv.r = static_cast<GLubyte>(255.0 * r);
            rv.g = static_cast<GLubyte>(255.0 * g);
            rv.b = static_cast<GLubyte>(255.0 * b);
            rv.a = static_cast<GLubyte>(255.0 * a);
            return rv;
        }

        Rgba32() = default;

        constexpr Rgba32(GLubyte r_, GLubyte g_, GLubyte b_, GLubyte a_) noexcept :
            r{r_},
            g{g_},
            b{b_},
            a{a_} {
        }
    };

    struct Rgb24 final {
        GLubyte r;
        GLubyte g;
        GLubyte b;

        Rgb24() = default;

        constexpr Rgb24(GLubyte r_, GLubyte g_, GLubyte b_) noexcept :
            r{r_},
            g{g_},
            b{b_} {
        }

        explicit constexpr Rgb24(glm::vec3 const& v) noexcept :
            r{static_cast<GLubyte>(255.0f * v.r)},
            g{static_cast<GLubyte>(255.0f * v.g)},
            b{static_cast<GLubyte>(255.0f * v.b)} {
        }
    };

    class Instance_flags final {
        GLubyte flags = 0x00;
        static constexpr GLubyte draw_lines_mask = 0x80;
        static constexpr GLubyte skip_shading_mask = 0x40;
        static constexpr GLubyte skip_vp_mask = 0x20;

    public:
        [[nodiscard]] constexpr GLenum mode() const noexcept {
            return flags & draw_lines_mask ? GL_LINES : GL_TRIANGLES;
        }

        void set_draw_lines() noexcept {
            flags |= draw_lines_mask;
        }

        [[nodiscard]] constexpr bool skip_shading() const noexcept {
            return flags & skip_shading_mask;
        }

        void set_skip_shading() noexcept {
            flags |= skip_shading_mask;
        }

        [[nodiscard]] constexpr bool skip_vp() const noexcept {
            return flags & skip_vp_mask;
        }

        void set_skip_vp() noexcept {
            flags |= skip_vp_mask;
        }

        [[nodiscard]] constexpr bool operator<(Instance_flags other) const noexcept {
            return flags < other.flags;
        }

        [[nodiscard]] constexpr bool operator==(Instance_flags other) const noexcept {
            return flags == other.flags;
        }

        [[nodiscard]] constexpr bool operator!=(Instance_flags other) const noexcept {
            return flags != other.flags;
        }
    };

    template<typename T, typename Derived>
    class Safe_index {
    public:
        using value_type = T;
        static constexpr value_type invalid_value = -1;
        static_assert(std::is_signed_v<value_type>);
        static_assert(std::is_integral_v<value_type>);

    private:
        value_type v;

    public:
        [[nodiscard]] static constexpr Derived from_index(size_t i) {
            if (i > std::numeric_limits<T>::max()) {
                throw std::runtime_error{"tried to create a Safe_index with a value that is too high for the underlying storage"};
            }
            return Derived{static_cast<T>(i)};
        }

        constexpr Safe_index() noexcept : v{invalid_value} {
        }

        explicit constexpr Safe_index(value_type v_) noexcept : v{v_} {
        }

        [[nodiscard]] constexpr value_type get() const noexcept {
            return v;
        }

        [[nodiscard]] constexpr bool is_valid() const noexcept {
            return v >= 0;
        }

        [[nodiscard]] constexpr size_t as_index() const {
            if (!is_valid()) {
                throw std::runtime_error{"tried to convert a Safe_index with an invalid value into an index: this could cause runtime errors and has been disallowed"};
            }
            return static_cast<size_t>(v);
        }

        [[nodiscard]] constexpr bool operator<(Safe_index<T, Derived> other) const noexcept {
            return v < other.v;
        }

        [[nodiscard]] constexpr bool operator==(Safe_index<T, Derived> other) const noexcept {
            return v == other.v;
        }

        [[nodiscard]] constexpr bool operator!=(Safe_index<T, Derived> other) const noexcept {
            return v != other.v;
        }
    };

    class Meshidx : public Safe_index<short, Meshidx> {
        using Safe_index<short, Meshidx>::Safe_index;
    };

    class Texidx : public Safe_index<short, Texidx> {
        using Safe_index<short, Texidx>::Safe_index;
    };

    // create a normal transform from a model transform matrix
    template<typename Mtx>
    static constexpr glm::mat3 normal_matrix(Mtx&& m) noexcept {
        glm::mat3 top_left{m};
        return glm::inverse(glm::transpose(top_left));
    }

    struct Mesh_instance final {
        glm::mat4x3 model_xform;
        glm::mat3 normal_xform;
        Rgba32 rgba;

        union {
            struct {
                GLubyte b0;
                GLubyte b1;
                GLubyte rim_alpha;
            } passthrough;
            Rgb24 passthrough_as_color;

            static_assert(sizeof(passthrough_as_color) == sizeof(passthrough));
        };

        Instance_flags flags;
        Texidx texidx;
        Meshidx meshidx;

        Mesh_instance() noexcept : passthrough_as_color{0x00, 0x00, 0x00} {
        }

        [[nodiscard]] constexpr bool is_opaque() const noexcept {
            return rgba.a == 0xff || texidx.is_valid();
        }
    };

    // list of instances to draw in one renderer drawcall
    struct Drawlist final {
        // note: treat as private, because the implementation might
        // optimize this in various ways
        std::vector<std::vector<Mesh_instance>> _opaque_by_meshidx;
        std::vector<std::vector<Mesh_instance>> _nonopaque_by_meshidx;

        [[nodiscard]] size_t size() const noexcept {
            size_t acc = 0;
            for (auto const& lst : _opaque_by_meshidx) {
                acc += lst.size();
            }
            for (auto const& lst : _nonopaque_by_meshidx) {
                acc += lst.size();
            }
            return acc;
        }

        void push_back(Mesh_instance const& mi) {
            auto& lut = mi.is_opaque() ? _opaque_by_meshidx : _nonopaque_by_meshidx;

            size_t meshidx = mi.meshidx.as_index();

            size_t minsize = meshidx + 1;
            if (lut.size() < minsize) {
                lut.resize(minsize);
            }

            lut[meshidx].push_back(mi);
        }

        void clear() {
            for (auto& lst : _opaque_by_meshidx) {
                lst.clear();
            }
            for (auto& lst : _nonopaque_by_meshidx) {
                lst.clear();
            }
        }

        template<typename Callback>
        void for_each(Callback f) {
            for (auto& lst : _opaque_by_meshidx) {
                for (auto& mi : lst) {
                    f(mi);
                }
            }
            for (auto& lst : _nonopaque_by_meshidx) {
                for (auto& mi : lst) {
                    f(mi);
                }
            }
        }
    };

    // optimize a drawlist
    //
    // (what is optimized is an internal detail: just assume that this function
    //  mutates the drawlist in some way to make a subsequent render call optimal)
    void optimize(Drawlist&) noexcept;

    // a mesh, stored on the GPU
    //
    // not in any particular format - depends on which CPU data was passed
    // into its constructor
    struct GPU_mesh final {
        gl::Array_buffer<GLubyte> verts;
        gl::Element_array_buffer<elidx_t> indices;
        gl::Array_buffer<Mesh_instance, GL_DYNAMIC_DRAW> instances;
        gl::Vertex_array main_vao;
        gl::Vertex_array normal_vao;
        bool is_textured : 1;

        GPU_mesh(Untextured_mesh const&);
        GPU_mesh(Textured_mesh const&);
    };

    struct Gouraud_mrt_shader;
    struct Normals_shader;
    struct Plain_texture_shader;
    struct Colormapped_plain_texture_shader;
    struct Edge_detection_shader;
    struct Skip_msxaa_blitter_shader;

    // storage for GPU data. Used by renderer to load relevant data at runtime
    // (e.g. shaders, programs, mesh data)
    struct GPU_storage final {
        std::unique_ptr<Gouraud_mrt_shader> shader_gouraud;
        std::unique_ptr<Normals_shader> shader_normals;
        std::unique_ptr<Plain_texture_shader> shader_pts;
        std::unique_ptr<Colormapped_plain_texture_shader> shader_cpts;
        std::unique_ptr<Edge_detection_shader> shader_eds;
        std::unique_ptr<Skip_msxaa_blitter_shader> shader_skip_msxaa;

        std::vector<GPU_mesh> meshes;
        std::vector<gl::Texture_2d> textures;
        std::unordered_map<std::string, Meshidx> path_to_meshidx;

        // preallocated meshes
        Meshidx simbody_sphere_idx;
        Meshidx simbody_cylinder_idx;
        Meshidx simbody_cube_idx;
        Meshidx floor_quad_idx;
        Meshidx grid_25x25_idx;
        Meshidx yline_idx;
        Meshidx quad_idx;

        // preallocated textures
        Texidx chequer_idx;

        // debug quad
        gl::Array_buffer<Textured_vert> quad_vbo;

        // VAOs for debug quad
        gl::Vertex_array eds_quad_vao;
        gl::Vertex_array skip_msxaa_quad_vao;
        gl::Vertex_array pts_quad_vao;
        gl::Vertex_array cpts_quad_vao;

        GPU_storage();
        GPU_storage(GPU_storage const&) = delete;
        GPU_storage(GPU_storage&&) noexcept;
        GPU_storage& operator=(GPU_storage const&) = delete;
        GPU_storage& operator=(GPU_storage&&) noexcept;
        ~GPU_storage() noexcept;
    };

    // output target for a scene drawcall
    struct Render_target final {
        // dimensions of buffers
        int w;
        int h;

        // number of multisamples for multisampled buffers
        int samples;

        // raw scene output
        gl::Render_buffer scene_rgba;
        gl::Texture_2d_multisample scene_passthrough;
        gl::Render_buffer scene_depth24stencil8;
        gl::Frame_buffer scene_fbo;

        // passthrough resolution (intermediate data)
        gl::Texture_2d passthrough_nomsxaa;
        gl::Frame_buffer passthrough_fbo;
        std::array<gl::Pixel_pack_buffer<GLubyte, GL_STREAM_READ>, 2> passthrough_pbos;
        int passthrough_pbo_cur;

        // outputs
        gl::Texture_2d scene_tex_resolved;
        gl::Frame_buffer scene_fbo_resolved;
        gl::Texture_2d passthrough_tex_resolved;
        gl::Frame_buffer passthrough_fbo_resolved;
        Rgb24 hittest_result;

        Render_target(int w, int h, int samples);
        void reconfigure(int w, int h, int samples);

        [[nodiscard]] constexpr float aspect_ratio() const noexcept {
            return static_cast<float>(w) / static_cast<float>(h);
        }

        [[nodiscard]] gl::Texture_2d& main() noexcept {
            return scene_tex_resolved;
        }

        [[nodiscard]] constexpr glm::vec2 dimensions() const noexcept {
            return glm::vec2{static_cast<float>(w), static_cast<float>(h)};
        }
    };

    // flags for a scene drawcall
    using DrawcallFlags = int;
    enum DrawcallFlags_ {
        DrawcallFlags_None = 0 << 0,

        // draw meshes in wireframe mode
        DrawcallFlags_WireframeMode = 1 << 0,

        // draw mesh normals on top of render
        DrawcallFlags_ShowMeshNormals = 1 << 1,

        // draw selection rims
        DrawcallFlags_DrawRims = 1 << 2,

        // draw debug quads (development)
        RawRendererFlags_DrawDebugQuads = 1 << 3,

        // perform hit testing on Raw_mesh_instance passthrough data
        RawRendererFlags_PerformPassthroughHitTest = 1 << 4,

        // use optimized hit testing (which might arrive a frame late)
        RawRendererFlags_UseOptimizedButDelayed1FrameHitTest = 1 << 5,

        // draw the scene
        RawRendererFlags_DrawSceneGeometry = 1 << 6,

        // use instanced (optimized) rendering
        RawRendererFlags_UseInstancedRenderer = 1 << 7,

        RawRendererFlags_Default = DrawcallFlags_DrawRims | RawRendererFlags_DrawDebugQuads |
                                   RawRendererFlags_PerformPassthroughHitTest |
                                   RawRendererFlags_UseOptimizedButDelayed1FrameHitTest |
                                   RawRendererFlags_DrawSceneGeometry | RawRendererFlags_UseInstancedRenderer
    };

    // parameters for a scene drawcall
    struct Render_params final {
        glm::mat4 view_matrix;
        glm::mat4 projection_matrix;
        glm::vec3 view_pos;
        glm::vec3 light_dir;
        glm::vec3 light_rgb;
        glm::vec4 background_rgba;
        glm::vec4 rim_rgba;

        DrawcallFlags flags;
        int passthrough_hittest_x;
        int passthrough_hittest_y;
    };

    // draw a scene into the specified render target
    void draw_scene(GPU_storage&, Render_params const&, Drawlist&, Render_target&);
}
