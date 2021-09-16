#pragma once

#include "src/3D/Gl.hpp"
#include "src/3D/ShaderLocationIndex.hpp"

namespace osc {
    // uses a geometry shader to render normals as lines
    struct NormalsShader final {
        gl::Program program;

        static constexpr gl::AttributeVec3 aPos = SHADER_LOC_VERTEX_POSITION;
        static constexpr gl::AttributeVec3 aNormal = SHADER_LOC_VERTEX_NORMAL;

        gl::UniformMat4 uModelMat;
        gl::UniformMat4 uViewMat;
        gl::UniformMat4 uProjMat;
        gl::UniformMat4 uNormalMat;

        NormalsShader();
    };
}
