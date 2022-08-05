#include "SceneViewer.hpp"

#include "src/Bindings/ImGuiHelpers.hpp"
#include "src/Graphics/SceneDecorationNew.hpp"
#include "src/Graphics/SceneRendererNew.hpp"
#include "src/Graphics/SceneRendererNewParams.hpp"

#include <glm/vec2.hpp>
#include <imgui.h>
#include <nonstd/span.hpp>

#include <memory>
#include <utility>

class osc::SceneViewer::Impl final {
public:
    void draw(nonstd::span<SceneDecorationNew const> els, SceneRendererNewParams const& params)
	{
		m_Renderer.draw(els, params);

		// emit the texture to ImGui
		osc::DrawTextureAsImGuiImage(m_Renderer.updOutputTexture(), m_Renderer.getDimensions());
		m_IsHovered = ImGui::IsItemHovered();
		m_IsLeftClicked = ImGui::IsItemHovered() && osc::IsMouseReleasedWithoutDragging(ImGuiMouseButton_Left);
		m_IsRightClicked = ImGui::IsItemHovered() && osc::IsMouseReleasedWithoutDragging(ImGuiMouseButton_Right);
	}

	bool isHovered() const
	{
		return m_IsHovered;
	}

	bool isLeftClicked() const
	{
		return m_IsLeftClicked;
	}

	bool isRightClicked() const
	{
		return m_IsRightClicked;
	}

private:
	SceneRendererNew m_Renderer;
	bool m_IsHovered = false;
	bool m_IsLeftClicked = false;
	bool m_IsRightClicked = false;
};


// public API (PIMPL)

osc::SceneViewer::SceneViewer() :
	m_Impl{new Impl{}}
{
}

osc::SceneViewer::SceneViewer(SceneViewer&& tmp) noexcept :
	m_Impl{std::exchange(tmp.m_Impl, nullptr)}
{
}

osc::SceneViewer& osc::SceneViewer::operator=(SceneViewer&& tmp) noexcept
{
	std::swap(tmp.m_Impl, m_Impl);
	return *this;
}

osc::SceneViewer::~SceneViewer() noexcept
{
	delete m_Impl;
}

void osc::SceneViewer::draw(nonstd::span<SceneDecorationNew const> els, SceneRendererNewParams const& params)
{
    m_Impl->draw(std::move(els), params);
}

bool osc::SceneViewer::isHovered() const
{
	return m_Impl->isHovered();
}

bool osc::SceneViewer::isLeftClicked() const
{
	return m_Impl->isLeftClicked();
}

bool osc::SceneViewer::isRightClicked() const
{
	return m_Impl->isRightClicked();
}
