#include "window.hpp"

Window::Window(const char* title, int width, int height)
{
    m_window = SDL_CreateWindow(title,
                                SDL_WINDOWPOS_UNDEFINED,
                                SDL_WINDOWPOS_UNDEFINED,
                                width,
                                height,
                                SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
    if (m_window) {
        m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    }
}
// ====================================================================================================================
Window::~Window()
{
    if (m_window) {
        SDL_DestroyRenderer(m_renderer);
        SDL_DestroyWindow(m_window);
    }
}
// ====================================================================================================================
