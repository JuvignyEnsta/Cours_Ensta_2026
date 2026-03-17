#pragma once
#include <SDL2/SDL.h>

class Window
{
public:
    Window(const char* title, int width, int height);
    Window(const Window&) = delete;
    Window(Window&&) = delete;
    ~Window();

    Window& operator = (const Window&) = delete;
    Window& operator = (Window&&) = delete;

    SDL_Window* get() { return m_window; }
    SDL_Surface* getSurface() { return SDL_GetWindowSurface(m_window); }

    void set_pen( Uint8 r, Uint8 g, Uint8 b ) {
        SDL_SetRenderDrawColor( SDL_GetRenderer( m_window ), r, g, b, 255 );
    }

    void pset( int x, int y ) {
        SDL_RenderDrawPoint( SDL_GetRenderer( m_window ), x, y );
    }

    void clear() {
        SDL_RenderClear( SDL_GetRenderer( m_window ) );
    }

    void draw( SDL_Point const* points, int count ) {
        SDL_RenderDrawPoints( SDL_GetRenderer( m_window ), points, count );
    }

    void line( int x1, int y1, int x2, int y2 ) {
        SDL_RenderDrawLine( SDL_GetRenderer( m_window ), x1, y1, x2, y2 );
    }

    void blit() {
        SDL_RenderPresent( SDL_GetRenderer( m_window ) );
    }

    std::pair<int, int> size() {
        int w, h;
        SDL_GetWindowSize( m_window, &w, &h );
        return { w, h };
    }

private:
    SDL_Window* m_window{ nullptr };
    SDL_Renderer* m_renderer{ nullptr };
};