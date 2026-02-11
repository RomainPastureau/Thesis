import pygame, os, sys
from pygame.locals import *


def toggle_fullscreen():
    screen = pygame.display.get_surface()
    tmp = screen.convert()
    caption = pygame.display.get_caption()
    cursor = pygame.mouse.get_cursor()

    w, h = screen.get_width(), screen.get_height()
    flags = screen.get_flags()
    bits = screen.get_bitsize()

    screen = pygame.display.set_mode((w, h), flags ^ FULLSCREEN, bits)
    screen.blit(tmp, (0, 0))
    pygame.display.set_caption(*caption)

    pygame.key.set_mods(0)

    pygame.mouse.set_cursor(*cursor)

    return screen


def display_countup(window, font, t):
    minutes = abs(t // 1000) // 60
    seconds = abs(t // 1000) % 60

    minutes_str = "{:02d}".format(minutes)
    seconds_str = "{:02d}".format(seconds)
    t_str = minutes_str + ":" + seconds_str

    txt = font.render(t_str, 1, (255, 255, 255))

    window.blit(txt, (window.get_width() // 2 - txt.get_width() // 2,
                      window.get_height() // 2 - txt.get_height() // 2))


def main():
    pygame.init()
    info = pygame.display.Info()
    window = pygame.display.set_mode((info.current_w, info.current_h))

    toggle_fullscreen()
    program = True
    font = pygame.font.Font("consolab.ttf", 400)

    clock = pygame.time.Clock()
    start = -5000
    t = start
    tick = 0

    base_color = (20, 30, 40)
    colors = [(0, 0, 128), (117, 156, 0), (209, 192, 0), (209, 153, 0), (209, 0, 0)]
    delays = [start // 1000, 0, 45, 60, 75]
    current_delay = -1
    phase = "OFF"

    while program:
        for event in pygame.event.get():

            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                program = False
                pygame.quit()
                sys.exit()
                break

            if event.type == KEYDOWN and event.key == K_SPACE:
                if phase == "OFF":
                    t = start
                    current_delay = -1
                    phase = "ON"
                elif phase == "ON":
                    phase = "OFF"

        if current_delay < len(delays) - 1:
            if t > delays[current_delay + 1] * 1000:
                current_delay += 1
                window.fill(colors[current_delay])

        if phase == "OFF":
            window.fill(base_color)

        t += tick

        if phase == "ON":
            if current_delay == -1:
                window.fill(base_color)
            else:
                window.fill(colors[current_delay])
            display_countup(window, font, t)

        tick = clock.tick(30)
        pygame.display.flip()


if __name__ == '__main__':
    main()
