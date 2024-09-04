import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 50
CELL_SIZE = 9
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Colors
COLORS = {
    'player': GREEN,
    'flag': RED,
    'door': BLACK
}

# Initialize screen
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Grid Game')


def draw_grid():
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, WHITE, rect, 1)


def draw_player(position):
    x, y = position
    y = 49 - y
    pygame.draw.rect(screen, COLORS['player'], pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_flag(position):
    x, y = position
    y = 49 - y
    pygame.draw.rect(screen, COLORS['flag'], pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_door(doors):
    for (x, col, direction, is_open) in doors:
        y = 49-col
        if direction == 'left':
            start_pos = (x * CELL_SIZE, y * CELL_SIZE)
            end_pos = (x * CELL_SIZE, y * CELL_SIZE + CELL_SIZE)
        elif direction == 'right':
            start_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE)
            end_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE + CELL_SIZE)
        elif direction == 'up':
            start_pos = (x * CELL_SIZE, y * CELL_SIZE)
            end_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE)
        elif direction == 'down':
            start_pos = (x * CELL_SIZE, y * CELL_SIZE+CELL_SIZE)
            end_pos = (x * CELL_SIZE + CELL_SIZE, y * CELL_SIZE + CELL_SIZE)

        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)


def main():
    # Sample input
    player_position = (1, 1)
    flag_position = (9, 9)
    doors = [
        (1, 1, 'left', False)
    ]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(WHITE)
        draw_grid()
        draw_player(player_position)
        draw_flag(flag_position)
        draw_door(doors)

        pygame.display.flip()


if __name__ == "__main__":
    main()
