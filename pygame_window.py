import pygame
import numpy as np
import redis
import pickle

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def visualize():
    pygame.init()
    window_size = 500
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Minigrid Visualization")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if redis_client.llen('frames') > 0:
            frame = redis_client.rpop('frames')
            print("POPPIN'")
            img = pickle.loads(frame)
            surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            surface = pygame.transform.scale(surface, (window_size, window_size))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    visualize()
